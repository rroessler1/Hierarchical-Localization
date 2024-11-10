import argparse
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import os
import pycolmap
import torch
from gluefactory.utils import image
from gluefactory.utils.image import ImagePreprocessor
from scipy.io import loadmat
from scipy.ndimage import zoom
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from . import logger
from .utils.parsers import names_to_pair, parse_retrieval
from .image_timestamp_mapping import create_image_timestamp_map, create_timestamp_trajectory_map, get_depths


def interpolate_scan(scan, kp):
    h, w, c = scan.shape
    kp = kp / np.array([[w - 1, h - 1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None]
    scan = scan.double()
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(scan, kp, align_corners=True, mode="bilinear")[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        scan, kp, align_corners=True, mode="nearest"
    )[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    kp3d = interp.T.numpy()
    valid = valid.numpy()
    return kp3d, valid


def get_scan_pose(dataset_dir, rpath):
    split_image_rpath = rpath.split("/")
    floor_name = split_image_rpath[-3]
    scan_id = split_image_rpath[-2]
    image_name = split_image_rpath[-1]
    building_name = image_name[:3]

    path = Path(
        dataset_dir,
        "database/alignments",
        floor_name,
        f"transformations/{building_name}_trans_{scan_id}.txt",
    )
    with open(path) as f:
        raw_lines = f.readlines()

    P_after_GICP = np.array(
        [
            np.fromstring(raw_lines[7], sep=" "),
            np.fromstring(raw_lines[8], sep=" "),
            np.fromstring(raw_lines[9], sep=" "),
            np.fromstring(raw_lines[10], sep=" "),
        ]
    )

    return P_after_GICP

def depth_to_point_cloud(depth_image, fx, fy, cx, cy):
    """
    Convert a depth image to a 3D point cloud.

    Parameters:
    - depth_image (numpy.ndarray): 2D depth map with depth values in meters.
    - fx, fy: Focal lengths of the camera in pixels.
    - cx, cy: Optical center (principal point) of the camera in pixels.

    Returns:
    - points (numpy.ndarray): Array of 3D points of shape (N, 3).
    """
    h, w = depth_image.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # Convert depth values to real-world 3D coordinates
    Z = depth_image
    X = (i - cx) * Z / fx
    Y = (j - cy) * Z / fy

    # Stack into HxWx3 array of 3D coordinates
    points = np.stack((X, Y, Z), axis=-1) #.reshape(-1, 3)
    return points

    # # Filter out points with invalid depth
    # valid_points = points[~np.isnan(Z.flatten()) & (Z.flatten() > 0)]
    # return valid_points

def depthmap_img_to_points_3d(depthmap_path, lut):
    depthmap = np.array(
      cv2.imread(str(depthmap_path), cv2.IMREAD_ANYDEPTH),
      dtype=float
    )
    DEPTHMAP_SCALING_FACTOR = 1000  # mm to m
    depthmap /= DEPTHMAP_SCALING_FACTOR
    is_valid = np.logical_not(depthmap == -1)

    ## Backproject to 3D.
    num_valid_pixels = np.sum(is_valid)
    valid_depths = depthmap[is_valid]
    normalized_pixels = lut[is_valid]
    normalized_pixels_hom = np.hstack([normalized_pixels, np.ones([num_valid_pixels, 1])])
    points_3d = normalized_pixels_hom * (
      valid_depths[:, np.newaxis] / np.linalg.norm(normalized_pixels_hom, axis=-1, keepdims=True)
    )
    points_3d = points_3d.reshape((*depthmap.shape, 3))

    new_H, new_W = 432, 480
    # Compute zoom factors for each dimension
    zoom_factors = (new_H / points_3d.shape[0], new_W / points_3d.shape[1], 1)  # Keep 3rd dimension (2 channels) unchanged
    # Apply zoom with bilinear interpolation
    points_3d = zoom(points_3d, zoom_factors, order=1)  # order=1 for bilinear
    return points_3d

def load_and_rescale_depth_image(file_path):
    depth_image = image.load_image(file_path, grayscale=True)
    imsize = 480
    depth_process = ImagePreprocessor({'resize':imsize , "side": "long", "square_pad": False})
    new_depth_image = depth_process(depth_image)['image']
    return new_depth_image[0, :, :].numpy()

def get_3d_points_matching_2d_points(mkp3d, mkpr):
    h, w, c = mkp3d.shape
    cols = mkpr[:, 0].astype(int)
    rows = mkpr[:, 1].astype(int)

    # Check bounds
    vertical_offset = 140
    valid_mask = (rows >= 0 + vertical_offset) & (rows < h + vertical_offset) & (cols >= 0) & (cols < w)
    rows[~valid_mask] = vertical_offset
    cols[~valid_mask] = vertical_offset

    # Select only valid points
    selected_points = mkp3d[rows - vertical_offset, cols]
    return selected_points, valid_mask

def pose_from_cluster(dataset_dir, q, retrieved, feature_file, match_file, skip=None):
    height, width = cv2.imread(str(dataset_dir / q)).shape[:2]
    cx = 0.5 * width
    cy = 0.5 * height
    focal_length = 4032.0 * 28.0 / 36.0

    all_mkpq = []
    all_mkpr = []
    all_mkp3d = []
    all_indices = []
    kpq = feature_file[q]["keypoints"].__array__()
    num_matches = 0
    session_folder = "hl_2022-01-14-14-38-01-427"
    depth_folder = "raw_data/depths"

    # Reads the metadata files and ultimately creates a mapping from image name to the trajectory.
    image_timestamp_map = create_image_timestamp_map(os.path.join(dataset_dir, session_folder, "images.txt"))
    timestamp_trajectory_map = create_timestamp_trajectory_map(os.path.join(dataset_dir, session_folder, "trajectories.txt"))
    depth_filenames = get_depths(os.path.join(dataset_dir, session_folder, "depths.txt"))
    depth_lut = np.load(os.path.join(dataset_dir, session_folder, "depth_LUT.npy"))

    for i, r in enumerate(retrieved):
        file_name = os.path.join(os.path.basename(os.path.dirname(r)), os.path.basename(r))
        depth_filename = depth_filenames[np.searchsorted(depth_filenames, os.path.basename(file_name))]
        try:
            trajectory = timestamp_trajectory_map[image_timestamp_map[file_name]['timestamp']]
            kpr = feature_file[r]["keypoints"].__array__()
            pair = names_to_pair(q, r)
            # m is an array of indices for which point from r matches to q
            # or -1 if there is no match
            m = match_file[pair]["matches0"].__array__()
            v = m > -1

            if skip and (np.count_nonzero(v) < skip):
                continue
            # so then, this gets all the valid match points from q, and the corresponding match from r
            mkpq, mkpr = kpq[v], kpr[m[v]]
            num_matches += len(mkpq)

            depth_file_path = os.path.join(dataset_dir, session_folder, depth_folder, depth_filename)
            point_cloud = depthmap_img_to_points_3d(depth_file_path, depth_lut)
            mkp3d, valid = get_3d_points_matching_2d_points(point_cloud, mkpr)
            # mkp3d = point_cloud # , valid = interpolate_scan(point_cloud, mkpr)
            # valid = [True] * mkpr.shape[0]
            quat = [trajectory['qw'], trajectory['qx'], trajectory['qy'], trajectory['qz']]
            translation = np.array([trajectory['tx'], trajectory['ty'], trajectory['tz']])
            # Tr = get_scan_pose(dataset_dir, r)
            mkp3d = (R.from_quat(quat).as_matrix() @ mkp3d.T + translation[:, np.newaxis]).T

            all_mkpq.append(mkpq[valid])
            all_mkpr.append(mkpr[valid])
            all_mkp3d.append(mkp3d[valid])
            all_indices.append(np.full(np.count_nonzero(valid), i))
        except Exception as e:
            print(repr(e))
            continue

    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    cfg = {
        "model": "SIMPLE_PINHOLE",
        "width": width,
        "height": height,
        "params": [focal_length, cx, cy],
    }
    ret = pycolmap.absolute_pose_estimation(all_mkpq, all_mkp3d, cfg, 48.00)
    ret["cfg"] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches


def main(dataset_dir, retrieval, features, matches, results, skip_matches=None):
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())[:100]

    feature_file = h5py.File(features, "r", libver="latest")
    match_file = h5py.File(matches, "r", libver="latest")

    poses = {}
    logs = {
        "features": features,
        "matches": matches,
        "retrieval": retrieval,
        "loc": {},
    }
    logger.info("Starting localization...")
    i = 0
    for q in tqdm(queries):
        db = retrieval_dict[q]
        if i > 100:
            break
        try:
            ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_cluster(
                dataset_dir, q, db, feature_file, match_file, skip_matches
            )

            poses[q] = (ret["qvec"], ret["tvec"])
            logs["loc"][q] = {
                "db": db,
                "PnP_ret": ret,
                "keypoints_query": mkpq,
                "keypoints_db": mkpr,
                "3d_points": mkp3d,
                "indices_db": indices,
                "num_matches": num_matches,
            }
            i += 1
        except ValueError:
            continue
        except KeyError:
            continue

    logger.info(f"Writing poses to {results}...")
    with open(results, "w") as f:
        for q in queries:
            if q in poses:
                qvec, tvec = poses[q]
                qvec = " ".join(map(str, qvec))
                tvec = " ".join(map(str, tvec))
                name = q.split("/")[-1]
                f.write(f"{name} {qvec} {tvec}\n")
            else:
                print(f"Couldn't localize {q}")

    logs_path = f"{results}_logs.pkl"
    logger.info(f"Writing logs to {logs_path}...")
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--skip_matches", type=int)
    args = parser.parse_args()
    main(**args.__dict__)
