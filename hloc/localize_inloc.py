import argparse
import pickle
from pathlib import Path

import cv2
import glob
import h5py
import itertools
import numpy as np
import os
import pycolmap
import torch
import traceback

from gluefactory.utils import image
from gluefactory.utils.image import ImagePreprocessor
from scipy.io import loadmat
from scipy.ndimage import zoom
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from . import logger
from .utils.parsers import names_to_pair, parse_retrieval
from .image_timestamp_mapping import create_image_timestamp_map, create_timestamp_trajectory_map, create_rig_transform_map, CaptureData


def create_transform(trajectory):
    rotation = R.from_quat([trajectory['qw'], trajectory['qx'], trajectory['qy'], trajectory['qz']], scalar_first=True).as_matrix()
    translation = np.array([trajectory['tx'], trajectory['ty'], trajectory['tz']])

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform

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

def create_normalized_lut(width, height):
    """
    Create a W x H x 2 NumPy array where each (i, j) entry contains the
    normalized coordinates of the image pixel (x, y) in the range [-1, 1].

    Args:
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        np.ndarray: A (W x H x 2) array, where each pixel position
                    holds its normalized coordinates [x_norm, y_norm].
    """
    # Create a grid of pixel indices for (x, y) coordinates
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Normalize x and y to the range [-1, 1]
    x_norm = 2 * x_coords / (width - 1) - 1  # Normalize x to [-1, 1]
    y_norm = 2 * y_coords / (height - 1) - 1  # Normalize y to [-1, 1]

    # Stack the x and y coordinates along a new axis to get W x H x 2 array
    normalized_coords = np.stack([x_norm, y_norm], axis=2)  # Shape (W, H, 2)

    return normalized_coords

def depthmap_img_to_points_3d(depthmap_path, lut, new_H, new_W):
    depthmap = np.array(
      cv2.imread(str(depthmap_path), cv2.IMREAD_ANYDEPTH),
      dtype=float
    )
    DEPTHMAP_SCALING_FACTOR = 1000  # mm to m
    depthmap /= DEPTHMAP_SCALING_FACTOR
    is_valid = np.logical_not(depthmap == -1)

    # for iOS
    if lut is None:
        lut = create_normalized_lut(width=depthmap.shape[1], height=depthmap.shape[0])

    ## Backproject to 3D.
    num_valid_pixels = np.sum(is_valid)
    valid_depths = depthmap[is_valid]
    normalized_pixels = lut[is_valid]
    normalized_pixels_hom = np.hstack([normalized_pixels, np.ones([num_valid_pixels, 1])])
    points_3d = normalized_pixels_hom * (
      valid_depths[:, np.newaxis] / np.linalg.norm(normalized_pixels_hom, axis=-1, keepdims=True)
    )
    points_3d = points_3d.reshape((*depthmap.shape, 3))

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

def get_3d_points_matching_2d_points(mkp3d, mkpr, from_hl):
    h, w, c = mkp3d.shape
    cols = mkpr[:, 0].astype(int)
    rows = mkpr[:, 1].astype(int)

    # Check bounds
    vertical_offset = 140 if from_hl else 0
    horiz_offset = 20 if from_hl else 0
    valid_mask = (rows >= 0 + vertical_offset) & (rows < h + vertical_offset) & (cols >= 0 + horiz_offset) & (cols < w + horiz_offset)
    rows[~valid_mask] = vertical_offset
    cols[~valid_mask] = horiz_offset

    # Select only valid points
    selected_points = mkp3d[rows - vertical_offset, cols - horiz_offset]
    valid_mask = np.logical_and(valid_mask, np.all(selected_points != 0, axis=1))
    return selected_points, valid_mask

def pose_from_cluster(dataset_dir, query_dir, q, q_feature_file, retrieved, feature_file, match_file, capture_data_dict, skip=None):
    height, width = cv2.imread(str(query_dir / q)).shape[:2]
    # cx = 0.5 * width
    # cy = 0.5 * height
    # TODO: These might be different in Magic Leap (These are for session hl_2020-12-13-10-20-30-996/hetlf)
    # cx = 235.031
    # cy = 288.286
    fx = 100
    fy = 100

    cx = 320
    cy = 240
    # fx = 363.32
    # fy = 360.299

    all_mkpq = []
    all_mkpr = []
    all_mkp3d = []
    all_indices = []
    kpq = q_feature_file[q]["keypoints"].__array__()
    num_matches = 0

    for i, r in enumerate(retrieved):
        file_name = os.path.basename(r)
        # folder_and_file_name = os.path.join(os.path.basename(os.path.dirname(r)), file_name)
        # FIXME: right now we're only using the hetlf, so this is ok, but it's kinda a hack
        leading_folder_name = Path(r).parts[0]
        capture_data: CaptureData = capture_data_dict[leading_folder_name]
        file_timestamp = int(file_name.split('.')[0])
        depth_filename = f"{file_timestamp}.png"
        try:
            trajectory = capture_data.timestamp_trajectory_map[file_timestamp]
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

            depth_file_path = os.path.join(dataset_dir, leading_folder_name, 'processed_data', 'depths', depth_filename)

            point_cloud = depthmap_img_to_points_3d(depth_file_path, capture_data.depth_lut, height, width)
            file_from_hololens = leading_folder_name[:2] == "hl"
            mkp3d, valid = get_3d_points_matching_2d_points(point_cloud, mkpr, file_from_hololens)
            if skip and (np.sum(valid) < skip):
                # print(f"skipping {r}, Found {np.sum(valid)} valid points")
                continue
            # print(f"processing {r}, Found {np.sum(valid)} valid points")


            mkpq = mkpq[valid]
            mkpr = mkpr[valid]
            mkp3d = mkp3d[valid]

            rig_to_global = create_transform(trajectory)

            mkp3d = np.hstack([mkp3d, np.ones((len(mkp3d),1))])
            mkp3d = (rig_to_global @ capture_data.camera_to_rig @ mkp3d.T).T[:,:3]

            all_mkpq.append(mkpq)
            all_mkpr.append(mkpr)
            all_mkp3d.append(mkp3d)
            all_indices.append(np.full(np.count_nonzero(valid), i))
        except Exception as e:
            print(traceback.format_exc())
            continue

    NUM_IMAGES_TO_USE = 3
    all_mkpq = sorted(all_mkpq, key=lambda x: x.shape[0], reverse=True)[:NUM_IMAGES_TO_USE]
    all_mkpr = sorted(all_mkpr, key=lambda x: x.shape[0], reverse=True)[:NUM_IMAGES_TO_USE]
    all_mkp3d = sorted(all_mkp3d, key=lambda x: x.shape[0], reverse=True)[:NUM_IMAGES_TO_USE]
    all_indices = sorted(all_indices, key=lambda x: x.shape[0], reverse=True)[:NUM_IMAGES_TO_USE]

    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    cfg = {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "params": [fx, fy, cx, cy],
    }

    # ret = pycolmap.absolute_pose_estimation(all_mkpq.astype(np.float64), all_mkp3d, pycolmap.Camera(**cfg), 48.00)
    ret = pycolmap.absolute_pose_estimation(all_mkpq, all_mkp3d, cfg, estimation_options={'ransac': {'max_error': 12.0}})
    ret["cfg"] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches


def main(dataset_dir, q_dir, retrieval, q_features, features, matches, results, skip_matches=None):
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())[:100]

    r_paths = list(itertools.chain(*retrieval_dict.values()))
    r_folders = list(set([Path(r).parts[0] for r in r_paths]))
    capture_data_dict = {}

    for folder in r_folders:
        image_timestamp_map = create_image_timestamp_map(glob.glob(f"{os.path.join(dataset_dir, folder)}/**/images.txt", recursive=True)[0])
        timestamp_trajectory_map = create_timestamp_trajectory_map(glob.glob(f"{os.path.join(dataset_dir, folder)}/**/global_trajectories.txt", recursive=True)[0])

        rig_file = f"{os.path.join(dataset_dir, folder)}/rigs.txt"
        if os.path.exists(rig_file):
            rig_transform_map = create_rig_transform_map(rig_file)
            camera_to_rig = create_transform(rig_transform_map["hetlf"])
        else:
            camera_to_rig = np.eye(4)

        depth_LUT_files = glob.glob(f"{os.path.join(dataset_dir, folder)}/**/depth_LUT.npy", recursive=True)
        depth_LUT = np.load(depth_LUT_files[0]) if len(depth_LUT_files) > 0 else None
        capture_data_dict[folder] = CaptureData(image_timestamp_map, timestamp_trajectory_map, depth_LUT, camera_to_rig)


    feature_file = h5py.File(features, "r", libver="latest")
    q_feature_file = h5py.File(q_features, "r", libver="latest")
    match_file = h5py.File(matches, "r", libver="latest")

    logs = {
        "query features": q_features,
        "features": features,
        "matches": matches,
        "retrieval": retrieval,
        "loc": {},
    }
    logger.info("Starting localization...")
    for q in tqdm(queries):
        db = retrieval_dict[q]
        try:
            ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_cluster(
                dataset_dir, q_dir, q, q_feature_file, db, feature_file, match_file, capture_data_dict, skip_matches
            )

            camera_transform =  ret["cam_from_world"].inverse()
            qvec = camera_transform.rotation.quat
            tvec = camera_transform.translation

            logs["loc"][q] = {
                "t": tvec,
                "q": qvec,
                "db": db,
                "PnP_ret": ret,
                "keypoints_query": mkpq,
                "keypoints_db": mkpr,
                "3d_points": mkp3d,
                "indices_db": indices,
                "num_matches": num_matches,
            }
        except ValueError:
            continue
        except KeyError:
            continue

    logger.info(f"Writing poses to {results}...")
    with open(results, "w") as f:
        for q in queries:
            if q in logs["loc"]:
                tvec = logs["loc"][q]["t"]
                qvec = logs["loc"][q]["q"]
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

    return logs["loc"]


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
