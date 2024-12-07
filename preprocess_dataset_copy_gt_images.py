import argparse
from natsort import natsorted
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from hloc.image_timestamp_mapping import create_image_timestamp_map, create_timestamp_trajectory_map, get_depths
# from hloc.localize_inloc import depthmap_img_to_points_3d

camera = "hetlf"
intermediate_path = f"raw_data/images/{camera}"

def get_gt_folders(gt_file_path: str):
    df = pd.read_csv(gt_file_path, delimiter=',', names=['timestamp', 'device_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'], skiprows=1, skipinitialspace=True)
    return set(df['device_id'].apply(lambda x: x.split(".")[0]))

def get_gt_filenames(gt_file_path: str):
    df = pd.read_csv(gt_file_path, delimiter=',', names=['timestamp', 'device_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'], skiprows=1, skipinitialspace=True)
    df['device_id'] = df['device_id'].apply(lambda x: x.split(".")[0])
    df['timestamp'] = df['timestamp'].astype(str)
    paths = set(df.apply(lambda row: os.path.join(row['device_id'], intermediate_path, f"{row['timestamp']}.jpg"), axis=1))
    return paths

def get_gt_image_timestamps(gt_filenames, session_dir):
    timestamps = []
    for f in gt_filenames:
        base_dir = f.split("/")[0]
        if base_dir == os.path.basename(session_dir):
            timestamp = os.path.basename(f).split(".")[0]
            timestamps.append(timestamp)
    return natsorted(timestamps)

def process_session(session_dir, gt_folders, gt_filenames):

    raw_folder = os.path.join(session_dir, "raw_data")
    depth_file = os.path.join(session_dir, "depths.txt")
    if not os.path.isfile(depth_file) or not os.path.isdir(raw_folder):
        print(f"session {session_dir} is invalid")
        return
    if os.path.basename(session_dir) not in gt_folders:
        print(f"No ground truth poses for {session_dir}")
        return
    gt_image_timestamps = get_gt_image_timestamps(gt_filenames, session_dir)

    processed_data_path = os.path.join(session_dir, "processed_data")
    processed_image_path = os.path.join(processed_data_path, "images")
    processed_depth_path = os.path.join(processed_data_path, "depth")
    # processed_pointcloud_path = os.path.join(processed_data_path, "pointcloud")


    for path in [processed_data_path, processed_image_path, processed_depth_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # depth_lut = np.load(os.path.join(session_dir, "depth_LUT.npy"))
    for i in gt_image_timestamps:
        image_path = os.path.join(session_dir, "raw_data", "images", camera, f"{i}.jpg")
        # shutil.copy(depth_path, processed_depth_path)
        # shutil.copy(image_path, processed_image_path)
        print(f"Copying {image_path} to {processed_image_path}")

    return

def main(args):
    gt_folders = set.union(*[get_gt_folders(f) for f in args["gt"]])
    gt_filenames = set.union(*[get_gt_filenames(f) for f in args["gt"]])
    session_paths = [f.path for f in os.scandir(args["dataset"]) if f.is_dir()]
    for session_dir in session_paths:
        print(f"processing {session_dir}")
        process_session(session_dir, gt_folders, gt_filenames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gt", type=Path, nargs='+', required=True)
    args = parser.parse_args()
    main(args.__dict__)