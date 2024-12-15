import argparse
from natsort import natsorted
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from hloc.image_timestamp_mapping import get_depths
# from hloc.localize_inloc import depthmap_img_to_points_3d

camera = "hetlf"
intermediate_path = f"raw_data/images/{camera}"

def extract_session_id(s: str):
    if s.startswith("ios"):
        return s.split('/')[0].rsplit('_', 1)[0]
    else:
        return s.rsplit(".")[0]

def create_gt_file(dataset_dir: str, gt_file_path: str) -> dict[str, dict[str, float]]:
    df = pd.read_csv(gt_file_path, delimiter=',', names=['timestamp', 'device_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'], skiprows=1, skipinitialspace=True)
    df['device_id'] = df['device_id'].apply(lambda x: extract_session_id(x))
    df.set_index('timestamp')
    for session, session_df in df.groupby('device_id'):
        if session.startswith("ios"):
            session_dir = os.path.join(dataset_dir, session)
            if not os.path.exists(session_dir):
                continue
            session_gt_file_path = os.path.join(dataset_dir, session, "global_trajectories.txt")
            session_df.to_csv(session_gt_file_path, index=False)

def get_gt_image_timestamps(gt_file_path):
    df = pd.read_csv(gt_file_path, delimiter=',', names=['timestamp', 'device_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'], skiprows=1, skipinitialspace=True)
    df = df[(df['tz'] > 0.9)]
    df = df[(df['ty'] > -50) & (df['ty'] < 20)]
    df = df[(df['tx'] > -10) & (df['tx'] < 65)]
    return natsorted(df['timestamp'])

def process_session(session_dir):
    is_ios = "ios_" in session_dir
    raw_folder = os.path.join(session_dir, "raw_data")
    depth_file = os.path.join(session_dir, "depths.txt")
    gt_file = os.path.join(session_dir, "global_trajectories.txt")

    if not os.path.isfile(depth_file) or not os.path.isdir(raw_folder):
        print(f"session {session_dir} is invalid")
        return
    if not os.path.exists(gt_file):
        print(f"No ground truth poses for {session_dir}")
        return
    gt_timestamps = get_gt_image_timestamps(gt_file)

    processed_data_path = os.path.join(session_dir, "processed_data")
    processed_image_path = os.path.join(processed_data_path, "images")
    processed_depth_path = os.path.join(processed_data_path, "depths")

    for path in [processed_data_path, processed_image_path, processed_depth_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    depth_files = get_depths(f"{session_dir}/depths.txt", skiprows=1 if is_ios else 0)
    depth_files = natsorted(depth_files)
    depth_timestamps = np.array([int(filename.split('.')[0]) for filename in depth_files], dtype=np.int64)
    depth_timestamps = natsorted(depth_timestamps)

    for t in gt_timestamps:
        image_path = os.path.join(session_dir, "raw_data", "images", f"{t}.jpg") if is_ios else os.path.join(session_dir, "raw_data", "images", camera, f"{t}.jpg")

        if not os.path.exists(image_path):
            print(f"{image_path} doesn't exist")
            continue

        processed_depth_filename = os.path.join(processed_depth_path, f"{t}.png")

        idx = np.searchsorted(depth_timestamps, t)
        if idx == len(depth_timestamps) or (idx > 0 and t - depth_timestamps[idx-1] < depth_timestamps[idx] - t):
            idx -= 1
        depth_path = os.path.join(session_dir, "raw_data", "depth" if is_ios else "depths", depth_files[idx])

        shutil.copy(image_path, processed_image_path)
        print(f"Copying {image_path} to {processed_image_path}")
        shutil.copy(depth_path, processed_depth_filename)
        print(f"Copying {depth_path} to {processed_depth_filename}")

    return

def main(args):
    for f in args["gt"]:
        create_gt_file(args["dataset"], f)

    session_paths = [f.path for f in os.scandir(args["dataset"]) if f.is_dir()]
    for session_dir in session_paths:
        print(f"processing {session_dir}")
        process_session(session_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gt", type=Path, nargs='+', required=True)
    args = parser.parse_args()
    main(args.__dict__)