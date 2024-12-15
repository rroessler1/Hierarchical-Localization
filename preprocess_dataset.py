import argparse
import os
import shutil
from pathlib import Path
import numpy as np
from hloc.image_timestamp_mapping import create_image_timestamp_map, create_timestamp_trajectory_map, get_depths
# from hloc.localize_inloc import depthmap_img_to_points_3d

camera = "hetlf"

def process_session(session_dir):

    raw_folder = os.path.join(session_dir, "raw_data")
    depth_file = os.path.join(session_dir, "depths.txt")
    if not os.path.isfile(depth_file) or not os.path.isdir(raw_folder):
        print(f"session {session_dir} is invalid")
        return
    image_timestamp_map = create_image_timestamp_map(os.path.join(session_dir, "images.txt"))

    timestamp_to_image = {}
    for image, value in image_timestamp_map.items():
        if os.path.dirname(image) == camera:
            timestamp_to_image[int(value["timestamp"])] = image
    image_timestamps = list(timestamp_to_image.keys())
    depth_filenames = get_depths(depth_file)
    depth_to_image = {}
    for depth_filename in depth_filenames:
        depth_time = int(os.path.basename(depth_filename).split(".")[0])
        i = np.searchsorted(image_timestamps, depth_time)
        # if time is closer to the left edge
        if i == len(image_timestamps) or (i > 0 and depth_time - image_timestamps[i-1] < image_timestamps[i] - depth_time):
            i -= 1
        depth_to_image[depth_filename] = timestamp_to_image[image_timestamps[i]]

    processed_data_path = os.path.join(session_dir, "processed_data")
    processed_image_path = os.path.join(processed_data_path, "images")
    processed_depth_path = os.path.join(processed_data_path, "depth")
    # processed_pointcloud_path = os.path.join(processed_data_path, "pointcloud")


    for path in [processed_data_path, processed_image_path, processed_depth_path]:
        if not os.path.exists(path):
            os.makedirs(path)


    # depth_lut = np.load(os.path.join(session_dir, "depth_LUT.npy"))

    for i, (depth, image) in enumerate(depth_to_image.items()):
        depth_path = os.path.join(session_dir, "raw_data", "depths", depth)
        image_path = os.path.join(session_dir, "raw_data", "images", image)
        # point_cloud = depthmap_img_to_points_3d(depth_path, depth_lut)

        shutil.copy(depth_path, processed_depth_path)
        shutil.copy(image_path, processed_image_path)
        # np.save(os.path.join(processed_pointcloud_path, f"{Path(image_path).stem}.npy"), point_cloud)

    return

def main(args):
    session_paths = [f.path for f in os.scandir(args["dataset"]) if (f.is_dir() and "hl_" in f.path)]
    for session_dir in session_paths:
        print(f"processing {session_dir}")
        process_session(session_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    main(args.__dict__)