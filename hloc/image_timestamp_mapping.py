import os
import pandas as pd

from dataclasses import dataclass
from numpy import ndarray

@dataclass
class CaptureData:
    image_timestamp_map: pd.DataFrame
    timestamp_trajectory_map: pd.DataFrame
    depth_lut: any
    camera_to_rig: ndarray

def create_image_timestamp_map(path_to_image_file):
    df = pd.read_csv(path_to_image_file, delimiter=',', names=['timestamp', 'device_id', 'image_name'], skiprows=1, skipinitialspace=True)
    df['image_name'] = df['image_name'].apply(lambda x: os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x)))
    df = df.set_index('image_name')
    return df.to_dict('index')

def create_timestamp_trajectory_map(path_to_trajectory_file):
    df = pd.read_csv(path_to_trajectory_file, delimiter=',', names=['timestamp', 'device_id', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'], skiprows=1, skipinitialspace=True)
    df = df.set_index(['timestamp'])
    return df.to_dict('index')

def get_depths(path_to_depths_txt_file, skiprows=0):
    df = pd.read_csv(path_to_depths_txt_file, delimiter=',', names=['timestamp', 'label', 'image_name'], skipinitialspace=True, skiprows=skiprows)
    df['image_name'] = df['image_name'].apply(lambda x: os.path.basename(x))
    return df['image_name'].to_numpy()

def create_rig_transform_map(path_to_rig_file):
    df = pd.read_csv(path_to_rig_file, delimiter=',', names=['device', 'sensor', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'], skiprows=1, skipinitialspace=True)
    df = df.set_index(['sensor'])
    return df.to_dict('index')

# This stuff was valid for lamar data
# def create_image_timestamp_map(path_to_image_file):
#     df = pd.read_csv(path_to_image_file, delimiter=',', names=['timestamp', 'device_id', 'image_name'], skiprows=1, skipinitialspace=True)
#     df['image_name'] = df['image_name'].apply(lambda x: os.path.basename(x))
#     df = df.set_index('image_name')
#     return df.to_dict('index')

# def create_timestamp_trajectory_map(path_to_trajectory_file):
#     df = pd.read_csv(path_to_trajectory_file, delimiter=',', skipinitialspace=True)
#     df = df.rename(columns={'# timestamp': 'timestamp', '*covar': 'covar'})
#     df = df.set_index(['timestamp', 'device_id'])
#     return df.to_dict('index')