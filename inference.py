from pathlib import Path
import numpy as np
import os
import glob
import torch
from PIL import Image
import pprint
from tqdm import tqdm
import h5py
from functools import partial

from hloc import extract_features, match_features, localize_inloc, visualization, pairs_from_retrieval, extractors, logger, matchers
from hloc.utils.io import list_h5_names
from hloc.extract_features import ImageDataset
from hloc.match_features import FeaturePairsDataset, WorkQueue, find_unique_new_pairs, writer_fn
from hloc.utils.base_model import dynamic_load
from hloc.utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval
from hloc.localize_inloc import create_transform

import matplotlib.pyplot as plt

DATASET = Path("datasets/lamar-hg/")  # change this if your dataset is somewhere else

QUERY_IMAGE_DIR = Path("temp/input/")
QUERY_IMAGE_PATH =  Path("processed_data/images")
QUERY_OUTPUT = Path("temp/output/")
QUERY_PAIRS = Path("temp/pairs/generated-pairs.txt")
QUERY_RESULT = Path("temp/output/InLoc_hloc_superpoint+superglue_netvlad40.txt")

DB_DESCRIPTORS = Path("outputs/lamar-hg/global-feats-netvlad.h5")
DB_FEATURE_PATH = Path("outputs/lamar-hg/feats-superpoint-n4096-r1600.h5")

FEATURE_CONF = extract_features.confs["superpoint_inloc"]
RETRIEVAL_CONF = extract_features.confs["netvlad"]
MATCHER_CONF = match_features.confs["superglue"]


os.makedirs(QUERY_IMAGE_DIR / QUERY_IMAGE_PATH, exist_ok = True)
os.makedirs(QUERY_PAIRS.parent, exist_ok = True)
os.makedirs(QUERY_OUTPUT, exist_ok = True)

class FeatureExtractor:
    @torch.no_grad()
    def __init__(self, conf, as_half = True, overwrite = False,) -> None:
        self.conf = conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, self.conf["model"]["name"])
        self.model = Model(self.conf["model"]).eval().to(self.device)
        self.as_half = as_half
        self.overwrite = overwrite

    @torch.no_grad()
    def extract_features(self, image_dir, export_dir):
        logger.info(
            "Extracting local features with configuration:" f"\n{pprint.pformat(self.conf)}"
        )

        # Looks like this is just a glob for all the images in image_dir
        dataset = ImageDataset(image_dir, self.conf["preprocessing"], None)
        feature_path = Path(export_dir, self.conf["output"] + ".h5")
        feature_path.parent.mkdir(exist_ok=True, parents=True)
        skip_names = set(
            list_h5_names(feature_path) if feature_path.exists() and not self.overwrite else ()
        )
        # and then skip_names is the ones that have already been processed and are in this .h5 file
        dataset.names = [n for n in dataset.names if n not in skip_names]
        if len(dataset.names) == 0:
            logger.info("Skipping the extraction.")
            return feature_path

        loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, pin_memory=True
        )
        for idx, data in enumerate(tqdm(loader)):
            name = dataset.names[idx]
            pred = self.model({"image": data["image"].to(self.device, non_blocking=True)})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

            pred["image_size"] = original_size = data["original_size"][0].numpy()
            if "keypoints" in pred:
                size = np.array(data["image"].shape[-2:][::-1])
                scales = (original_size / size).astype(np.float32)
                pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
                if "scales" in pred:
                    pred["scales"] *= scales.mean()
                # add keypoint uncertainties scaled to the original resolution
                uncertainty = getattr(self.model, "detection_noise", 1) * scales.mean()

            if self.as_half:
                for k in pred:
                    dt = pred[k].dtype
                    if (dt == np.float32) and (dt != np.float16):
                        pred[k] = pred[k].astype(np.float16)

            with h5py.File(str(feature_path), "a", libver="latest") as fd:
                try:
                    if name in fd:
                        del fd[name]
                    grp = fd.create_group(name)
                    for k, v in pred.items():
                        grp.create_dataset(k, data=v)
                    if "keypoints" in pred:
                        grp["keypoints"].attrs["uncertainty"] = uncertainty
                except OSError as error:
                    if "No space left on device" in error.args[0]:
                        logger.error(
                            "Out of disk space: storing features on disk can take "
                            "significant space, did you enable the as_half flag?"
                        )
                        del grp, fd[name]
                    raise error

            del pred
        logger.info("Finished exporting features.")
        return feature_path

class Matcher:
    @torch.no_grad()
    def __init__(self, conf, overwrite) -> None:
        self.conf = conf
        self.overwrite = overwrite

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(matchers, conf["model"]["name"])
        self.model = Model(conf["model"]).eval().to(self.device)

    @torch.no_grad()
    def match(self, pairs_path: Path, export_dir: Path, features: str, feature_path_ref: Path,):
        logger.info(
            "Matching local features with configuration:" f"\n{pprint.pformat(self.conf)}"
        )

        feature_path_q = Path(export_dir, features + ".h5")

        match_path = Path(export_dir, f'{features}_{self.conf["output"]}_{QUERY_PAIRS.stem}.h5')

        if not feature_path_q.exists():
            raise FileNotFoundError(f"Query feature file {feature_path_q}.")
        if not feature_path_ref.exists():
            raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
        match_path.parent.mkdir(exist_ok=True, parents=True)

        assert pairs_path.exists(), pairs_path
        pairs = parse_retrieval(pairs_path)
        pairs = [(q, r) for q, rs in pairs.items() for r in rs]
        pairs = find_unique_new_pairs(pairs, None if self.overwrite else match_path)
        if len(pairs) == 0:
            logger.info("Skipping the matching.")
            return
        
        dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=True
        )
        writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

        for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
            data = {
                k: v if k.startswith("image") else v.to(self.device, non_blocking=True)
                for k, v in data.items()
            }
            pred = self.model(data)
            pair = names_to_pair(*pairs[idx])
            writer_queue.put((pair, pred))
        writer_queue.join()
        logger.info("Finished exporting matches.")

        return match_path


retrieval_feature_extractor = FeatureExtractor(RETRIEVAL_CONF, overwrite=True)
feature_extractor = FeatureExtractor(FEATURE_CONF, overwrite=True)
matcher = Matcher(MATCHER_CONF, overwrite=True)

def localize_image(image, groundtruth):
    image_path = QUERY_IMAGE_DIR / QUERY_IMAGE_PATH / "image.jpg"
    Image.fromarray(image).save(image_path)
    results: dict = localize_image_(image_path, groundtruth)
    return results.get("q",None), results.get("t", None)

def localize_image_(image_path: Path, groundtruth):

    query_descriptors = retrieval_feature_extractor.extract_features(QUERY_IMAGE_DIR, QUERY_OUTPUT)
    
    pairs_from_retrieval.main(
        query_descriptors,
        QUERY_PAIRS,
        40,
        query_list=[image_path.relative_to(QUERY_IMAGE_DIR).as_posix()],
        db_descriptors=DB_DESCRIPTORS
    )

    # this removes all pairs found with sessions without global alignment
    global_timestamp_trajectory_map = create_timestamp_trajectory_map("./datasets/HGE/sessions/map/trajectories.txt") | create_timestamp_trajectory_map("./datasets/HGE/sessions/query_val_hololens/proc/alignment_trajectories.txt")
    global_aligment_sessions = set()
    for trajectory in global_timestamp_trajectory_map.values():
        folder = trajectory["device_id"].split(".")[0]
        global_aligment_sessions.add(folder)

    with open(QUERY_PAIRS, "r") as f:
        lines = f.readlines()
    with open(QUERY_PAIRS, "w") as f:
        for line in lines:
            session = line.split(' ')[-1].split('/')[0]
            if session in global_aligment_sessions:
                f.write(line)
            else:
                print("deleted line: ", line)


    query_features = feature_extractor.extract_features(QUERY_IMAGE_DIR, QUERY_OUTPUT)

    match_path = matcher.match(QUERY_PAIRS, QUERY_OUTPUT, FEATURE_CONF["output"], DB_FEATURE_PATH)

    results = localize_inloc.main(
        DATASET, QUERY_IMAGE_DIR, QUERY_PAIRS, query_features, DB_FEATURE_PATH, match_path, QUERY_RESULT, skip_matches=20
    )  # skip database images with too few matches

    image_name = image_path.relative_to(QUERY_IMAGE_DIR).as_posix()

    if image_name in results:
        points_3d = results[image_name]["3d_points"]
        tvec = results[image_name]["t"]

        global_timestamp_trajectory_map = create_timestamp_trajectory_map("./datasets/HGE/sessions/map/trajectories.txt") | create_timestamp_trajectory_map("./datasets/HGE/sessions/query_val_hololens/proc/alignment_trajectories.txt")
        global_xyz = np.array([[trajectory['tx'], trajectory['ty'], trajectory['tz']] for trajectory in global_timestamp_trajectory_map.values()])

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        # ax.plot(coorx, coory, coorz, markersize = 1, marker = 'o', alpha = 1, c = 'white', zorder = 0, linestyle = '', alpha = 1)
        ax.scatter([groundtruth[0]], [groundtruth[1]], [groundtruth[2]], c = 'red', zorder=4)
        ax.scatter([tvec[0]], [tvec[1]], [tvec[2]], c = 'green', zorder=3)
        ax.scatter(points_3d[:,0],points_3d[:,1],points_3d[:,2], c = 'orange', zorder=2, alpha=0.01) # these alpha values are necessary since scatter doesn't respect z-order
        ax.scatter(global_xyz[:,0], global_xyz[:,1], global_xyz[:,2], c = 'blue', zorder=1, alpha=0.01)
        
        plt.show()

    return results.get(image_name, {})


from hloc.image_timestamp_mapping import create_image_timestamp_map, create_timestamp_trajectory_map

test_folder = "./datasets/lamar-hg/hl_2020-12-13-10-20-30-996"

for img_path in glob.glob(os.path.join(test_folder, "processed_data/images/*.jpg"))[:10]:

    image_timestamp_map = create_image_timestamp_map(glob.glob(f"{test_folder}/**/images.txt", recursive=True)[0])
    timestamp_trajectory_map = create_timestamp_trajectory_map(glob.glob(f"{test_folder}/**/trajectories.txt", recursive=True)[0])
    image_name = os.path.join("hetlf", Path(img_path).name)
    image_timestamp = image_timestamp_map[image_name]["timestamp"]
    groundtruth_trajectory = timestamp_trajectory_map[image_timestamp]
    groundtruth_local = np.array([groundtruth_trajectory["tx"], groundtruth_trajectory["ty"], groundtruth_trajectory["tz"], 1])

    global_timestamp_trajectory_map = create_timestamp_trajectory_map("./datasets/HGE/sessions/map/trajectories.txt") | create_timestamp_trajectory_map("./datasets/HGE/sessions/query_val_hololens/proc/alignment_trajectories.txt")
    global_transform = None
    min_time_diff = float('inf')
    for timestamp, trajectory in global_timestamp_trajectory_map.items():
        folder, device_id = trajectory["device_id"].split("/", maxsplit=1)
        folder = folder.split(".")[0]
        if folder == Path(test_folder).name:
            curr_time_diff = abs(image_timestamp - timestamp)
            if curr_time_diff < min_time_diff:
                min_time_diff = curr_time_diff
                local_transform = create_transform(timestamp_trajectory_map[timestamp])
                global_transform = create_transform(trajectory) @ np.linalg.inv(local_transform) # make global_transform local -> global

    groundtruth_global = (global_transform @ groundtruth_local.T).T

    print("groundtruth, global transform")
    print(groundtruth_global[:3], global_transform)

    image = np.asarray(Image.open(img_path))
    q, t = localize_image(image, groundtruth_global[:3])
    print("prediction: ", q, t)
