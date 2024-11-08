from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, localize_inloc, visualization
from hloc import pairs_from_retrieval

dataset = Path("datasets/lamar/")  # change this if your dataset is somewhere else

pairs = Path("pairs/lamar/")
loc_pairs = pairs / "generated-pairs.txt"

outputs = Path("outputs/lamar/")  # where everything will be saved
results = outputs / "InLoc_hloc_superpoint+superglue_netvlad40.txt"  # the result file

# list the standard configurations available
print(f"Configs for feature extractors:\n{pformat(extract_features.confs)}")
print(f"Configs for feature matchers:\n{pformat(match_features.confs)}")

# pick one of the configurations for extraction and matching
# you can also simply write your own here!
feature_conf = extract_features.confs["superpoint_inloc"]
matcher_conf = match_features.confs["superglue"]
retrieval_conf = extract_features.confs["netvlad"]
reference_sfm = outputs / "sfm_superpoint+superglue"  # the SfM model we will build

global_descriptors = extract_features.main(retrieval_conf, dataset, outputs)

pairs_from_retrieval.main(
    global_descriptors,
    loc_pairs,
    5,
    query_prefix="fake",
    db_prefix="hl_2022-01-14-14-38-01-427/raw_data/images/hetlf/"
)

feature_path = extract_features.main(feature_conf, dataset, outputs)

match_path = match_features.main(
    matcher_conf, loc_pairs, feature_conf["output"], outputs
)

localize_inloc.main(
    dataset, loc_pairs, feature_path, match_path, results, skip_matches=1
)  # skip database images with too few matches

visualization.visualize_loc(results, dataset, n=1, top_k_db=1, seed=2)
