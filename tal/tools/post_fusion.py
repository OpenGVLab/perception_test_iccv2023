import os.path
import mmengine

import numpy as np

from libs.utils import batched_nms

result_root = "/mnt/petrelfs/chenguo/workspace/actionformer_release_PT/ckpt/"

result_file_list = [
    # pre-fusion
    "perception_tal_mm_train_o/eval_results_e25_inter",
    # single-feature
    "tal_umt_k700_train_o/eval_results_e35_inter",
]

all_segs = dict()
all_scores = dict()
all_labels = dict()
all_fps = dict()
all_durations = dict()
all_feat_strides = dict()
all_feat_num_frames = dict()

for result_file in result_file_list:
    results_path = os.path.join(result_root, result_file)
    print(results_path)
    for file in mmengine.list_dir_or_file(results_path):
        file_name = file.split(".")[0]
        npz_file = os.path.join(results_path, file)
        npz = np.load(npz_file)
        segs = npz["segs"]
        scores = npz["scores"]
        labels = npz["labels"]
        fps = npz["fps"]
        durations = npz["durations"]
        feat_strides = npz["feat_strides"]
        print(file_name, segs.shape, scores.shape, labels.shape, fps.shape, durations.shape, feat_strides.shape)
        break


