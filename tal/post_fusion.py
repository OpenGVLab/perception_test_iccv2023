import collections
import os.path
import mmengine

import numpy as np
import torch
from libs.core import load_config
from libs.utils import batched_nms
from libs.datasets import make_dataset, make_data_loader
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed

result_root = "/mnt/petrelfs/chenguo/workspace/actionformer_release_PT/ckpt/"
config = "./configs/perception_tal_mm_valid.yaml"

result_file_list = [
    # pre-fusion
    "perception_tal_mm_train_o/eval_results_e25_inter",
    # single-feature
    "tal_umt_k700_train_o/eval_results_e35_inter",
]

all_segs = collections.defaultdict(list)
all_scores = collections.defaultdict(list)
all_labels = collections.defaultdict(list)
all_fps = collections.defaultdict(float)
all_durations = collections.defaultdict(float)
all_feat_strides = collections.defaultdict(float)
all_feat_num_frames = collections.defaultdict(float)

for result_file in result_file_list:
    results_path = os.path.join(result_root, result_file)
    # print(results_path)
    for file in mmengine.list_dir_or_file(results_path):
        file_name = file.split(".")[0]
        npz_file = os.path.join(results_path, file)
        npz = np.load(npz_file)
        # print(npz.files)
        segs = npz["segs"]
        scores = npz["scores"]
        labels = npz["labels"]
        fps = npz["fps"]
        vlen = npz["vlen"]
        stride = npz["stride"]
        nframes = npz["nframes"]

        # print(type(segs), segs.shape)
        all_fps[file_name] = fps
        all_durations[file_name] = vlen
        all_feat_strides[file_name] = stride
        all_feat_num_frames[file_name] = nframes
        all_segs[file_name].append(segs)
        all_scores[file_name].append(scores)
        all_labels[file_name].append(labels)
        # break

test_iou_threshold = 0.1
test_min_score = 0.001
test_max_seg_num = 500
test_nms_method = "soft"
test_multiclass_nms = True
test_nms_sigma = 0.4
test_voting_thresh = 0.9

processed_results = []
for video_name in all_segs.keys():
    segs = np.concatenate(all_segs[video_name], axis=0)
    scores = np.concatenate(all_scores[video_name], axis=0)
    labels = np.concatenate(all_labels[video_name], axis=0)

    segs = torch.from_numpy(segs)
    scores = torch.from_numpy(scores)
    labels = torch.from_numpy(labels)

    stride = all_feat_strides[video_name]
    nframes = all_feat_num_frames[video_name]
    fps = all_fps[video_name]
    vlen = all_durations[video_name]

    stride = float(stride)
    nframes = float(nframes)
    fps = float(fps)
    vlen = float(vlen)
    # print(video_name, segs.shape, scores.shape, labels.shape)
    # break
    segs, scores, labels = batched_nms(
        segs, scores, labels,
        test_iou_threshold,
        test_min_score,
        test_max_seg_num,
        use_soft_nms=(test_nms_method == 'soft'),
        multiclass=test_multiclass_nms,
        sigma=test_nms_sigma,
        voting_thresh=test_voting_thresh
    )
    if segs.shape[0] > 0:
        segs = (segs * stride + 0.5 * nframes) / fps
        # truncate all boundaries within [0, duration]
        segs[segs <= 0.0] *= 0.0
        segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen

    # 4: repack the results
    processed_results.append(
        {'video_id': video_name,
         'segments': segs,
         'scores': scores,
         'labels': labels}
    )

# format anet
results = {
    'video-id': [],
    't-start': [],
    't-end': [],
    'label': [],
    'score': []
}

# unpack the results into ANet format
num_vids = len(processed_results)
for vid_idx in range(num_vids):
    if processed_results[vid_idx]['segments'].shape[0] > 0:
        results['video-id'].extend(
            [processed_results[vid_idx]['video_id']] *
            processed_results[vid_idx]['segments'].shape[0]
        )
        results['t-start'].append(processed_results[vid_idx]['segments'][:, 0])
        results['t-end'].append(processed_results[vid_idx]['segments'][:, 1])
        results['label'].append(processed_results[vid_idx]['labels'])
        results['score'].append(processed_results[vid_idx]['scores'])

# gather all stats and evaluate 2
results['t-start'] = torch.cat(results['t-start']).numpy()
results['t-end'] = torch.cat(results['t-end']).numpy()
results['label'] = torch.cat(results['label']).numpy()
results['score'] = torch.cat(results['score']).numpy()

if os.path.isfile(config):
    cfg = load_config(config)
else:
    raise ValueError("Config file does not exist.")

val_dataset = make_dataset(
    cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
)
# set up evaluator
det_eval = ANETdetection(
    val_dataset.json_file,
    val_dataset.task,
    val_dataset.label_dict,
    val_dataset.split[0],
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]
)

_, mAP, _ = det_eval.evaluate(results, verbose=True)
