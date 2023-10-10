import os
import json
import h5py
import numpy as np
import random
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations

@register_dataset("perception")
class ActivityNetDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json file for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len
        input_modality,   # input modality ['video', 'audio', 'multi']
        mm_feat_folder,    # if using multiple input modalities specify other 
        task                  #   modality features
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        
        if input_modality not in ['video', 'audio', 'multi']:
            raise ValueError('Input modality is not a valid modality.')
        if input_modality == 'multi' and mm_feat_folder is None:
            raise ValueError('Please specify a second feature folder for multimodal input.')
        self.input_modality = input_modality
        
        if task not in ['action_localisation', 'sound_localisation']:
            raise ValueError('Please select a valid task from ["action_localisation", "sound_localisation"]')
        self.task = task
        
        self.feat_folder = feat_folder
        self.mm_feat_folder = mm_feat_folder

        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        # proposal vs action categories
      
        self.data_list = dict_db #[0:100]
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'Perception Test',
            #'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': []
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_db = json.load(fid)

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value[self.task]:
                    act['label'] = act['label']
                    act['label_id'] = act['label_id']
                    act['segment'] = [x/1e6 for x in act['timestamps']]
                    del act['timestamps']
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['metadata']['split'].lower() not in self.split:
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            duration = value['metadata']['num_frames'] * value['metadata']['frame_rate']

            # get annotations if available
            if (len(value[self.task]) > 0):
                valid_acts = value[self.task] #remove_duplicate_annotations(value[self.task])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    if self.num_classes == 1:
                        labels[idx] = 0
                    else:
                        labels[idx] = label_dict[act['label']]
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                          'fps' : fps,
                          'duration' : duration,
                          'segments' : segments,
                          'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        for i in range(0, 50):
            try:
                video_item = self.data_list[idx]
                filename = os.path.join(self.feat_folder + '/'
                                            + video_item['id'] + self.file_ext)
                feats = np.load(filename).astype(np.float32)
                if self.input_modality == 'multi': 

                    # mm_filename = os.path.join(self.mm_feat_folder + '/'+ video_item['id'] + self.file_ext)
                    # mm_feats = np.load(mm_filename).astype(np.float32)
                    mm_filename = os.path.join(self.mm_feat_folder + '/'+ video_item['id'] + '.pt')
                    mm_feats = torch.load(mm_filename).numpy().astype(np.float32)
                    x = np.array(range(mm_feats.shape[0]))
                    xnew = np.linspace(x.min(), x.max(), feats.shape[0])
                    # apply the interpolation to each column
                    f = interp1d(x, mm_feats, axis=0)
                    mm_feats = f(xnew)
                    feats = np.concatenate([feats,mm_feats],axis=1).astype(np.float32)

                # we support both fixed length features / variable length features
                # case 1: variable length features for training
                if self.feat_stride > 0 and (not self.force_upsampling):
                    # var length features
                    feat_stride, num_frames = self.feat_stride, self.num_frames
                    # only apply down sampling here
                    if self.downsample_rate > 1:
                        feats = feats[::self.downsample_rate, :]
                        feat_stride = self.feat_stride * self.downsample_rate
                # case 2: variable length features for input, yet resized for training
                elif self.feat_stride > 0 and self.force_upsampling:
                    feat_stride = float(
                            (feats.shape[0] - 1) * self.feat_stride + self.num_frames
                    ) / self.max_seq_len
                    # center the features
                    num_frames = feat_stride
                # case 3: fixed length features for input
                else:
                    # deal with fixed length feature, recompute feat_stride, num_frames
                    seq_len = feats.shape[0]
                    assert seq_len <= self.max_seq_len
                    if self.force_upsampling:
                        # reset to max_seq_len
                        seq_len = self.max_seq_len
                    feat_stride = video_item['duration'] * video_item['fps'] / seq_len
                    # center the features
                    num_frames = feat_stride
                feat_offset = 0.5 * num_frames / feat_stride

                # T x C -> C x T
                feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

                # resize the features if needed
                if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                    resize_feats = F.interpolate(
                        feats.unsqueeze(0),
                        size=self.max_seq_len,
                        mode='linear',
                        align_corners=False
                    )
                    feats = resize_feats.squeeze(0)
                # convert time stamp (in second) into temporal feature grids
                # ok to have small negative values here
                if video_item['segments'] is not None:
                    segments = torch.from_numpy(
                        video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
                    )
                    labels = torch.from_numpy(video_item['labels'])
                else:
                    segments, labels = None, None
                # return a data dict
                data_dict = {'video_id'        : video_item['id'],
                            'feats'           : feats,      # C x T
                            'segments'        : segments,   # N x 2
                            'labels'          : labels,     # N
                            'fps'             : video_item['fps'],
                            'duration'        : video_item['duration'],
                            'feat_stride'     : feat_stride,
                            'feat_num_frames' : num_frames}
                # no truncation is needed
                # truncate the features during training
                if self.is_training and (segments is not None):
                    data_dict = truncate_feats(
                        data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
                    )
                return data_dict
            except Exception as e:
                # print(e, flush=True)
                idx = random.randint(0, len(self.data_list))
        raise NotImplementedError

