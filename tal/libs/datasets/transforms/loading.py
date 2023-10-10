import io
import os

import decord
import mmcv
import mmengine
import numpy as np
import torch
from decord import VideoReader
from mmengine import FileClient
from mmengine.registry import TRANSFORMS


class VideoChunkCollector:
    def __init__(self, chunk_video_dir):
        super().__init__()
        self.chunk_video_dir = chunk_video_dir

        meta = mmengine.load(os.path.join(chunk_video_dir, "meta.json"))

        files = []
        for file in mmengine.list_dir_or_file(chunk_video_dir):
            if file.endswith(".mp4"):
                files.append(file)

        self.num_chunks = len(files)
        self.chunk_size = meta["chunk_size"]
        self.origin_size = meta["origin_size"]
        self.origin_fps = meta["origin_fps"]
        self.chunk_fps = meta["chunk_fps"]
        self.chunk_duration = meta["chunk_duration"]
        self.total_duration = meta["total_duration"]
        self.chunk_num = meta["chunk_num"]

        self.chunk_num_frames = self.chunk_duration * self.chunk_fps

    def __len__(self):
        return int(self.total_duration * self.chunk_fps)

    def get_batch(self, frame_ids):
        pools = [list() for _ in range(self.num_chunks)]
        for id in frame_ids:
            chunk_id = id // self.chunk_num_frames
            frame_id = id % self.chunk_num_frames
            pools[chunk_id].append(frame_id)
        all_frames = []
        for i, pool in enumerate(pools):
            if len(pool) > 0:
                video_data = mmengine.get(os.path.join(self.chunk_video_dir, "%03d.mp4" % i))
                video_reader = decord.VideoReader(io.BytesIO(video_data))
                # print(self.chunk_video_dir, i, len(video_reader), frame_ids, pool)
                frames = video_reader.get_batch(pool).asnumpy()
                all_frames.append(frames)
        all_frames = np.concatenate(all_frames)
        return all_frames


@TRANSFORMS.register_module()
class LoadFramesDecord:
    def __init__(self, data_path=None, file_ext=".mp4", file_client=dict()):
        super().__init__()
        self.data_path = data_path
        self.file_ext = file_ext
        self.file_client = FileClient(**file_client)

    def __call__(self, video_name, start_frame=None, end_frame=None):
        if self.data_path is None:
            video_path = video_name
        else:
            video_path = os.path.join(self.data_path, video_name + self.file_ext)
        video_data = self.file_client.get(video_path)
        reader = VideoReader(io.BytesIO(video_data))

        frames = reader.get_batch(np.arange(len(reader))).asnumpy()
        frames = torch.Tensor(frames).float().permute(3, 0, 1, 2)  # c t h w
        return frames


@TRANSFORMS.register_module()
class LoadFramesDecordV2:
    def __init__(self, data_path=None, file_ext=".mp4", file_client=dict()):
        super().__init__()
        self.data_path = data_path
        self.file_ext = file_ext
        self.file_client = FileClient(**file_client)

    def __call__(self, video_name, start_frame=None, end_frame=None):
        if self.data_path is None:
            video_path = video_name
        else:
            video_path = os.path.join(self.data_path, video_name + self.file_ext)
        video_data = self.file_client.get(video_path)
        reader = VideoReader(io.BytesIO(video_data))

        total_frames = len(reader)

        frames = reader.get_batch(np.arange(len(reader))).asnumpy()
        frames = torch.Tensor(frames).float().permute(3, 0, 1, 2)  # c t h w
        return frames


@TRANSFORMS.register_module()
class LoadFramesDir:
    def __init__(self, data_path, file_temp="img_%05d.jpg", start_index=0, file_client=dict(), num_frames=None):
        super().__init__()
        self.data_path = data_path
        self.file_temp = file_temp
        self.start_index = start_index
        self.file_client = FileClient(**file_client)
        self.num_frames = num_frames

    def __call__(self, video_name):
        total_frames = len(os.listdir(os.path.join(self.data_path, video_name)))

        if self.num_frames:
            sampler = SampleFrames(clip_len=1, frame_interval=1, num_clips=self.num_frames)
            frame_ids = sampler(total_frames, self.start_index)
        else:
            frame_ids = self.start_index + np.arange(0, total_frames)

        frames = []
        for id in frame_ids:
            img_path = os.path.join(self.data_path, video_name, self.file_temp % id)
            img_bytes = self.file_client.get(img_path)
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            frames.append(cur_frame)
        frames = torch.Tensor(frames).float().permute(3, 0, 1, 2)  # c t h w
        return frames


class SampleFrames:
    """Sample frames from the video.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Defaults to 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Defaults to False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Defaults to False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Defaults to 'loop'.
        test_mode (bool): Store True when building tests or validation dataset.
            Defaults to False.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Defaults to False.
    """

    def __init__(self,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 test_mode: bool = False,
                 keep_tail_frames: bool = False,
                 **kwargs) -> None:

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames: int) -> np.array:
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int) -> np.array:
        """Get clip offsets in tests mode.

        If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in tests mode.
        """
        if self.clip_len == 1:  # 2D recognizer
            # assert self.frame_interval == 1
            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:  # 3D recognizer
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1
                offset_between = max_offset / float(num_segments)
                clip_offsets = np.arange(num_clips) * offset_between
                clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int) -> np.array:
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, total_frames: int, start_index=0, ) -> dict:
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        frame_inds = np.concatenate(frame_inds) + start_index

        return frame_inds
