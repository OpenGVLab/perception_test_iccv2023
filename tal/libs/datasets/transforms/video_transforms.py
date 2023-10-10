import random
import warnings

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo, RandomResizedCropVideo, \
    RandomHorizontalFlipVideo, RandomCropVideo, ToTensorVideo

NormalizeVideo = NormalizeVideo
CenterCropVideo = CenterCropVideo
RandomResizedCropVideo = RandomResizedCropVideo
RandomHorizontalFlipVideo = RandomHorizontalFlipVideo
RandomCropVideo = RandomCropVideo
ToTensorVideo = ToTensorVideo


class TemporalResize(torch.nn.Module):

    def __init__(self, temporal_scale):
        super().__init__()

        self.temporal_scale = temporal_scale

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): 4D tensor  (C T H W)

        Returns:
            tensor (Tensor): 4D tensor
        """
        if isinstance(self.temporal_scale, list) or isinstance(self.temporal_scale, tuple):
            tscale = random.choice(self.temporal_scale)
        elif isinstance(self.temporal_scale, int):
            tscale = self.temporal_scale
        else:
            raise NotImplementedError

        sscale = tensor.shape[-2:]
        tensor = F.interpolate(tensor.unsqueeze(0), size=(tscale, *sscale), mode="trilinear").squeeze(0)

        return tensor
