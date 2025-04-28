"""Module containing pre and post processing functions."""

import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
)

from agitation import config


class SoftLabelToPred:
    """Converts model predictions to a single scalar."""

    def __init__(
        self,
        bin_range: tuple[float, float] = config.MOTION_BIN_RANGE,
        bin_number: int = config.MOTION_N_BINS,
    ):
        """
        Initialize.

        bin_range: (start, end), size-2 tuple
        bin_number: number of bins in the prediction
        """
        self.bin_start = bin_range[0]
        self.bin_end = bin_range[1]
        self.bin_number = bin_number
        self.bin_length = self.bin_end - self.bin_start
        self.bin_step = self.bin_length / self.bin_number
        self.bin_centers = (
            self.bin_start
            + float(self.bin_step) / 2
            + self.bin_step * torch.arange(self.bin_number)
        )

    def softmax_to_hardlabel(self, x: torch.Tensor) -> torch.Tensor:
        """Convert soft label to hard label.

        Args:
            x (torch.Tensor): Vector of soft label or single soft label

        Returns:
            torch.Tensor: Vector or single hard label
        """
        if x.get_device() != -1:
            x = x.cpu()
        pred = x @ self.bin_centers

        return pred

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call the `softmax_to_hardlabel` function."""
        return self.softmax_to_hardlabel(x)


class LoadVolume(Compose):
    """Transform to load data for inference."""

    def __init__(self):
        """Initialize basic loading."""
        keys = "data"
        tsf = [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=True),
            Orientationd(keys=keys, axcodes="RAS"),
            CenterSpatialCropd(keys=keys, roi_size=(160, 192, 160)),
            ScaleIntensityd(keys=keys, minv=0, maxv=1),
        ]
        super().__init__(tsf)
