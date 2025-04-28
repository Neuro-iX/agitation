"""Test function defined in `processing`."""

import torch
import torch.distributions as dist

from agitation import config
from agitation.processing import LoadVolume, SoftLabelToPred


def compute_prob(x):
    """Utility function to get a distribution for a value."""
    # Define Gaussian
    normal_dist = dist.Normal(x, 0.1)

    # Create bin edges
    bin_edges = torch.linspace(
        config.MOTION_BIN_RANGE[0], config.MOTION_BIN_RANGE[1], config.MOTION_N_BINS + 1
    )

    # Compute CDF at each edge
    cdf_vals = normal_dist.cdf(bin_edges)

    # Probability per bin = CDF difference
    return cdf_vals[1:] - cdf_vals[:-1]


def test_softlabeltopred():
    """Test `SoftLabelToPred`."""
    converter = SoftLabelToPred()

    should_be_one = compute_prob(1)
    should_be_four = compute_prob(4)
    should_be_zero = compute_prob(0)

    assert abs(converter(should_be_one) - 1) < 0.01
    assert abs(converter(should_be_four) - 4) < 0.01
    assert abs(converter(should_be_zero) - 0) < 0.01


def test_softlabeltopred_cuda():
    """Test `SoftLabelToPred` using gpu tensors."""
    converter = SoftLabelToPred()

    should_be_one = compute_prob(1).cuda()
    converted = converter(should_be_one)
    assert abs(converted - 1) < 0.01
    assert converted.device.type == "cpu"


def test_loadvolume():
    """Test `LoadVolume`"""
    load = LoadVolume()
    data = {
        "data": "tests/data/clinica/subjects/sub-000103/ses-headmotion2/t1_linear/"
        "sub-000103_ses-headmotion2_acq_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
    }
    loaded = load(data)
    vol = loaded["data"]
    assert vol.max() == 1
    assert vol.min() == 0
    assert tuple(vol.shape[1:]) == config.VOLUME_SHAPE
