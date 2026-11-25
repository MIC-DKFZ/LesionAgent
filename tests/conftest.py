"""Shared fixtures: a synthetic three-visit study (CPU only, no model weights)."""
import numpy as np
import pytest
import SimpleITK as sitk


def write_ct(path, shape=(12, 48, 48), spacing=(1.0, 1.0, 2.0), direction=None, value=-1000):
    img = sitk.GetImageFromArray(np.full(shape, value, np.int16))
    img.SetSpacing(spacing)
    if direction is not None:
        img.SetDirection(direction)
    sitk.WriteImage(img, str(path))
    return path


@pytest.fixture
def study(tmp_path):
    from lesionagent.core.study import Study

    return Study.from_files([write_ct(tmp_path / f"case_{i}.nii.gz") for i in range(3)], days=[0, 60, 120])
