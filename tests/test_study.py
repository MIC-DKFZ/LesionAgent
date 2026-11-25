"""Loading scans: unique case ids and the checks on what the tools can read."""
import numpy as np
import pytest
import SimpleITK as sitk
from conftest import write_ct

from lesionagent.core.study import Study
from lesionagent.reader import task_text


def test_case_ids_are_unique(tmp_path):
    for d in ("baseline", "followup"):
        (tmp_path / d).mkdir()
    same = [write_ct(tmp_path / d / "ct.nii.gz") for d in ("baseline", "followup")]
    assert [v.case_id for v in Study.from_files(same).visits] == ["v1_ct", "v2_ct"]
    dotted = [write_ct(tmp_path / n) for n in ("1.2.840.1.nii.gz", "1.2.840.2.nii.gz")]
    assert [v.case_id for v in Study.from_files(dotted).visits] == ["1.2.840.1", "1.2.840.2"]


def test_days_and_task_text(tmp_path, study):
    assert task_text(study) == ("Perform a complete RECIST 1.1 assessment of this patient "
                                "(3 visits: v1 day 0, v2 day 60, v3 day 120).")
    files = [v.path for v in study.visits]
    assert task_text(Study.from_files(files)).endswith("(3 visits: v1, v2, v3).")
    with pytest.raises(ValueError, match="3 image"):
        Study.from_files(files, days=[0, 60])
    with pytest.raises(ValueError, match="chronological"):
        Study.from_files(files, days=[0, 120, 60])
    with pytest.raises(FileNotFoundError):
        Study.from_files([tmp_path / "missing.nii.gz"])


def test_refuses_what_the_tools_cannot_read(tmp_path):
    sagittal = write_ct(tmp_path / "sag.nii.gz", direction=(0, 0, 1, 1, 0, 0, 0, 1, 0))
    with pytest.raises(ValueError, match="axial orientation"):
        Study.from_files([sagittal])
    flipped = write_ct(tmp_path / "flip.nii.gz", direction=(-1, 0, 0, 0, -1, 0, 0, 0, -1))
    Study.from_files([flipped])  # flipped axes are fine: the views label them
    series = sitk.GetImageFromArray(np.zeros((2, 4, 8, 8), np.int16), isVector=False)
    sitk.WriteImage(series, str(tmp_path / "4d.nii.gz"))
    with pytest.raises(ValueError, match="one 3D volume"):
        Study.from_files([tmp_path / "4d.nii.gz"])


def test_warns_when_values_are_not_hounsfield_units(tmp_path):
    st = Study.from_files([write_ct(tmp_path / "mr.nii.gz", value=300)])
    with pytest.warns(UserWarning, match="Hounsfield"):
        st.volume(1)
