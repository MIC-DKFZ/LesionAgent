"""The task of a read: the RECIST read by default, a built-in task, a text file or the text itself; and how the
command line ends a read without a RECIST report."""
from types import SimpleNamespace

import pytest
from conftest import write_ct

from lesionagent import cli, reader, tasks
from lesionagent.core.study import Study
from lesionagent.reader import task_text


@pytest.fixture
def scan(tmp_path):
    write_ct(tmp_path / "ct.nii.gz")
    return Study.from_files([tmp_path / "ct.nii.gz"])


def test_built_in_tasks(scan, study):
    assert tasks.names() == ["lesions", "lung-nodules"]
    for name in tasks.names():
        text = task_text(scan, name)
        assert "no RECIST report" in text and "  " not in text
        assert all("\n" not in paragraph for paragraph in text.split("\n\n"))  # the files are wrapped, the text is not
    assert "RECIST conventions do not apply" in task_text(scan, "lesions")
    with pytest.raises(ValueError, match="one scan"):
        task_text(study, "lesions")


def test_task_from_a_file_or_as_text(scan, tmp_path):
    (tmp_path / "task.txt").write_text("Segment the lesion at [5, 20, 20].\n")
    assert task_text(scan, str(tmp_path / "task.txt")) == "Segment the lesion at [5, 20, 20]."
    assert task_text(scan, "Segment the lesion at [5, 20, 20].") == "Segment the lesion at [5, 20, 20]."
    assert task_text(scan, "x" * 5000) == "x" * 5000
    assert task_text(scan).startswith("Perform a complete RECIST 1.1 assessment")


@pytest.mark.parametrize("task, report, finished, code", [(None, True, True, 0), (None, False, True, 1),
                                                           ("lesions", False, True, 0), ("lesions", False, False, 1)])  # fmt: skip
def test_command_line_ending(scan, tmp_path, monkeypatch, capsys, task, report, finished, code):
    """Only the default task must end with a RECIST report; another task ends with the lesions it found."""
    ws = SimpleNamespace(study=scan, registry=SimpleNamespace(lesions={}))
    if report:
        (tmp_path / "report.md").write_text("RECIST 1.1 report")
    result = reader.Read(tmp_path, {} if report else None, "", {"finished": finished}, ws)
    monkeypatch.setattr(reader, "read", lambda *a, **k: result)
    monkeypatch.setattr(cli, "_llm", lambda a: None)
    assert cli.main(["run", "--images", str(scan.visits[0].path)] + (["--task", task] if task else [])) == code
    out = capsys.readouterr()
    assert ("RECIST 1.1 report" in out.out) == report
    assert ("no lesions found" in out.out) == (task is not None and finished)
    assert ("did not finish" in out.err) == (code == 1)
