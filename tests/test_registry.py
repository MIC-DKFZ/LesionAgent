"""Registry / RECIST engine / target selection edge cases on synthetic data (CPU only)."""
import numpy as np
import pytest

from lesionagent.core import recist
from lesionagent.core.registry import LesionRegistry


def _box(shape, z, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[z, y0:y1, x0:x1] = True
    return m


def _target(reg, st, ld_px=(20, 30)):
    """A liver target measured at every visit: a rectangle of ld_px[k] x 6 voxels on one slice."""
    l = reg.new_lesion("t", "tumor", "liver", "target")
    shape = st.volume(1).shape
    for k, v in enumerate(st.visits[:2]):
        reg.set_mask(l.id, v, _box(shape, 5, 10, 10 + ld_px[k], 10, 16))
        reg.finding(l.id, v).status = "present"
        reg.remeasure(l.id, v)
    return l


def test_report_is_repeatable_and_does_not_judge_visits(study):
    reg = LesionRegistry(study)
    _target(reg, study, ld_px=(20, 20))  # SD by the targets: PD only through the new lesion
    new = reg.new_lesion("n", "tumor", "liver", "new")
    reg.finding(new.id, 2).status = "present"
    first = reg.report()
    assert reg.judgements == {}  # reading the report records no judgement
    assert reg.report() == first
    assert first["timepoints"][0]["new_lesion"] and first["timepoints"][0]["overall"] == "PD"


def test_present_target_without_mask_is_not_assessed(study):
    reg = LesionRegistry(study)
    l = _target(reg, study)
    v3 = study.visit(3)
    reg.finding(l.id, v3).status = "present"
    m = reg.remeasure(l.id, v3)  # empty mask
    assert m["n_voxels"] == 0
    base, visits, _ = reg.engine_inputs()
    assert visits[1].targets[0].status == "not_assessed"
    tp = reg.report()["timepoints"][1]
    assert tp["target_response"] == "NE" and tp["overall"] == "NE" and tp["sld_mm"] is None
    # no measurement at all, or a zero measurement handed to the engine directly: NE as well, never CR/PR
    reg.lesions[l.id].findings[3].measurement = {}
    assert reg.report()["timepoints"][1]["target_response"] == "NE"
    v = recist.Visit(2, 60, [recist.TargetMeasurement(1, False, "present", ld_mm=0.0)])
    assert recist.evaluate([recist.TargetMeasurement(1, False, ld_mm=30)], [v])[0].target_response == "NE"
    # absent still counts 0 mm
    reg.lesions[l.id].findings[3].status = "absent"
    assert reg.report()["timepoints"][1]["target_response"] == "CR"


def test_lesion_ids_are_not_reused(study, tmp_path):
    reg = LesionRegistry(study)
    a, b = reg.new_lesion(), reg.new_lesion()
    reg.delete(b.id)
    c = reg.new_lesion()
    assert (a.id, b.id, c.id) == (1, 2, 3)
    reg.delete(c.id)
    reg.save(tmp_path / "out")
    back = LesionRegistry.load(study, tmp_path / "out")
    assert back.new_lesion().id == 4
    # registry.json without a counter: continue after the largest id
    import json

    f = tmp_path / "out" / "registry.json"
    d = json.loads(f.read_text())
    d.pop("next_id")
    f.write_text(json.dumps(d))
    assert LesionRegistry.load(study, tmp_path / "out").new_lesion().id == 2
    # ids set from outside (imported label maps) are respected
    reg.lesions[10] = reg.lesions.pop(a.id)
    reg.lesions[10].id = 10
    assert reg.new_lesion().id == 11


def test_gpu_models_stay_resident_and_evict_each_other_on_oom(monkeypatch):
    from lesionagent.backends import models

    class OutOfMemoryError(RuntimeError):
        pass

    monkeypatch.setattr(models, "_MODELS", {})
    monkeypatch.setattr(models, "EXCLUSIVE", False)
    for n in ("a", "b", "c"):
        models.register(n, lambda n=n: type(n, (), {})())
    models.get("a"), models.get("b")
    assert models.loaded() == ["a", "b"]  # resident side by side
    calls = []

    def step(fail_times):
        def fn(model):
            calls.append(models.loaded())
            if len(calls) <= fail_times:
                raise RuntimeError("wrapped") from OutOfMemoryError("CUDA out of memory")
            return "ok"
        return fn

    assert models.run("c", step(1)) == "ok"  # OOM -> the idle others are released, one retry
    assert calls == [["a", "b", "c"], ["c"]]
    calls.clear()
    with pytest.raises(RuntimeError):
        models.run("c", step(2))  # nothing left to release: no second retry, the caller decides
    assert len(calls) == 1
    models.get("a")
    assert not models.retry("a", lambda: models.evict_others("c"))  # a model in use is never released
    assert sorted(models.loaded()) == ["a", "c"]
    with pytest.raises(ValueError):
        models.run("c", lambda m: (_ for _ in ()).throw(ValueError("not a GPU error")))
    monkeypatch.setattr(models, "EXCLUSIVE", True)  # opt-in: one model at a time
    models.get("b")
    assert models.loaded() == ["b"]
