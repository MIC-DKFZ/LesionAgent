from lesionagent.core.recist import (Candidate, TargetMeasurement as T, Visit, best_overall_response, evaluate,
                                      overall_response, suggest_targets, validate_targets)  # fmt: skip


def bl(*lds, node=False):
    return [T(i + 1, node, "present", ld_mm=v, sa_mm=v) for i, v in enumerate(lds)]


def fu(*lds, node=False, **kw):
    return Visit(kw.pop("visit", 2), kw.pop("day", 60), [T(i + 1, node, "present", ld_mm=v, sa_mm=v)
                                                          for i, v in enumerate(lds)], **kw)  # fmt: skip


def test_pr_sd_pd_thresholds():
    assert evaluate(bl(50, 50), [fu(35, 35)])[0].target_response == "PR"      # -30 % exactly
    assert evaluate(bl(50, 50), [fu(36, 35)])[0].target_response == "SD"
    assert evaluate(bl(50, 50), [fu(60, 60)])[0].target_response == "PD"      # +20 %, +20 mm
    # +20 % but only +4 mm -> not PD
    assert evaluate(bl(10, 10), [fu(12, 12)])[0].target_response == "SD"


def test_pd_relative_to_nadir():
    r = evaluate(bl(50, 50), [fu(30, 30, visit=2), fu(36, 36, visit=3, day=120)])
    assert r[0].target_response == "PR" and r[1].target_response == "PD"     # 72 vs nadir 60: +20 %, +12 mm


def test_cr_with_nodes():
    v = Visit(2, 60, [T(1, False, "absent"), T(2, True, "present", sa_mm=8)])
    r = evaluate([T(1, False, ld_mm=30), T(2, True, sa_mm=20)], [v])[0]
    assert r.target_response == "CR" and r.sld_mm == 8                        # sum non-zero at CR is allowed


def test_too_small_and_missing():
    v = Visit(2, 60, [T(1, False, "too_small"), T(2, False, "present", ld_mm=20)])
    assert evaluate(bl(30, 30), [v])[0].sld_mm == 25
    v = Visit(2, 60, [T(1, False, "not_assessed"), T(2, False, "present", ld_mm=20)])
    assert evaluate(bl(30, 30), [v])[0].target_response == "NE"
    v = Visit(2, 60, [T(1, False, "not_assessed"), T(2, False, "present", ld_mm=80)])
    assert evaluate(bl(30, 30), [v])[0].target_response == "PD"              # measured part alone is PD


def test_overall_table():
    assert overall_response("CR", "NonCR/NonPD", False) == "PR"
    assert overall_response("SD", "NonCR/NonPD", True) == "PD"
    assert overall_response("PR", "PD", False) == "PD"
    assert overall_response(None, "NonCR/NonPD", False) == "NonCR/NonPD"
    assert overall_response("NE", "NonCR/NonPD", False) == "NE"


def test_new_lesion_and_best_response():
    r = evaluate(bl(40), [fu(20, day=56), fu(20, visit=3, day=112, new_lesion=True), fu(10, visit=4, day=170)])
    assert [x.overall for x in r] == ["PR", "PD", "PR"] and "after the first PD" in r[2].notes[0]
    assert best_overall_response(r) == "PR"
    assert best_overall_response(r, confirm=True) == "SD"                     # PR not confirmed


def test_target_rules():
    c = [Candidate(1, "liver", False, 40, 30), Candidate(2, "liver", False, 30, 20), Candidate(3, "liver", False, 25, 20),
         Candidate(4, "lung", False, 8, 6), Candidate(5, "x", True, 25, 18), Candidate(6, "y", True, 22, 12)]  # fmt: skip
    assert suggest_targets(c) == [1, 2, 5]
    errs = validate_targets([c[0], c[1], c[2], c[3], c[5]])
    assert any("3 targets in liver" in e for e in errs) and any("lesion 4 not measurable" in e for e in errs)
    assert any("lesion 6 not measurable" in e for e in errs)
