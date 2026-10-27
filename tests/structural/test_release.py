import copy
import json
from pathlib import Path

import pytest

from structural import release
from structural.project import sha256
from structural.results import REQUIRED_RESULT_GATE_NAMES


def _case(name):
    return {**{key: True for key in release.BINDINGS}, **{key: key for key in release.IDENTITY_KEYS},
            "numerically_accepted": True, "friction_coefficient": 0.,
            "source_files": {"results": {"path": name, "sha256": name}},
            "single_run_acceptance": {"accepted": True, "diagnostic_only": False, "solver_status": "completed",
                "assessment": {"status": "numerically_accepted", "gates": [
                    {"name": name, "status": "passed"} for name in REQUIRED_RESULT_GATE_NAMES]}}}


def _sections(monkeypatch):
    # This test isolates final linkage; spatial numerical checks have their own
    # independent tests and the file assembler freshly recomputes all sections.
    monkeypatch.setattr(release, "assess_mesh_release", lambda *a, **kw: {"full_local_spatial_release_accepted": True})
    baseline, final = _case("baseline.json"), _case("final.json")
    friction = [final, {**final, "friction_coefficient": .1}, {**final, "friction_coefficient": .2}]
    studies = {"mesh": {"cases": [baseline]},
               "contact": {"study": "contact", "numerical_convergence_accepted": True, "cases": [final]},
               "friction": {"study": "friction", "status": "sensitivity_completed", "cases": friction}}
    return [baseline, final, studies, {"accepted": True, "reference_is_final_candidate": True},
            {"verified": True}, {"verified": True}]


def test_complete_linkage_is_software_release_not_design_or_equivalence(monkeypatch):
    result = release.evaluate_sections(*_sections(monkeypatch))
    assert result["phase_0_to_6_release_accepted"]
    assert result["design_adequacy"] == result["load_rating"] == "not_assessed"
    assert result["cross_code_equivalence_claimed"] is False


@pytest.mark.parametrize("binding", release.BINDINGS)
def test_stale_final_binding_cannot_be_promoted(monkeypatch, binding):
    sections = _sections(monkeypatch)
    sections[1][binding] = False
    assert not release.evaluate_sections(*sections)["phase_0_to_6_release_accepted"]


def test_frictionless_only_evidence_cannot_close_approved_full_scope(monkeypatch):
    sections = _sections(monkeypatch)
    sections[2]["friction"]["cases"] = [sections[1]]
    result = release.evaluate_sections(*sections)
    assert not result["phase_0_to_6_release_accepted"]
    assert next(g for g in result["gates"] if g["name"] == "friction_sensitivity")["status"] == "failed"


def test_penalty_or_repeatability_on_different_mesh_does_not_close_release(monkeypatch):
    sections = _sections(monkeypatch)
    sections[2]["contact"]["cases"] = [_case("other.json")]
    sections[3]["reference_is_final_candidate"] = False
    result = release.evaluate_sections(*sections)
    failed = {g["name"] for g in result["gates"] if g["status"] == "failed"}
    assert {"final_candidate_in_penalty_study", "repeatability_on_final_candidate"} <= failed


def test_missing_actual_gate_rejects_accepted_flag(monkeypatch):
    sections = _sections(monkeypatch)
    sections[1]["single_run_acceptance"]["assessment"]["gates"].pop()
    assert not release.evaluate_sections(*sections)["phase_0_to_6_release_accepted"]


def test_software_receipt_detects_changed_and_new_sources(tmp_path):
    files = [tmp_path / "structural/example.py", tmp_path / "tests/structural/test_example.py",
             tmp_path / "structural-fea-environment/verify-structural.ps1"]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("source", encoding="utf-8")
    evidence = tmp_path / "verification"
    evidence.mkdir()
    hashes = {str(p.resolve()): sha256(p) for p in files}
    for name in ("before.json", "after.json"):
        (evidence / name).write_text(json.dumps(hashes), encoding="utf-8")
    for name in ("regression.log", "desktop-smoke.log"):
        (evidence / name).write_text("passed", encoding="utf-8")
    receipt = {"regression_exit_code": 0, "desktop_exit_code": 0, "desktop_requested": True,
               "source_files_unchanged": True, "source_file_count": len(hashes),
               "source_hashes_before": "before.json", "source_hashes_after": "after.json"}
    path = evidence / "verification.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert release.verify_software_snapshot(path, tmp_path)["verified"]
    files[0].write_text("changed", encoding="utf-8")
    assert not release.verify_software_snapshot(path, tmp_path)["verified"]
    files[0].write_text("source", encoding="utf-8")
    (tmp_path / "structural/new.py").write_text("new module", encoding="utf-8")
    assert not release.verify_software_snapshot(path, tmp_path)["verified"]


def test_retained_source_changes_are_rejected(tmp_path):
    file = tmp_path / "result.json"
    file.write_text("original", encoding="utf-8")
    record = {"path": str(file), "sha256": sha256(file)}
    ledger = {}
    release._collect_sources({"nested": [record]}, ledger)
    file.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale evidence"):
        release._collect_sources({"nested": [record]}, {})


def _comparison():
    screens = [{"quantity": quantity, "distance_mm": distance, "observed_within_tolerance": False,
                "candidate": [1., 2., 3.], "reference": [1., 2., 4.]}
               for distance in (35., 75., 125., 175.) for quantity in
               ("section_mean_displacement_vector_mm", "section_force_vector_N", "section_moment_vector_Nmm")]
    screens += [{"quantity": "pipe_max_displacement_mm", "observed_within_tolerance": False,
                 "candidate": 1., "reference": 2.}]
    screens += [{"quantity": "finite_volume_mean_von_mises_MPa", "region": region,
                 "observed_within_tolerance": False, "candidate": 1., "reference": 2.}
                for region in ("center", "slots", "seats")]
    components = {name: {"status": "available", "full_reference_probe_coverage": True,
                        "reference_valid_probe_count": 25, "joint_valid_probe_count": 25}
                  for name in ("pipe", "screw_1", "screw_2", "screw_3")}
    components["pipe"]["finite_volume_regions"] = {
        side: {region: {"volume_mm3": 1., "volume_mean_nodal_von_mises_mpa": 2.}
               for region in ("center", "slots", "seats")} for side in ("reference", "candidate")}
    return {"states": {state: {"screens": copy.deepcopy(screens), "components": copy.deepcopy(components)}
                       for state in ("preload", "loaded")}}


def test_complete_comparison_can_disagree_without_being_missing():
    result = release.comparison_coverage(_comparison())
    assert result["complete"]
    assert len(result["observed_screen_failures"]) == 32


@pytest.mark.parametrize("missing", ["state", "component", "screen", "nonfinite", "volume"])
def test_incomplete_comparison_cannot_qualify_interpretation(missing):
    report = _comparison()
    state = report["states"]["loaded"]
    if missing == "state":
        del report["states"]["preload"]
    elif missing == "component":
        state["components"]["screw_2"]["joint_valid_probe_count"] = 24
    elif missing == "screen":
        state["screens"].pop()
    elif missing == "nonfinite":
        state["screens"][0]["candidate"][1] = float("nan")
    else:
        state["components"]["pipe"]["finite_volume_regions"]["reference"]["slots"]["volume_mm3"] = 0
    assert not release.comparison_coverage(report)["complete"]


def test_repeatability_flat_hash_map_is_consumed_and_rechecked(tmp_path):
    file = tmp_path / "repeat.dat"
    file.write_text("native", encoding="utf-8")
    hashes = {str(file.resolve()): sha256(file)}
    ledger = {}
    release.collect_hash_map(hashes, ledger)
    assert ledger == hashes
    file.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale evidence"):
        release.collect_hash_map(hashes, {})


def test_interpretation_must_match_actual_inputs_and_current_reader(tmp_path):
    text = tmp_path / "interpretation.md"
    text.write_text("Discussed preload mismatch and unverified model identity.", encoding="utf-8")
    source = tmp_path / "baseline-results.json"
    source.write_text("{}", encoding="utf-8")
    expected = {"baseline_results": release._file(source)}
    receipt = {"schema_version": 1, "cross_code_equivalence_claimed": False,
               "reviewed_discrepancies": ["Preload discrepancy remains documented."],
               "section_screen_extension_sha256": "extension", "interpretation": release._file(text), **expected}
    file = tmp_path / "interpretation-receipt.json"
    file.write_text(json.dumps(receipt), encoding="utf-8")
    reports = {"baseline35": {"section_screen_extension_sha256": "extension"},
               "matched30": {"section_screen_extension_sha256": "extension"}}
    assert release.verify_interpretation_receipt(file, expected, reports, {})
    with pytest.raises(ValueError, match="another case"):
        release.verify_interpretation_receipt(file, {"baseline_results": {"path": "different", "sha256": "other"}}, reports, {})
    text.write_text("Changed without review receipt.", encoding="utf-8")
    with pytest.raises(ValueError, match="path/hash"):
        release.verify_interpretation_receipt(file, expected, reports, {})
