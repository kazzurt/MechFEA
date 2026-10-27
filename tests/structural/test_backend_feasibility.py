"""Acceptance must depend on unchanged, accepted evidence and verified binaries."""
import json
from pathlib import Path
import sys

import pytest

from structural.backends.calculix import DEFAULT_EXECUTABLE, ROOT, sha256
from structural.backends.feasibility import verify_backend_feasibility


@pytest.fixture
def report():
    path = ROOT / "Dodec FEA/structural-fea-validation-deliverables/backend-feasibility.json"
    if not path.is_file() or not DEFAULT_EXECUTABLE.is_file():
        pytest.skip("Retained native backend verification is not available")
    return json.loads(path.read_text())


def replace_evidence(tmp_path, report, key, mutate):
    data = json.loads(Path(report[key]["path"]).read_text())
    mutate(data)
    path = tmp_path / f"modified-{key}.json"
    path.write_text(json.dumps(data))
    report[key] = {**report[key], "path":str(path), "sha256":sha256(path)}


def test_retained_evidence_verifies_both_pinned_builds(report):
    for executable in (DEFAULT_EXECUTABLE, DEFAULT_EXECUTABLE.with_name("ccx_MT.exe")):
        verified = verify_backend_feasibility(report, executable)
        assert verified["verified"]
        assert len(verified["verified_json_references"]) == 8
        assert verified["selected_executable_sha256"] == sha256(executable)


def test_boolean_pass_cannot_hide_changed_evidence(report):
    report["generic_fixture_summary"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash changed"):
        verify_backend_feasibility(report)


@pytest.mark.parametrize("change,expected", [
    (lambda data: data.pop("preload_contact_open"), "required generic fixture is absent"),
    (lambda data: data["preload_lock"].update(accepted=False), "not accepted"),
    (lambda data: data["contact"]["checks"].update(gap_sign_verified=False), "unmet numerical check"),
    (lambda data: data["tension"]["run"].update(status="failed"), "did not complete"),
])
def test_rehashed_but_incomplete_or_failed_proof_is_rejected(tmp_path, report, change, expected):
    replace_evidence(tmp_path, report, "generic_fixture_summary", change)
    with pytest.raises(ValueError, match=expected):
        verify_backend_feasibility(report)


def test_unverified_selected_solver_is_rejected(report):
    with pytest.raises(ValueError, match="selected executable is not a verified build"):
        verify_backend_feasibility(report, sys.executable)


@pytest.mark.parametrize("key,value,expected", [
    ("required_assembly_pretension_id", "native_pretension_section", "explicit paired-node"),
    ("required_assembly_anchorage", "projected_tie", "fixed rigid master"),
])
def test_different_formulation_cannot_reuse_fixture_gate(report, key, value, expected):
    report[key] = value
    with pytest.raises(ValueError, match=expected):
        verify_backend_feasibility(report)
