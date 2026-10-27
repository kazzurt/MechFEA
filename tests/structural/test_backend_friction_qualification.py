"""Reject known-unqualified friction histories before expensive assembly work."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from structural.assembly import bind_mesh_evidence, write_assembly_deck
from structural.backends.friction_qualification import (
    BUNDLED_QUALIFICATION, BUNDLED_QUALIFICATION_SHA256, verify_friction_qualification)
from structural.project import StructuralProject, sha256


def test_frictionless_verification_does_not_change_project_or_require_files():
    project=StructuralProject();before=copy.deepcopy(project.to_dict());fingerprint=project.fingerprint()
    project.solver["friction_qualification_manifest"]={"path":"missing", "sha256":"invalid"}
    assert verify_friction_qualification(project) is None
    del project.solver["friction_qualification_manifest"]
    assert project.to_dict()==before and project.fingerprint()==fingerprint
    assert not any("friction" in issue.lower() for issue in project.readiness_issues())


def test_frozen_failed_native_reclosure_proof_blocks_nonzero_friction():
    assert sha256(BUNDLED_QUALIFICATION)==BUNDLED_QUALIFICATION_SHA256
    project=StructuralProject();project.contacts.friction_coefficient=.2
    with pytest.raises(ValueError,match="opening/reclosure history qualification has not passed"):
        verify_friction_qualification(project)
    assert any("opening/reclosure history" in issue for issue in project.readiness_issues())


def test_explicit_invalid_manifest_is_not_replaced_by_bundled_proof(tmp_path):
    project=StructuralProject();project.contacts.friction_coefficient=.1
    project.solver["friction_qualification_manifest"]={"path":str(tmp_path/'missing.json'),"sha256":"0"*64}
    with pytest.raises(ValueError,match="missing qualification_manifest"):
        verify_friction_qualification(project)


def test_rehashed_top_level_pass_does_not_hide_failed_reclosure_checks(tmp_path):
    data=json.loads(BUNDLED_QUALIFICATION.read_text());data["all_passed"]=True
    # This proof must not depend on the original machine's retained run files.
    # A hash-valid fixture with failed law checks must still block a rehashed
    # top-level "pass", even when the native result tree is absent.
    failed = tmp_path / 'failed-native-checks.json'
    failed.write_text(json.dumps({"accepted": False, "checks": {"opened_after_slip/all_contact_force_zero": False}}))
    data["fixtures"] = {name: {"path": str(failed), "sha256": sha256(failed)} for name in data["fixtures"]}
    candidate=tmp_path/'falsely-passed-qualification.json';candidate.write_text(json.dumps(data))
    project=StructuralProject();project.contacts.friction_coefficient=.2
    project.solver["friction_qualification_manifest"]={"path":str(candidate),"sha256":sha256(candidate)}
    with pytest.raises(ValueError,match="has missing or failed checks"):
        verify_friction_qualification(project)


def test_friction_bind_rejects_before_touching_mesh_and_preserves_original():
    project=StructuralProject();project.contacts.friction_coefficient=.1;before=copy.deepcopy(project.to_dict())
    with pytest.raises(ValueError,match="opening/reclosure history"):
        bind_mesh_evidence(project,object(),backend_fixture_report={})
    assert project.to_dict()==before


def test_friction_deck_rejects_before_mesh_processing_or_output(tmp_path,monkeypatch):
    project=StructuralProject();project.contacts.friction_coefficient=.1
    monkeypatch.setattr(project,"validate",lambda **kwargs:None)
    path=tmp_path/'forbidden-friction.inp'
    with pytest.raises(ValueError,match="opening/reclosure history"):
        write_assembly_deck(project,object(),path)
    assert not path.exists() and not (tmp_path/'solver-mesh.npz').exists()


@pytest.mark.parametrize("reference",[None,{}, {"path":"x","sha256":"no"}])
def test_positive_friction_explicit_malformed_reference_is_rejected(reference):
    project=StructuralProject();project.contacts.friction_coefficient=.1
    project.solver["friction_qualification_manifest"]=reference
    with pytest.raises(ValueError,match="hash-pinned"):
        project.validate()
