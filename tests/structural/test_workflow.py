from pathlib import Path
import pytest
from structural.project import StructuralProject
from structural.workflow import freeze_run, new_run_directory, verify_evidence


def test_run_manifest_cannot_be_reused(tmp_path):
    project = StructuralProject()
    # This regression tests immutable run manifests, so bind a synthetic source.
    # Real CAD import/identity has separate optional retained-source checks.
    from structural.project import sha256
    source = tmp_path / "source.step"
    source.write_text("synthetic source identity fixture")
    project.source_path = str(source)
    project.source_sha256 = sha256(source)
    output = new_run_directory("preload diagnostic", tmp_path)
    manifest = freeze_run(project, output, purpose="unit-test")
    assert manifest["numerical_acceptance"] == "unaccepted"
    assert manifest["solver_completion"] == "not_run"
    with pytest.raises(FileExistsError):
        freeze_run(project, output, purpose="overwrite-attempt")


def test_evidence_missing_is_not_ready():
    assert verify_evidence(StructuralProject())
