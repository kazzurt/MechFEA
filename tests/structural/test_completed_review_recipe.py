import importlib.util
import json
from pathlib import Path

import pytest

from structural.project import sha256


@pytest.fixture
def recipe():
    path=Path(__file__).resolve().parents[2]/"tools/review_completed_assembly.py"
    spec=importlib.util.spec_from_file_location("completed_review_recipe",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def retained_run(tmp_path):
    run=tmp_path/"retained-run";run.mkdir()
    deck=run/"assembly.inp";deck.write_text("*HEADING\nretained fixture\n")
    for name in ("project.json","structural-mesh.npz","solver-mesh.npz","assembly.assembly.json","assembly.dat","assembly.frd","assembly.sta"):
        (run/name).write_text("retained fixture")
    (run/"assembly.run.json").write_text(json.dumps({"status":"completed","returncode":0,"input_path":str(deck),"input_sha256":sha256(deck)}))
    vtu=tmp_path/"reference.vtu";vtu.write_text("reference fixture")
    reference=tmp_path/"reference.json"
    reference.write_text(json.dumps({"states":{state:{"source":{"path":str(vtu),"sha256":sha256(vtu)}} for state in ("preload","loaded")}}))
    return run,reference


@pytest.mark.parametrize("invalid_evidence", [None, "numerically_accepted", "retained_result_source_binding_verified", "retained_component_field_binding_verified"])
def test_review_cannot_promote_accepted_report_with_stale_current_evidence(recipe, monkeypatch, invalid_evidence):
    import structural.results
    monkeypatch.setattr(structural.results, "report_is_accepted", lambda report: True)
    metrics = {key: True for key in ("numerically_accepted", "retained_result_source_binding_verified",
                                    "retained_component_field_binding_verified", "geometry_revision_binding_verified")}
    if invalid_evidence:
        metrics[invalid_evidence] = False
    summary = recipe.concise_review({"solver_status": "completed", "assessment": {"gates": []}}, metrics, {})
    assert summary["numerically_accepted"] is (invalid_evidence is None)


def test_review_rejects_live_manifest_before_creating_output(recipe,tmp_path):
    run,reference=retained_run(tmp_path)
    manifest=run/"assembly.run.json";data=json.loads(manifest.read_text());data["status"]="running";manifest.write_text(json.dumps(data))
    output=tmp_path/"review"
    with pytest.raises(ValueError,match="completed retained"):
        recipe.main([str(run),"--output",str(output),"--reference",str(reference)])
    assert not output.exists()


@pytest.mark.parametrize("mutation",[None,"raw","reader","reference"])
def test_review_preserves_unaccepted_diagnostics_and_rejects_source_changes(recipe,tmp_path,monkeypatch,mutation):
    run,reference=retained_run(tmp_path)
    calls=[]
    reader_hash={"results.py":"original reader"}
    monkeypatch.setattr(recipe,"reader_source_hashes",lambda:reader_hash.copy())
    def stage(command,directory,name):
        assert "solve" not in command
        assert command[command.index("--stage")+1]==name
        destination=Path(command[command.index("--output")+1]);destination.mkdir()
        calls.append(name)
        if name=="postprocess":
            (destination/"results.json").write_text(json.dumps({"solver_status":"completed","accepted":False,"diagnostic_only":True,
                "assessment":{"gates":[{"name":"penetration","status":"failed","measured":.008,"limit":.0075}]},"stages":{}}))
            return 2  # Unaccepted physical result is retained, not a reader crash.
        if name=="metrics":
            (destination/"study-summary.json").write_text(json.dumps({"geometry_revision_binding_verified":True}))
        else:
            (destination/"comparison.json").write_text(json.dumps({"screen_status":"diagnostic_only_identity_or_numerical_gate_unmet"}))
            if mutation=="raw":(run/"assembly.dat").write_text("raw data changed during review")
            elif mutation=="reader":reader_hash["results.py"]="changed reader"
            elif mutation=="reference":(tmp_path/"reference.vtu").write_text("changed reference")
        return 0
    monkeypatch.setattr(recipe,"run_stage",stage)
    output=tmp_path/"review"
    arguments=[str(run),"--output",str(output),"--reference",str(reference)]
    if mutation:
        with pytest.raises(RuntimeError,match="changed during review"):recipe.main(arguments)
        manifest=json.loads((output/"review-manifest.json").read_text())
        key={"raw":"source_files_unchanged","reader":"reader_sources_unchanged","reference":"reference_sources_unchanged"}[mutation]
        assert manifest["status"]=="failed" and not manifest[key]
        assert not (output/"review-summary.json").exists()
    else:
        assert recipe.main(arguments)==2
        manifest=json.loads((output/"review-manifest.json").read_text())
        assert manifest["status"]=="completed" and manifest["source_files_unchanged"]
        summary=json.loads((output/"review-summary.json").read_text())
        assert not summary["numerically_accepted"] and summary["gate_counts"]["failed"]==1
        assert summary["convergence_acceptance"]=="not_established_by_one_run"
    assert calls==["postprocess","metrics","compare"] and manifest["solver_invoked"] is False
    # A changed reference is rejected at preflight; restore it to test the
    # independent no-overwrite guarantee after the completed/failed review.
    if mutation=="reference":(tmp_path/"reference.vtu").write_text("reference fixture")
    with pytest.raises(FileExistsError):recipe.main(arguments)
    assert calls==["postprocess","metrics","compare"]
