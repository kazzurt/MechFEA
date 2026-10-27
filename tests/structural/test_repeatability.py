import copy
import importlib
import json
from pathlib import Path
import sys

import pytest

from structural import repeatability as repeat
from structural.project import StructuralProject,sha256
from structural.provenance import module_code_sha256
from structural.results import REQUIRED_RESULT_GATE_NAMES


def metrics():
    state={"max_displacement_mm":.5,"bolt_force_n":[100.,200.,300.],
        "contact_force_n":{key:[0.,100.,0.] for key in ["PIPE_BORE__HUB_SNOUT",*[f"SCREW_{i}_SEAT__SCREW_{i}_CONE" for i in (1,2,3)]]},
        "section_force_n":{str(d):[0.,1000.,10.] for d in repeat.CONTRACT["section_distances_mm"]},
        "section_moment_nmm":{str(d):[100000.,0.,0.] for d in repeat.CONTRACT["section_distances_mm"]},
        "regions":{name:{"volume_mean_nodal_von_mises_mpa":20.} for name in repeat.CONTRACT["regions"]}}
    return {"states":{name:copy.deepcopy(state) for name in repeat.CONTRACT["states"]}}


@pytest.fixture
def retained_cases(tmp_path,monkeypatch):
    executable=tmp_path/"pinned_solver.exe";executable.write_bytes(b"fixture executable; never invoked")
    rows={};paths=[]
    project=StructuralProject();project.solver["threads"]=4
    reader={}
    for key,module_name in {"results":"structural.results","comparison":"structural.comparison",
                           "backend_calculix":"structural.backends.calculix","validation":"structural.validation"}.items():
        module=importlib.import_module(module_name)
        reader[key]={"module":module_name,"source_sha256_at_import":sha256(module.__file__),
                     "loaded_function_code_sha256":module_code_sha256(vars(module)),"python_version":sys.version}
    for label in ("reference","repeat"):
        folder=tmp_path/label;folder.mkdir()
        project.save(folder/"project.json")
        deck=folder/"assembly.inp";deck.write_bytes(b"*HEADING\nexact deck\n*STEP\n*STATIC\n*END STEP\n*STEP\n*STATIC\n*END STEP\n")
        metadata={"prepared_mesh_fingerprint":"prepared","solver_mesh":"solver-mesh.npz","expected_step_times":[1.,1.]}
        deck.with_suffix(".assembly.json").write_text(json.dumps(metadata))
        for name in ("structural-mesh.npz","solver-mesh.npz"):(folder/name).write_bytes(name.encode())
        run={"input_path":str(deck),"executable":str(executable),"executable_sha256":sha256(executable),
             "solver_version":"2.23","threads":4,"expected_steps":2,"completed_steps":[1,2],"expected_step_times":[1.,1.]}
        deck.with_suffix(".run.json").write_text(json.dumps(run))
        for ext in (".dat",".frd",".sta",".stdout.log",".stderr.log"):deck.with_suffix(ext).write_bytes(ext.encode())
        fields=folder/"pipe.vtu";fields.write_bytes(b"bound field fixture")
        results=folder/"results.json"
        report={"accepted":True,"diagnostic_only":False,"solver_status":"completed","friction_coefficient":0.,
                "source_run":str(deck.with_suffix(".run.json")),"implementation_provenance":reader,
                "assessment":{"status":"numerically_accepted","gates":[{"name":name,"status":"passed"} for name in REQUIRED_RESULT_GATE_NAMES]}}
        results.write_text(json.dumps(report))
        summary=metrics()
        summary.update(numerically_accepted=True,solver_sha256=sha256(executable),mesh_fingerprint="solver",geometry_revision=None,
                       numerical_controls_fingerprint="exact controls",metric_definition_sha256="fixed metrics")
        for key in ("retained_result_source_binding_verified","retained_component_field_binding_verified","project_binding_verified",
                    "mesh_binding_verified","geometry_revision_binding_verified","step_controls_binding_verified","contact_controls_binding_verified"):
            summary[key]=True
        summary["source_files"]={path.name:{"path":str(path),"sha256":sha256(path)} for path in folder.iterdir() if path.is_file()}
        rows[str(results.resolve())]=summary;paths.append(results)
    monkeypatch.setattr(repeat,"summarize_run",lambda path:copy.deepcopy(rows[str(Path(path).resolve())]))
    return paths,rows


def test_predeclared_rule_uses_absolute_tolerance_and_vector_norm_without_cancellation():
    assert repeat.contract_evidence()["definition"]==repeat.CONTRACT
    assert repeat._leaf_changes(0.,1e-7,"zero U",1e-7)[0]["status"]=="passed"
    assert repeat._leaf_changes(0.,1.01e-7,"zero U",1e-7)[0]["status"]=="failed"
    assert repeat._leaf_changes([0.,100.,0.],[.101,100.,0.],"non-Y force",1e-3,vector=True)[0]["status"]=="failed"
    rows=repeat._leaf_changes([100.,100.],[100.2,99.8],"screws",1e-3)
    assert len(rows)==2 and all(row["status"]=="failed" for row in rows)
    assert repeat._leaf_changes(None,0.,"missing",1e-3)[0]["status"]=="missing"


def test_assessor_accepts_same_exact_inputs_and_all_38_response_entries(retained_cases,tmp_path):
    paths,_=retained_cases
    report=repeat.assess_repeatability(*paths,tmp_path/"assessment")
    assert report["accepted"] and report["status"]=="passed" and len(report["changes"])==38
    assert report["solver_invoked"] is False
    assert report["identity"]["reference"]==report["identity"]["repeat"]
    assert report["records"]["reference_results"]==str(paths[0])
    with pytest.raises(FileExistsError):repeat.assess_repeatability(*paths,tmp_path/"assessment")
    assert not repeat.assess_repeatability(paths[0],paths[0])["accepted"]


@pytest.mark.parametrize("flag",["numerically_accepted","retained_result_source_binding_verified","retained_component_field_binding_verified",
    "geometry_revision_binding_verified","contact_controls_binding_verified"])
def test_unaccepted_or_unbound_run_cannot_establish_repeatability(retained_cases,flag):
    paths,rows=retained_cases;rows[str(paths[1])][flag]=False
    assert not repeat.assess_repeatability(*paths)["accepted"]


@pytest.mark.parametrize("kind",["missing_section","opposed_screws","non_y_force","region_stress","reader_version","changed_dat","changed_vtu","project_bytes"])
def test_repeatability_rejects_missing_response_changes_and_stale_or_different_identity(retained_cases,kind):
    paths,rows=retained_cases;summary=rows[str(paths[1])];loaded=summary["states"]["loaded"]
    if kind=="missing_section":
        for row in rows.values():row["states"]["loaded"]["section_force_n"].pop("175.0")
    elif kind=="opposed_screws":loaded["bolt_force_n"]=[100.2,199.8,300.]
    elif kind=="non_y_force":loaded["section_force_n"]["175.0"][0]=1.01
    elif kind=="region_stress":loaded["regions"]["slots"]["volume_mean_nodal_von_mises_mpa"]=20.021
    elif kind=="reader_version":
        data=json.loads(paths[1].read_text());data["implementation_provenance"]["results"]["loaded_function_code_sha256"]="changed"
        paths[1].write_text(json.dumps(data));summary["source_files"]["results.json"]["sha256"]=sha256(paths[1])
    elif kind=="project_bytes":
        path=paths[1].parent/"project.json";path.write_bytes(path.read_bytes()+b"\n")
        summary["source_files"]["project.json"]["sha256"]=sha256(path)
    else:
        path=paths[1].parent/("assembly.dat" if kind=="changed_dat" else "pipe.vtu")
        path.write_bytes(path.read_bytes()+b" changed")
    assert not repeat.assess_repeatability(*paths)["accepted"]


def test_prepare_exact_replay_copies_bytes_and_never_rebinds_or_launches(retained_cases,tmp_path,monkeypatch):
    from structural.backends import calculix
    paths,_=retained_cases
    monkeypatch.setattr(calculix.CalculixRunner,"run",lambda *args,**kwargs:pytest.fail("Preparation must never solve"))
    output=tmp_path/"prepared-repeat"
    receipt=repeat.prepare_exact_replay(paths[0],output)
    assert receipt["status"]=="prepared_not_run" and not receipt["solver_invoked"] and not receipt["rebind_performed"]
    assert len(receipt["copied_inputs"])==5
    for target,digest in receipt["copied_inputs"].items():
        assert Path(target).read_bytes()==(paths[0].parent/Path(target).name).read_bytes()
        assert digest==sha256(target)
    with pytest.raises(FileExistsError):repeat.prepare_exact_replay(paths[0],output)
    (output/"project.json").write_bytes((output/"project.json").read_bytes()+b"\n")
    with pytest.raises(ValueError,match="changed"):repeat.execute_prepared_replay(output/"exact-replay-manifest.json")


def test_explicit_execution_uses_pinned_runner_without_deck_generation(retained_cases,tmp_path,monkeypatch):
    from structural.backends import calculix
    paths,_=retained_cases
    output=tmp_path/"explicit-repeat";receipt=repeat.prepare_exact_replay(paths[0],output)
    calls=[]
    def fake_run(runner,deck,**kwargs):
        calls.append((runner,Path(deck),kwargs))
        assert sha256(deck)==receipt["reference_identity"]["byte_inputs"]["deck"]
        return calculix.RunResult("completed",str(deck),str(runner.executable),returncode=0)
    monkeypatch.setattr(calculix.CalculixRunner,"run",fake_run)
    result=repeat.execute_prepared_replay(output/"exact-replay-manifest.json")
    assert result.completed and len(calls)==1
    assert calls[0][0].threads==4 and calls[0][2]["expected_steps"]==2
    assert calls[0][2]["expected_step_times"]==[1.,1.]
    with pytest.raises(FileExistsError):repeat.execute_prepared_replay(output/"exact-replay-manifest.json")


def test_replay_refuses_updated_receipt_hash_for_changed_copied_deck(retained_cases,tmp_path,monkeypatch):
    paths,_=retained_cases;output=tmp_path/"tampered-repeat"
    receipt=repeat.prepare_exact_replay(paths[0],output)
    deck=Path(receipt["input_path"]);deck.write_bytes(deck.read_bytes()+b"** changed\n")
    receipt["copied_inputs"][str(deck)]=sha256(deck)
    manifest=output/"exact-replay-manifest.json";manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="changed"):repeat.execute_prepared_replay(manifest)
