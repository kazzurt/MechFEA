"""Frozen same-input assembly repeatability and byte-preserving replay.

Preparing or assessing a replay never invokes a solver. The separate execution
entry point is deliberately explicit and refuses any changed retained input.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

from .project import StructuralProject, sha256, write_json
from .provenance import capture_module_source, module_code_sha256
from .results import report_is_accepted
from .studies import summarize_run

IMPLEMENTATION_PROVENANCE=capture_module_source(__file__,__name__)
ROOT=Path(__file__).resolve().parents[1]
CONTRACT_PATH=ROOT/"structural/contracts/repeatability-contract-v1.json"
CONTRACT={
    "schema_version":1,"id":"same_input_assembly_repeatability_v1","declared_date":"2026-10-18",
    "declaration":"Defined before the first full-assembly repeat; not fitted to repeat results",
    "relative_tolerance":.001,
    "absolute_tolerances":{"displacement_mm":1e-7,"force_n":1e-3,"moment_nmm":.1,"region_stress_mpa":1e-3},
    "rule":"absolute scalar or Euclidean vector difference <= max(absolute tolerance, relative tolerance times reference magnitude)",
    "states":["preload","loaded"],"section_distances_mm":[35.,75.,125.,175.],
    "regions":["center","slots","seats"],"required_threads":4,"required_solver_version":"2.23",
    "identity":"Exact bytes of deck, frozen project, assembly metadata, prepared mesh and solver mesh; same pinned executable, threads, controls, reader implementations and metric definition",
    "scope":"Repeatability of fixed both-state global/integrated/finite-volume response metrics; not bitwise native outputs, every nodal value, convergence, cross-code identity or design adequacy",
}


def contract_evidence():
    declared=json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if declared!=CONTRACT:raise ValueError("Frozen repeatability contract differs from the implemented version1 definition")
    return {"path":str(CONTRACT_PATH),"sha256":sha256(CONTRACT_PATH),"definition":copy.deepcopy(CONTRACT)}


def reader_source_hashes():
    """Pin every structural Python source, including imported verification helpers."""
    return {str(path.resolve()):sha256(path) for path in sorted((ROOT/"structural").rglob("*.py"))}


def _remember(files,path,digest=None):
    path=Path(path).resolve()
    current=sha256(path)
    if digest is not None and current!=digest:raise ValueError(f"Retained source changed: {path}")
    if str(path) in files and files[str(path)]!=current:raise ValueError(f"Source changed during review: {path}")
    files[str(path)]=current
    return current


def _nested_hash_records(value,files):
    if isinstance(value,dict):
        if isinstance(value.get("path"),str) and isinstance(value.get("sha256"),str):
            _remember(files,value["path"],value["sha256"])
        for child in value.values():_nested_hash_records(child,files)
    elif isinstance(value,list):
        for child in value:_nested_hash_records(child,files)


def _reader_identity(report):
    required={"results":"structural.results","comparison":"structural.comparison",
              "backend_calculix":"structural.backends.calculix","validation":"structural.validation"}
    if report.get("friction_required") is True:required["friction_qualification"]="structural.backends.friction_qualification"
    identity={};current=True
    for key,name in required.items():
        row=report.get("implementation_provenance",{}).get(key,{})
        module=importlib.import_module(name)
        source=Path(module.__file__).resolve()
        loaded=module_code_sha256(vars(module))
        expected={"source_sha256_at_import":sha256(source),"loaded_function_code_sha256":loaded,"python_version":sys.version}
        identity[key]={field:row.get(field) for field in expected}
        current=current and row.get("module")==name and identity[key]==expected
    return {"verified_current":bool(current),"identity":identity}


def _case_evidence(results_path):
    results_path=Path(results_path).resolve()
    report=json.loads(results_path.read_text(encoding="utf-8"))
    summary=summarize_run(results_path)
    source_run=Path(report["source_run"]).resolve()
    run=json.loads(source_run.read_text(encoding="utf-8"))
    deck=Path(run["input_path"]).resolve();directory=deck.parent
    metadata=json.loads(deck.with_suffix(".assembly.json").read_text(encoding="utf-8"))
    project=StructuralProject.load(directory/"project.json")
    files={}
    _nested_hash_records(summary,files)
    byte_inputs={name:_remember(files,path) for name,path in {
        "deck":deck,"project":directory/"project.json","assembly":deck.with_suffix(".assembly.json"),
        "prepared_mesh":directory/"structural-mesh.npz","solver_mesh":directory/"solver-mesh.npz"}.items()}
    _nested_hash_records(project.to_dict(),files)
    _nested_hash_records(metadata,files)
    executable=Path(run["executable"]).resolve()
    executable_sha=_remember(files,executable,run.get("executable_sha256"))
    readers=_reader_identity(report)
    checks={"single_run_acceptance":report_is_accepted(report) and summary.get("numerically_accepted") is True,
        **{key:summary.get(key) is True for key in (
            "retained_result_source_binding_verified","retained_component_field_binding_verified","project_binding_verified",
            "mesh_binding_verified","geometry_revision_binding_verified","step_controls_binding_verified","contact_controls_binding_verified")},
        "current_pinned_readers":readers["verified_current"],
        "pinned_backend":run.get("solver_version")==CONTRACT["required_solver_version"] and executable_sha==summary.get("solver_sha256"),
        "four_threads":run.get("threads")==CONTRACT["required_threads"] and project.solver.get("threads",1)==CONTRACT["required_threads"],
        "both_native_steps":run.get("completed_steps")==[1,2] and run.get("expected_steps")==2
            and run.get("expected_step_times")==[1.,1.] and metadata.get("expected_step_times")==[1.,1.],
        "fixed_metric_coverage":_metric_coverage(summary)}
    identity={"byte_inputs":byte_inputs,"solver_sha256":executable_sha,"solver_version":run.get("solver_version"),
        "threads":run.get("threads"),"geometry_revision":summary.get("geometry_revision"),
        "prepared_mesh_fingerprint":metadata.get("prepared_mesh_fingerprint"),"solver_mesh_fingerprint":summary.get("mesh_fingerprint"),
        "numerical_controls_fingerprint":summary.get("numerical_controls_fingerprint"),
        "metric_definition_sha256":summary.get("metric_definition_sha256"),"reader_implementations":readers["identity"]}
    return {"results":str(results_path),"run":str(source_run),"deck":str(deck),"checks":checks,"identity":identity,
            "summary":summary,"source_files":files,"execution":{"executable":str(executable),"expected_version":run.get("solver_version"),
                "threads":run.get("threads"),"timeout_s":project.solver.get("timeout_s",1800),"expected_steps":2,"expected_step_times":[1.,1.]}}


def _metric_coverage(summary):
    sections={str(value) for value in CONTRACT["section_distances_mm"]}
    pairs={"PIPE_BORE__HUB_SNOUT",*[f"SCREW_{i}_SEAT__SCREW_{i}_CONE" for i in (1,2,3)]}
    for state in CONTRACT["states"]:
        row=summary.get("states",{}).get(state,{})
        if not isinstance(row.get("bolt_force_n"),(list,tuple)) or len(row["bolt_force_n"])!=3:return False
        if set(row.get("contact_force_n") or {})!=pairs:return False
        if any(set(row.get(key) or {})!=sections for key in ("section_force_n","section_moment_nmm")):return False
        if set(row.get("regions") or {})!=set(CONTRACT["regions"]):return False
    return True


def _leaf_changes(reference,repeat,name,absolute,*,vector=False):
    if isinstance(reference,dict) and isinstance(repeat,dict):
        if not reference or reference.keys()!=repeat.keys():return [{"metric":name,"status":"missing","reason":"Corresponding metric entries differ or are absent"}]
        return [entry for key in reference for entry in _leaf_changes(reference[key],repeat[key],name+"/"+key,absolute,vector=vector)]
    if not vector and isinstance(reference,(list,tuple)) and isinstance(repeat,(list,tuple)):
        if not reference or len(reference)!=len(repeat):return [{"metric":name,"status":"missing","reason":"Corresponding scalar entries are absent"}]
        return [entry for i,(left,right) in enumerate(zip(reference,repeat),1) for entry in _leaf_changes(left,right,f"{name}/{i}",absolute)]
    try:
        a=np.asarray(reference,dtype=float);b=np.asarray(repeat,dtype=float)
        if a.shape!=b.shape or (a.shape!=(3,) if vector else a.shape!=()) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError()
    except (ValueError,TypeError):return [{"metric":name,"status":"missing","reason":"Finite corresponding scalar/vector required"}]
    difference=float(np.linalg.norm(b-a));magnitude=float(np.linalg.norm(a))
    limit=max(absolute,CONTRACT["relative_tolerance"]*magnitude)
    return [{"metric":name,"reference":a.tolist(),"repeat":b.tolist(),"absolute_difference":difference,
             "relative_difference":difference/magnitude if magnitude else None,"absolute_tolerance":absolute,"limit":limit,
             "status":"passed" if difference<=limit else "failed"}]


def _response_changes(reference,repeat):
    changes=[];floors=CONTRACT["absolute_tolerances"]
    for state in CONTRACT["states"]:
        left=reference.get("states",{}).get(state,{})
        right=repeat.get("states",{}).get(state,{})
        for key,units,vector in (("max_displacement_mm","displacement_mm",False),("bolt_force_n","force_n",False),
            ("contact_force_n","force_n",True),("section_force_n","force_n",True),("section_moment_nmm","moment_nmm",True)):
            changes.extend(_leaf_changes(left.get(key),right.get(key),state+"/"+key,floors[units],vector=vector))
        for region in CONTRACT["regions"]:
            values=[(item.get("regions") or {}).get(region,{}).get("volume_mean_nodal_von_mises_mpa") for item in (left,right)]
            changes.extend(_leaf_changes(*values,state+"/region/"+region,floors["region_stress_mpa"]))
    return changes


def assess_repeatability(reference_results,repeat_results,output_dir=None):
    """Freshly assess two retained reports; never solve, rebind or edit inputs."""
    contract=contract_evidence();reader_before=reader_source_hashes()
    records={"reference_results":str(Path(reference_results).resolve()),"repeat_results":str(Path(repeat_results).resolve())}
    cases=[];gates=[];files={str(CONTRACT_PATH):contract["sha256"]}
    def gate(name,passed,detail):gates.append({"name":name,"status":"passed" if passed is True else "failed","detail":detail})
    for label,path in (("reference",reference_results),("repeat",repeat_results)):
        try:
            case=_case_evidence(path)
            for name,passed in case["checks"].items():gate(label+"/"+name,passed,"Fresh retained numerical and identity evidence")
            for source,digest in case["source_files"].items():_remember(files,source,digest)
            cases.append(case)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            gate(label+"/evidence_available",False,str(exc));cases.append(None)
    changes=[];identity={}
    if all(cases):
        first,second=cases
        gate("distinct_native_run_locations",first["run"]!=second["run"],"One retained run cannot be compared with itself as a repeat")
        identity={"reference":first["identity"],"repeat":second["identity"]}
        for key in first["identity"]:
            gate("identical/"+key,(key=="geometry_revision" or first["identity"][key] is not None) and first["identity"][key]==second["identity"][key],
                 "No normalization, rebinding or reserialization of frozen input identity")
        changes=_response_changes(first["summary"],second["summary"])
        gate("both_state_repeatability",bool(changes) and all(row["status"]=="passed" for row in changes),"Every corresponding fixed response uses the frozen 0.1%/absolute-tolerance rule")
    unchanged=all(Path(path).is_file() and sha256(path)==digest for path,digest in files.items())
    reader_after=reader_source_hashes()
    gate("retained_sources_unchanged",unchanged,"Raw outputs, fields, exact inputs and referenced evidence remained unchanged during assessment")
    gate("reader_sources_unchanged",reader_before==reader_after,"All structural Python implementations remained unchanged during assessment")
    accepted=bool(gates) and all(row["status"]=="passed" for row in gates)
    report={"schema_version":1,"status":"passed" if accepted else "unaccepted","accepted":accepted,"gates":gates,
        "contract":contract,"identity":identity,"source_files":files,"reader_source_hashes":reader_before,"records":records,"changes":changes,
        "implementation_provenance":{**IMPLEMENTATION_PROVENANCE,"loaded_function_code_sha256":module_code_sha256(globals())},
        "solver_invoked":False,"scope":CONTRACT["scope"],"design_adequacy":"not_assessed"}
    if output_dir is not None:
        output=Path(output_dir);output.mkdir(parents=True,exist_ok=False);write_json(output/"repeatability-assessment.json",report)
    return report


def prepare_exact_replay(reference_results,output_dir):
    """Copy exact retained inputs into a new directory; never launch or rebind."""
    contract=contract_evidence();readers_before=reader_source_hashes();case=_case_evidence(reference_results)
    if not all(case["checks"].values()):raise ValueError("Replay preparation requires a fully accepted, currently bound reference run: "+", ".join(k for k,v in case["checks"].items() if not v))
    deck=Path(case["deck"]);text=deck.read_text(encoding="utf-8")
    if re.search(r"(?im)^\s*\*(?:INCLUDE\b|RESTART\s*,[^\n]*READ\b)",text):
        raise ValueError("Version1 exact replay requires a self-contained deck without INCLUDE or restart reads")
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    copied={}
    for name in (deck.name,deck.with_suffix(".assembly.json").name,"project.json","structural-mesh.npz","solver-mesh.npz"):
        source=deck.parent/name;target=output/name;shutil.copyfile(source,target)
        copied[str(target)]=_remember({},target,case["source_files"][str(source)])
    for path,digest in case["source_files"].items():_remember({},path,digest)
    if readers_before!=reader_source_hashes():raise ValueError("Reader sources changed during exact replay preparation")
    receipt={"schema_version":1,"status":"prepared_not_run","solver_invoked":False,"contract":contract,
        "reference_results":str(Path(reference_results).resolve()),"reference_identity":case["identity"],
        "reference_source_files":case["source_files"],"copied_inputs":copied,"input_path":str(output/deck.name),
        "execution":case["execution"],"reader_source_hashes":readers_before,
        "implementation_provenance":{**IMPLEMENTATION_PROVENANCE,"loaded_function_code_sha256":module_code_sha256(globals())},
        "rebind_performed":False,"note":"Original project and assembly receipts are copied byte-for-byte, including absent/null annotations; external absolute proof references remain in place and hash-pinned."}
    write_json(output/"exact-replay-manifest.json",receipt)
    return receipt


def execute_prepared_replay(manifest_path):
    """Explicit opt-in execution entry; not called by preparation or assessment."""
    from .backends.calculix import CalculixRunner
    manifest_path=Path(manifest_path).resolve();receipt=json.loads(manifest_path.read_text(encoding="utf-8"))
    if receipt.get("status")!="prepared_not_run" or receipt.get("solver_invoked") is not False:raise ValueError("Only a new prepared replay can be executed")
    if receipt.get("contract")!=contract_evidence():raise ValueError("Replay contract changed")
    if receipt.get("reader_source_hashes")!=reader_source_hashes():raise ValueError("Reader/runner sources changed since replay preparation")
    for group in ("reference_source_files","copied_inputs"):
        if not receipt.get(group):raise ValueError("Missing pinned replay inputs")
        for path,digest in receipt[group].items():_remember({},path,digest)
    deck=Path(receipt["input_path"]).resolve()
    if deck.parent!=manifest_path.parent or str(deck) not in receipt["copied_inputs"]:raise ValueError("Replay deck is outside its prepared directory")
    execution=receipt["execution"]
    expected=receipt["reference_identity"]
    current=_case_evidence(receipt["reference_results"])
    if not all(current["checks"].values()) or current["identity"]!=expected or current["execution"]!=execution:
        raise ValueError("Reference acceptance, identity or execution configuration changed since preparation")
    for role,name in {"deck":deck.name,"assembly":deck.with_suffix(".assembly.json").name,"project":"project.json",
                      "prepared_mesh":"structural-mesh.npz","solver_mesh":"solver-mesh.npz"}.items():
        target=deck.parent/name
        if str(target) not in receipt["copied_inputs"]:raise ValueError("Missing required exact replay input")
        _remember({},target,expected["byte_inputs"][role])
    if (sha256(execution["executable"])!=expected["solver_sha256"] or execution.get("threads")!=4
        or execution.get("expected_version")!=CONTRACT["required_solver_version"]
        or execution.get("expected_steps")!=2 or execution.get("expected_step_times")!=[1.,1.]):
        raise ValueError("Replay execution differs from the pinned backend/thread/step contract")
    started=manifest_path.parent/"exact-replay-execution.json"
    if started.exists():raise FileExistsError(started)
    state={"status":"starting","solver_invoked":True,"replay_manifest_sha256":sha256(manifest_path),"execution":execution}
    write_json(started,state)
    try:
        runner=CalculixRunner(execution["executable"],expected_version=execution["expected_version"],threads=execution["threads"])
        run=runner.run(deck,expected_steps=2,expected_step_times=[1.,1.],timeout_s=execution["timeout_s"])
        state.update(status=run.status,run=run.to_dict())
        state["reader_sources_unchanged"]=receipt["reader_source_hashes"]==reader_source_hashes()
        state["input_sources_unchanged"]=all(Path(path).is_file() and sha256(path)==digest
            for group in ("reference_source_files","copied_inputs") for path,digest in receipt[group].items())
        if not state["reader_sources_unchanged"] or not state["input_sources_unchanged"]:
            raise ValueError("Pinned replay inputs or implementation sources changed during execution")
        return run
    except BaseException as exc:
        state.update(status="failed",error=str(exc));raise
    finally:write_json(started,state)
