"""Change one numerical control family on the retained exact shoulder fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import psutil

from .calculix import CalculixRunner, DEFAULT_EXECUTABLE, ROOT, parse_frd, parse_sta, sha256
from .short_screw_fixture import evaluate_short_screw


def _canonical_physics(text):
    lines=text.splitlines(); skip=False; result=[]
    for line in lines:
        if skip:
            result.append("NUMERICAL_TIME_INCREMENT_CONTROLS");skip=False
        else:
            result.append(line)
            skip=line.strip().upper()=="*STATIC"
    return hashlib.sha256("\n".join(result).encode()).hexdigest()


def compare_nodal_frames(before,after):
    comparisons={}
    for time in (1.,2.):
        a={f["name"]:f for f in before["fields"] if f["time"]==time}
        b={f["name"]:f for f in after["fields"] if f["time"]==time}
        for name in sorted(set(a)|set(b)):
            key=f"{time}:{name}"
            if name not in a or name not in b or set(a[name]["values"])!=set(b[name]["values"]):
                comparisons[key]={"complete":False};continue
            ids=sorted(a[name]["values"])
            av=np.asarray([a[name]["values"][n] for n in ids]);bv=np.asarray([b[name]["values"][n] for n in ids])
            if av.shape!=bv.shape:
                comparisons[key]={"complete":False};continue
            delta=np.abs(av-bv)
            floor=.001 if name=="DISP" else 1.
            comparisons[key]={"complete":True,"nodes_compared":len(ids),"components":a[name]["components"][:av.shape[1]],
                "maximum_absolute_difference_per_component":delta.max(axis=0).tolist(),
                "maximum_reference_magnitude_per_component":np.abs(av).max(axis=0).tolist(),
                "normalized_maximum_difference":float(delta.max()/max(np.abs(av).max(),floor)),
                "normalization_floor":floor,"acceptance_field":name in {"DISP","STRESS","FORC"}}
    return comparisons


def verify(output, *, mode="timestep", threads=4):
    if mode=="timestep" and threads!=4:
        raise ValueError("Timestep verification holds threads at the verified baseline value4")
    baseline=ROOT/"Dodec FEA/structural-fea-benchmark-runs/fixtures/short-screw-explicit-jump-multithread4"
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=baseline/"short_screw.inp";original=source.read_text()
    lines=original.splitlines();changes=[];step=0
    for index,line in enumerate(lines):
        if line.strip().upper()=="*STATIC":
            step+=1
            if mode=="timestep":
                replacement=".1,1.,1.e-6,.25" if step==1 else ".05,1.,1.e-6,.25"
                changes.append({"step":step,"before":lines[index+1],"after":replacement})
                lines[index+1]=replacement
    if step!=2:raise ValueError("Representative fixture must contain exactly two static stages")
    candidate="\n".join(lines)+"\n"
    inp=output/"short_screw.inp";inp.write_text(candidate,encoding="ascii")
    shutil.copyfile(baseline/"mesh-manifest.json",output/"mesh-manifest.json")
    observed={};monitor=None;parent=psutil.Process()
    def progress(record):
        nonlocal monitor
        try:
            if monitor is None:
                monitor=next((p for p in parent.children() if Path(p.cwd()).resolve()==output and p.name().lower()=="ccx_mt.exe"),None)
            if monitor is not None:
                observed["sampled_cpu_seconds_lower_bound"]=sum(monitor.cpu_times()[:2])
                observed["peak_observed_process_threads"]=max(observed.get("peak_observed_process_threads",0),monitor.num_threads())
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    run=CalculixRunner(DEFAULT_EXECUTABLE.with_name("ccx_MT.exe"),threads=threads).run(inp,expected_steps=2,timeout_s=180,on_progress=progress)
    checked=evaluate_short_screw(inp) if run.completed else {"accepted_fixture":False}
    old=json.loads((baseline/"verification.json").read_text())
    nodal=compare_nodal_frames(parse_frd(baseline/"short_screw.frd"),parse_frd(inp.with_suffix(".frd"))) if run.completed else {}
    metric_changes={}
    if run.completed:
        for key,value in old["verification"]["metrics"].items():
            reference=np.asarray(value);current=np.asarray(checked["metrics"][key]);delta=np.abs(current-reference)
            metric_changes[key]={"max_absolute_difference":float(delta.max()),"reference":value,"current":checked["metrics"][key]}
    checks={"completed_cleanly":run.completed and not run.errors and not run.warnings,
            "fixture_acceptance":checked["accepted_fixture"],
            "physical_deck_unchanged":_canonical_physics(original)==_canonical_physics(candidate),
            "all_endpoint_nodal_fields_compared":bool(nodal) and all(row["complete"] for row in nodal.values()),
            "endpoint_displacement_stress_force_within_0_1_percent":bool(nodal) and all(row.get("normalized_maximum_difference",float("inf"))<=.001 for row in nodal.values() if row.get("acceptance_field")),
            "endpoint_scalar_force_tightening_within_0_1_percent":run.completed and all(
                row["max_absolute_difference"]<=.001*max(float(np.max(np.abs(row["reference"]))),1e-6 if "adjustment" in key else 1.)
                for key,row in metric_changes.items() if key!="service_axial_imbalance_N")}
    result={"accepted_numerical_control_verification":all(checks.values()),"checks":checks,"mode":mode,
            "baseline":{"path":str(baseline),"input_sha256":sha256(source),"verification_sha256":sha256(baseline/"verification.json"),
                        "elapsed_s":old["run"]["elapsed_s"],"peak_rss_bytes":old["run"]["peak_rss_bytes"],"threads":old["run"]["threads"]},
            "static_control_changes":changes,"run":run.to_dict(),"cpu_observations":observed,
            "verification":checked,"metric_changes":metric_changes,"endpoint_nodal_field_comparison":nodal,
            "increments":parse_sta(inp.with_suffix(".sta")),
            "normalization":"Nodal U max component uses .001mm floor; nodal stress/RF use1MPa/1N floor. CONTACT/ERROR fields are fully compared but not normalized acceptance fields.",
            "scope":"Same representative shoulder only; no assembly runtime or convergence guarantee. Automatic cutbacks and convergence tolerances unchanged.",
            "manual":{"path":str(ROOT/"structural-fea-environment/solver/documentation/CalculiX/ccx_2.23/doc/ccx/node350.html"),
                      "source_archive_url":"https://www.dhondt.de/ccx_2.23.htm.tar.bz2"}}
    (output/"numerical-control-verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("accepted_numerical_control_verification","checks","mode","run","cpu_observations","metric_changes")},indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True)
    parser.add_argument("--mode",choices=("timestep","threads"),default="timestep")
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    raise SystemExit(0 if verify(args.output,mode=args.mode,threads=args.threads)["accepted_numerical_control_verification"] else 1)
