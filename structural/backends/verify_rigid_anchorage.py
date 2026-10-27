"""Verify exact elimination of continuity MPCs with a fixed rigid master.

An anchorage tie enforces u_slave = sum(N_master * u_master). When every master
DOF is prescribed zero and ADJUST=NO, this reduces identically to u_slave=0.
Only nodes on the physically selected anchorage are constrained.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from .calculix import CalculixRunner, DEFAULT_EXECUTABLE, parse_frd, sha256
from .short_screw_fixture import evaluate_short_screw
from ..mesh import TET10_FACES


def verify(output):
    root=Path(__file__).resolve().parents[2]
    baseline=root/"Dodec FEA/structural-fea-benchmark-runs/fixtures/short-screw-explicit-jump-multithread4"
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    source=(baseline/"short_screw.inp").read_text().splitlines()
    elements={}
    anchors=[]
    mode=None
    for line in source:
        if line.startswith("*"):
            mode="element" if line.startswith("*ELEMENT,") else "anchor" if line=="*SURFACE,NAME=ANCHOR" else None
        elif mode=="element":
            values=list(map(int,line.split(",")))
            elements[values[0]]=values[1:]
        elif mode=="anchor":
            element,face=line.split(",")
            anchors.append((int(element),int(face[1:])))
    nodes=sorted({elements[e][i] for e,f in anchors for i in TET10_FACES[f]})
    lines=[]
    skip=False
    inserted=False
    for line in source:
        if skip:
            skip=False
            continue
        if line.startswith("*TIE,NAME=FULL_CYLINDER"):
            skip=True
            continue
        if line.startswith("*STEP") and not inserted:
            lines += ["** Exact elimination of continuity to the fixed rigid hub; original anchor surface only",
                      "*NSET,NSET=RIGID_ANCHORAGE"]
            lines += [",".join(map(str,nodes[i:i+16])) for i in range(0,len(nodes),16)]
            lines += ["*BOUNDARY","RIGID_ANCHORAGE,1,3"]
            inserted=True
        lines.append(line)
    inp=output/"short_screw.inp"
    inp.write_text("\n".join(lines)+"\n",encoding="ascii")
    (output/"mesh-manifest.json").write_text((baseline/"mesh-manifest.json").read_text())
    run=CalculixRunner(DEFAULT_EXECUTABLE.with_name("ccx_MT.exe"),threads=4).run(inp,expected_steps=2,timeout_s=120)
    check=evaluate_short_screw(inp) if run.completed else {"accepted_fixture":False}
    differences={}
    if run.completed:
        before=parse_frd(baseline/"short_screw.frd")
        after=parse_frd(inp.with_suffix(".frd"))
        for tm in (1.,2.):
            a=next(f for f in before["fields"] if f["name"]=="DISP" and f["time"]==tm)["values"]
            b=next(f for f in after["fields"] if f["name"]=="DISP" and f["time"]==tm)["values"]
            differences[str(tm)]=max(float(np.linalg.norm(np.asarray(a[n])-b[n])) for n in a if n in b)
    previous=json.loads((baseline/"verification.json").read_text())["verification"]["metrics"]
    compared=("preload_force_N","service_force_N","adjustment_preload_mm","adjustment_service_mm","service_axial_imbalance_N")
    scalar_diffs={key:abs(check["metrics"][key]-previous[key]) for key in compared} if run.completed else {}
    exact=bool(run.completed and check["accepted_fixture"] and max(differences.values())<=1e-8 and
               scalar_diffs["service_force_N"]<=1e-5 and scalar_diffs["adjustment_service_mm"]<=1e-10)
    result={"accepted_equivalence":exact,"mathematical_identity":"u_slave = sum(N_master*u_master) = 0 when all rigid hub master DOFs are fixed and ADJUST=NO",
            "baseline":str(baseline.resolve()),"baseline_input_sha256":sha256(baseline/"short_screw.inp"),
            "anchorage_selected_nodes":len(nodes),"only_original_anchor_nodes_fixed":True,
            "max_nodal_displacement_difference_mm_by_time":differences,"absolute_scalar_metric_differences":scalar_diffs,
            "run":run.to_dict(),"verification":check,
            "scope":"Fixed rigid hub only; never substitute for a deformable or moving master. No additional pipe supports."}
    (output/"equivalence-verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    raise SystemExit(0 if verify(args.output)["accepted_equivalence"] else 1)
