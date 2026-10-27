"""Reassess retained spatial studies and an optional separate combined candidate.

No meshing/solving, source edits or overwrites. A spatial pass does not replace
contact-penalty qualification, benchmark interpretation or design assessment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE))
from structural.project import sha256,write_json
from structural.studies import assess_existing_runs,assess_mesh_release,summarize_run


def retained_records(report):
    """Recover immutable inputs; never treat imported summary flags as evidence."""
    records=[]
    for case in report.get("cases",[]):
        sources=case.get("source_files",{})
        if not all(name in sources for name in ("results","project","mesh","run")):
            raise ValueError("Every imported case requires retained results/project/mesh/run source records; regenerate the study assessment")
        for name,source in sources.items():
            path=Path(source["path"])
            if not path.is_file() or sha256(path)!=source.get("sha256"):
                raise ValueError(f"Retained {name} source no longer matches its study hash: {path}")
        records.append({name:sources[name]["path"] for name in ("results","project","mesh")})
    if not records:raise ValueError("Study has no retained case records")
    return records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wall_study",type=Path)
    parser.add_argument("screw_contact_study",type=Path)
    parser.add_argument("bore_contact_study",type=Path)
    parser.add_argument("--slot-study",type=Path,help="Retained independent slot study; required for full version2 local spatial release")
    parser.add_argument("--candidate-results",type=Path,help="Retained results.json for the separate combined spatial configuration")
    parser.add_argument("--output",type=Path,required=True,help="New assessment directory")
    args=parser.parse_args(argv)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source_names=("studies.py","results.py","comparison.py","mesh.py","project.py","validation.py","provenance.py","geometry_revision.py","backends/calculix.py","backends/friction_qualification.py","backends/contact_controls.py")
    recipe=Path(__file__).resolve()
    receipt={"status":"running","solver_invoked":False,"recipe_sha256":sha256(recipe),
             "sources":{str(path):sha256(path) for path in [recipe,*[WORKSPACE/"structural"/name for name in source_names]]}}
    write_json(output/"assessment-manifest.json",receipt)
    try:
        reassessed=[]
        families=[("mesh",args.wall_study),("contact_mesh",args.screw_contact_study),("bore_contact_mesh",args.bore_contact_study)]
        if args.slot_study:families.append(("slot_mesh",args.slot_study))
        for kind,path in families:
            source=json.loads(path.read_text(encoding="utf-8"))
            if source.get("study")!=kind:raise ValueError(f"Expected {kind} study: {path}")
            receipt["sources"][str(path.resolve())]=sha256(path)
            records=retained_records(source)
            for case in source["cases"]:
                for evidence in case["source_files"].values():receipt["sources"][evidence["path"]]=evidence["sha256"]
            current=assess_existing_runs(records,kind)
            for case in current.get("cases",[]):
                for evidence in case.get("source_files",{}).values():receipt["sources"][evidence["path"]]=evidence["sha256"]
            write_json(output/(kind+"-reassessed.json"),current);reassessed.append(current)
            print(json.dumps({"study":kind,"status":current["status"]}),flush=True)
        candidate=None
        if args.candidate_results:
            receipt["sources"][str(args.candidate_results.resolve())]=sha256(args.candidate_results)
            candidate=summarize_run(args.candidate_results)
            write_json(output/"combined-candidate-summary.json",candidate)
            for evidence in candidate["source_files"].values():receipt["sources"][evidence["path"]]=evidence["sha256"]
        report=assess_mesh_release(*reassessed[:3],candidate,**({"slot_mesh_study":reassessed[3]} if args.slot_study else {}))
        unchanged=all(Path(path).is_file() and sha256(path)==digest for path,digest in receipt["sources"].items())
        receipt["retained_sources_unchanged"]=unchanged
        if not unchanged:raise ValueError("Retained input changed while reassessing; spatial release cannot be promoted")
        write_json(output/"spatial-release-assessment.json",report)
        receipt.update(status="completed",release_mesh_accepted=report["release_mesh_accepted"])
        print(json.dumps({"output":str(output),"release_mesh_accepted":report["release_mesh_accepted"],
                          "independent_spatial_studies_accepted":report["independent_spatial_studies_accepted"],
                          "combined_finest_configuration_verified":report["combined_finest_configuration_verified"],"issues":report["issues"]},indent=2))
        return 0 if report["release_mesh_accepted"] else 2
    except BaseException as exc:
        receipt.update(status="failed",error=str(exc));raise
    finally:write_json(output/"assessment-manifest.json",receipt)


if __name__=="__main__":raise SystemExit(main())
