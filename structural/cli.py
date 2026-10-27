"""Command-line structural workflow; setup, provenance and failure states stay explicit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .project import DODEC, SOURCE, StructuralProject
from .workflow import initialize_project_from_evidence, new_run_directory, verify_evidence


def build_parser():
    parser = argparse.ArgumentParser(description="KurtsFEM static structural workflow (N-mm-MPa)")
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="Create a reproducible Dodec project from existing audit evidence")
    new.add_argument("project", type=Path)
    new.add_argument("--patch-width", type=float, choices=(30.0,35.0), default=35.0)
    check = sub.add_parser("check", help="Validate project and report missing physical preparation")
    check.add_argument("project", type=Path)
    audit = sub.add_parser("audit", help="Run kernel input audit without altering source CAD")
    audit.add_argument("--source", type=Path, default=SOURCE)
    audit.add_argument("--output", type=Path)
    fixtures = sub.add_parser("fixtures", help="Run analytic/contact/retained-tightening fixtures")
    fixtures.add_argument("--output", type=Path)
    fixtures.add_argument("--executable", type=Path)
    mesh = sub.add_parser("mesh", help="Prepare a registered, separate-interface quadratic pilot mesh")
    mesh.add_argument("project", type=Path)
    mesh.add_argument("--output", type=Path)
    mesh.add_argument("--enforce-wall-layers",action=argparse.BooleanOptionalAction,default=True,
                      help="Partition wall internally to enforce ceil(1.5/thickness_target) continuous radial layers")
    solve = sub.add_parser("solve", help="Run a verified mesh through preload and retained-tightening service stages")
    solve.add_argument("project", type=Path)
    solve.add_argument("mesh", type=Path)
    solve.add_argument("--output", type=Path)
    solve.add_argument("--cancel-file", type=Path)
    results = sub.add_parser("results", help="Reprocess retained assembly results with explicit acceptance gates")
    results.add_argument("mesh", type=Path)
    results.add_argument("input", type=Path)
    results.add_argument("--output", type=Path)
    compare=sub.add_parser("compare",help="Compare saved preload/service fields with registered COMSOL reference")
    compare.add_argument("results",type=Path)
    compare.add_argument("--reference",type=Path,default=DODEC/"structural-fea-validation-deliverables/comsol-reference-manifest.json")
    compare.add_argument("--output",type=Path)
    study=sub.add_parser("study",help="Run controlled mesh, contact stiffness or patch-width cases")
    study.add_argument("project",type=Path)
    study.add_argument("kind",choices=("mesh","contact_mesh","bore_contact_mesh","slot_mesh","contact","patch","friction","load","preload"))
    study.add_argument("--output",type=Path,required=True)
    study.add_argument("--mesh",type=Path)
    assess=sub.add_parser("assess-study",help="Assess retained runs without repeating their solves")
    assess.add_argument("kind",choices=("mesh","contact_mesh","bore_contact_mesh","slot_mesh","contact","patch","friction","load","preload"))
    assess.add_argument("results",type=Path,nargs="+",help="Ordered results.json paths; mesh cases coarse to fine")
    assess.add_argument("--output",type=Path,required=True)
    assess.add_argument("--baseline-index",type=int,help="Zero-based baseline case index for sensitivity comparisons")
    sub.add_parser("gui", help="Open the separate structural desktop workflow")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "new":
            if args.project.exists():
                raise FileExistsError("Project already exists; choose a new file or edit it in the application")
            project = initialize_project_from_evidence(patch_width_mm=args.patch_width)
            project.save(args.project)
            print(json.dumps({"project":str(args.project.resolve()), "fingerprint":project.fingerprint(),
                              "assembly_readiness_issues":project.readiness_issues()}, indent=2))
        elif args.command == "check":
            project = StructuralProject.load(args.project, verify_source=True)
            issues = verify_evidence(project) + project.readiness_issues()
            print(json.dumps({"configuration_valid":True, "fingerprint":project.fingerprint(),
                              "ready_for_assembly":not issues, "issues":issues}, indent=2))
            return 2 if issues else 0
        elif args.command == "audit":
            from .geometry import audit_step
            output = args.output or new_run_directory("cad-audit", DODEC / "structural-fea-model-preparation")
            report = audit_step(args.source, output)
            print(json.dumps({"output":str(output), "dimensions":report["dimensions"],
                              "excluded_duplicates":report["excluded_duplicates"],
                              "unresolved_interferences":report["unresolved_interferences"]}, indent=2))
        elif args.command == "fixtures":
            from .fixtures import run_fixtures
            output = args.output or new_run_directory("analytical-contact-preload-fixtures")
            report = run_fixtures(output, executable=args.executable)
            print(json.dumps(report, indent=2, default=str))
            if isinstance(report, dict) and any(isinstance(value, dict) and value.get("accepted") is False for value in report.values()):
                return 2
        elif args.command == "mesh":
            from .preparation import generate_pilot_mesh, prepare_registered_geometry
            from .workflow import freeze_run
            import inspect
            project = StructuralProject.load(args.project, verify_source=True)
            if project.geometry_revision:
                raise ValueError("This project uses a derived seat-trim geometry revision. Load its separately verified prepared mesh; generic CAD meshing does not apply the retained repair recipe.")
            output = args.output or new_run_directory("assembly-mesh", DODEC / "structural-fea-model-preparation")
            project.preparation["mesh_wall_layer_policy"]={"enforced":args.enforce_wall_layers,"definition":"ceil(1.5 mm / thickness target); shared quadratic internal interfaces"}
            freeze_run(project, output, purpose="derived-mesh-preparation")
            registered = DODEC / "structural-fea-model-preparation/registered-geometry/registered-geometry-map.json"
            audit = Path(project.preparation["audit"]["path"])
            if not registered.exists():
                prepare_registered_geometry(project.source_path, audit, registered.parent)
            kwargs = dict(far_size_mm=project.mesh.global_size_mm, contact_size_mm=project.mesh.contact_size_mm,
                          patch_projected_width_mm=project.loads.patch_width_mm, order=2)
            if "thickness_size_mm" in inspect.signature(generate_pilot_mesh).parameters:
                kwargs["thickness_size_mm"] = project.mesh.thickness_size_mm
            if "max_nodes" in inspect.signature(generate_pilot_mesh).parameters:
                kwargs["max_nodes"] = project.mesh.max_nodes
            kwargs["threads"]=project.mesh.threads
            kwargs["enforce_wall_layers"]=args.enforce_wall_layers
            if project.mesh.bore_snout_size_mm is not None:
                kwargs["bore_snout_size_mm"]=project.mesh.bore_snout_size_mm
            if project.mesh.slot_size_mm is not None:
                kwargs["slot_size_mm"]=project.mesh.slot_size_mm
            report = generate_pilot_mesh(registered, audit, output, **kwargs)
            print(json.dumps({"output":str(output), **{k:report.get(k) for k in ("status","retained_counts","mesh_quality","retained_mesh_quality","resource_limit_passed","elapsed_seconds")}},indent=2))
        elif args.command == "solve":
            from .workflow import freeze_run
            from .mesh import SolidMesh
            from .assembly import bind_mesh_evidence, write_assembly_deck
            from .backends.calculix import CalculixRunner, DEFAULT_EXECUTABLE
            from .results import postprocess_assembly, report_is_accepted
            from .project import write_json
            import threading
            project = StructuralProject.load(args.project, verify_source=True)
            mesh = SolidMesh.load(args.mesh)
            evidence_path = DODEC / "structural-fea-validation-deliverables/backend-feasibility.json"
            if not evidence_path.exists():
                raise ValueError("Representative screw pretension/contact feasibility is not yet verified")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            project = bind_mesh_evidence(project, mesh, backend_fixture_report=evidence)
            issues = verify_evidence(project) + project.readiness_issues()
            if issues:
                raise ValueError("\n".join(issues))
            output = args.output or new_run_directory("dodec-assembly")
            freeze_run(project, output, purpose="two-stage-assembly")
            mesh.save(output / "structural-mesh.npz")
            deck = output / "dodec_assembly.inp"
            write_assembly_deck(project, mesh, deck)
            mesh = SolidMesh.load(output / "solver-mesh.npz")
            cancellation = threading.Event()
            finished = threading.Event()
            if args.cancel_file:
                def monitor_cancel():
                    while not finished.wait(.2):
                        if args.cancel_file.exists():
                            cancellation.set()
                            break
                threading.Thread(target=monitor_cancel,daemon=True).start()
            try:
                last_report=[-10.]
                def report_progress(progress):
                    if progress["elapsed_s"]-last_report[0]>=5:
                        print(json.dumps({"elapsed_s":progress["elapsed_s"],"last_increment":progress["increments"][-1] if progress["increments"] else None}),flush=True)
                        last_report[0]=progress["elapsed_s"]
                executable=DEFAULT_EXECUTABLE.with_name("ccx_MT.exe") if project.solver.get("variant")=="multithread" else DEFAULT_EXECUTABLE
                run = CalculixRunner(executable,threads=project.solver.get("threads",1)).run(deck,expected_steps=2,timeout_s=project.solver.get("timeout_s",3600),cancel_event=cancellation,
                    on_progress=report_progress)
            finally:
                finished.set()
            manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
            # Persist solver outcome before postprocessing, including failed/truncated runs.
            manifest.update(status=run.status,solver_completion=run.status,numerical_acceptance="unaccepted",accepted=False)
            write_json(output/"manifest.json",manifest)
            report = postprocess_assembly(mesh,deck)
            manifest.update(numerical_acceptance=report.get("assessment",{}).get("status","unaccepted"),
                            accepted=report_is_accepted(report),results_report="structural-results/results.json")
            write_json(output/"manifest.json",manifest)
            print(json.dumps({"output":str(output),"solver_status":report["solver_status"],"accepted":report_is_accepted(report),"assessment":report["assessment"],"metrics":report["metrics"]},indent=2,default=str))
            return 0 if report_is_accepted(report) else 2
        elif args.command == "results":
            from .mesh import SolidMesh
            from .results import postprocess_assembly, report_is_accepted
            report = postprocess_assembly(SolidMesh.load(args.mesh),args.input,output_dir=args.output)
            print(json.dumps({"solver_status":report["solver_status"],"accepted":report_is_accepted(report),"assessment":report["assessment"],"metrics":report["metrics"]},indent=2,default=str))
            return 0 if report_is_accepted(report) else 2
        elif args.command == "compare":
            from .comparison import compare_run
            report=compare_run(args.reference,args.results,args.output)
            print(json.dumps({"screen_status":report["screen_status"],"candidate_numerically_accepted":report["candidate_numerically_accepted"],
                              "comparison_classification":report["comparison_classification"],"output":str(args.output or args.results.parent/"comsol-comparison")},indent=2))
            return 0
        elif args.command == "gui":
            from .structural_main import main as gui_main
            return gui_main()
        elif args.command == "study":
            from .studies import run_study
            report=run_study(StructuralProject.load(args.project,verify_source=True),args.kind,args.output,existing_mesh=args.mesh)
            print(json.dumps(report,indent=2,default=str))
            return 0 if report["status"] in ("passed","sensitivity_completed") else 2
        elif args.command == "assess-study":
            from .studies import assess_existing_runs
            if args.baseline_index is not None and not 0<=args.baseline_index<len(args.results):
                raise ValueError("Baseline index must identify one of the supplied result reports")
            report=assess_existing_runs(args.results,args.kind,args.output,baseline_index=args.baseline_index)
            print(json.dumps({"status":report["status"],"numerical_convergence_accepted":report["numerical_convergence_accepted"],
                              "output":str(args.output)},indent=2))
            return 0 if report["status"] in ("passed","sensitivity_completed") else 2
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"Structural workflow stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
