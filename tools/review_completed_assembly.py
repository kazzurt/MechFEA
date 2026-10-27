"""Fresh retained-run review: results, geometry binding, study metrics and COMSOL.

Run only after solver completion; this recipe never starts a solver or polls one.
Usage: python review_completed_assembly.py RUN_DIRECTORY --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from structural.project import sha256, write_json


def reader_source_hashes():
    names=("results.py","comparison.py","studies.py","mesh.py","project.py",
           "validation.py","provenance.py","geometry_revision.py","backends/calculix.py","backends/friction_qualification.py",
           "backends/contact_controls.py")
    return {name:sha256(WORKSPACE/"structural"/name) for name in names}


def reference_source_hashes(manifest, *, verify_declared=True):
    manifest=Path(manifest).resolve()
    data=json.loads(manifest.read_text(encoding="utf-8"))
    result={}
    for state in ("preload","loaded"):
        source=data["states"][state]["source"]
        path=Path(source["path"])
        if not path.is_absolute():path=manifest.parent/path
        digest=sha256(path)
        if verify_declared and digest!=source.get("sha256"):
            raise ValueError(f"{state} reference VTU does not match its retained manifest hash")
        result[str(path.resolve())]=digest
    return result


def source_run(directory):
    directory = Path(directory).resolve()
    manifests = list(directory.glob("*.run.json"))
    if len(manifests) != 1:
        raise ValueError("Run directory must contain exactly one retained .run.json")
    run = json.loads(manifests[0].read_text(encoding="utf-8"))
    if run.get("status") != "completed" or run.get("returncode") != 0:
        raise ValueError("Review requires a completed retained native run; live/failed runs are not processed by this recipe")
    deck = Path(run["input_path"]).resolve()
    if deck.parent != directory or not deck.is_file() or sha256(deck) != run.get("input_sha256"):
        raise ValueError("Retained input path/hash does not match the supplied completed run")
    required = [directory / name for name in ("project.json", "structural-mesh.npz", "solver-mesh.npz")]
    required += [deck.with_suffix(ext) for ext in (".assembly.json", ".dat", ".frd", ".sta", ".run.json")]
    if any(not path.is_file() for path in required):
        raise ValueError("Completed-run review requires frozen project, prepared/solver meshes and all retained final outputs")
    optional = [deck.with_suffix(ext) for ext in (".stdout.log", ".stderr.log", ".cvg")]
    optional += [directory / name for name in ("manifest.json", "provenance-correction.json")]
    files = [deck, *required, *(path for path in optional if path.is_file())]
    return deck, {str(path): sha256(path) for path in files}


def worker(args):
    """One fresh reader process per expensive stage; no native solve invocation."""
    deck, _ = source_run(args.run_directory)
    output = args.output.resolve()
    if args.stage == "postprocess":
        from structural.mesh import SolidMesh
        from structural.results import postprocess_assembly, report_is_accepted
        report = postprocess_assembly(SolidMesh.load(deck.parent / "solver-mesh.npz"), deck, output_dir=output)
        print(json.dumps({"results": str(output / "results.json"), "accepted": report_is_accepted(report)}))
        return 0 if report_is_accepted(report) else 2
    if args.stage == "metrics":
        from structural.studies import summarize_run
        output.mkdir(parents=True, exist_ok=False)
        summary = summarize_run(args.results, project_path=deck.parent / "project.json", mesh_path=deck.parent / "solver-mesh.npz")
        write_json(output / "study-summary.json", summary)
        print(json.dumps({"summary": str(output / "study-summary.json"),
                          "geometry_binding_verified": summary.get("geometry_revision_binding_verified")}))
        return 0
    from structural.comparison import compare_run
    report = compare_run(args.reference, args.results, output, make_plots=not args.no_plots)
    print(json.dumps({"comparison": str(output / "comparison.json"), "screen_status": report["screen_status"]}))
    return 0


def stage_command(args, stage, output, results=None):
    command = [sys.executable, str(Path(__file__).resolve()), str(args.run_directory.resolve()),
               "--output", str(output), "--stage", stage, "--reference", str(args.reference.resolve())]
    if results is not None:
        command += ["--results", str(results)]
    if args.no_plots:
        command.append("--no-plots")
    return command


def run_stage(command, directory, name):
    stdout, stderr = directory / f"{name}.stdout.log", directory / f"{name}.stderr.log"
    record = {"arguments": command, "cwd": str(WORKSPACE), "status": "running",
              "stdout": str(stdout), "stderr": str(stderr)}
    receipt = directory / f"{name}.command.json"
    write_json(receipt, record)
    try:
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            process = subprocess.run(command, cwd=WORKSPACE, stdout=out, stderr=err, check=False,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        record.update(status="finished", exit_code=process.returncode)
        return process.returncode
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        write_json(receipt, record)


def concise_review(report, metrics, comparison):
    from structural.results import report_is_accepted
    gates = report.get("assessment", {}).get("gates", [])
    summary = {"solver_status": report.get("solver_status"), "numerically_accepted": report_is_accepted(report)
                   and metrics.get("numerically_accepted") is True
                   and metrics.get("retained_result_source_binding_verified") is True
                   and metrics.get("retained_component_field_binding_verified") is True,
               "retained_result_source_binding_verified": metrics.get("retained_result_source_binding_verified") is True,
               "retained_result_source_binding": metrics.get("retained_result_source_binding"),
               "retained_component_field_binding_verified": metrics.get("retained_component_field_binding_verified") is True,
               "retained_component_field_binding": metrics.get("retained_component_field_binding"),
               "geometry_revision": report.get("geometry_revision"),
               "geometry_binding_verified": metrics.get("geometry_revision_binding_verified") is True,
               "geometry_binding": metrics.get("geometry_revision_evidence"),
               "gate_counts": {status: sum(gate.get("status") == status for gate in gates) for status in ("passed", "failed", "missing")},
               "gates": gates, "pretension": report.get("pretension"), "stages": {},
               "comparison_status": comparison.get("screen_status"),
               "comparison_identity_verified": comparison.get("geometry_settings_identity_verified"),
               "comparison_classification": comparison.get("comparison_classification"),
               "convergence_acceptance": "not_established_by_one_run", "design_adequacy": "not_assessed"}
    for name, stage in report.get("stages", {}).items():
        summary["stages"][name] = {key: stage.get(key) for key in
            ("time", "equilibrium", "contact", "raw_stress_complete_c3d10", "raw_stress_integration_point_rows")}
        summary["stages"][name]["components"] = {component: {key: values.get(key) for key in
            ("status", "max_displacement_mm", "raw_nodal_peak_von_mises_MPa", "stress_location", "reason")}
            for component, values in stage.get("components", {}).items()}
    return summary


def markdown_review(summary):
    def value(item):
        if item is None:
            return "unavailable"
        return f"{item:.9g}" if isinstance(item, float) else json.dumps(item) if isinstance(item, (dict, list)) else str(item)
    lines = ["# Completed assembly review", "",
             f"Numerical acceptance: **{summary['numerically_accepted']}**. Geometry binding: **{summary['geometry_binding_verified']}**.",
             f"Comparison: `{summary['comparison_status']}`; classification `{summary['comparison_classification']}`.",
             "One run does not establish mesh/contact convergence or design adequacy.", "",
             "| Gate | Status | Measured | Limit |", "|---|---|---:|---:|"]
    for gate in summary["gates"]:
        lines.append(f"| {gate['name']} | {gate['status']} | {value(gate.get('measured'))} | {value(gate.get('limit'))} |")
    lines += ["", "Geometry revision: `" + value(summary["geometry_revision"]) + "`.", "",
              "| State | Force residual fraction | Moment residual fraction | Penetration, mm | Raw IP fields complete |",
              "|---|---:|---:|---:|---|"]
    for state, row in summary["stages"].items():
        eq, contact = row.get("equilibrium") or {}, row.get("contact") or {}
        lines.append(f"| {state} | {value(eq.get('force_imbalance_fraction'))} | {value(eq.get('moment_imbalance_fraction'))} | {value(contact.get('max_physical_penetration_mm'))} | {value(row.get('raw_stress_complete_c3d10'))} |")
    lines += ["", "Cut forces, saved adjustment drift, contact-pair vectors/gaps, component extrema and geometry evidence are retained in `review-summary.json`.",
              "Fixed physical section and finite-volume metrics are in `study-metrics/study-summary.json`; both-state cross-code diagnostics are in `comsol-diagnostics/`.",
              "Native raw files and prior reports were not modified. Undefined gaps and stresses remain undefined; no geometric tolerance is subtracted from penetration."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New review root; existing directories are refused")
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("comsol-reference-manifest.json"))
    parser.add_argument("--no-plots", action="store_true", help="Keep comparison tables but omit figures")
    parser.add_argument("--stage", choices=("postprocess", "metrics", "compare"), help=argparse.SUPPRESS)
    parser.add_argument("--results", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.stage:
        return worker(args)
    _, before = source_run(args.run_directory)
    if not args.reference.is_file():
        raise ValueError("Reference manifest is required before starting a review")
    reader_before=reader_source_hashes()
    reference_before=reference_source_hashes(args.reference)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = output / "review-commands"
    commands.mkdir()
    receipt = {"schema_version": 1, "status": "running", "run_directory": str(args.run_directory.resolve()),
               "started_utc": datetime.now(timezone.utc).isoformat(), "recipe_sha256": sha256(__file__),
               "reference_manifest_sha256": sha256(args.reference), "source_hashes_before": before,
               "reader_source_hashes_before":reader_before,"reference_source_hashes_before":reference_before,
               "solver_invoked": False, "output": str(output), "stages": []}
    write_json(output / "review-manifest.json", receipt)
    try:
        results = output / "fresh-results/results.json"
        for stage, destination in (("postprocess", "fresh-results"), ("metrics", "study-metrics"), ("compare", "comsol-diagnostics")):
            code = run_stage(stage_command(args, stage, output / destination, None if stage == "postprocess" else results), commands, stage)
            receipt["stages"].append({"stage": stage, "exit_code": code})
            write_json(output / "review-manifest.json", receipt)
            print(json.dumps({"review_stage": stage, "exit_code": code}), flush=True)
            if code not in ((0, 2) if stage == "postprocess" else (0,)):
                raise RuntimeError(f"{stage} failed with exit code {code}; inspect retained review-commands logs")
        report = json.loads(results.read_text(encoding="utf-8"))
        metrics = json.loads((output / "study-metrics/study-summary.json").read_text(encoding="utf-8"))
        comparison = json.loads((output / "comsol-diagnostics/comparison.json").read_text(encoding="utf-8"))
        _, after = source_run(args.run_directory)
        reader_after=reader_source_hashes()
        reference_after=reference_source_hashes(args.reference,verify_declared=False)
        receipt.update(source_hashes_after=after, source_files_unchanged=before == after,
                       reader_source_hashes_after=reader_after,reader_sources_unchanged=reader_before==reader_after,
                       reference_source_hashes_after=reference_after,reference_sources_unchanged=reference_before==reference_after,
                       recipe_unchanged=receipt["recipe_sha256"] == sha256(__file__),
                       reference_manifest_unchanged=receipt["reference_manifest_sha256"] == sha256(args.reference))
        if before != after:
            raise RuntimeError("Retained source files changed during review; generated artifacts cannot be promoted")
        if not receipt["recipe_unchanged"] or not receipt["reference_manifest_unchanged"]:
            raise RuntimeError("Review recipe or reference manifest changed during processing; generated artifacts cannot be promoted")
        if not receipt["reader_sources_unchanged"] or not receipt["reference_sources_unchanged"]:
            raise RuntimeError("Reader implementation or reference VTU changed during review; generated artifacts cannot be promoted")
        summary = concise_review(report, metrics, comparison)
        write_json(output / "review-summary.json", summary)
        (output / "review-summary.md").write_text(markdown_review(summary), encoding="utf-8")
        receipt.update(status="completed", numerically_accepted=summary["numerically_accepted"],
                       geometry_binding_verified=summary["geometry_binding_verified"])
        print(json.dumps({"review": str(output), "gate_counts": summary["gate_counts"],
                          "numerically_accepted": summary["numerically_accepted"],
                          "geometry_binding_verified": summary["geometry_binding_verified"],
                          "comparison_status": summary["comparison_status"]}, indent=2))
        return 0 if summary["numerically_accepted"] and summary["geometry_binding_verified"] else 2
    except BaseException as exc:
        receipt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "review-manifest.json", receipt)


if __name__ == "__main__":
    raise SystemExit(main())
