"""Link fresh numerical assessments into a Phase 0–6 software release record.

This is a read-only evidence operation. It neither launches a solver nor assigns
a load rating. Missing friction, spatial or repeatability evidence keeps the
approved full scope incomplete, even when a frictionless case is useful.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .project import sha256, write_json
from .results import report_is_accepted
from .studies import assess_mesh_release, assess_study, summarize_run


FAMILIES = ("mesh", "contact_mesh", "bore_contact_mesh", "slot_mesh", "contact", "friction")
IDENTITY_KEYS = ("physical_model_fingerprint", "solver_sha256", "metric_definition_sha256",
                 "numerical_controls_fingerprint")
BINDINGS = ("project_binding_verified", "mesh_binding_verified", "geometry_revision_binding_verified",
            "step_controls_binding_verified", "contact_controls_binding_verified", "mesh_settings_verified",
            "retained_result_source_binding_verified", "retained_component_field_binding_verified")


def _accepted(case):
    return (bool(case) and case.get("numerically_accepted") is True
            and all(case.get(key) is True for key in BINDINGS)
            and report_is_accepted(case.get("single_run_acceptance", {})))


def _same_identity(a, b):
    return bool(a and b) and all(a.get(key) is not None and a.get(key) == b.get(key) for key in IDENTITY_KEYS)


def _same_result(a, b):
    left = a.get("source_files", {}).get("results", {})
    right = b.get("source_files", {}).get("results", {})
    return bool(left.get("sha256") and left == right)


def evaluate_sections(baseline, candidate, studies, repeatability, comparison, software):
    """Assess freshly computed sections; use assemble_release_record for files."""
    gates = []

    def gate(name, condition, detail):
        gates.append({"name": name, "status": "passed" if condition is True else "failed", "detail": detail})

    gate("baseline", _accepted(baseline), "Baseline requires all assembly gates and current input/field/control bindings")
    gate("final_candidate", _accepted(candidate), "Final combined candidate independently passes the same requirements")
    gate("baseline_final_identity", _same_identity(baseline, candidate),
         "Baseline and final mesh share model, solver build, numerical controls and metric definitions")
    wall = studies.get("mesh", {})
    gate("baseline_in_wall_study", any(_same_result(baseline, row) for row in wall.get("cases", [])),
         "The declared baseline is a retained case in the three-level wall study")
    spatial = assess_mesh_release(wall, studies.get("contact_mesh", {}), studies.get("bore_contact_mesh", {}),
                                  candidate or None, slot_mesh_study=studies.get("slot_mesh", {}))
    gate("full_spatial_release", spatial.get("full_local_spatial_release_accepted") is True,
         "All four independent spatial families and their separate combined candidate must pass")
    penalty = studies.get("contact", {})
    gate("penalty_study", penalty.get("study") == "contact" and penalty.get("numerical_convergence_accepted") is True,
         "Recomputed same-mesh frictionless penalty bracket passes both-state 2%/5% criteria")
    gate("final_candidate_in_penalty_study", any(_same_result(candidate, row) for row in penalty.get("cases", [])),
         "Penalty evidence includes the actual final candidate")
    friction = studies.get("friction", {})
    coefficients = {row.get("friction_coefficient") for row in friction.get("cases", [])}
    gate("friction_sensitivity", friction.get("study") == "friction" and friction.get("status") == "sensitivity_completed"
         and {0., .1, .2}.issubset(coefficients),
         "Approved 0/0.1/0.2 sensitivity requires qualified native history and every independent assembly gate")
    gate("final_candidate_in_friction_study", any(_same_result(candidate, row) for row in friction.get("cases", [])),
         "Friction sensitivity includes the actual final frictionless candidate")
    gate("same_input_repeatability", repeatability.get("accepted") is True,
         "Fresh exact-input assessment passes its separately declared solver tolerance")
    gate("repeatability_on_final_candidate", repeatability.get("reference_is_final_candidate") is True,
         "Repeatability is demonstrated on the final combined assembly")
    gate("benchmark_interpretation", comparison.get("verified") is True,
         "Both 35 mm baseline and 30 mm diagnostic are freshly compared and tied to a retained interpretation")
    gate("current_software_verification", software.get("verified") is True,
         "Regression and native desktop smoke pass on the current, unchanged source snapshot")
    accepted = all(row["status"] == "passed" for row in gates)
    return {"schema_version": 1, "scope": "approved_phase_0_to_6", "status": "passed" if accepted else "incomplete",
            "phase_0_to_6_release_accepted": accepted, "gates": gates, "spatial_release": spatial,
            "design_adequacy": "not_assessed", "load_rating": "not_assessed",
            "cross_code_equivalence_claimed": False,
            "interpretation": "Numerical software release is separate from cross-code identity and physical design acceptance"}


def verify_software_snapshot(receipt_path, workspace):
    """Reject stale or partial source inventories, including newly added modules."""
    path = Path(receipt_path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    before_path = path.parent / data.get("source_hashes_before", "missing-before")
    after_path = path.parent / data.get("source_hashes_after", "missing-after")
    before = json.loads(before_path.read_text(encoding="utf-8-sig"))
    after = json.loads(after_path.read_text(encoding="utf-8-sig"))
    current_files = set()
    for relative in ("structural", "tests/structural"):
        current_files.update(str(p.resolve()) for p in (workspace / relative).rglob("*.py"))
    for relative in ("Dodec FEA/structural-fea-validation-deliverables", "Dodec FEA/structural-fea-validation-working"):
        current_files.update(str(p.resolve()) for p in (workspace / relative).glob("*.py"))
    current_files.add(str((workspace / "structural-fea-environment/verify-structural.ps1").resolve()))
    current = (set(before) == current_files and before == after
               and all(Path(p).is_file() and sha256(p) == digest for p, digest in before.items()))
    passed = (data.get("regression_exit_code") == 0 and data.get("desktop_exit_code") == 0
              and data.get("desktop_requested") is True and data.get("source_files_unchanged") is True
              and data.get("source_file_count") == len(before) and current)
    logs = [path.parent / "regression.log", path.parent / "desktop-smoke.log"]
    passed = passed and all(p.is_file() and p.stat().st_size > 0 for p in logs)
    return {"verified": bool(passed), "current_source_snapshot": bool(current), "receipt": data,
            "source_files": [_file(p) for p in (path, before_path, after_path, *logs)], "source_hashes": before}


def _file(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha256(p)}


def _collect_sources(value, ledger):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            p = str(Path(value["path"]).resolve())
            digest = value["sha256"]
            if p in ledger and ledger[p] != digest:
                raise ValueError(f"Conflicting source hashes: {p}")
            if not Path(p).is_file() or sha256(p) != digest:
                raise ValueError(f"Stale evidence source: {p}")
            ledger[p] = digest
        for item in value.values():
            _collect_sources(item, ledger)
    elif isinstance(value, list):
        for item in value:
            _collect_sources(item, ledger)


def collect_hash_map(records, ledger):
    """Consume the flat absolute-path maps used by repeatability receipts."""
    if not isinstance(records, dict) or not records:
        raise ValueError("Expected a nonempty path-to-hash source map")
    for path, digest in records.items():
        if not Path(path).is_absolute() or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Source hash map requires absolute paths and SHA256 values")
        _collect_sources({"path": path, "sha256": digest}, ledger)


def comparison_coverage(report):
    """Require complete diagnostics without converting agreement into validity."""
    failures = []
    observed_failures = []
    required = {(quantity, distance, None) for distance in (35., 75., 125., 175.)
                for quantity in ("section_mean_displacement_vector_mm", "section_force_vector_N", "section_moment_vector_Nmm")}
    required |= {("pipe_max_displacement_mm", None, None)}
    required |= {("finite_volume_mean_von_mises_MPa", None, region) for region in ("center", "slots", "seats")}

    def finite(value):
        if isinstance(value, list):
            return bool(value) and all(finite(v) for v in value)
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    if set(report.get("states", {})) != {"preload", "loaded"}:
        failures.append("Both saved comparison states are required")
    for state in ("preload", "loaded"):
        data = report.get("states", {}).get(state, {})
        components = data.get("components", {})
        for name in ("pipe", "screw_1", "screw_2", "screw_3"):
            item = components.get(name, {})
            if not (item.get("status") == "available" and item.get("full_reference_probe_coverage") is True
                    and isinstance(item.get("reference_valid_probe_count"), int) and item["reference_valid_probe_count"] > 0
                    and item.get("joint_valid_probe_count") == item["reference_valid_probe_count"]):
                failures.append(f"{state}/{name}: reference-valid probes are incomplete")
        screens = data.get("screens", [])
        keys = [(row.get("quantity"), row.get("distance_mm"), row.get("region")) for row in screens]
        if len(keys) != len(required) or set(keys) != required:
            failures.append(f"{state}: four vector sections, displacement and three regions are required exactly once")
        for row in screens:
            if not (type(row.get("observed_within_tolerance")) is bool and finite(row.get("candidate"))
                    and finite(row.get("reference"))):
                failures.append(f"{state}: missing/nonfinite comparison value")
            elif row["observed_within_tolerance"] is False:
                observed_failures.append({"state": state, **row})
        for side in ("reference", "candidate"):
            regions = components.get("pipe", {}).get("finite_volume_regions", {}).get(side, {})
            for region in ("center", "slots", "seats"):
                item = regions.get(region, {})
                volume = item.get("volume_mm3")
                if not (finite(volume) and volume > 0 and finite(item.get("volume_mean_nodal_von_mises_mpa"))):
                    failures.append(f"{state}/{side}/{region}: missing finite-volume measure")
    return {"complete": not failures, "failures": failures, "observed_screen_failures": observed_failures}


def verify_interpretation_receipt(path, expected, reports, ledger):
    """Bind the scientific review to both actual cases and their reader version."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1 or data.get("cross_code_equivalence_claimed") is not False:
        raise ValueError("Interpretation requires schema1 and explicit diagnostic-only scope")
    for key, record in expected.items():
        if data.get(key) != record:
            raise ValueError(f"Interpretation is stale or belongs to another case: {key}")
    contract_hashes = {report.get("section_screen_extension_sha256") for report in reports.values()}
    if len(contract_hashes) != 1 or None in contract_hashes or data.get("section_screen_extension_sha256") not in contract_hashes:
        raise ValueError("Interpretation does not match the current comparison extension")
    if not isinstance(data.get("reviewed_discrepancies"), list) or not data["reviewed_discrepancies"] or not all(
            isinstance(item, str) and item.strip() for item in data["reviewed_discrepancies"]):
        raise ValueError("Interpretation must identify reviewed discrepancies and identity limitations")
    review_file = data.get("interpretation", {})
    if not review_file.get("path") or review_file != _file(review_file["path"]):
        raise ValueError("Interpretation text requires a current path/hash binding")
    _collect_sources(review_file, ledger)
    if not review_file.get("path") or not Path(review_file["path"]).read_text(encoding="utf-8-sig").strip():
        raise ValueError("Missing scientific interpretation")
    _collect_sources(data, ledger)
    _collect_sources(_file(path), ledger)
    return data


def assemble_release_record(manifest_path, output_dir):
    """Freshly reassess an explicit evidence manifest; never solve or overwrite."""
    from .comparison import compare_run
    from .repeatability import assess_repeatability

    manifest_path = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = Path(__file__).resolve().parents[1]
    ledger = {str(manifest_path): sha256(manifest_path)}
    for p in (workspace / "structural").rglob("*.py"):
        ledger[str(p.resolve())] = sha256(p)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Release manifest requires schema_version 1")
    errors = []
    summaries = {}

    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()

    def summarize(value):
        path = resolve(value)
        key = str(path)
        if key not in summaries:
            summaries[key] = summarize_run(path)
            _collect_sources(summaries[key], ledger)
        return summaries[key]

    def attempt(name, function, fallback):
        try:
            return function()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"section": name, "error": str(exc)})
            return fallback

    baseline = attempt("baseline", lambda: summarize(manifest["baseline_results"]), {})
    candidate = attempt("final_candidate", lambda: summarize(manifest["final_results"]), {})
    studies = {}
    for kind in FAMILIES:
        def compute(kind=kind):
            records = manifest.get("studies", {}).get(kind, [])
            rows = [summarize(record) for record in records]
            return assess_study(rows, kind)
        studies[kind] = attempt(kind, compute, {})
        write_json(output / (kind + "-reassessed.json"), studies[kind])

    def repeat():
        data = assess_repeatability(resolve(manifest["final_results"]), resolve(manifest["repeat_results"]))
        data["reference_is_final_candidate"] = True
        collect_hash_map(data["source_files"], ledger)
        _collect_sources(data, ledger)
        return data
    repeatability = attempt("repeatability", repeat, {})
    write_json(output / "repeatability-reassessed.json", repeatability)

    def comparisons():
        config = manifest["comparison"]
        reference = resolve(config["reference_manifest"])
        interpretation = resolve(config["interpretation_receipt"])
        matched = summarize(config["matched30_results"])
        compatible = (_accepted(baseline) and _accepted(matched)
                      and baseline.get("patch_width_mm") == 35 and matched.get("patch_width_mm") == 30
                      and baseline.get("controlled_fingerprints", {}).get("patch") is not None
                      and baseline.get("controlled_fingerprints", {}).get("patch") == matched.get("controlled_fingerprints", {}).get("patch")
                      and all(baseline.get(k) is not None and baseline.get(k) == matched.get(k)
                              for k in ("solver_sha256", "metric_definition_sha256", "numerical_controls_fingerprint",
                                        "thickness_size_mm", "contact_size_mm", "effective_bore_snout_size_mm", "effective_slot_size_mm",
                                        "far_size_mm", "anchor_size_mm")))
        reports = {}
        for label, result in (("baseline35", manifest["baseline_results"]), ("matched30", config["matched30_results"])):
            reports[label] = compare_run(reference, resolve(result), output / (label + "-comparison"), make_plots=False)
            _collect_sources(reports[label], ledger)
        refs = json.loads(reference.read_text(encoding="utf-8-sig"))
        _collect_sources(refs, ledger)
        coverage = {key: comparison_coverage(report) for key, report in reports.items()}
        expected = {"baseline_results": _file(resolve(manifest["baseline_results"])),
                    "matched30_results": _file(resolve(config["matched30_results"])),
                    "reference_manifest": _file(reference),
                    "comparison_reader": _file(workspace / "structural/comparison.py"),
                    "study_reader": _file(workspace / "structural/studies.py")}
        interpreted = verify_interpretation_receipt(interpretation, expected, reports, ledger)
        evidence = {"verified": bool(compatible and all(row["complete"] for row in coverage.values())),
                    "classification": "diagnostic_comparison_with_documented_identity_limits",
                    "interpretation": interpreted, "coverage": coverage, "reports": reports}
        _collect_sources(evidence, ledger)
        return evidence
    comparison = attempt("comparison", comparisons, {})
    write_json(output / "comparison-evidence.json", comparison)
    software = attempt("software", lambda: verify_software_snapshot(resolve(manifest["software_verification"]), workspace), {})
    _collect_sources(software, ledger)
    if software.get("source_hashes"):
        collect_hash_map(software["source_hashes"], ledger)
    write_json(output / "software-evidence.json", software)
    report = evaluate_sections(baseline, candidate, studies, repeatability, comparison, software)
    unchanged = all(Path(p).is_file() and sha256(p) == digest for p, digest in ledger.items())
    if software.get("verified") is True:
        unchanged = unchanged and verify_software_snapshot(resolve(manifest["software_verification"]), workspace)["verified"]
    report["gates"].append({"name": "retained_sources_unchanged", "status": "passed" if unchanged else "failed",
                            "detail": "All consumed source files and current reader modules are hash-bound before/after assessment"})
    if not unchanged or errors:
        report.update(status="incomplete", phase_0_to_6_release_accepted=False)
    report.update(recorded_utc=datetime.now(timezone.utc).isoformat(), solver_invoked=False, errors=errors,
                  source_files=[{"path": path, "sha256": digest} for path, digest in sorted(ledger.items())])
    write_json(output / "release-record.json", report)
    lines = ["# Phase 0–6 release evidence", "", f"Status: **{report['status']}**. No solver was invoked.", "",
             "| Requirement | Status |", "|---|---|"]
    lines += [f"| {row['name']} | {row['status']} |" for row in report["gates"]]
    lines += ["", "This record does not claim cross-code equivalence, local strength or a load rating."]
    (output / "release-record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
