"""Verify retained numerical evidence before binding it to an assembly project."""
from __future__ import annotations

import json
from pathlib import Path

from .calculix import DEFAULT_EXECUTABLE, sha256


VERIFIED_VERSION = "2.23"
SERIAL_SHA256 = "27a9426a1ee2f72c38fbe892e0ffaf38e14867628ffd0235770261b6f3ec9c5f"
MULTITHREAD_SHA256 = "62349096f3b76cef2f0e094d03f747187016683a233ec7f07a363c7089804876"
PRETENSION_ID = "explicit_paired_node_axial_jump"
GENERIC_FIXTURES = frozenset(("tension", "bending", "contact", "contact_stiffer",
                            "preload_lock", "preload_contact", "preload_contact_open"))


def verify_backend_feasibility(report: dict, executable: str | Path = DEFAULT_EXECUTABLE,
                               *, expected_version: str = VERIFIED_VERSION) -> dict:
    """Return a verification receipt, or raise ValueError on missing/stale evidence.

    Every path/hash reference in the report is checked, including retained failed
    diagnoses. Acceptance is then checked independently in each required result;
    the top-level ``all_passed`` flag alone cannot establish feasibility. This
    verifies backend fixtures only, never the physical assembly's acceptance.
    """
    def require(condition, message):
        if not condition:
            raise ValueError("Backend feasibility evidence invalid: " + message)

    require(isinstance(report, dict), "report must be an object")
    require(report.get("schema_version") == 1, "unsupported evidence schema")
    require(report.get("all_passed") is True, "backend fixtures have not all passed")
    require(report.get("version") == expected_version == VERIFIED_VERSION, "solver version is not verified")
    require(report.get("required_assembly_pretension_id") == PRETENSION_ID,
            "explicit paired-node pretension is required")
    require(report.get("required_assembly_anchorage") == "fixed_master_elimination",
            "fixed rigid master elimination is required")
    require(report.get("solver_executable_sha256") == SERIAL_SHA256, "serial build hash is not verified")
    references = {}
    contents = {}

    def visit(value, label):
        if isinstance(value, dict):
            if "path" in value or "sha256" in value:
                require(isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str),
                        f"incomplete path/hash reference at {label}")
                path = Path(value["path"])
                require(path.is_file(), f"referenced file missing at {label}: {path}")
                require(sha256(path) == value["sha256"], f"referenced file hash changed at {label}: {path}")
                try:
                    contents[label] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Backend feasibility JSON unreadable at {label}: {path}") from exc
                references[label] = {"path": str(path.resolve()), "sha256": value["sha256"]}
            for key, item in value.items():
                visit(item, f"{label}.{key}" if label else key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{label}[{index}]")

    visit(report, "")

    def evidence(label):
        require(label in contents and isinstance(contents[label], dict), f"missing result {label}")
        return contents[label]

    def accepted(result, key, label):
        require(result.get(key) is True, f"{label} not accepted")
        checks = result.get("checks")
        if checks is not None:
            require(isinstance(checks, dict) and bool(checks) and all(v is True for v in checks.values()),
                    f"{label} has an unmet numerical check")

    def clean_run(run, expected_hash, label):
        require(isinstance(run, dict), f"missing run record for {label}")
        require(run.get("status") == "completed" and run.get("returncode") == 0,
                f"{label} solver did not complete cleanly")
        require(not run.get("errors") and not run.get("warnings") and not run.get("critical_warnings"),
                f"{label} has solver warnings or errors")
        require(run.get("solver_version") == VERIFIED_VERSION and run.get("executable_sha256") == expected_hash,
                f"{label} used an unverified solver build")
        steps = run.get("expected_steps")
        require(isinstance(steps, int) and steps > 0 and run.get("completed_steps") == list(range(1, steps + 1)),
                f"{label} lacks completed stages")
        inp = Path(run.get("input_path", ""))
        require(inp.is_file() and sha256(inp) == run.get("input_sha256"), f"{label} input is missing or changed")

    generic = evidence("generic_fixture_summary")
    require(GENERIC_FIXTURES.issubset(generic), "a required generic fixture is absent")
    for name in GENERIC_FIXTURES:
        accepted(generic[name], "accepted", name)
        clean_run(generic[name].get("run"), SERIAL_SHA256, name)
    analytic = evidence("explicit_analytic_fixture")
    accepted(report["explicit_analytic_fixture"], "accepted", "analytic report reference")
    accepted(analytic, "accepted", "explicit analytic fixture")
    clean_run(analytic.get("run"), SERIAL_SHA256, "explicit analytic fixture")
    shoulder = evidence("shoulder_fixture")
    accepted(report["shoulder_fixture"], "accepted", "shoulder report reference")
    accepted(shoulder, "accepted_fixture", "shoulder fixture")
    clean_run(evidence("shoulder_fixture.run"), SERIAL_SHA256, "shoulder fixture")
    mesh = evidence("shoulder_fixture.mesh")
    require(mesh.get("order") == 2 and mesh.get("full_cylinder_tied") is True and
            mesh.get("quadratic_cut_adjacent_contact_elements") == [], "shoulder mesh compatibility failed")
    require(str(mesh.get("pretension_formulation", "")).startswith("explicit paired-node displacement jump"),
            "shoulder mesh used a different pretension formulation")
    anchor = evidence("rigid_anchorage_equivalence")
    accepted(report["rigid_anchorage_equivalence"], "accepted", "anchorage report reference")
    accepted(anchor, "accepted_equivalence", "rigid anchorage equivalence")
    require(anchor.get("only_original_anchor_nodes_fixed") is True, "anchorage selection changed")
    accepted(anchor.get("verification", {}), "accepted_fixture", "rigid anchorage fixture")
    clean_run(anchor.get("run"), MULTITHREAD_SHA256, "rigid anchorage equivalence")
    mt = report.get("verified_multithread_build", {})
    require(mt.get("version") == VERIFIED_VERSION and mt.get("executable_sha256") == MULTITHREAD_SHA256,
            "multithread build hash/version is not verified")
    require(mt.get("matches_serial_printed_metrics") is True, "multithread equivalence is absent")
    mt_evidence = evidence("verified_multithread_build.evidence")
    accepted(mt_evidence.get("verification", {}), "accepted_fixture", "multithread shoulder")
    clean_run(mt_evidence.get("run"), MULTITHREAD_SHA256, "multithread shoulder")
    require(mt_evidence["verification"].get("metrics") == shoulder.get("metrics"),
            "serial and multithread shoulder metrics differ")
    executable = Path(executable)
    require(executable.is_file(), f"selected executable is missing: {executable}")
    selected_hash = sha256(executable)
    require(selected_hash in {SERIAL_SHA256, MULTITHREAD_SHA256}, "selected executable is not a verified build")
    return {"verified": True, "version": VERIFIED_VERSION,
            "selected_executable": str(executable.resolve()), "selected_executable_sha256": selected_hash,
            "pretension_formulation": PRETENSION_ID, "anchorage_enforcement": "fixed_master_elimination",
            "verified_json_references": references, "scope": "backend fixtures only; assembly acceptance remains separate"}
