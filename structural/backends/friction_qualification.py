"""Additive, hash-bound native friction qualification; μ=0 remains unchanged."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .calculix import DEFAULT_EXECUTABLE, sha256
from .feasibility import MULTITHREAD_SHA256, VERIFIED_VERSION


COULOMB_ABSOLUTE_TOLERANCE_MPA = 1e-5
COULOMB_RELATIVE_TOLERANCE = 1e-6
BUNDLED_QUALIFICATION = Path(__file__).resolve().parents[2]/"structural/contracts/backend-friction-qualification-v1.json"
BUNDLED_QUALIFICATION_SHA256 = "fff6b041f8ede58cddf3064829cb4267b7e2f7464d6d6cbe3fd3ca6fed484b66"
REQUIRED_FIXTURES = frozenset((
    "hex20_k_over_10", "hex20_k_over_100", "hex20_frictionless",
    "tetra10_k_over_10", "tetra10_k_over_100", "tetra10_mu_010", "tetra10_frictionless",
    "tetra10_cycle_k_over_10", "tetra10_cycle_k_over_100", "pretension_shoulder"))
HISTORY_CHECKS = frozenset((
    "stick/integrated_shear", "positive_sliding/integrated_shear",
    "elastic_unloading/integrated_shear", "reversed_sliding/integrated_shear",
    "opened_after_slip/all_contact_force_zero", "opened_after_slip/no_tangential_traction",
    "all_increments/coulomb_bound", "all_increments/normal_penalty_law",
    "all_increments/upper_contact_force_balance", "all_increments/global_force_balance"))
CYCLE_CHECKS = frozenset((
    "cycle/same_end_drive", "cycle/same_end_contact_force", "cycle/nonnegative_dissipated_work",
    "cycle/analytic_dissipated_work", "reclosed_at_fixed_drive/integrated_shear",
    "reclosed_stick/integrated_shear", "reclosed_stick/local_elastic_law",
    "reclosed_sliding/integrated_shear"))
SHOULDER_CHECKS = frozenset((
    "native_completed", "preload/direct_force", "preload/contact_axial_transfer",
    "preload/global_force_balance", "preload/global_moment_balance", "preload/coulomb_bound",
    "loaded/global_force_balance", "loaded/global_moment_balance", "loaded/coulomb_bound",
    "preload/nonzero_local_friction", "loaded/nonzero_local_friction",
    "service/locked_adjustment", "service/redistribution", "service/axial_equilibrium"))


def validate_friction_reference(reference):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference.get("path"), str) or not reference["path"]
            or not isinstance(reference.get("sha256"), str) or len(reference["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in reference["sha256"])):
        raise ValueError("Nonzero friction requires a hash-pinned friction_qualification_manifest")


def verify_friction_qualification(project, executable=None):
    """Recheck native law fixtures and selected build before any frictional deck.

    This verifies the numerical law, not the physical coefficient for a material
    pair or assembly spatial/increment convergence. No files or default fields
    are created for frictionless projects.
    """
    mu = project.contacts.friction_coefficient
    if mu == 0:
        return None

    def require(condition, message):
        if not condition:
            raise ValueError("Friction qualification invalid: " + message)

    require(isinstance(mu, (int, float)) and not isinstance(mu, bool) and math.isfinite(mu) and mu > 0,
            "coefficient must be finite and positive")
    reference = project.solver.get("friction_qualification_manifest")
    if "friction_qualification_manifest" not in project.solver:
        reference = {"path":str(BUNDLED_QUALIFICATION), "sha256":BUNDLED_QUALIFICATION_SHA256}
    validate_friction_reference(reference)
    checked = {}

    def check_file(ref, label):
        require(isinstance(ref, dict) and isinstance(ref.get("path"), str)
                and isinstance(ref.get("sha256"), str), f"missing path/hash for {label}")
        path = Path(ref["path"])
        require(path.is_file(), f"missing {label}: {path}")
        require(sha256(path) == ref["sha256"], f"changed {label}: {path}")
        checked[label] = {"path":str(path.resolve()), "sha256":ref["sha256"]}
        return path

    receipt_path = check_file(reference, "qualification_manifest")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema_version") == 1 and receipt.get("qualification_id") == "calculix_2_23_friction_v1",
            "unsupported qualification identity")
    require(receipt.get("all_passed") is True,
            "nonzero-friction assembly is unavailable: opening/reclosure history qualification has not passed")
    require(receipt.get("version") == project.solver.get("expected_version") == VERIFIED_VERSION,
            "solver version differs from qualified version")
    require(receipt.get("solver_executable_sha256") == MULTITHREAD_SHA256,
            "receipt does not identify the qualified native build")
    require(receipt.get("threads") == project.solver.get("threads") == 4,
            "friction fixtures qualify the four-thread build configuration")
    require(receipt.get("coulomb_absolute_tolerance_mpa") == COULOMB_ABSOLUTE_TOLERANCE_MPA
            and receipt.get("coulomb_relative_tolerance") == COULOMB_RELATIVE_TOLERANCE,
            "local law tolerance differs from the verified precision policy")
    fixtures = receipt.get("fixtures", {})
    require(isinstance(fixtures, dict) and set(fixtures) == REQUIRED_FIXTURES, "required native fixture is missing")
    required_run_artifacts = {}
    for name, ref in fixtures.items():
        report_path = check_file(ref, "fixture/"+name)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checks = report.get("checks", {})
        required = SHOULDER_CHECKS if name == "pretension_shoulder" else HISTORY_CHECKS
        if "cycle" in name:
            required = required | CYCLE_CHECKS
        require(report.get("accepted") is True and isinstance(checks, dict)
                and required.issubset(checks) and all(v is True for v in checks.values()),
                f"{name} has missing or failed checks")
        run = report.get("run", {})
        steps = 2 if name == "pretension_shoulder" else 10 if "cycle" in name else 6
        require(run.get("status") == "completed" and run.get("returncode") == 0
                and run.get("expected_steps") == steps and run.get("completed_steps") == list(range(1,steps+1))
                and not run.get("errors") and not run.get("warnings") and not run.get("critical_warnings"),
                f"{name} did not complete cleanly")
        require(run.get("solver_version") == VERIFIED_VERSION and run.get("executable_sha256") == MULTITHREAD_SHA256
                and run.get("threads") == 4, f"{name} build/configuration mismatch")
        raw = report.get("source_hashes", {})
        stem = "short_screw" if name == "pretension_shoulder" else "friction"
        target = report.get("targets", {})
        expected_mu = 0. if "frictionless" in name else .1 if name == "tetra10_mu_010" else .2
        expected_stiffness = 1e7 if name == "pretension_shoulder" else 1e5
        expected_slope = expected_stiffness/(100 if "over_100" in name else 10)
        require(target.get("mu") == expected_mu and target.get("normal_stiffness_MPa_per_mm") == expected_stiffness
                and target.get("stick_slope_MPa_per_mm") == expected_slope, f"{name} parameter identity mismatch")
        for filename in (stem+".inp", stem+".dat", stem+".frd", "targets.json"):
            require(filename in raw, f"{name} lacks raw source hash for {filename}")
            check_file({"path":str(report_path.parent/filename), "sha256":raw[filename]}, f"{name}/{filename}")
        require(run.get("input_sha256") == raw[stem+".inp"], f"{name} input identity mismatch")
        for suffix in (".run.json", ".stdout.log", ".stderr.log", ".sta", ".cvg"):
            required_run_artifacts[str((report_path.parent/(stem+suffix)).resolve())] = name
    artifacts = receipt.get("retained_artifacts", [])
    require(isinstance(artifacts, list) and len(artifacts) >= len(fixtures)*3,
            "raw run/log/recipe provenance is incomplete")
    artifact_paths = set()
    for i, ref in enumerate(artifacts):
        artifact = check_file(ref, f"artifact/{i}")
        require(str(artifact.resolve()) not in artifact_paths, "duplicate retained artifact")
        artifact_paths.add(str(artifact.resolve()))
    require(set(required_run_artifacts).issubset(artifact_paths), "native run/log/convergence artifact is absent")
    if executable is None:
        variant = project.solver.get("variant", "serial")
        require(variant in {"serial", "multithread"}, "unknown solver variant")
        executable = DEFAULT_EXECUTABLE.with_name("ccx_MT.exe") if variant == "multithread" else DEFAULT_EXECUTABLE
    executable = Path(executable)
    require(executable.is_file() and sha256(executable) == MULTITHREAD_SHA256,
            "selected executable is not the friction-qualified build")
    stiffness = project.contacts.normal_stiffness_mpa_per_mm
    require(isinstance(stiffness, (int, float)) and math.isfinite(stiffness) and stiffness > 0,
            "invalid normal stiffness")
    return {"verified":True, "qualification_id":receipt["qualification_id"],
            "qualification_manifest":checked["qualification_manifest"], "version":VERIFIED_VERSION,
            "selected_executable":str(executable.resolve()), "selected_executable_sha256":MULTITHREAD_SHA256,
            "threads":4, "friction_coefficient":mu, "normal_contact_stiffness_mpa_per_mm":stiffness,
            "friction_stick_slope_mpa_per_mm":stiffness/10,
            "coulomb_absolute_tolerance_mpa":COULOMB_ABSOLUTE_TOLERANCE_MPA,
            "coulomb_relative_tolerance":COULOMB_RELATIVE_TOLERANCE,
            "verified_references":checked,
            "scope":"Native numerical Coulomb law and explicit pretension coupling; material coefficient and assembly convergence remain unverified."}
