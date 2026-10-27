"""Frozen development gates from the approved plan, never a load rating.

Unavailable diagnostics are missing, not zero. A completed solver run is not
accepted unless every required numerical diagnostic has passed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any
from .provenance import capture_module_source

IMPLEMENTATION_PROVENANCE=capture_module_source(__file__,__name__)


@dataclass(frozen=True)
class Gate:
    name: str
    status: str
    measured: float | bool | None
    limit: float | bool | None
    detail: str


def upper_gate(name: str, value: Any, limit: float, detail: str) -> Gate:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return Gate(name, "missing", None, limit, detail)
    return Gate(name, "passed" if 0 <= value <= limit else "failed", value, limit, detail)


def bool_gate(name: str, value: Any, detail: str) -> Gate:
    return Gate(name, "passed" if value is True else "failed" if value is False else "missing", value if isinstance(value, bool) else None, True, detail)


def assess_assembly(metrics: dict[str, Any], solver_status: str) -> dict[str, Any]:
    """Metrics have explicit meanings/units; fractions are dimensionless, not percent."""
    gates = [
        bool_gate("solver_completed", solver_status == "completed", "Both preload and service states must converge to their final target times"),
        bool_gate("model_preparation", metrics.get("model_preparation_verified"), "Source, registration, selections, cuts and applied load resultant checked"),
        upper_gate("force_equilibrium", metrics.get("force_imbalance_fraction"), .005, "Norm of all external force/reaction imbalance divided by characteristic force"),
        upper_gate("moment_equilibrium", metrics.get("moment_imbalance_fraction"), .005, "All reactions and applied moments about registered midpoint, divided by F_characteristic * 233.68 mm"),
        upper_gate("preload", metrics.get("max_preload_relative_error"), .01, "Direct solver forces for each of the three complete screw cuts"),
        upper_gate("tightening_drift_mm", metrics.get("max_tightening_drift_mm"), 1e-8, "Locked cut adjustment difference between stages; absolute solver constraint check"),
        bool_gate("screw_force_redistribution", metrics.get("screw_forces_free_to_redistribute"), "Service step locks adjustment and removes force-control on each pretension DOF"),
        bool_gate("no_final_artificial_support", metrics.get("no_final_artificial_support"), "No artificial final restraint or unremoved stabilization"),
        upper_gate("physical_penetration_mm", metrics.get("max_physical_penetration_mm"), .0075, "Maximum physical overlap including initial gaps and any offsets"),
        upper_gate("open_interface_force_fraction", metrics.get("open_interface_force_fraction"), .005, "Regularization force on open interfaces divided by characteristic force"),
        bool_gate("compression_only_contact", metrics.get("compression_only_contact"), "Normal traction/gap sign audited on every separating pair"),
    ]
    if metrics.get("friction_required") is True or metrics.get("friction_coefficient",0) not in (None,0):
        gates.append(bool_gate("local_coulomb_bound",metrics.get("local_coulomb_bound"),
            "Nonzero friction requires a freshly verified native history qualification and both-state paired finite tractions with ||t|| <= mu*max(p,0) + 1e-5 MPa + 1e-6*max(mu*abs(p),||t||)"))
    accepted = all(g.status == "passed" for g in gates)
    return {"status": "numerically_accepted" if accepted else "unaccepted", "solver_status": solver_status,
            "gates": [asdict(g) for g in gates], "design_adequacy": "not_assessed",
            "scope": "Single-mesh numerical checks; mesh/contact sensitivity remains a separate release gate"}


def relative_change(previous: float, current: float, absolute_floor: float) -> float:
    if absolute_floor <= 0 or not all(math.isfinite(x) for x in (previous, current, absolute_floor)):
        raise ValueError("Finite metrics and a positive physically defined normalization floor are required")
    return abs(current - previous) / max(abs(current), absolute_floor)


def assess_mesh_study(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Ordered coarse-to-fine runs with identical physical model fingerprints."""
    if len(runs) < 3:
        return {"status": "incomplete", "reason": "At least three controlled meshes are required"}
    if any(r.get("acceptance_status") != "numerically_accepted" for r in runs):
        return {"status": "unaccepted", "reason": "Every study mesh must pass numerical checks"}
    identities = {r.get("physical_model_fingerprint") for r in runs}
    if len(identities) != 1 or None in identities:
        return {"status": "unaccepted", "reason": "Physical models or metric definitions differ or lack provenance"}
    sizes = [r.get("thickness_size_mm") for r in runs]
    if any(not isinstance(x, (int, float)) or x <= 0 for x in sizes) or any(a <= b for a, b in zip(sizes, sizes[1:])):
        return {"status": "unaccepted", "reason": "Meshes must have strictly decreasing thickness size"}
    definitions = {"max_displacement_mm": (.02, .001), "bolt_force_n": (.02, 1.0),
                   "contact_force_n": (.02, 1.0), "seat_region_stress_mpa": (.05, .1),
                   "slot_region_stress_mpa": (.05, .1)}
    gates = []
    for key, (limit, floor) in definitions.items():
        a, b = runs[-2].get(key), runs[-1].get(key)
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a:
            change = max(relative_change(x, y, floor) for x, y in zip(a, b))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            change = relative_change(a, b, floor)
        else:
            change = None
        gates.append(upper_gate(key, change, limit, f"Last-two mesh relative change; normalization floor {floor}"))
    return {"status": "passed" if all(g.status == "passed" for g in gates) else "unaccepted", "gates": [asdict(g) for g in gates]}
