"""Project evidence, immutable run directories and staged workflow entry points."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import uuid

from .project import DODEC, ROOT, StructuralProject, sha256, write_json


def new_run_directory(label: str, root: Path | None = None) -> Path:
    import re
    safe = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if not safe:
        raise ValueError("A descriptive run name is required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parent = root or DODEC / "structural-fea-benchmark-runs"
    path = Path(parent) / f"{safe}-{stamp}-{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def evidence_reference(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def initialize_project_from_evidence(*, patch_width_mm=35.0) -> StructuralProject:
    project = StructuralProject()
    project.loads.patch_width_mm = patch_width_mm
    if patch_width_mm == 30:
        project.name = "Dodec 30 mm COMSOL partition diagnostic"
    audit_path = DODEC / "structural-fea-model-preparation/cad-audit.json"
    registration_path = DODEC / "structural-fea-validation-deliverables/cad-reference-registration.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["source_sha256"] != project.source_sha256:
            raise ValueError("CAD audit refers to another source revision")
        project.preparation["audit"] = evidence_reference(audit_path)
        project.preparation["cad_audit_passed"] = all(b["valid"] for b in audit["bodies"]) and not audit["unresolved_interferences"]
        project.preparation["three_physical_screws_verified"] = len(audit["derived_body_map"]) == 5
        project.preparation["contact_clearances_verified"] = False  # pair distances alone do not verify all contact extents
        project.preparation["source_named_selections"] = audit["named_selections"]
    if registration_path.exists():
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        if registration["source_cad"]["sha256"] != project.source_sha256:
            raise ValueError("Registration refers to another source revision")
        project.registration = {"status": "verified", "evidence": evidence_reference(registration_path),
                                **registration["source_cad_to_reference"],
                                "geometry_equivalence": "not_yet_verified",
                                "comparison_classification": "revised_model" if patch_width_mm == 35 else "matched_partition_diagnostic"}
    project.validate(verify_source=True)
    return project


def freeze_run(project: StructuralProject, output: Path, *, purpose: str) -> dict:
    project.validate(verify_source=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "project.json").exists() or (output / "manifest.json").exists():
        raise FileExistsError("This run already has a manifest; create a new run directory")
    project.save(output / "project.json")
    manifest = {"schema_version": 1, "status": "prepared", "purpose": purpose,
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "project_fingerprint": project.fingerprint(), "source_sha256": project.source_sha256,
                "units": project.units, "python": platform.python_version(),
                "dependencies": evidence_reference(ROOT / "structural-fea-environment/requirements-lock.txt"),
                "solver_completion": "not_run", "numerical_acceptance": "unaccepted", "design_adequacy": "not_assessed"}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_evidence(project: StructuralProject) -> list[str]:
    issues = []
    for evidence in (project.preparation.get("audit"), project.registration.get("evidence")):
        if not evidence:
            issues.append("Required audit/registration evidence missing")
            continue
        try:
            if sha256(evidence["path"]) != evidence["sha256"]:
                issues.append(f"Evidence changed since project creation: {evidence['path']}")
        except (KeyError, FileNotFoundError):
            issues.append(f"Evidence file unavailable: {evidence.get('path')}")
    return issues
