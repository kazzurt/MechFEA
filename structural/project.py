"""Versioned, reproducible structural projects with explicit physical units."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DODEC = ROOT / "Dodec FEA"
SOURCE = DODEC / "Dodec Hub Conformal and 1.5mm pipe half.step"
SOURCE_HASH = "ca9207ebd900aec8013ddad45af0ac59b767afcb5f8ef2a2121909c665622d6f"
REFERENCE = Path(os.environ.get("MECHFEA_REFERENCE_DIR", str(DODEC / "reference")))
SCHEMA_VERSION = 1


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    """Atomic JSON replacement; never emit nonstandard NaN/Infinity tokens."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


@dataclass
class Material:
    name: str
    youngs_modulus_mpa: float
    poisson_ratio: float
    evidence: str = "Recorded COMSOL assumption; exact entered values unverified"

    def validate(self) -> None:
        if not math.isfinite(self.youngs_modulus_mpa) or self.youngs_modulus_mpa <= 0:
            raise ValueError("Young's modulus must be finite and positive in MPa")
        if not math.isfinite(self.poisson_ratio) or not -1 < self.poisson_ratio < 0.5:
            raise ValueError("Poisson ratio must be finite and between -1 and 0.5")


@dataclass
class MeshSettings:
    element_order: int = 2
    global_size_mm: float = 4.0
    thickness_size_mm: float = 0.75
    contact_size_mm: float = 0.75
    max_nodes: int = 500000
    threads: int = 2
    bore_snout_size_mm: float | None = None
    slot_size_mm: float | None = None


@dataclass
class LoadSettings:
    screw_preloads_n: list[float] = field(default_factory=lambda: [100.0, 100.0, 100.0])
    service_force_n: float = 1112.0554
    direction_reference: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    patch_width_mm: float = 35.0
    patch_axial_length_mm: float = 25.0
    patch_width_convention: str = "projected"
    patch_center_angle_deg: float | None = -90.0
    lock_tightening: bool = True


@dataclass
class ContactSettings:
    friction_coefficient: float = 0.0
    normal_stiffness_mpa_per_mm: float = 100000.0
    geometric_nonlinearity: bool = True
    anchorage: str = "tied_smooth_cylinder"
    hub: str = "fixed_rigid"
    final_artificial_supports: list[str] = field(default_factory=list)


@dataclass
class StructuralProject:
    schema_version: int = SCHEMA_VERSION
    name: str = "Dodec half-pipe baseline"
    units: dict[str, str] = field(default_factory=lambda: {"length": "mm", "force": "N", "stress": "MPa"})
    source_path: str = str(SOURCE)
    source_sha256: str = SOURCE_HASH
    geometry_revision: dict[str, Any] | None = None
    half_length_mm: float = 233.68
    pipe_od_mm: float = 38.0
    pipe_id_mm: float = 35.0
    pipe: Material = field(default_factory=lambda: Material("Pipe elastic assumption", 205000.0, 0.29))
    screw: Material = field(default_factory=lambda: Material("17-4 PH H1150", 201000.0, 0.291))
    mesh: MeshSettings = field(default_factory=MeshSettings)
    loads: LoadSettings = field(default_factory=LoadSettings)
    contacts: ContactSettings = field(default_factory=ContactSettings)
    registration: dict[str, Any] = field(default_factory=dict)
    selections: dict[str, Any] = field(default_factory=dict)
    preparation: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=lambda: {"backend": "calculix", "expected_version": "2.23", "timeout_s": 3600, "variant":"multithread", "threads":4})
    reference: dict[str, Any] = field(default_factory=lambda: {
        "preload": str(REFERENCE / "DodecV2_500lbf_100Nboltpreload_first.vtu"),
        "loaded": str(REFERENCE / "DodecV2_500lbf_100Nboltpreload_last.vtu"),
        "coordinate_units": "mm", "displacement_units": "mm", "stress_units": "Pa",
    })

    def validate(self, verify_source: bool = False) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported project schema {self.schema_version}; expected {SCHEMA_VERSION}")
        if self.units != {"length": "mm", "force": "N", "stress": "MPa"}:
            raise ValueError("Structural decks require explicit N-mm-MPa units")
        if self.geometry_revision is not None:
            revision = self.geometry_revision
            if (not isinstance(revision, dict)
                    or set(revision) != {"schema_version", "kind", "definition_sha256"}
                    or type(revision["schema_version"]) is not int or revision["schema_version"] != 1
                    or revision["kind"] != "derived_seat_trim_intersection"
                    or not isinstance(revision["definition_sha256"], str)
                    or len(revision["definition_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in revision["definition_sha256"])):
                raise ValueError("Derived geometry requires an identified seat-trim intersection definition")
        self.pipe.validate()
        self.screw.validate()
        for name, expected in (("half_length_mm", 233.68), ("pipe_od_mm", 38.0), ("pipe_id_mm", 35.0)):
            if not math.isclose(getattr(self, name), expected, abs_tol=1e-7, rel_tol=0):
                raise ValueError(f"V1 geometry is fixed: {name} must be {expected}; use an explicit geometry revision")
        if self.mesh.element_order != 2:
            raise ValueError("Assembly V1 requires quadratic solid elements")
        for size in (self.mesh.global_size_mm, self.mesh.thickness_size_mm, self.mesh.contact_size_mm):
            if not math.isfinite(size) or size <= 0:
                raise ValueError("Mesh sizes must be finite and positive")
        bore_size=self.mesh.bore_snout_size_mm
        if bore_size is not None and (isinstance(bore_size,bool) or not isinstance(bore_size,(int,float)) or not math.isfinite(bore_size) or bore_size<=0):
            raise ValueError("Optional bore/snout mesh size must be finite and positive")
        slot_size=self.mesh.slot_size_mm
        if slot_size is not None and (isinstance(slot_size,bool) or not isinstance(slot_size,(int,float)) or not math.isfinite(slot_size) or slot_size<=0):
            raise ValueError("Optional slot mesh size must be finite and positive")
        if self.mesh.max_nodes < 1 or self.mesh.threads < 1:
            raise ValueError("Mesh node limit and threads must be positive")
        if len(self.loads.screw_preloads_n) != 3 or any(not math.isfinite(x) or x < 0 for x in self.loads.screw_preloads_n):
            raise ValueError("Exactly three nonnegative whole-screw preloads are required")
        if not math.isfinite(self.loads.service_force_n) or self.loads.service_force_n < 0:
            raise ValueError("Service force must be finite and nonnegative")
        if len(self.loads.direction_reference) != 3 or not all(math.isfinite(x) for x in self.loads.direction_reference):
            raise ValueError("Load direction must be a finite 3-vector")
        if not math.isclose(sum(x*x for x in self.loads.direction_reference), 1, abs_tol=1e-8):
            raise ValueError("Load direction must be unit length (no implicit resultant rescaling)")
        if self.loads.patch_width_convention not in {"unresolved", "projected", "arc"}:
            raise ValueError("Patch convention must be unresolved, projected or arc")
        if not (0 < self.loads.patch_width_mm < self.pipe_od_mm) or not (0 < self.loads.patch_axial_length_mm <= self.half_length_mm):
            raise ValueError("Load footprint must fit on the modeled outer half-cylinder")
        if not self.loads.lock_tightening:
            raise ValueError("Service loading must retain tightening, leaving screw forces free")
        if self.contacts.anchorage != "tied_smooth_cylinder" or self.contacts.hub != "fixed_rigid":
            raise ValueError("V1 baseline requires smooth tied anchorage and a fixed rigid hub")
        if self.contacts.final_artificial_supports:
            raise ValueError("Artificial supports are forbidden in the final baseline")
        if not self.contacts.geometric_nonlinearity:
            raise ValueError("The matched baseline requires geometric nonlinearity")
        if not math.isfinite(self.contacts.friction_coefficient) or self.contacts.friction_coefficient < 0:
            raise ValueError("Friction coefficient must be finite and nonnegative")
        if not math.isfinite(self.contacts.normal_stiffness_mpa_per_mm) or self.contacts.normal_stiffness_mpa_per_mm <= 0:
            raise ValueError("Normal contact stiffness must be finite and positive")
        if self.contacts.friction_coefficient > 0 and "friction_qualification_manifest" in self.solver:
            from .backends.friction_qualification import validate_friction_reference
            validate_friction_reference(self.solver["friction_qualification_manifest"])
        if self.solver.get("backend") != "calculix":
            raise ValueError("Only the CalculiX backend is implemented")
        timeout=self.solver.get("timeout_s",3600)
        if not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError("Solver timeout must be finite and positive in seconds")
        threads=self.solver.get("threads",1)
        if not isinstance(threads,int) or isinstance(threads,bool) or threads<1:
            raise ValueError("Solver thread count must be a positive integer")
        self.static_step_controls()
        self.contact_count_relative_tolerance()
        if verify_source and sha256(self.source_path) != self.source_sha256:
            raise ValueError("Source CAD hash changed: rerun geometry audit and registration")

    def static_step_controls(self):
        """Numerical load-ramp controls; omitted fields preserve the legacy schedule."""
        controls=self.solver.get("step_controls",{
            "preload":{"initial_increment":.05,"minimum_increment":1e-6,"maximum_increment":.1},
            "service":{"initial_increment":.025,"minimum_increment":1e-6,"maximum_increment":.1}})
        if not isinstance(controls,dict) or set(controls)!={"preload","service"}:
            raise ValueError("Step controls require preload and service stages")
        for stage,values in controls.items():
            if not isinstance(values,dict) or set(values)!={"initial_increment","minimum_increment","maximum_increment"}:
                raise ValueError(f"Incomplete numerical increment controls for {stage}")
            if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) for value in values.values()):
                raise ValueError("Increment controls must be finite numbers")
            if not 0<values["minimum_increment"]<=values["initial_increment"]<=values["maximum_increment"]<=1:
                raise ValueError("Increment controls require 0 < minimum <= initial <= maximum <= 1")
        return controls

    def contact_count_relative_tolerance(self) -> float | None:
        """Optional tighter native active-contact-count convergence criterion.

        The CalculiX default is 0.001. This checks count changes, not the
        identities of active points or the final pressure sign.
        """
        value = self.solver.get("contact_count_relative_tolerance")
        if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= .001):
            raise ValueError("Contact count relative tolerance must be finite, positive and no greater than the native default 0.001")
        return value

    def readiness_issues(self) -> list[str]:
        """Physical preparation is required independently of a syntactically valid project."""
        issues = []
        if self.contacts.friction_coefficient > 0:
            from .backends.friction_qualification import verify_friction_qualification
            try:
                verify_friction_qualification(self)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                issues.append(str(exc))
        for key in ("cad_audit_passed", "three_physical_screws_verified", "cuts_verified", "contact_clearances_verified", "backend_fixtures_passed"):
            if self.preparation.get(key) is not True:
                issues.append(f"Missing verified preparation: {key}")
        revision_check=self.preparation.get("geometry_revision_verification", {})
        if self.geometry_revision and (revision_check.get("verified") is not True
                or revision_check.get("geometry_revision") != self.geometry_revision):
            issues.append("Derived geometry revision has no matching verified mesh binding")
        if self.geometry_revision is None and revision_check.get("geometry_revision") is not None:
            issues.append("Derived geometry evidence exists without its explicit project revision")
        if self.registration.get("status") != "verified":
            issues.append("CAD/reference registration remains unverified")
        if self.loads.patch_width_convention != "projected":
            issues.append("This preparation requires projected straight-line patch width")
        if self.loads.patch_center_angle_deg != -90.0:
            issues.append("This preparation requires the patch on reference -Y (center angle -90 degrees)")
        if self.loads.patch_axial_length_mm != 25.0:
            issues.append("This preparation requires the verified 25 mm axial patch length")
        for name in ("pipe", "hub", "midpoint", "load_patch", "pipe_bore", "hub_snout"):
            if not self.selections.get(name):
                issues.append(f"Missing named selection: {name}")
        for i in range(1, 4):
            for kind in ("body", "anchor", "cone", "seat", "cut"):
                if not self.selections.get(f"screw_{i}_{kind}"):
                    issues.append(f"Missing named selection: screw_{i}_{kind}")
        return issues

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data=asdict(self)
        # Optional refinement must not change fingerprints of retained projects
        # created before this setting existed.
        if data['mesh']['bore_snout_size_mm'] is None:
            del data['mesh']['bore_snout_size_mm']
        if data['mesh']['slot_size_mm'] is None:
            del data['mesh']['slot_size_mm']
        if data['geometry_revision'] is None:
            del data['geometry_revision']
        if data['solver'].get('contact_count_relative_tolerance') is None:
            data['solver'].pop('contact_count_relative_tolerance', None)
        return data

    def fingerprint(self) -> str:
        data = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(data.encode()).hexdigest()

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path, verify_source: bool = False) -> "StructuralProject":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or "schema_version" not in data or "source_sha256" not in data:
            raise ValueError("Project requires explicit schema_version and source_sha256")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown project fields: {sorted(unknown)}")
        for key, kind in (("pipe", Material), ("screw", Material), ("mesh", MeshSettings), ("loads", LoadSettings), ("contacts", ContactSettings)):
            if key in data:
                data[key] = kind(**data[key])
        project = cls(**data)
        project.validate(verify_source=verify_source)
        return project
