"""Registered, body-aware comparison of structural fields in N, mm and MPa.

COMSOL's VTU is a sampled linear tetrahedral field, not the solver mesh.
Interpolation is restricted to containing export tetrahedra in explicitly
selected domains. Holes, contact interfaces and pretension jumps are never
bridged by a Delaunay reconstruction, nearest-neighbour fill or node welding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
from .provenance import capture_module_source, function_code_sha256

IMPLEMENTATION_PROVENANCE=capture_module_source(__file__,__name__)


COMSOL_BODY_DOMAINS = {
    "hub": (1,), "pipe": (2,), "bolt_lower_left": (3, 4),
    "bolt_upper_left": (5, 6), "bolt_right": (7, 8),
}
REFERENCE_REGIONS_MM = {"center": (0.0, 25.0), "slots": (190.0, 201.0),
                        "seats": (208.0, 226.0)}
STRESS_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")
SECTION_SCREEN_EXTENSION_V1 = {
    "schema_version":1,
    "contract_id":"comsol_section_vector_screen_extension_v1",
    "declared_utc":"2026-10-18T07:35:20Z",
    "timing":"Defined after the original assembly solve; not part of the original pre-solve screen freeze",
    "free_span_distances_mm":[35.,75.,125.,175.],
    "section_coordinate_tolerance_mm":1e-9,
    "vector_rule":"Euclidean norm(candidate-reference) <= max(absolute tolerance, relative tolerance*Euclidean norm(reference))",
    "displacement_tolerances":"Unchanged original reference-manifest displacement relative/absolute tolerances: 0.05 and 1e-5 mm",
    "force_tolerances":"Unchanged original reference-manifest section-force relative/absolute tolerances: 0.05 and 1 N",
    "moment_relative_tolerance":.05,
    "moment_characteristic_length_mm":233.68,
    "moment_absolute_tolerance_definition":"Original 1 N section-force tolerance multiplied by existing 233.68 mm half-length = 233.68 Nmm",
    "moment_origin":"Registered section center [0,0,-distance_mm], matching the retained reference reconstruction",
    "other_sections":"Retained as diagnostics; excluded from section-screen acceptance",
    "scope":"Both final states; four required free-span stations; maximum pipe displacement and fixed-volume stress screens remain unchanged",
    "prior_artifacts":"Original reference manifest and earlier comparison reports remain unchanged",
}


@dataclass(frozen=True)
class RigidTransform:
    """Map source to target: x_target = R @ x_source + t, without scaling."""

    rotation: np.ndarray
    translation_mm: np.ndarray

    def __post_init__(self):
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation_mm, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("Rigid transform needs a 3x3 rotation and 3-vector translation")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("Transform must be finite")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10, rtol=0):
            raise ValueError("Rotation must be orthonormal; scale/shear is forbidden")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0):
            raise ValueError("Registration must preserve handedness")
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "translation_mm", translation.copy())

    def points(self, points_mm):
        return np.asarray(points_mm) @ self.rotation.T + self.translation_mm

    def vectors(self, vectors):
        return np.asarray(vectors) @ self.rotation.T

    def tensors(self, tensors):
        return np.einsum("ij,...jk,lk->...il", self.rotation, tensors, self.rotation)

    def inverse(self):
        return RigidTransform(self.rotation.T, -self.rotation.T @ self.translation_mm)

    def to_dict(self):
        return {"convention": "target_mm = rotation @ source_mm + translation_mm",
                "rotation": self.rotation.tolist(), "translation_mm": self.translation_mm.tolist()}


def fit_rigid_transform(source_landmarks_mm, target_landmarks_mm):
    """Fit paired, noncollinear physical landmarks; do not fit scale or reflect."""
    source = np.asarray(source_landmarks_mm, dtype=float)
    target = np.asarray(target_landmarks_mm, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Landmarks must be matching N x 3 arrays")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("At least three finite paired landmarks are needed")
    a, b = source - source.mean(0), target - target.mean(0)
    if min(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)) < 2:
        raise ValueError("Collinear landmarks do not resolve rotation about the pipe")
    u, _, vh = np.linalg.svd(a.T @ b)
    correction = np.diag([1.0, 1.0, np.linalg.det(vh.T @ u.T)])
    rotation = vh.T @ correction @ u.T
    transform = RigidTransform(rotation, target.mean(0) - rotation @ source.mean(0))
    residual = np.linalg.norm(transform.points(source) - target, axis=1)
    return transform, {"rms_residual_mm": float(np.sqrt(np.mean(residual ** 2))),
                       "max_residual_mm": float(residual.max()), "landmark_count": len(source)}


def symmetric_tensor(components):
    """Convert [..., xx yy zz xy xz yz] components to true tensor shear."""
    c = np.asarray(components, dtype=float)
    if c.shape[-1] != 6:
        raise ValueError("Six stress components required")
    result = np.zeros(c.shape[:-1] + (3, 3))
    result[..., 0, 0], result[..., 1, 1], result[..., 2, 2] = c[..., 0], c[..., 1], c[..., 2]
    result[..., 0, 1] = result[..., 1, 0] = c[..., 3]
    result[..., 0, 2] = result[..., 2, 0] = c[..., 4]
    result[..., 1, 2] = result[..., 2, 1] = c[..., 5]
    return result


def von_mises(stress_mpa):
    s = np.asarray(stress_mpa)
    dev = s - np.trace(s, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))


def file_provenance(path):
    path = Path(path).resolve()
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


@dataclass
class FieldMesh:
    points_mm: np.ndarray
    tetrahedra: np.ndarray
    cell_domains: np.ndarray
    displacement_mm: np.ndarray
    stress_mpa: np.ndarray
    point_fields: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points_mm = np.asarray(self.points_mm, dtype=float)
        self.tetrahedra = np.asarray(self.tetrahedra, dtype=np.int64)
        self.cell_domains = np.asarray(self.cell_domains, dtype=np.int64)
        self.displacement_mm = np.asarray(self.displacement_mm, dtype=float)
        self.stress_mpa = np.asarray(self.stress_mpa, dtype=float)
        n = len(self.points_mm)
        if self.points_mm.shape != (n, 3) or self.tetrahedra.ndim != 2 or self.tetrahedra.shape[1] != 4:
            raise ValueError("FieldMesh requires points and linear tetrahedral connectivity")
        if self.displacement_mm.shape != (n, 3) or self.stress_mpa.shape != (n, 3, 3):
            raise ValueError("Displacement/stress shapes must correspond to points")
        if self.cell_domains.shape != (len(self.tetrahedra),):
            raise ValueError("Each tetrahedron needs an explicit physical domain")
        if not np.isfinite(self.points_mm).all():
            raise ValueError("Mesh coordinates must be finite")
        if self.tetrahedra.size and (self.tetrahedra.min() < 0 or self.tetrahedra.max() >= n):
            raise ValueError("Invalid tetrahedral point indices")

    def transformed(self, transform: RigidTransform):
        return FieldMesh(transform.points(self.points_mm), self.tetrahedra.copy(),
                         self.cell_domains.copy(), transform.vectors(self.displacement_mm),
                         transform.tensors(self.stress_mpa), dict(self.point_fields),
                         {**self.provenance, "applied_transform": transform.to_dict()})

    def body(self, domains: Iterable[int]):
        domains = tuple(domains)
        ids = np.flatnonzero(np.isin(self.cell_domains, domains))
        if not len(ids):
            raise ValueError(f"No tetrahedra in selected domains {domains}")
        used, inverse = np.unique(self.tetrahedra[ids], return_inverse=True)
        return FieldMesh(self.points_mm[used], inverse.reshape(-1, 4), self.cell_domains[ids],
                         self.displacement_mm[used], self.stress_mpa[used],
                         {key: np.asarray(value)[used] for key, value in self.point_fields.items()},
                         {**self.provenance, "selected_domains": list(domains)})


def read_comsol_vtu(path, *, coordinates_to_mm=1.0, displacement_to_mm=1.0, stress_to_mpa=1e-6):
    """Read explicit assumed units; no magnitude-based auto-detection or welding."""
    import meshio

    if min(coordinates_to_mm, displacement_to_mm, stress_to_mpa) <= 0:
        raise ValueError("Unit conversion factors must be positive")
    mesh = meshio.read(path)
    if any(block.type != "tetra" for block in mesh.cells):
        raise ValueError("COMSOL reader accepts sampled linear tetrahedra only")
    tetrahedra = np.vstack([block.data for block in mesh.cells])
    domains = np.asarray(mesh.point_data["Entity_index"]).reshape(-1)
    if not np.isfinite(domains).all() or not np.equal(domains, np.rint(domains)).all():
        raise ValueError("Entity_index must contain finite integer domain labels")
    cell_labels = domains[tetrahedra]
    if np.any(np.ptp(cell_labels, axis=1) != 0):
        raise ValueError("A sampled tetrahedron crosses domain labels")
    displacement = np.column_stack([mesh.point_data[f"Displacement_field,_{c}-component"].reshape(-1)
                                    for c in "XYZ"]) * displacement_to_mm
    stress = symmetric_tensor(np.column_stack([
        mesh.point_data[f"Stress_tensor,_{c}-component"].reshape(-1) for c in STRESS_COMPONENTS
    ]) * stress_to_mpa)
    fields = {}
    for name in ("von_Mises_stress", "von_Mises_stress_Gp"):
        if name in mesh.point_data:
            fields[name + "_MPa"] = np.asarray(mesh.point_data[name]).reshape(-1) * stress_to_mpa
    provenance = file_provenance(path)
    provenance.update({"reader": "meshio", "source_representation": "exported_linear_tetrahedral_samples",
                       "unit_factors": {"coordinates_to_mm": coordinates_to_mm,
                                        "displacement_to_mm": displacement_to_mm, "stress_to_mpa": stress_to_mpa},
                       "source_units_explicit_in_vtu": False})
    return FieldMesh(mesh.points * coordinates_to_mm, tetrahedra, cell_labels[:, 0].astype(int),
                     displacement, stress, fields, provenance)


class BodyInterpolator:
    """Containing-cell interpolation on one selected body/domain set.

    A probe on a discontinuous cut is invalid if the two traces disagree.
    Callers can select one partition explicitly to request a one-sided value.
    """

    def __init__(self, mesh: FieldMesh, domains: Iterable[int]):
        from scipy.spatial import cKDTree

        self.mesh = mesh.body(domains)
        self.q = self.mesh.points_mm[self.mesh.tetrahedra]
        self.centers = self.q.mean(1)
        matrix = np.transpose(self.q[:, 1:] - self.q[:, :1], (0, 2, 1))
        determinant = np.linalg.det(matrix)
        if np.any(np.abs(determinant) <= 1e-14):
            raise ValueError("Selected export contains degenerate tetrahedra")
        self.inverse = np.linalg.inv(matrix)
        self.max_radius = np.linalg.norm(self.q - self.centers[:, None], axis=2).max()
        self.tree = cKDTree(self.centers)

    def sample(self, points_mm, *, barycentric_tolerance=1e-9, trace_tolerance=1e-7):
        points = np.atleast_2d(np.asarray(points_mm, dtype=float))
        if points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Probes must be finite N x 3 coordinates")
        values_u = np.full((len(points), 3), np.nan)
        values_s = np.full((len(points), 3, 3), np.nan)
        valid = np.zeros(len(points), dtype=bool)
        ambiguous = np.zeros(len(points), dtype=bool)
        candidates_count = np.zeros(len(points), dtype=int)
        for i, point in enumerate(points):
            ids = np.asarray(self.tree.query_ball_point(point, self.max_radius * (1 + 1e-10)), dtype=int)
            if not len(ids):
                continue
            b123 = np.einsum("nij,nj->ni", self.inverse[ids], point - self.q[ids, 0])
            weights = np.column_stack((1 - b123.sum(1), b123))
            inside = np.all(weights >= -barycentric_tolerance, axis=1) & np.all(weights <= 1 + barycentric_tolerance, axis=1)
            ids, weights = ids[inside], weights[inside]
            candidates_count[i] = len(ids)
            if not len(ids):
                continue
            nodes = self.mesh.tetrahedra[ids]
            u = np.einsum("ni,nij->nj", weights, self.mesh.displacement_mm[nodes])
            s = np.einsum("ni,nijk->njk", weights, self.mesh.stress_mpa[nodes])
            if not np.isfinite(u).all() or not np.isfinite(s).all():
                continue
            # Relative/absolute trace agreement prevents averaging pretension jumps.
            if not (np.allclose(u, u[0], atol=trace_tolerance, rtol=trace_tolerance)
                    and np.allclose(s, s[0], atol=trace_tolerance, rtol=trace_tolerance)):
                ambiguous[i] = True
                continue
            values_u[i], values_s[i], valid[i] = u[0], s[0], True
        return {"points_mm": points, "displacement_mm": values_u, "stress_mpa": values_s,
                "valid": valid, "ambiguous_trace": ambiguous, "containing_cells": candidates_count}


def integrate_section(mesh: FieldMesh, domains, origin_mm, normal, *, moment_origin_mm=None):
    """Integrate sigma.n and r cross sigma.n over an undeformed planar section.

    Triangles use exact integration for linear stress/displacement and the
    quadratic moment integrand. Results describe exported stress samples;
    they are not direct solver reactions. Select one side at pretension cuts.
    """
    body = mesh.body(domains)
    origin = np.asarray(origin_mm, dtype=float)
    normal = np.asarray(normal, dtype=float)
    if origin.shape != (3,) or normal.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError("Finite section origin and nonzero normal required")
    normal = normal / np.linalg.norm(normal)
    moment_origin = origin if moment_origin_mm is None else np.asarray(moment_origin_mm, dtype=float)
    q = body.points_mm[body.tetrahedra]
    distances = (q - origin) @ normal
    selected = np.flatnonzero((distances.min(1) <= 1e-10) & (distances.max(1) >= -1e-10))
    helper = np.eye(3)[np.argmin(np.abs(normal))]
    e1 = np.cross(normal, helper); e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    area_total = 0.0
    force = np.zeros(3); moment = np.zeros(3); displacement_integral = np.zeros(3)
    seen_faces = {}
    triangle_count = 0
    for cell in selected:
        d = distances[cell]
        weights = []
        for a in range(4):
            if abs(d[a]) <= 1e-10:
                weights.append(np.eye(4)[a])
            for b in range(a + 1, 4):
                if d[a] * d[b] < -1e-20:
                    w = np.zeros(4); w[a] = d[b] / (d[b] - d[a]); w[b] = 1 - w[a]
                    weights.append(w)
        if len(weights) < 3:
            continue
        weights = np.asarray(weights)
        xyz = weights @ q[cell]
        # An interior face cut appears once for each adjacent tetrahedron.
        key = tuple(sorted(tuple(np.round(p, 9)) for p in xyz))
        nodes = body.tetrahedra[cell]
        u = weights @ body.displacement_mm[nodes]
        s = np.einsum("ni,ijk->njk", weights, body.stress_mpa[nodes])
        if not np.isfinite(u).all() or not np.isfinite(s).all():
            raise ValueError("Section intersects undefined field values")
        if key in seen_faces:
            previous = seen_faces[key]
            current = {tuple(np.round(p, 9)): (ui, si) for p, ui, si in zip(xyz, u, s)}
            if any(not (np.allclose(current[k][0], previous[k][0], atol=1e-7, rtol=1e-7) and
                            np.allclose(current[k][1], previous[k][1], atol=1e-7, rtol=1e-7)) for k in current):
                raise ValueError("Section lies on a discontinuous interface; select one partition")
            continue
        seen_faces[key] = {tuple(np.round(p, 9)): (ui, si) for p, ui, si in zip(xyz, u, s)}
        center = xyz.mean(0)
        order = np.argsort(np.arctan2((xyz - center) @ e2, (xyz - center) @ e1))
        xyz, u, s = xyz[order], u[order], s[order]
        for j in range(1, len(xyz) - 1):
            ids = [0, j, j + 1]
            p, us, stress = xyz[ids], u[ids], s[ids]
            area = np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) / 2
            traction = stress @ normal
            r = p - moment_origin
            area_total += area
            force += area * traction.mean(0)
            moment += area / 12 * (np.cross(r.sum(0), traction.sum(0)) + np.cross(r, traction).sum(0))
            displacement_integral += area * us.mean(0)
            triangle_count += 1
    if area_total == 0:
        raise ValueError("Section does not intersect positive material area")
    return {"area_mm2": float(area_total), "force_N": force.tolist(), "moment_Nmm": moment.tolist(),
            "mean_displacement_mm": (displacement_integral / area_total).tolist(),
            "normal": normal.tolist(), "origin_mm": origin.tolist(),
            "moment_origin_mm": moment_origin.tolist(), "triangles": triangle_count,
            "interpretation": "reconstructed from undeformed linear export samples; not solver reactions"}


def compare_scalar(candidate, reference, *, relative_tolerance, absolute_tolerance):
    """Near-zero criteria are explicit and frozen before any candidate solve."""
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Tolerances must be nonnegative")
    if not np.isfinite([candidate, reference]).all():
        return {"passed": False, "reason": "nonfinite metric"}
    difference = abs(candidate - reference)
    allowed = max(absolute_tolerance, relative_tolerance * abs(reference))
    return {"candidate": float(candidate), "reference": float(reference), "absolute_difference": float(difference),
            "allowed_difference": float(allowed), "passed": bool(difference <= allowed)}


def compare_vector(candidate,reference,*,relative_tolerance,absolute_tolerance):
    """Compare registered three-vectors objectively, including non-Y changes."""
    if not np.isfinite([relative_tolerance,absolute_tolerance]).all() or min(relative_tolerance,absolute_tolerance)<0:
        raise ValueError("Finite nonnegative tolerances are required")
    try:a=np.asarray(candidate,dtype=float);b=np.asarray(reference,dtype=float)
    except (ValueError,TypeError):return {"passed":None,"reason":"missing or malformed vector"}
    if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"passed":None,"reason":"three finite vector components required"}
    delta=a-b;difference=float(np.linalg.norm(delta));reference_norm=float(np.linalg.norm(b))
    allowed=max(absolute_tolerance,relative_tolerance*reference_norm)
    return {"candidate":a.tolist(),"reference":b.tolist(),"difference_vector":delta.tolist(),
            "absolute_difference":difference,"reference_magnitude":reference_norm,
            "relative_tolerance":relative_tolerance,"absolute_tolerance":absolute_tolerance,
            "allowed_difference":float(allowed),"passed":bool(difference<=allowed),"comparison_norm":"Euclidean"}


def section_vector_screens(candidate,reference,tolerance,*,eligible=False):
    """Return scoped vector comparisons without promoting loaded/joint sections."""
    distance=reference["distance_mm"]
    free_span=any(np.isclose(distance,value,rtol=0,atol=SECTION_SCREEN_EXTENSION_V1["section_coordinate_tolerance_mm"])
                  for value in SECTION_SCREEN_EXTENSION_V1["free_span_distances_mm"])
    origin=[0.,0.,-distance]
    def same_geometry(key,expected):
        try:return all(np.asarray(data[key]).shape==(3,) and np.isfinite(data[key]).all()
            and np.allclose(data[key],expected,rtol=0,atol=1e-9) for data in (candidate,reference))
        except (KeyError,TypeError,ValueError):return False
    frame_verified=same_geometry("origin_mm",origin) and same_geometry("normal",[0.,0.,1.])
    moments_verified=frame_verified and same_geometry("moment_origin_mm",origin)
    definitions=(
        ("section_mean_displacement_vector_mm","mean_displacement_mm",tolerance["displacement_relative_tolerance"],tolerance["displacement_absolute_tolerance_mm"],frame_verified),
        ("section_force_vector_N","force_N",tolerance["section_force_relative_tolerance"],tolerance["section_force_absolute_tolerance_N"],frame_verified),
        ("section_moment_vector_Nmm","moment_Nmm",SECTION_SCREEN_EXTENSION_V1["moment_relative_tolerance"],
         tolerance["section_force_absolute_tolerance_N"]*SECTION_SCREEN_EXTENSION_V1["moment_characteristic_length_mm"],moments_verified))
    rows=[]
    for metric,key,rtol,atol,geometry_verified in definitions:
        check=compare_vector(candidate.get(key),reference.get(key),relative_tolerance=rtol,absolute_tolerance=atol)
        if not geometry_verified:check.update(passed=None,reason="Section plane/normal or moment origin does not match the registered retained definition")
        observed=check.pop("passed")
        rows.append({"quantity":metric,"distance_mm":distance,"observed_within_tolerance":observed,
                     "contributes_to_screen":free_span,"scope":"free_span_screen" if free_span else "other_section_diagnostic",
                     "status":"missing" if observed is None else "eligible" if eligible and free_span else "diagnostic_only",**check})
    return rows


def compare_probe_fields(candidate: FieldMesh, reference: FieldMesh, *, candidate_domains,
                         reference_domains, points_reference_mm, candidate_to_reference,
                         geometry_identity_verified=False, permit_revised_model=False):
    """Compare fixed physical probes with explicit geometry-equivalence status.

    Tensor differences here compare two codes at the *same load state*. They
    must not be interpreted as a nonlinear service-only stress solution.
    Invalid/outside/discontinuous probes remain invalid and are not filled.
    """
    if not geometry_identity_verified and not permit_revised_model:
        raise ValueError("Geometry/settings identity must pass before a matched-model comparison; explicitly permit a revised-model diagnostic")
    mapped = candidate.transformed(candidate_to_reference)
    a = BodyInterpolator(mapped, candidate_domains).sample(points_reference_mm)
    b = BodyInterpolator(reference, reference_domains).sample(points_reference_mm)
    valid = a["valid"] & b["valid"]
    return {"classification": "matched_model" if geometry_identity_verified else "revised_model_diagnostic",
            "screen_eligible": bool(geometry_identity_verified and valid.all()),
            "points_reference_mm": np.asarray(points_reference_mm), "valid": valid,
            "candidate_displacement_mm": a["displacement_mm"], "reference_displacement_mm": b["displacement_mm"],
            "displacement_difference_mm": a["displacement_mm"] - b["displacement_mm"],
            "candidate_stress_mpa": a["stress_mpa"], "reference_stress_mpa": b["stress_mpa"],
            "stress_tensor_difference_mpa": a["stress_mpa"] - b["stress_mpa"],
            "candidate_valid": a["valid"], "reference_valid": b["valid"],
            "candidate_ambiguous_trace": a["ambiguous_trace"], "reference_ambiguous_trace": b["ambiguous_trace"]}


def cached_reference_deformable(source, cache_directory):
    """Compact derived reference cache tied to freshly verified original hashes."""
    import json
    cache_directory=Path(cache_directory)
    current=file_provenance(source["path"])
    if current["sha256"]!=source["sha256"]:
        raise ValueError("Original COMSOL source changed since reference manifest creation")
    cache_directory.mkdir(parents=True,exist_ok=True)
    cache=cache_directory/(source["sha256"][:16]+"-deformable-fields.npz")
    metadata=cache.with_suffix(".json")
    if cache.exists():
        if not metadata.exists():raise ValueError("Reference field cache has no provenance")
        description=json.loads(metadata.read_text(encoding="utf-8"))
        if description["source_sha256"]!=source["sha256"] or file_provenance(cache)["sha256"]!=description["cache_sha256"]:
            raise ValueError("Reference field cache provenance mismatch")
        with np.load(cache,allow_pickle=False) as data:
            fields={key[6:]:data[key] for key in data.files if key.startswith("field_")}
            return FieldMesh(data["points"],data["tetrahedra"],data["domains"],data["u"],data["stress"],fields,current)
    mesh=read_comsol_vtu(source["path"]).body([2,3,4,5,6,7,8])
    np.savez_compressed(cache,points=mesh.points_mm,tetrahedra=mesh.tetrahedra,domains=mesh.cell_domains,
                        u=mesh.displacement_mm,stress=mesh.stress_mpa,
                        **{"field_"+key:value for key,value in mesh.point_fields.items()})
    metadata.write_text(json.dumps({"source_sha256":source["sha256"],"cache_sha256":file_provenance(cache)["sha256"],
                                   "recipe":"read_comsol_vtu(source).body([2,3,4,5,6,7,8]); no welding or field changes"},indent=2),encoding="utf-8")
    return mesh


def _bolt_probes(component):
    axes={"screw_1":[.9995507658422011,-.029971094479004787,0.],
          "screw_2":[-.5257311121191417,-.8506508083520347,0.],
          "screw_3":[-.4738196537230589,.8806219028310396,0.]}
    axis=np.asarray(axes[component]);tangent=np.array([-axis[1],axis[0],0.])
    # Avoid the pretension plane at radial17.05mm and all contacting outer profiles.
    return np.array([axis*station+[0,0,-217.88]+tangent*a+np.array([0,0,b])
                     for station in [12.05,15.05,16.55,17.55,18.05]
                     for a,b in [(0,0),(2,0),(-2,0),(0,2),(0,-2)]])


def cached_reference_region_measures(source,reference_mesh,cache_directory):
    """Freeze reusable physical-volume reference metrics with reader/recipe hashes."""
    import hashlib
    import json
    from .studies import region_measures, STUDY_METRIC_DEFINITION
    loaded_functions={"region_measures":function_code_sha256(region_measures),"von_mises":function_code_sha256(von_mises)}
    recipe=hashlib.sha256((json.dumps(loaded_functions,sort_keys=True)+json.dumps(STUDY_METRIC_DEFINITION,sort_keys=True)).encode()).hexdigest()
    cache=Path(cache_directory)/(source["sha256"][:16]+"-finite-volume-"+recipe[:12]+".json")
    if cache.is_file():
        data=json.loads(cache.read_text(encoding="utf-8"))
        actual=hashlib.sha256(json.dumps(data["regions"],sort_keys=True).encode()).hexdigest()
        if data["source_sha256"]!=source["sha256"] or data["recipe_sha256"]!=recipe or data["metrics_sha256"]!=actual:
            raise ValueError("Reference finite-volume metric provenance mismatch")
        return data
    regions=region_measures(reference_mesh.body([2]))
    data={"source_sha256":source["sha256"],"recipe_sha256":recipe,
          "metrics_sha256":hashlib.sha256(json.dumps(regions,sort_keys=True).encode()).hexdigest(),
          "definition":STUDY_METRIC_DEFINITION,"loaded_function_code_sha256":loaded_functions,
          "reference_representation":"COMSOL exported linear tetrahedral stress tensor samples","regions":regions}
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")
    return data


def compare_run(reference_manifest_path,results_report_path,output_dir=None,*,make_plots=True,reference_cache_directory=None):
    """Compare both final states at fixed same-body locations; no implicit acceptance.

    A35mm candidate remains a revised model; a30mm partition candidate still
    requires settings/geometry identity evidence. Failed numerical runs can
    produce diagnostic comparisons but cannot pass a cross-code screen.
    """
    import csv
    import json
    from .results import read_component_vtu,report_is_accepted,retained_result_source_binding,retained_component_field_binding
    from .studies import region_measures

    reference_path=Path(reference_manifest_path).resolve();run_path=Path(results_report_path).resolve()
    reference=json.loads(reference_path.read_text(encoding="utf-8"));candidate=json.loads(run_path.read_text(encoding="utf-8"))
    output=Path(output_dir) if output_dir is not None else run_path.parent/"comsol-comparison"
    output.mkdir(parents=True,exist_ok=False)
    cache=Path(reference_cache_directory) if reference_cache_directory is not None else Path(__file__).resolve().parents[1]/"Dodec FEA/structural-fea-validation-working/reference-field-cache"
    identity=candidate.get("comparison_identity_verified") is True
    source_binding=retained_result_source_binding(candidate,run_path)
    field_binding=retained_component_field_binding(candidate,run_path)
    numerical=report_is_accepted(candidate) and source_binding["verified"] and field_binding["verified"]
    eligible=numerical and identity
    mapping={"pipe":(2,),"screw_1":(7,8),"screw_2":(3,4),"screw_3":(5,6)}
    report={"schema_version":1,"reference_manifest":file_provenance(reference_path),"candidate_report":file_provenance(run_path),
            "candidate_solver_status":candidate.get("solver_status"),"candidate_numerically_accepted":numerical,
            "candidate_retained_result_source_binding":source_binding,
            "candidate_component_field_binding":field_binding,
            "section_screen_extension_contract":SECTION_SCREEN_EXTENSION_V1,
            "section_screen_extension_sha256":sha256(json.dumps(SECTION_SCREEN_EXTENSION_V1,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "comparison_classification":candidate.get("comparison_classification","identity_unverified"),
            "candidate_geometry_revision":candidate.get("geometry_revision"),
            "geometry_settings_identity_verified":identity,"screen_eligible":eligible,"states":{},
            "interpretation":"Same load-state cross-code diagnostics; not a service-only stress subtraction or design acceptance",
            "limitations":["Reference contact pressure/gap, exact bolt forces and reactions are absent.",
                           "Fixed probes are accepted only inside tetrahedra of the same physical body; holes and pretension cuts are never bridged.",
                           "Candidate quadratic nodal display fields are sampled through eight linear subtetrahedra; reference is exported linear sampled data.",
                           "Raw stress maxima remain diagnostic; finite-volume stress screens use separately integrated matched physical windows."]}
    probe_rows=[];section_rows=[];region_rows=[]
    tolerance=reference["comparison_screen_frozen_before_assembly_solve"]
    for state in ("preload","loaded"):
        ref_state=reference["states"][state]
        candidate_stage=candidate.get("stages",{}).get(state,{})
        state_result={"components":{},"screens":[],"section_diagnostics":[]}
        available=any(item.get("status")=="available" for name,item in candidate_stage.get("components",{}).items() if name in mapping)
        ref_mesh=cached_reference_deformable(ref_state["source"],cache) if available else None
        for name,domains in mapping.items():
            item=candidate_stage.get("components",{}).get(name,{})
            if item.get("status")!="available" or not item.get("vtu"):
                state_result["components"][name]={"status":"missing","reason":"Candidate final component fields unavailable"};continue
            source=Path(item["vtu"])
            if not source.is_absolute():source=run_path.parent/source
            try:
                component=read_component_vtu(source,name,time=candidate_stage["time"],solver_status=candidate.get("solver_status","unknown"))
                sampled=component.field_mesh()
                points=np.asarray(ref_state["fixed_probes"]["points_mm"]) if name=="pipe" else _bolt_probes(name)
                a=BodyInterpolator(sampled,[component.domain]).sample(points)
                b=BodyInterpolator(ref_mesh,domains).sample(points)
                valid=a["valid"] & b["valid"]
                n_valid=int(valid.sum());n_reference=int(b["valid"].sum())
                delta_u=a["displacement_mm"]-b["displacement_mm"]
                delta_s=a["stress_mpa"]-b["stress_mpa"]
                info={"status":"available","source_vtu":file_provenance(source),"fixed_probe_count":len(points),
                      "joint_valid_probe_count":n_valid,"reference_valid_probe_count":n_reference,
                      "full_reference_probe_coverage":n_valid==n_reference,
                      "max_fixed_probe_displacement_vector_difference_mm":float(np.max(np.linalg.norm(delta_u[valid],axis=1))) if n_valid else None,
                      "rms_fixed_probe_displacement_vector_difference_mm":float(np.sqrt(np.mean(np.sum(delta_u[valid]**2,axis=1)))) if n_valid else None,
                      "max_fixed_probe_tensor_difference_frobenius_MPa":float(np.max(np.linalg.norm(delta_s[valid],axis=(1,2)))) if n_valid else None}
                for i,point in enumerate(points):
                    row={"state":state,"component":name,"probe":i,"x_mm":point[0],"y_mm":point[1],"z_mm":point[2],
                         "reference_valid":bool(b["valid"][i]),"candidate_valid":bool(a["valid"][i]),"joint_valid":bool(valid[i])}
                    for prefix,values in (("reference_u",b["displacement_mm"]),("candidate_u",a["displacement_mm"]),("delta_u",delta_u)):
                        row.update({f"{prefix}_{axis}_mm":float(value) if np.isfinite(value) else None for axis,value in zip("xyz",values[i])})
                    row["reference_displacement_magnitude_mm"]=float(np.linalg.norm(b["displacement_mm"][i])) if b["valid"][i] else None
                    row["candidate_displacement_magnitude_mm"]=float(np.linalg.norm(a["displacement_mm"][i])) if a["valid"][i] else None
                    row["reference_mises_MPa"]=float(von_mises(b["stress_mpa"][i])) if b["valid"][i] else None
                    row["candidate_mises_MPa"]=float(von_mises(a["stress_mpa"][i])) if a["valid"][i] else None
                    row["tensor_difference_frobenius_MPa"]=float(np.linalg.norm(delta_s[i])) if valid[i] else None
                    probe_rows.append(row)
                if name=="pipe":
                    info["sections"]=[]
                    free_span_counts={distance:0 for distance in SECTION_SCREEN_EXTENSION_V1["free_span_distances_mm"]}
                    for ref_section in ref_state["sections"]:
                        distance=ref_section["distance_mm"]
                        actual=integrate_section(sampled,[component.domain],[0,0,-distance],[0,0,1])
                        section={"distance_mm":distance,"candidate":actual,"reference":ref_section}
                        info["sections"].append(section)
                        checks=section_vector_screens(actual,ref_section,tolerance,eligible=eligible)
                        free_span=checks[0]["contributes_to_screen"]
                        row={"state":state,"distance_mm":distance,"section_screen_scope":"free_span" if free_span else "diagnostic_only"}
                        for side,data in (("reference",ref_section),("candidate",actual)):
                            row[f"{side}_area_mm2"]=data["area_mm2"]
                            for key,prefix in (("force_N","force_N"),("moment_Nmm","moment_Nmm"),("mean_displacement_mm","mean_u_mm")):
                                row.update({f"{side}_{prefix}_{axis}":value for axis,value in zip("xyz",data[key])})
                        section_rows.append(row)
                        state_result["screens" if free_span else "section_diagnostics"].extend(checks)
                        if free_span:
                            station=min(free_span_counts,key=lambda value:abs(value-distance));free_span_counts[station]+=1
                    if any(count!=1 for count in free_span_counts.values()):
                        state_result["screens"].append({"quantity":"free_span_section_coverage","observed_within_tolerance":None,
                            "status":"missing","required_station_counts":free_span_counts,
                            "reason":"Each of the four declared free-span sections must occur exactly once"})
                    check=compare_scalar(float(np.linalg.norm(component.displacement_mm,axis=1).max()),ref_state["pipe_max_displacement_mm"],
                        relative_tolerance=tolerance["displacement_relative_tolerance"],absolute_tolerance=tolerance["displacement_absolute_tolerance_mm"])
                    state_result["screens"].append({"quantity":"pipe_max_displacement_mm","observed_within_tolerance":check.pop("passed"),
                                                    "status":"eligible" if eligible else "diagnostic_only",**check})
                    info["region_raw_peaks"]={}
                    distance=-sampled.points_mm[:,2];vm=von_mises(sampled.stress_mpa)
                    for region,(lo,hi) in REFERENCE_REGIONS_MM.items():
                        take=(distance>=lo-1e-8)&(distance<=hi+1e-8)
                        info["region_raw_peaks"][region]={"candidate_nodal_display_MPa":float(vm[take].max()) if take.any() else None,
                            "reference_sampled_peak_MPa":ref_state["pipe_regions"][region]["raw_peak_mises_mpa"],"acceptance_metric":False}
                    reference_regions=cached_reference_region_measures(ref_state["source"],ref_mesh,cache)
                    candidate_regions=region_measures(sampled)
                    info["finite_volume_regions"]={"reference":reference_regions["regions"],"candidate":candidate_regions,
                        "reference_recipe_sha256":reference_regions["recipe_sha256"],"reference_metrics_sha256":reference_regions["metrics_sha256"]}
                    for name_region,actual in candidate_regions.items():
                        ref=reference_regions["regions"][name_region]
                        value=actual["volume_mean_nodal_von_mises_mpa"];target=ref["volume_mean_nodal_von_mises_mpa"]
                        row={"state":state,"region":name_region,"distance_start_mm":actual["distance_window_mm"][0],
                             "distance_end_mm":actual["distance_window_mm"][1],"candidate_volume_mm3":actual["volume_mm3"],
                             "reference_volume_mm3":ref["volume_mm3"],"candidate_volume_mean_von_mises_MPa":value,
                             "reference_volume_mean_von_mises_MPa":target,"candidate_raw_peak_MPa":actual["raw_display_peak_mpa"],
                             "reference_raw_peak_MPa":ref["raw_display_peak_mpa"]}
                        region_rows.append(row)
                        if value is not None and target is not None:
                            check=compare_scalar(value,target,relative_tolerance=tolerance["finite_region_stress_relative_tolerance"],absolute_tolerance=tolerance["stress_absolute_tolerance_MPa"])
                            state_result["screens"].append({"quantity":"finite_volume_mean_von_mises_MPa","region":name_region,
                                "observed_within_tolerance":check.pop("passed"),"status":"eligible" if eligible else "diagnostic_only",**check})
                        else:
                            state_result["screens"].append({"quantity":"finite_volume_mean_von_mises_MPa","region":name_region,
                                "observed_within_tolerance":None,"status":"missing"})
                state_result["components"][name]=info
            except (ValueError,KeyError,FileNotFoundError) as exc:
                state_result["components"][name]={"status":"missing","reason":str(exc)}
        report["states"][state]=state_result
    for filename,rows in (("fixed-probe-comparison.csv",probe_rows),("section-comparison.csv",section_rows),("finite-volume-region-comparison.csv",region_rows)):
        if rows:
            with (output/filename).open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    all_components=all(info.get("status")=="available" and info.get("full_reference_probe_coverage") is True
                       for state in report["states"].values() for info in state["components"].values())
    all_checks=[check["observed_within_tolerance"] for state in report["states"].values() for check in state["screens"]]
    report["screen_eligible"] &= all_components and bool(all_checks) and all(value is not None for value in all_checks)
    if not report["screen_eligible"]:
        for state in report["states"].values():
            for check in state["screens"]:
                if check["status"]!="missing":check["status"]="diagnostic_only"
    report["screen_status"]=("passed" if all(all_checks) else "investigate_discrepancy") if report["screen_eligible"] else "diagnostic_only_identity_or_numerical_gate_unmet"
    if make_plots and section_rows:
        try:report["plots"]=_plot_run_comparison(section_rows,probe_rows,output,report,region_rows=region_rows)
        except (ImportError,ValueError,OSError) as exc:report["plot_error"]=str(exc)
    def clean(value):
        if isinstance(value,np.generic):value=value.item()
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,dict):return {key:clean(v) for key,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    (output/"comparison.json").write_text(json.dumps(clean(report),indent=2,allow_nan=False),encoding="utf-8")
    return report


def _plot_run_comparison(section_rows,probe_rows,output,report,*,region_rows=None):
    import os
    mpl_cache=Path(__file__).resolve().parents[1]/"Dodec FEA/structural-fea-validation-working/comparison-matplotlib-cache"
    mpl_cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(mpl_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for r,state in enumerate(("preload","loaded")):
        rows=[row for row in section_rows if row["state"]==state]
        for c,(key,label,scale) in enumerate((("mean_u_mm_y","Mean section Uy [mm]",1.),("force_N_y","Section shear Fy [N]",1.),("moment_Nmm_x","Section bending Mx [N m]",.001))):
            ax=axes[r,c]
            for side,color in (("reference","#176e85"),("candidate","#b66a36")):
                ax.plot([row["distance_mm"] for row in rows],[row[f"{side}_{key}"]*scale for row in rows],"o-",label=side,color=color)
            ax.set_title(f"{state}: {label}");ax.set_xlabel("Distance from midpoint [mm]");ax.grid(alpha=.2)
            if r==0 and c==0:ax.legend()
    fig.suptitle(f"COMSOL / CalculiX section comparison · {report['comparison_classification']}\n{report['screen_status']} · exported stress reconstruction is not a reaction force",fontsize=11)
    sections=output/"section-response-comparison.png";fig.savefig(sections,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for r,state in enumerate(("preload","loaded")):
        rows=[row for row in probe_rows if row["state"]==state and row["component"]=="pipe"]
        maximum=max([row[f"{side}_displacement_magnitude_mm"] for row in rows for side in ("reference","candidate") if row[f"{side}_displacement_magnitude_mm"] is not None],default=1.)
        for c,side in enumerate(("reference","candidate")):
            selected=[row for row in rows if row[f"{side}_displacement_magnitude_mm"] is not None]
            ax=axes[r,c];scatter=ax.scatter([-row["z_mm"] for row in selected],
                [np.degrees(np.arctan2(row["y_mm"],row["x_mm"])) for row in selected],
                c=[row[f"{side}_displacement_magnitude_mm"] for row in selected],vmin=0,vmax=maximum,s=14,cmap="viridis")
            ax.set_title(f"{state} · {side}");ax.set_xlabel("Distance from midpoint [mm]");ax.set_ylabel("Angle atan2(Y,X) [deg]")
            fig.colorbar(scatter,ax=ax,label="Displacement magnitude [mm]")
    fig.suptitle("Registered fixed material probes at radius18.25mm; holes/outside points omitted\n"+report["screen_status"],fontsize=11)
    probes=output/"registered-displacement-probes.png";fig.savefig(probes,dpi=160);plt.close(fig)
    plots=[sections.name,probes.name]
    if region_rows:
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
        for r,state in enumerate(("preload","loaded")):
            for c,region in enumerate(("center","slots","seats")):
                row=next((row for row in region_rows if row["state"]==state and row["region"]==region),None)
                if row is None:continue
                ax=axes[r,c];values=[row[f"{side}_volume_mean_von_mises_MPa"] for side in ("reference","candidate")]
                if any(value is None or not np.isfinite(value) for value in values):
                    ax.text(.5,.5,"Finite-volume result unavailable",ha="center",va="center",transform=ax.transAxes)
                    continue
                ax.bar(["reference","candidate"],values,color=["#176e85","#b66a36"])
                ax.set_title(f"{state} · {region}");ax.set_ylabel("Volume mean von Mises [MPa]");ax.grid(axis="y",alpha=.2)
        fig.suptitle("Fixed physical volumes; tensor interpolation before invariant; no edge-peak trimming\n"+report["screen_status"],fontsize=11)
        path=output/"finite-volume-stress-comparison.png";fig.savefig(path,dpi=160);plt.close(fig);plots.append(path.name)
    return plots
