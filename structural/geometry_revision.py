"""Verify a declared, reversible seat-support mesh revision before deck creation.

This module deliberately uses no Gmsh or OCP library: GUI binding may occur after
Qt initialization. Native quality is bound to exact geometry through the hashed
audit's semantic mesh fingerprint. Independent NumPy checks cover coordinates,
topology, named selections and the original analytic support constraints.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .mesh import SolidMesh, mesh_fingerprint

KIND = "derived_seat_trim_intersection"
MAXIMUM_REVISION_SHIFT_MM = .010  # Explicit candidate safeguard, not CAD tolerance.
MAXIMUM_SUPPORT_ERROR_MM = 1e-7


def _require(condition, message):
    if not condition:
        raise ValueError("Geometry revision: " + message)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _finite_positive(value, name, ceiling):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and 0 < value <= ceiling,
             f"{name} must be finite, positive and <= {ceiling:g}")
    return float(value)


def _file(record, path_key, hash_key):
    path = Path(record.get(path_key, ""))
    expected = record.get(hash_key)
    _require(path.is_absolute() and path.is_file(), f"retained absolute file required: {path_key}")
    _require(isinstance(expected, str) and len(expected) == 64 and _sha(path) == expected,
             f"file SHA mismatch: {path_key}")
    return path


def _orphan_keys(record):
    return [key for key in record if "repair" in key.lower()
            or key.startswith("derived_geometry") or key == "geometry_revision_evidence"]


def _topology(mesh):
    return {"element_sets": {k: sorted(v) for k, v in mesh.element_sets.items()},
            "node_sets": {k: sorted(v) for k, v in mesh.node_sets.items()},
            "surfaces": {k: sorted(map(tuple, v)) for k, v in mesh.surfaces.items()}}


def _cone_error(points, cone):
    axis = np.asarray(cone["axis"], dtype=float)
    apex = np.asarray(cone["apex_mm"], dtype=float)
    angle = float(cone["semi_angle_radians"])
    _require(axis.shape == apex.shape == (3,) and np.isfinite(axis).all() and np.isfinite(apex).all()
             and np.isclose(np.linalg.norm(axis), 1, atol=1e-10, rtol=0)
             and math.isfinite(angle) and 0 < angle < math.pi/2, "invalid original cone support")
    relative = points-apex
    axial = relative@axis
    _require(np.all(axial > 0), "seat nodes lie on the wrong cone nappe")
    radial = np.linalg.norm(relative-axial[:, None]*axis, axis=1)
    return np.abs(math.cos(angle)*radial-math.sin(angle)*axial)


def verify_mesh_geometry_revision(project, mesh, manifest):
    """Return concrete verification evidence, or raise ValueError before binding.

    ``mesh`` must be the prepared structural mesh, before pretension duplication.
    Legacy meshes have no revision on any participant and no repair metadata.
    Revised meshes require identical three-key identities and identical hashed
    audit pointers in mesh provenance and the generation manifest.
    """
    revision = getattr(project, "geometry_revision", None)
    identities = (revision, mesh.provenance.get("geometry_revision"), manifest.get("geometry_revision"))
    if all(item is None for item in identities):
        _require(not _orphan_keys(mesh.provenance) and not _orphan_keys(manifest),
                 "orphan repair metadata without an explicit geometry revision")
        return {"verified": True, "kind": "legacy_original", "geometry_revision": None,
                "checks": {"all_three_revision_identities_absent": True, "orphan_repair_metadata_absent": True}}
    _require(all(item == revision for item in identities) and isinstance(revision, dict),
             "project, mesh and manifest geometry_revision must be explicitly equal")
    _require(set(revision) == {"schema_version", "kind", "definition_sha256"}
             and type(revision["schema_version"]) is int and revision["schema_version"] == 1
             and revision["kind"] == KIND, "unsupported geometry revision identity")
    evidence = mesh.provenance.get("geometry_revision_evidence")
    _require(isinstance(evidence, dict) and evidence == manifest.get("geometry_revision_evidence")
             and set(evidence) == {"audit_path", "audit_sha256"}, "mesh/manifest audit pointers differ or are incomplete")
    audit_path = _file(evidence, "audit_path", "audit_sha256")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _require(audit.get("schema_version") == 1 and audit.get("geometry_revision") == revision,
             "audit revision identity differs")
    files = {name: _file(audit, name+"_path", name+"_sha256") for name in
             ("definition", "parent_mesh", "changed_nodes_csv", "registered_geometry", "parent_manifest", "boundary_arrays")}
    _require(audit["definition_sha256"] == revision["definition_sha256"], "definition differs from approved revision")
    definition = json.loads(files["definition"].read_text(encoding="utf-8"))
    _require(definition.get("schema_version") == 1 and definition.get("kind") == KIND
             and definition.get("source_sha256") == project.source_sha256
             and definition.get("registered_geometry_sha256") == audit["registered_geometry_sha256"]
             and isinstance(definition.get("method"), str) and bool(definition["method"].strip()),
             "definition source/support identity is incomplete or differs")
    max_shift = _finite_positive(definition.get("max_node_shift_mm"), "maximum node shift", MAXIMUM_REVISION_SHIFT_MM)
    max_error = _finite_positive(definition.get("max_source_support_error_mm"), "source support error", MAXIMUM_SUPPORT_ERROR_MM)
    fingerprint = mesh_fingerprint(mesh)
    _require(audit.get("output_mesh_fingerprint") == fingerprint, "stale audit: output mesh fingerprint differs")
    parent = SolidMesh.load(files["parent_mesh"])
    _require(parent.provenance.get("source_sha256") == project.source_sha256
             and parent.provenance.get("geometry_revision") is None and not _orphan_keys(parent.provenance),
             "parent is not the identified original geometry mesh")
    parent_manifest = json.loads(files["parent_manifest"].read_text(encoding="utf-8"))
    geometry = json.loads(files["registered_geometry"].read_text(encoding="utf-8"))
    _require(Path(parent.provenance.get("mesh_manifest", "")).resolve() == files["parent_manifest"].resolve(),
             "parent manifest is not the parent mesh's retained generation manifest")
    _require(parent_manifest.get("source_sha256") == geometry.get("source_sha256") == project.source_sha256,
             "original support/parent source revision differs")
    for key in ("settings", "pipe_wall_resolution"):
        _require(isinstance(parent_manifest.get(key), dict) and bool(parent_manifest[key])
                 and manifest.get(key) == parent_manifest[key], f"derived manifest {key} differs from parent refinement")
    for name in ("node_ids", "element_ids", "tetra10"):
        _require(np.array_equal(getattr(parent, name), getattr(mesh, name)), f"topology changed: {name}")
    _require(_topology(parent) == _topology(mesh), "named component, node or surface selections changed")
    mesh.validate()
    delta = mesh.points_mm-parent.points_mm
    shifts = np.linalg.norm(delta, axis=1)
    changed = set(map(int, mesh.node_ids[np.any(delta != 0, axis=1)]))
    _require(changed and float(shifts.max()) <= max_shift, "coordinate shift exceeds declared bound or no nodes changed")
    with files["changed_nodes_csv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    row_ids = [int(row["node_id"]) for row in rows]
    _require(len(row_ids) == len(set(row_ids)) and set(row_ids) == changed,
             "changed-node CSV is incomplete, duplicated or does not match actual changes")
    _require(audit.get("changed_nodes") == len(changed)
             and np.isclose(audit.get("maximum_node_shift_mm", np.nan), shifts.max(), atol=1e-12, rtol=0),
             "changed-node count/maximum shift audit differs")
    seat_nodes = {f"screw_{i}": set(mesh.surface_nodes(f"screw_{i}_seat")) for i in (1, 2, 3)}
    _require(changed <= set.union(*seat_nodes.values()), "nodes outside the original seat selections moved")
    protected_names = [name for name in mesh.surfaces if name in ("load_patch", "midpoint")
                       or "_anchor" in name or "_cut" in name]
    protected = {node for name in protected_names for node in mesh.surface_nodes(name)}
    _require(not changed & protected, "protected load, midpoint, anchor or cut coordinates moved")
    pipe_nodes = {int(node) for element in mesh.element_sets["pipe"] for node in mesh.tetra10[mesh._elements[int(element)]]}
    _require(changed <= pipe_nodes, "non-pipe nodes moved")
    other_body_nodes = {int(node) for name in ("hub", "screw_1", "screw_2", "screw_3")
                        for element in mesh.element_sets.get(name, []) for node in mesh.tetra10[mesh._elements[int(element)]]}
    _require(not changed & other_body_nodes, "hub or screw body coordinates moved")
    radii = {"pipe_bore": project.pipe_id_mm/2, "pipe_outer_wall": project.pipe_od_mm/2}
    radii.update({f"pipe_layer_interface_{i}": float(radius) for i, radius in
                  enumerate(parent_manifest["pipe_wall_resolution"]["radial_partition_radii_mm"], 1)})
    _require(all(math.isfinite(radius) and project.pipe_id_mm/2 <= radius <= project.pipe_od_mm/2
                 for radius in radii.values()), "invalid original concentric-cylinder radius")
    with np.load(files["boundary_arrays"], allow_pickle=False) as arrays:
        _require(np.array_equal(arrays["node_ids"], parent.node_ids)
                 and np.allclose(arrays["nodes"], parent.points_mm, atol=1e-12, rtol=0),
                 "boundary topology arrays are not the original parent coordinates")
        cylinder_nodes = {name: set(map(int, arrays["boundary_"+name].ravel())) for name in radii}
    for row in rows:
        node = int(row["node_id"]); index = mesh._nodes[node]
        _require(row["seat"] in seat_nodes and node in seat_nodes[row["seat"]], "CSV seat membership differs")
        memberships = [name for name, nodes in cylinder_nodes.items() if node in nodes]
        _require(len(memberships) <= 1 and row["additional_support"] == (memberships[0] if memberships else ""),
                 "CSV cylinder membership differs from original boundary topology")
        for prefix, actual in (("original", parent.points_mm[index]), ("corrected", mesh.points_mm[index]), ("shift", delta[index])):
            recorded = np.array([float(row[f"{prefix}_{axis}_mm"]) for axis in "xyz"])
            _require(np.allclose(actual, recorded, atol=1e-12, rtol=0), f"CSV {prefix} coordinate differs at node {node}")
        _require(np.isclose(float(row["shift_distance_mm"]), shifts[index], atol=1e-12, rtol=0), "CSV shift norm differs")
    support_errors = {}; original_cones = {}
    for i in (1, 2, 3):
        bodies = [body for body in geometry["bodies"] if body["name"] == f"Bolt {i}"]
        _require(len(bodies) == 1, f"original Bolt {i} support missing")
        cones = [face for face in bodies[0]["registered_faces"] if face["type"] == "Cone"]
        _require(len(cones) == 1, f"original Bolt {i} cone ambiguous")
        original_cones[f"screw_{i}"] = cones[0]
        nodes = sorted(seat_nodes[f"screw_{i}"])
        support_errors[f"screw_{i}_cone"] = float(_cone_error(mesh.points_mm[[mesh._nodes[n] for n in nodes]], cones[0]).max())
    all_seat_nodes = set.union(*seat_nodes.values())
    for name, radius in radii.items():
        nodes = sorted(cylinder_nodes[name] & all_seat_nodes)
        support_errors[name] = (float(np.max(np.abs(np.linalg.norm(mesh.points_mm[[mesh._nodes[n] for n in nodes], :2], axis=1)-radius)))
                                if nodes else 0.)
    measured_error = max(support_errors.values())
    _require(measured_error <= max_error, "corrected nodes do not satisfy original analytic support bounds")
    # Nearest-point KKT stationarity: the correction must lie in the span of
    # the exact final support normals. Merely lying on the surfaces would also
    # permit arbitrary sliding along their intersection and is insufficient.
    stationarity_residuals = []
    for row in rows:
        index = mesh._nodes[int(row["node_id"])]; point = mesh.points_mm[index]
        cone = original_cones[row["seat"]]
        axis = np.asarray(cone["axis"]); relative = point-np.asarray(cone["apex_mm"])
        radial = relative-np.dot(relative, axis)*axis
        normal = np.cos(cone["semi_angle_radians"])*radial/np.linalg.norm(radial)-np.sin(cone["semi_angle_radians"])*axis
        normals = [normal]
        if row["additional_support"]:
            normals.append(np.r_[point[:2]/np.linalg.norm(point[:2]), 0.])
        basis = np.asarray(normals).T
        coefficients, _, rank, _ = np.linalg.lstsq(basis, delta[index], rcond=None)
        _require(rank == len(normals), "source support normals are rank deficient")
        stationarity_residuals.append(float(np.linalg.norm(delta[index]-basis@coefficients)))
    max_stationarity = max(stationarity_residuals)
    _require(max_stationarity <= max_error, "nearest-point stationarity failed: unexplained tangential node movement")
    audited_error = audit.get("max_source_support_error_mm")
    _require(not isinstance(audited_error, bool) and isinstance(audited_error, (int, float))
             and math.isfinite(audited_error) and 0 <= audited_error <= max_error
             and np.isclose(audited_error, measured_error, atol=1e-10, rtol=0), "source support error audit differs")
    quality = audit.get("mesh_quality", {})
    minimum = quality.get("min_scaled_jacobian")
    _require(not isinstance(minimum, bool) and isinstance(minimum, (int, float)) and math.isfinite(minimum)
             and minimum >= .1 and type(quality.get("nonpositive_elements")) is int and quality["nonpositive_elements"] == 0,
             "native minimum scaled Jacobian must be >= 0.1 with zero nonpositive elements")
    declared_quality = manifest.get("mesh_quality", {})
    _require(declared_quality.get("min_scaled_jacobian") == minimum
             and declared_quality.get("nonpositive_elements") == 0, "manifest native quality differs from hashed output audit")
    continuity = audit.get("shared_node_continuity", {})
    _require(continuity.get("verified") is True and type(continuity.get("disconnected_shared_node_pairs")) is int
             and continuity["disconnected_shared_node_pairs"] == 0, "shared-node continuity audit missing or failed")
    return {"verified": True, "kind": KIND, "geometry_revision": revision,
            "audit": {"path": str(audit_path), "sha256": evidence["audit_sha256"]},
            "files": {name: {"path": str(path), "sha256": audit[name+"_sha256"]} for name, path in files.items()},
            "output_mesh_fingerprint": fingerprint,
            "checks": {"topology_unchanged": True, "all_named_selections_unchanged": True,
                       "shared_global_node_ids_and_single_coordinates_preserved": True,
                       "shared_node_continuity": continuity, "protected_coordinates_unchanged": True,
                       "all_changes_on_pipe_seat_nodes": True, "changed_nodes": len(changed),
                       "maximum_node_shift_mm": float(shifts.max()), "declared_shift_limit_mm": max_shift,
                       "maximum_source_support_error_mm": measured_error, "declared_support_error_limit_mm": max_error,
                       "maximum_nearest_point_stationarity_residual_mm": max_stationarity,
                       "parent_manifest_path_bound": True, "parent_refinement_settings_unchanged": True,
                       "support_errors_mm": support_errors, "native_minimum_scaled_jacobian": minimum,
                       "nonpositive_elements": 0},
            "scope": "Prepared derived geometry only; not solver acceptance, mesh convergence or stored CAD tolerance compliance."}
