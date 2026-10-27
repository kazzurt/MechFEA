"""Reproducible registered CAD and separate-interface Gmsh preparation.

The imported body volumes are preserved. Explicit load-patch, screw shoulder,
anchorage-mouth and optional radial wall partitions are recorded. Gmsh never fragments different components
together, since that would merge nodes across separating-contact interfaces.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .geometry import (SOURCE_SHA256, read_step_bodies, shape_properties,
                       geometric_fingerprint, make_named_selections, describe_body)


def _process_memory_snapshot():
    """Measure native-inclusive peak RSS without an optional dependency."""
    if os.name != "nt":
        import resource
        return {"process_peak_resident_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    import ctypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (key, ctypes.c_size_t) for key in ("peak_rss", "rss", "peak_paged_pool", "paged_pool",
                                              "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private")]
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (key, ctypes.c_ulonglong) for key in ("total", "available", "total_page", "available_page",
                                                "total_virtual", "available_virtual", "extended")]
    process, system = Counters(), Memory()
    process.cb, system.length = ctypes.sizeof(process), ctypes.sizeof(system)
    ok_process = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(process), process.cb)
    ok_system = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(system))
    return {"process_measurement_succeeded": bool(ok_process), "system_measurement_succeeded": bool(ok_system),
            "process_peak_working_set_bytes": int(process.peak_rss), "process_working_set_bytes": int(process.rss),
            "system_available_physical_bytes": int(system.available)}


def source_to_reference_transform(midpoint):
    angle = math.radians(18)
    rotation = np.array([[math.sin(angle), math.cos(angle), 0],
                         [0, 0, 1], [math.cos(angle), -math.sin(angle), 0]])
    translation = -rotation @ np.asarray(midpoint)
    return rotation, translation


def prepare_registered_geometry(source, audit_path, output_dir):
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepTools import BRepTools
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Trsf
    import hashlib
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    report = json.loads(Path(audit_path).read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != report["source_sha256"] or digest != SOURCE_SHA256:
        raise ValueError("Source CAD changed since kernel audit")
    if report["unresolved_interferences"]:
        raise ValueError("Unresolved CAD interference blocks derived preparation")
    if not all(b["valid"] for b in report["bodies"]):
        raise ValueError("Invalid source solid blocks derived preparation")
    midpoint = report["source_frame"]["midpoint_center_mm"]
    rotation, translation = source_to_reference_transform(midpoint)
    transform = gp_Trsf()
    transform.SetValues(*(float(v) for row, offset in zip(rotation, translation) for v in [*row, offset]))
    bodies = read_step_bodies(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    builder, compound = BRep_Builder(), TopoDS_Compound()
    builder.MakeCompound(compound)
    body_map = []
    for body in bodies:
        if body["name"] in report["excluded_duplicates"]:
            continue
        shape = BRepBuilderAPI_Transform(body["shape"], transform, True).Shape()
        volume_before = shape_properties(body["shape"])["measure"]
        volume_after = shape_properties(shape)["measure"]
        if abs(volume_after - volume_before) > 1e-7 * volume_before:
            raise ValueError("Rigid registration changed component volume")
        path = output_dir / (body["name"].lower().replace(" ", "-") + ".brep")
        BRepTools.Write_s(shape, str(path))
        builder.Add(compound, shape)
        description = describe_body({**body, "shape": shape})
        body_map.append({"name": body["name"], "source_step_entity": body["source_step_entity"],
                         "path": str(path), "volume_mm3": volume_after,
                         "registered_faces": description["faces"]})
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    writer.Write(str(output_dir / "registered-deduplicated-mm.step"))
    manifest = {"schema_version": 1, "source": str(source), "source_sha256": digest,
                "rotation_source_to_reference": rotation.tolist(),
                "translation_source_to_reference_mm": translation.tolist(),
                "frame": {"midpoint": [0, 0, 0], "pipe_toward_hub": [0, 0, -1],
                          "service_force_direction": [0, 1, 0], "loaded_side": "-Y"},
                "registration_evidence": "Pipe axis, midpoint, named screw axes and asymmetric hub centroid; see reference registration audit",
                "bodies": body_map,
                "geometry_operations": ["rigid registration only; volume checked per body"],
                "source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == digest}
    (output_dir / "registered-geometry-map.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _face_geometry_match(gmsh, surface, original_faces, *, area_tolerance=1e-4, centroid_tolerance_mm=1e-4):
    """Match imported surfaces by center and area, avoiding transient tag assumptions."""
    center = np.array(gmsh.model.occ.getCenterOfMass(2, surface))
    area = gmsh.model.occ.getMass(2, surface)
    matches = [f for f in original_faces
               if np.linalg.norm(center - f["centroid_mm"]) < centroid_tolerance_mm
               and abs(area - f["area_mm2"]) < max(area_tolerance, area * 1e-6)]
    if len(matches) != 1:
        return None
    return matches[0]


def _partition_radial_layers(gmsh, pipe_volumes, radii):
    """Split all radii together; sequential nested booleans failed source validity."""
    if not radii:
        return list(pipe_volumes)
    if radii != sorted(set(radii)) or not all(17.5 < r < 19.0 for r in radii):
        raise ValueError("Radial partitions must strictly increase within the pipe wall")
    objects = list(pipe_volumes)
    tools = [(3, gmsh.model.occ.addCylinder(0, 0, -240, 0, 0, 241, radius)) for radius in radii]
    fragments, mapping = gmsh.model.occ.fragment(objects, tools)
    physical_pipe = sorted(set(v for mapped in mapping[:len(objects)] for v in mapped if v[0] == 3))
    gmsh.model.occ.remove([v for v in fragments if v not in set(physical_pipe)], recursive=True)
    return physical_pipe


SLOT_SOURCE_FACES = {"slot_1": list(range(5, 13)), "slot_2": list(range(13, 21)),
                     "slot_3": [*range(21, 28), 29]}


def _slot_wall_fragments(gmsh, pipe_surfaces, original_faces):
    """Identify unchanged slot supports after concentric wall fragmentation.

    An angular footprint disambiguates the three coplanar top/bottom walls.
    Sampling evaluates the analytic surface; it never replaces its trim curves.
    """
    descriptors = {face["index"]: face for face in original_faces}
    groups = list(SLOT_SOURCE_FACES)
    centers = np.array([descriptors[SLOT_SOURCE_FACES[name][0]]["centroid_mm"][:2] for name in groups])
    by_source = {index: [] for ids in SLOT_SOURCE_FACES.values() for index in ids}
    for tag in pipe_surfaces:
        box = gmsh.model.getBoundingBox(2, tag)
        if box[2] < -199.88-1e-4 or box[5] > -193.88+1e-4:
            continue
        center = np.asarray(gmsh.model.occ.getCenterOfMass(2, tag))
        group = groups[np.linalg.norm(centers-center[:2], axis=1).argmin()]
        low, high = map(np.asarray, gmsh.model.getParametrizationBounds(2, tag))
        points = np.array([gmsh.model.getValue(2, tag, low+(high-low)*[a,b])
                           for a,b in ((.21,.37),(.49,.61),(.73,.83))])
        matches = []
        for index in SLOT_SOURCE_FACES[group]:
            face = descriptors[index]
            delta = points-np.asarray(face["origin_mm"])
            axis = np.asarray(face["axis"])
            axial = delta@axis
            error = abs(axial) if face["type"] == "Plane" else abs(
                np.linalg.norm(delta-axial[:,None]*axis, axis=1)-face["radius_mm"])
            if float(error.max()) < 1e-7:
                matches.append(index)
        if len(matches) != 1:
            raise ValueError(f"Cannot identify slot-wall fragment {tag}: {matches}")
        by_source[matches[0]].append(tag)
    if not all(by_source.values()):
        raise ValueError("Slot sizing must cover every one of the 24 original slot faces")
    selections = {name+"_walls": sorted(tag for index in ids for tag in by_source[index])
                  for name, ids in SLOT_SOURCE_FACES.items()}
    evidence = {str(index): {"derived_surface_tags": tags,
                            "original_area_mm2": descriptors[index]["area_mm2"],
                            "fragment_area_mm2": sum(gmsh.model.occ.getMass(2, tag) for tag in tags)}
                for index, tags in by_source.items()}
    return selections, evidence


def surface_resolution(mesh):
    """Measure actual contact/anchor boundary triangles independently of targets."""
    measurements = {}
    names = ["hub_snout"]+[f"screw_{i}_{kind}" for i in range(1, 4) for kind in ("anchor", "cone", "seat")]
    for surface in names:
        corners = np.array([mesh.face_points(e, f)[:3] for e, f in mesh.surfaces[surface]])
        lengths = np.linalg.norm(corners-np.roll(corners, 1, axis=1), axis=2).ravel()
        measurements[surface] = {
            "quadratic_triangles": len(corners), "nodes": len(mesh.surface_nodes(surface)),
            "corner_edge_length_mm": {"minimum": float(lengths.min()), "median": float(np.median(lengths)),
                                       "percentile95": float(np.percentile(lengths, 95)), "maximum": float(lengths.max())}}
    return measurements


def _tetra_quadratic_ccx_permutation(gmsh):
    """Derive edge order from Gmsh reference coordinates, not a remembered table."""
    _, dim, order, count, coordinates, primary = gmsh.model.mesh.getElementProperties(11)
    if (dim, order, count, primary) != (3, 2, 10, 4):
        raise ValueError("Unexpected Gmsh second-order tetrahedron")
    coordinates = np.asarray(coordinates).reshape(10, 3)
    edges_ccx = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    permutation = list(range(4))
    for a, b in edges_ccx:
        midpoint = (coordinates[a] + coordinates[b]) / 2
        matches = np.flatnonzero(np.linalg.norm(coordinates - midpoint, axis=1) < 1e-10)
        if len(matches) != 1:
            raise ValueError("Cannot map Gmsh midside node to CalculiX edge")
        permutation.append(int(matches[0]))
    return permutation


def _adaptive_partition_volume_audit(gmsh, body_volumes, manifest, output_dir):
    """Avoid false volume alarms from nonadaptive integration on new trims."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopoDS import TopoDS_Shape, TopoDS_Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SOLID
    from .geometry import shape_parts
    def read(path):
        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
            raise ValueError("Cannot read derived partition BREP")
        return shape
    def volume(shape):
        properties = GProp_GProps()
        error = BRepGProp.VolumeProperties_s(shape, properties, 1e-10)
        return float(properties.Mass()), float(error)
    path = output_dir / "partitioned-registered-geometry.brep"
    gmsh.write(str(path))
    derived = list(shape_parts(read(path), TopAbs_SOLID))
    derived_properties = [shape_properties(s) for s in derived]
    invalid = [i for i, shape in enumerate(derived) if not BRepCheck_Analyzer(shape).IsValid()]
    if invalid:
        (output_dir / "partition-volume-audit-failure.json").write_text(json.dumps({
            "status": "rejected_invalid_partition_solid", "invalid_solid_indices": invalid,
            "derived_solid_properties": derived_properties}, indent=2)+"\n")
        raise ValueError(f"Partition created invalid material solid indices: {invalid}")
    centers = np.array([p["centroid_mm"] for p in derived_properties])
    masses = np.array([p["measure"] for p in derived_properties])
    results = {}
    consumed_matches = set()
    for body in manifest["bodies"]:
        if body["name"] not in body_volumes:
            continue
        source_shape = read(body["path"])
        source_volume, source_error = volume(source_shape)
        parts, errors, matched_shapes = [], [], []
        for dim, tag in body_volumes[body["name"]]:
            target = gmsh.model.occ.getCenterOfMass(dim, tag)
            distance = np.linalg.norm(centers - target, axis=1)
            target_mass = gmsh.model.occ.getMass(dim, tag)
            match = int(np.argmin(distance + abs(masses-target_mass)/max(abs(target_mass), 1)))
            if distance[match] > 1e-4 or abs(masses[match]-target_mass) > 1e-6*max(abs(target_mass), 1):
                raise ValueError("Cannot map derived partition for adaptive volume verification")
            if match in consumed_matches:
                raise ValueError("Derived partition solid matched more than one retained volume")
            consumed_matches.add(match)
            value, error = volume(derived[match])
            parts.append(value)
            errors.append(error)
            matched_shapes.append(derived[match])
        total = sum(parts)
        results[body["name"]] = {"adaptive_before_mm3": source_volume, "adaptive_after_mm3": total,
                                 "relative_error": abs(total - source_volume) / source_volume,
                                 "requested_integration_relative_tolerance": 1e-10,
                                 "integration_error_estimates": [source_error, *errors]}
        if abs(total - source_volume) > 1e-7 * source_volume:
            # Trimmed-surface quadrature can depend on partition even with a
            # tight requested Eps. Demand an independent boolean equivalence
            # check instead of treating its discrepancy as removed material.
            builder, compound = BRep_Builder(), TopoDS_Compound()
            builder.MakeCompound(compound)
            for part in matched_shapes:
                builder.Add(compound, part)
            removed = BRepAlgoAPI_Cut(source_shape, compound)
            added = BRepAlgoAPI_Cut(compound, source_shape)
            removed.Build(); added.Build()
            if not removed.IsDone() or not added.IsDone():
                raise ValueError("Partition symmetric-difference boolean failed")
            removed_volume = abs(shape_properties(removed.Shape())["measure"])
            added_volume = abs(shape_properties(added.Shape())["measure"])
            results[body["name"]]["boolean_removed_volume_mm3"] = removed_volume
            results[body["name"]]["boolean_added_volume_mm3"] = added_volume
            if max(removed_volume, added_volume) > 1e-6:
                raise ValueError(f"Partition changed material: {body['name']}: removed {removed_volume}, added {added_volume}")
            results[body["name"]]["integration_note"] = "Trim-dependent quadrature discrepancy; independent symmetric-difference boolean confirms no added or removed solid volume."
    return results


def _add_fixed_rigid_surface_carriers(gmsh, surface_geometry):
    """Give exact triangle6 interfaces fully prescribed C3D10 carrier elements.

    These disconnected, fully fixed interior tetrahedra carry no elastic-hub
    meaning. Their outer faces are exactly the quadratic CAD surface mesh.
    """
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    points = {int(n): p for n, p in zip(node_tags, np.asarray(coordinates).reshape(-1, 3))}
    next_node = max(points) + 1
    all_element_tags = gmsh.model.mesh.getElements()[1]
    next_element = max(int(v) for tags in all_element_tags for v in tags) + 1
    created_nodes, created_points, created_elements, created_cells = [], [], [], []
    for surface, geometry in surface_geometry.items():
        types, _, connectivity = gmsh.model.mesh.getElements(2, surface)
        if list(types) != [9]:
            raise ValueError("Fixed rigid carrier requires quadratic triangles")
        axis, origin = np.array(geometry["axis"]), np.array(geometry["origin_mm"])
        sense = -1 if geometry["orientation"] == "TopAbs_REVERSED" else 1
        for triangle in np.asarray(connectivity[0]).reshape(-1, 6):
            p = np.array([points[int(n)] for n in triangle])
            face_center = -p[:3].sum(axis=0) / 9 + 4 * p[3:].sum(axis=0) / 9
            delta = face_center - origin
            outward = sense * (delta - np.dot(delta, axis) * axis)
            outward /= np.linalg.norm(outward)
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), outward) > 0:
                order = [0, 2, 1, 5, 4, 3]
            else:
                order = [0, 1, 2, 3, 4, 5]
            ordered = triangle[order]
            q = np.array([points[int(n)] for n in ordered[:3]])
            height = np.mean([np.linalg.norm(q[a] - q[b]) for a, b in [(0, 1), (1, 2), (2, 0)]]) * .7
            apex = face_center - height * outward
            new_points = [apex, *((q + apex) / 2)]
            new_ids = list(range(next_node, next_node + 4))
            next_node += 4
            created_nodes.extend(new_ids)
            created_points.extend(new_points)
            # Native Gmsh node order: edge03, edge23, edge13 at positions7,8,9.
            cell = [*ordered[:3], new_ids[0], *ordered[3:], new_ids[1], new_ids[3], new_ids[2]]
            created_elements.append(next_element)
            next_element += 1
            created_cells.extend(cell)
    volume = gmsh.model.addDiscreteEntity(3)
    gmsh.model.mesh.addNodes(3, volume, created_nodes, np.asarray(created_points).ravel())
    gmsh.model.mesh.addElementsByType(volume, 11, created_elements, created_cells)
    for dim, tag in gmsh.model.getPhysicalGroups(3):
        if gmsh.model.getPhysicalName(dim, tag) == "Hub":
            gmsh.model.removePhysicalGroups([(dim, tag)])
    gmsh.model.addPhysicalGroup(3, [volume], name="Hub")
    return volume, len(created_elements)


def _tie_projection_diagnostic(mesh):
    """Conservative closest-distance upper bound to subdivided master faces."""
    from scipy.spatial import cKDTree
    result = {}
    for i in range(1, 4):
        ids = mesh.surface_nodes(f"screw_{i}_anchor")
        p = mesh.points_mm[[mesh._nodes[n] for n in ids]]
        triangles = np.concatenate([mesh.face_points(e, f)[[[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]]]
                                    for e, f in mesh.surfaces[f"hub_hole_{i}"]])
        centers = triangles.mean(axis=1)
        candidates = cKDTree(centers).query(p, k=min(24, len(centers)))[1]
        t = triangles[candidates]
        x = p[:, None, :]
        a, b, c = t[:, :, 0], t[:, :, 1], t[:, :, 2]
        ab, ac, ap = b-a, c-a, x-a
        normal = np.cross(ab, ac)
        n2 = np.sum(normal*normal, axis=-1)
        projection = x - (np.sum(ap*normal, axis=-1)/n2)[:, :, None]*normal
        v = projection-a
        d00, d01, d11 = np.sum(ab*ab, axis=-1), np.sum(ab*ac, axis=-1), np.sum(ac*ac, axis=-1)
        d20, d21 = np.sum(v*ab, axis=-1), np.sum(v*ac, axis=-1)
        denominator = d00*d11-d01*d01
        u, w = (d11*d20-d01*d21)/denominator, (d00*d21-d01*d20)/denominator
        inside = (u >= 0) & (w >= 0) & (u+w <= 1)
        distances = np.where(inside, np.sum((x-projection)**2, axis=-1), np.inf)
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end-start
            fraction = np.clip(np.sum((x-start)*edge, axis=-1)/np.sum(edge*edge, axis=-1), 0, 1)
            closest = start+fraction[:, :, None]*edge
            distances = np.minimum(distances, np.sum((x-closest)**2, axis=-1))
        nearest = np.sqrt(distances.min(axis=1))
        result[str(i)] = {"slave_nodes": len(ids), "maximum_distance_mm": float(nearest.max()),
                          "rms_distance_mm": float(np.sqrt(np.mean(nearest**2))),
                          "worst_slave_node": ids[int(nearest.argmax())]}
    maximum = max(v["maximum_distance_mm"] for v in result.values())
    return {"method": "Upper bound from 24 nearest centroid candidates of four planar subtriangles per quadratic master face; ADJUST=NO",
            "screws": result, "recommended_search_tolerance_mm": maximum*1.05 + 1e-6,
            "bounded_search_passed": maximum*1.05 + 1e-6 <= .01}


def qualify_existing_mesh(mesh_path, output_path):
    """Attach measured tie search bounds to an immutable completed pilot."""
    from .mesh import SolidMesh
    mesh = SolidMesh.load(mesh_path)
    diagnostic = _tie_projection_diagnostic(mesh)
    if not diagnostic["bounded_search_passed"]:
        raise ValueError("Measured tie projection exceeds the bounded 0.01 mm search tolerance")
    mesh.provenance["tie_projection_tolerance_mm"] = diagnostic["recommended_search_tolerance_mm"]
    mesh.provenance["qualified_from_mesh"] = str(Path(mesh_path).resolve())
    mesh.provenance["tie_projection_diagnostic"] = diagnostic
    mesh.save(output_path)
    Path(str(output_path) + ".qualification.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    return diagnostic


def generate_pilot_mesh(registered_manifest, audit_path, output_dir, *,
                        far_size_mm=3.0, contact_size_mm=1.0,
                        patch_projected_width_mm=35.0, order=2,
                        include_hub=True, threads=2, thickness_size_mm=1.5,
                        hub_size_mm=6.0, max_nodes=500000, retain_rigid_hub_carriers=True,
                        anchor_size_mm=0.5, enforce_wall_layers=False,
                        bore_snout_size_mm=None, slot_size_mm=None):
    """Bounded pilot with geometry/mesh provenance; not an accepted convergence mesh.

    Export NPZ stores actual node tags and C3D10 ordering. Boundary triangle6
    connectivity is exported by semantic group for independent face mapping.
    """
    import gmsh
    output_dir = Path(output_dir).resolve()
    if (output_dir / "mesh-manifest.json").exists():
        raise FileExistsError("Mesh runs are immutable; choose a new descriptive directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(registered_manifest).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if not 0 < patch_projected_width_mm < 38:
        raise ValueError("Projected load width must be between 0 and pipe OD")
    if order not in (1, 2):
        raise ValueError("Only first- or second-order structural tetrahedra are supported")
    if order != 2 and retain_rigid_hub_carriers:
        raise ValueError("Rigid surface carrier baseline requires quadratic elements")
    if not include_hub:
        raise ValueError("The assembly baseline requires its fixed hub interface carriers")
    if bore_snout_size_mm is not None and (not math.isfinite(bore_snout_size_mm) or bore_snout_size_mm <= 0):
        raise ValueError("Optional bore/snout size must be finite and positive")
    if slot_size_mm is not None and (isinstance(slot_size_mm, bool) or not isinstance(slot_size_mm, (int, float))
                                    or not math.isfinite(slot_size_mm) or slot_size_mm <= 0):
        raise ValueError("Optional slot size must be finite and positive")
    slot_scope = None if slot_size_mm is None else {
        "target_size_mm": slot_size_mm, "source_face_groups": SLOT_SOURCE_FACES,
        "distance_source": "all original slot-wall fragments, their curves and points",
        "minimum_size_distance_mm": .5, "grading_end_distance_mm": 3.5,
        "semantic_selections": [f"slot_{i}_walls" for i in range(1,4)],
        "scope": "independent axial/circumferential slot profile; radial layers retained",
        "geometry_and_contact_selections_changed": False,
        "slot_trim_projection_or_healing": False}
    bore_snout_scope = None if bore_snout_size_mm is None else {
        "distance_source_surface": "hub_snout", "target_size_mm": bore_snout_size_mm,
        "minimum_size_distance_mm": 2.0, "grading_end_distance_mm": 5.0,
        "paired_pipe_coverage": "2 mm region spans nominal 0.45 mm radial gap and 1.5 mm pipe wall",
        "geometry_and_contact_selections_changed": False}
    wall_layers = max(1, math.ceil(1.5/thickness_size_mm-1e-9)) if enforce_wall_layers else 1
    radial_partitions = [17.5 + 1.5*i/wall_layers for i in range(1, wall_layers)]
    start = time.monotonic()
    initial_memory = _process_memory_snapshot()
    gmsh.initialize()
    gmsh.logger.start()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", threads)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(contact_size_mm, thickness_size_mm, anchor_size_mm,
                                                       slot_size_mm if slot_size_mm is not None else math.inf))
        gmsh.option.setNumber("Mesh.MeshSizeMax", hub_size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.model.add("dodec-registered-separate-contact-pilot")
        body_volumes, original_faces, screw_cuts, head_volumes = {}, {}, {}, {}
        volume_checks = {}
        for body in manifest["bodies"]:
            if body["name"] == "Hub" and not include_hub:
                continue
            imported = gmsh.model.occ.importShapes(body["path"], highestDimOnly=True)
            if len(imported) != 1 or imported[0][0] != 3:
                raise ValueError(f"Expected one imported volume: {body['name']}")
            body_volumes[body["name"]] = imported
            original_faces[body["name"]] = body["registered_faces"]
            volume_checks[body["name"]] = {"before_mm3": body["volume_mm3"]}
        # Partition ONLY pipe with a projected-width box. Outside box fragments
        # are removed using OCC's explicit source-to-fragment map.
        half_width = patch_projected_width_mm / 2
        tool_box = gmsh.model.occ.addBox(-half_width, -40, -25, 2 * half_width, 80, 26)
        fragments, fragment_map = gmsh.model.occ.fragment(body_volumes["New Pipe"], [(3, tool_box)])
        pipe_fragments = list(fragment_map[0])
        retained = set(pipe_fragments)
        gmsh.model.occ.remove([x for x in fragments if x not in retained], recursive=True)
        body_volumes["New Pipe"] = pipe_fragments
        if radial_partitions:
            body_volumes["New Pipe"] = _partition_radial_layers(gmsh, body_volumes["New Pipe"], radial_partitions)
        # Reproduce the COMSOL shoulder cut: radius-4 disk at the junction of
        # the fully engaged smooth shaft and the larger conical head. The
        # 0.5504 mm shoulder annulus is preserved, not replaced by a free shank.
        for name in ("Bolt 1", "Bolt 2", "Bolt 3"):
            shaft = next(f for f in original_faces[name] if f["type"] == "Cylinder" and abs(f["radius_mm"] - 4) < 1e-5)
            cone = next(f for f in original_faces[name] if f["type"] == "Cone")
            center, inward_axis = np.array(cone["origin_mm"]), np.array(shaft["axis"])
            plane = gmsh.model.occ.addRectangle(-10, -10, 0, 20, 20)
            rotation_axis = np.cross([0, 0, 1], inward_axis)
            angle = math.acos(float(np.clip(inward_axis[2], -1, 1)))
            if np.linalg.norm(rotation_axis) > 1e-10:
                gmsh.model.occ.rotate([(2, plane)], 0, 0, 0, *rotation_axis, angle)
            gmsh.model.occ.translate([(2, plane)], *center)
            fragments, fragment_map = gmsh.model.occ.fragment(body_volumes[name], [(2, plane)])
            body_volumes[name] = [v for v in fragment_map[0] if v[0] == 3]
            if len(body_volumes[name]) != 2:
                raise ValueError(f"Shoulder plane did not split {name} into head and inner shaft")
            gmsh.model.occ.synchronize()
            volume_boundaries = set(gmsh.model.getBoundary(body_volumes[name], combined=False, oriented=False))
            remnants = [v for v in fragments if v[0] == 2 and v not in volume_boundaries]
            gmsh.model.occ.remove(remnants, recursive=True)
            # The hub-hole mouth is trimmed by the curved snout, not a plane.
            # Partition the screw by that exact cylindrical envelope so the
            # exposed shank crescents are not accidentally tied to the hub.
            before_envelope = list(body_volumes[name])
            envelope = gmsh.model.occ.addCylinder(0, 0, -230, 0, 0, 25, 17.05)
            envelope_fragments, envelope_map = gmsh.model.occ.fragment(before_envelope, [(3, envelope)])
            source_parts = sorted(set(v for mappings in envelope_map[:len(before_envelope)] for v in mappings if v[0] == 3))
            gmsh.model.occ.remove([v for v in envelope_fragments if v not in set(source_parts)], recursive=True)
            body_volumes[name] = source_parts
            shank_parts = [v for v in source_parts if np.dot(np.array(gmsh.model.occ.getCenterOfMass(*v)) - center, inward_axis) >= -1e-7]
            if len(shank_parts) > 1:
                # Keep the imprinted exterior mouth edge but discard the
                # artificial tangential internal envelope interface; otherwise
                # zero-angle crescent volumes unnecessarily degrade elements.
                previous_simplification = gmsh.option.getNumber("Geometry.OCCBooleanSimplify")
                gmsh.option.setNumber("Geometry.OCCBooleanSimplify", 0)
                joined, _ = gmsh.model.occ.fuse(shank_parts[:1], shank_parts[1:])
                gmsh.option.setNumber("Geometry.OCCBooleanSimplify", previous_simplification)
                body_volumes[name] = [v for v in source_parts if v not in shank_parts] + [v for v in joined if v[0] == 3]
            head_volumes[name] = [v[1] for v in body_volumes[name]
                                  if np.dot(np.array(gmsh.model.occ.getCenterOfMass(*v)) - center, inward_axis) < 0]
            screw_cuts[name] = {"center_mm": center.tolist(), "inward_axis": inward_axis.tolist()}
        gmsh.model.occ.synchronize()
        adaptive_volume_checks = _adaptive_partition_volume_audit(gmsh, body_volumes, manifest, output_dir)
        semantic_surfaces = {}
        body_surfaces = {}
        source_selection_refs = {}
        for name, volumes in body_volumes.items():
            volume_after = sum(gmsh.model.occ.getMass(dim, tag) for dim, tag in volumes)
            before = volume_checks[name]["before_mm3"]
            volume_checks[name]["after_mm3"] = volume_after
            if abs(volume_after - before) > 1e-4 * before:
                raise ValueError(f"Partition changed {name} volume: {before} -> {volume_after}, tags={volumes}")
            surfaces = [tag for dim, tag in gmsh.model.getBoundary(volumes, combined=True, oriented=False) if dim == 2]
            body_surfaces[name] = surfaces
            gmsh.model.addPhysicalGroup(3, [tag for _, tag in volumes], name=name)
            for tag in surfaces:
                matched = _face_geometry_match(gmsh, tag, original_faces[name])
                if matched:
                    source_selection_refs[(name, matched["index"])] = tag
            if name == "New Pipe":
                load_faces, midpoint_faces, bore_faces, outer_faces = [], [], [], []
                for tag in surfaces:
                    center = gmsh.model.occ.getCenterOfMass(2, tag)
                    box = gmsh.model.getBoundingBox(2, tag)
                    kind = gmsh.model.getType(2, tag)
                    if abs(box[2]) < 1e-5 and abs(box[5]) < 1e-5:
                        midpoint_faces.append(tag)
                    if kind != "Plane":
                        # Distinguish coaxial pipe cylinders from small slots by
                        # the radius of a real surface evaluation, not its COM.
                        uv_min, uv_max = gmsh.model.getParametrizationBounds(2, tag)
                        uv = [(uv_min[0] + uv_max[0]) / 2, (uv_min[1] + uv_max[1]) / 2]
                        samples = [gmsh.model.getValue(2, tag,
                                   [uv_min[0] + a * (uv_max[0]-uv_min[0]), uv_min[1] + b * (uv_max[1]-uv_min[1])])
                                   for a, b in [(0.21, 0.37), (0.49, 0.61), (0.73, 0.83)]]
                        radii = [math.hypot(p[0], p[1]) for p in samples]
                        if all(abs(r - 17.5) < 1e-4 for r in radii):
                            bore_faces.append(tag)
                        elif all(abs(r - 19) < 1e-4 for r in radii):
                            outer_faces.append(tag)
                            if box[2] >= -25 - 1e-5 and box[5] <= 1e-5 and box[0] >= -half_width - 1e-5 and box[3] <= half_width + 1e-5 and center[1] < 0:
                                load_faces.append(tag)
                if not load_faces or not midpoint_faces or not bore_faces:
                    diagnostic = [{"tag": tag, "type": gmsh.model.getType(2, tag),
                                   "center": gmsh.model.occ.getCenterOfMass(2, tag),
                                   "bounds": gmsh.model.getBoundingBox(2, tag)} for tag in surfaces]
                    (output_dir / "pipe-surface-selection-diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
                    raise ValueError(f"Derived surface selection failed: load={load_faces},midpoint={midpoint_faces},bore={bore_faces}")
                semantic_surfaces.update(load_patch=load_faces, midpoint_symmetry=midpoint_faces,
                                         pipe_bore=bore_faces, pipe_outer_wall=outer_faces)
                if slot_size_mm is not None:
                    slot_selections, slot_fragments = _slot_wall_fragments(gmsh, surfaces, original_faces[name])
                    semantic_surfaces.update(slot_selections)
                    slot_scope["source_face_fragment_map"] = slot_fragments
        for semantic, ref in audit["named_selections"].items():
            refs = ref if isinstance(ref, list) else [ref]
            tags = [source_selection_refs[(r["body"], r["face_index_at_audit"])]
                    for r in refs if (r["body"], r["face_index_at_audit"]) in source_selection_refs]
            if tags and semantic not in semantic_surfaces:
                semantic_surfaces[semantic] = tags
        for name, cut in screw_cuts.items():
            from collections import Counter
            face_occurrences = Counter(gmsh.model.getBoundary(body_volumes[name], combined=False, oriented=False))
            matches = [tag for (dim, tag), count in face_occurrences.items() if dim == 2 and count == 2
                       and gmsh.model.getType(2, tag) == "Plane"
                       and abs(np.dot(np.array(gmsh.model.occ.getCenterOfMass(2, tag)) - cut["center_mm"], cut["inward_axis"])) < 1e-5]
            if not matches or abs(sum(gmsh.model.occ.getMass(2, s) for s in matches) - math.pi*16) > 1e-4:
                raise ValueError(f"Failed to resolve radius-4 shoulder cut: {name}")
            semantic_surfaces[name.lower().replace(" ", "_") + "_cut"] = matches
        aliases = {"midpoint_symmetry": "midpoint", "hub_snout_candidates": "hub_snout"}
        for i in range(1, 4):
            aliases.update({f"bolt_{i}_shaft": f"screw_{i}_anchor", f"bolt_{i}_anchor_holes": f"hub_hole_{i}",
                            f"bolt_{i}_cone": f"screw_{i}_cone", f"bolt_{i}_pipe_seat": f"screw_{i}_seat",
                            f"bolt_{i}_cut": f"screw_{i}_cut"})
        semantic_surfaces = {aliases.get(k, k): v for k, v in semantic_surfaces.items()}
        if radial_partitions:
            # Internal layer cylinders split the existing seat faces. Recover
            # all pieces by their unchanged analytic cone equation.
            for i in range(1, 4):
                reference = audit["named_selections"][f"bolt_{i}_pipe_seat"][0]
                cone = next(f for f in original_faces["New Pipe"] if f["index"] == reference["face_index_at_audit"])
                origin, axis = np.array(cone["origin_mm"]), np.array(cone["axis"])
                matches = []
                for tag in body_surfaces["New Pipe"]:
                    if gmsh.model.getType(2, tag) == "Plane":
                        continue
                    low, high = gmsh.model.getParametrizationBounds(2, tag)
                    points = np.array([gmsh.model.getValue(2, tag, [low[0]+u*(high[0]-low[0]), low[1]+v*(high[1]-low[1])])
                                       for u, v in [(.21, .37), (.49, .61), (.73, .83)]])
                    delta = points-origin
                    axial = delta@axis
                    radial = np.linalg.norm(delta-axial[:, None]*axis, axis=1)
                    error = radial-(cone["reference_radius_mm"]+axial*math.tan(cone["semi_angle_radians"]))
                    if np.max(np.abs(error)) < 1e-5:
                        matches.append(tag)
                if not matches or abs(sum(gmsh.model.occ.getMass(2, s) for s in matches)-cone["area_mm2"]) > .01:
                    raise ValueError(f"Layered pipe seat {i} failed original-cone area/support check")
                semantic_surfaces[f"screw_{i}_seat"] = matches
            all_pipe_faces = set(gmsh.model.getBoundary(body_volumes["New Pipe"], combined=False, oriented=False))
            for layer, radius in enumerate(radial_partitions, 1):
                matches = []
                for dim, tag in all_pipe_faces:
                    if dim != 2 or gmsh.model.getType(2, tag) == "Plane":
                        continue
                    low, high = gmsh.model.getParametrizationBounds(2, tag)
                    points = [gmsh.model.getValue(2, tag, [low[0]+u*(high[0]-low[0]), low[1]+v*(high[1]-low[1])])
                              for u, v in [(.21, .37), (.49, .61), (.73, .83)]]
                    if all(abs(math.hypot(p[0], p[1])-radius) < 1e-5 for p in points):
                        if len(gmsh.model.getAdjacencies(2, tag)[0]) != 2:
                            raise ValueError("Radial layer interface is not shared by exactly two material volumes")
                        matches.append(tag)
                if not matches:
                    raise ValueError("Radial layer interface could not be selected")
                semantic_surfaces[f"pipe_layer_interface_{layer}"] = matches
        for i in range(1, 4):
            hole_refs = audit["named_selections"][f"bolt_{i}_anchor_holes"]
            if len(hole_refs) != 1:
                raise ValueError("Expected one physical hole per screw")
            hole_geometry = next(f for f in original_faces["Hub"] if f["index"] == hole_refs[0]["face_index_at_audit"])
            # Independent trim quadrature differs by ~0.0008 mm² and 0.0001
            # mm in centroid; preserve a narrow matching bound and separately
            # measure every slave-node distance to its meshed hole surface.
            matching_anchor = [s for s in body_surfaces[f"Bolt {i}"] if _face_geometry_match(
                gmsh, s, [hole_geometry], area_tolerance=.005, centroid_tolerance_mm=.001)]
            if len(matching_anchor) != 1:
                diagnostic = {"expected_hole": hole_geometry, "screw_surfaces": [{"tag": s, "area": gmsh.model.occ.getMass(2, s),
                              "center": gmsh.model.occ.getCenterOfMass(2, s), "type": gmsh.model.getType(2, s)} for s in body_surfaces[f"Bolt {i}"]]}
                (output_dir / f"screw-{i}-engagement-diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
                raise ValueError(f"Exact curved engagement selection unresolved for screw {i}: {matching_anchor}")
            semantic_surfaces[f"screw_{i}_anchor"] = matching_anchor
        # Only the outer 17.05-mm snout cylinder belongs to the bore contact.
        snout_reference = [r for r in audit["named_selections"]["hub_snout_candidates"] if abs(r["radius_mm"] - 17.05) < 1e-5]
        semantic_surfaces["hub_snout"] = [source_selection_refs[(r["body"], r["face_index_at_audit"])] for r in snout_reference]
        hub_surface_geometry = {}
        if retain_rigid_hub_carriers:
            keep_hub_surfaces = set(s for name in ["hub_snout", "hub_hole_1", "hub_hole_2", "hub_hole_3"] for s in semantic_surfaces[name])
            for surface in keep_hub_surfaces:
                match = _face_geometry_match(gmsh, surface, original_faces["Hub"])
                if not match or match["type"] != "Cylinder":
                    raise ValueError("Rigid carrier interface must match an audited analytic cylinder")
                hub_surface_geometry[surface] = match
            gmsh.model.occ.remove(body_volumes["Hub"], recursive=False)
            gmsh.model.occ.remove([(2, s) for s in body_surfaces["Hub"] if s not in keep_hub_surfaces], recursive=True)
            body_volumes["Hub"] = []
            body_surfaces["Hub"] = list(keep_hub_surfaces)
            gmsh.model.occ.synchronize()
        # Contact and screw regions are locally refined; pilot far span remains
        # coarse by design, so this result cannot pass thickness convergence.
        refine_surfaces = set()
        anchor_refine_surfaces = set()
        for key, surfaces in semantic_surfaces.items():
            if "cone" in key or "seat" in key or "cut" in key:
                refine_surfaces.update(surfaces)
        for name in ("Bolt 1", "Bolt 2", "Bolt 3"):
            head_surfaces = {s for dim, s in gmsh.model.getBoundary([(3, v) for v in head_volumes[name]],
                                                                   combined=True, oriented=False) if dim == 2}
            refine_surfaces.update(head_surfaces)
            anchor_refine_surfaces.update(set(body_surfaces[name])-head_surfaces)
        anchor_refine_surfaces.update(s for i in range(1, 4) for s in semantic_surfaces[f"hub_hole_{i}"])
        refine_points = gmsh.model.getBoundary([(2, s) for s in refine_surfaces], combined=False, oriented=False, recursive=True)
        gmsh.model.mesh.setSize(list(set(p for p in refine_points if p[0] == 0)), contact_size_mm)
        # Explicit spatial sizing prevents small unrelated rigid-hub fillets
        # from imposing their size throughout the complete hub interior.
        field = gmsh.model.mesh.field
        constant = field.add("MathEval")
        field.setString(constant, "F", str(hub_size_mm))
        fields = [constant]
        def surface_threshold(surfaces, size, distance):
            source = field.add("Distance")
            field.setNumbers(source, "SurfacesList", sorted(set(surfaces)))
            field.setNumber(source, "Sampling", 60)
            threshold = field.add("Threshold")
            field.setNumber(threshold, "InField", source)
            field.setNumber(threshold, "SizeMin", size)
            field.setNumber(threshold, "SizeMax", hub_size_mm)
            field.setNumber(threshold, "DistMin", distance)
            field.setNumber(threshold, "DistMax", distance + 3)
            fields.append(threshold)
        surface_threshold([s for n, surfaces in body_surfaces.items() if n != "Hub" for s in surfaces], far_size_mm, 1.5)
        surface_threshold(refine_surfaces, contact_size_mm, .3)
        surface_threshold(anchor_refine_surfaces, anchor_size_mm, .3)
        surface_threshold(semantic_surfaces["pipe_bore"] + semantic_surfaces["pipe_outer_wall"],
                          far_size_mm if enforce_wall_layers else thickness_size_mm, .8)
        if bore_snout_size_mm is not None:
            surface_threshold(semantic_surfaces["hub_snout"], bore_snout_size_mm, 2.0)
        if slot_size_mm is not None:
            slot_surfaces = sorted({tag for i in range(1,4) for tag in semantic_surfaces[f"slot_{i}_walls"]})
            slot_curves = sorted({tag for dim, tag in gmsh.model.getBoundary(
                [(2, tag) for tag in slot_surfaces], combined=False, oriented=False) if dim == 1})
            slot_points = sorted({tag for dim, tag in gmsh.model.getBoundary(
                [(1, tag) for tag in slot_curves], combined=False, oriented=False) if dim == 0})
            source = field.add("Distance")
            field.setNumbers(source, "SurfacesList", slot_surfaces)
            field.setNumbers(source, "CurvesList", slot_curves)
            field.setNumbers(source, "PointsList", slot_points)
            field.setNumber(source, "Sampling", 60)
            threshold = field.add("Threshold")
            for key, value in {"InField": source, "SizeMin": slot_size_mm, "SizeMax": hub_size_mm,
                               "DistMin": .5, "DistMax": 3.5}.items():
                field.setNumber(threshold, key, value)
            fields.append(threshold)
            slot_scope.update(derived_surface_tags=slot_surfaces, derived_curve_tags=slot_curves,
                              derived_point_tags=slot_points)
        background = field.add("Min")
        field.setNumbers(background, "FieldsList", fields)
        field.setAsBackgroundMesh(background)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), far_size_mm)
        gmsh.model.mesh.setSize(list(set(p for p in refine_points if p[0] == 0)), contact_size_mm)
        if slot_size_mm is not None:
            contact_points = {p[1] for p in refine_points if p[0] == 0}
            gmsh.model.mesh.setSize([(0,p) for p in slot_points if p not in contact_points], slot_size_mm)
            gmsh.model.mesh.setSize([(0,p) for p in slot_points if p in contact_points], min(slot_size_mm, contact_size_mm))
        for name, surfaces in semantic_surfaces.items():
            gmsh.model.addPhysicalGroup(2, surfaces, name=name)
        gmsh.model.mesh.generate(3)
        initial_element_ids = np.concatenate(gmsh.model.mesh.getElements(3)[1])
        initial_quality = np.asarray(gmsh.model.mesh.getElementQualities(initial_element_ids.tolist(), "minSJ"))
        optimization = {"nonpositive_before": int((initial_quality <= 0).sum()),
                        "below_scaled_jacobian_0_1_before": int((initial_quality < .1).sum()),
                        "minimum_before": float(initial_quality.min()), "method": "none"}
        if np.any(initial_quality < .1):
            gmsh.write(str(output_dir / "before-high-order-optimization.msh"))
            (output_dir / "pre-optimization-status.json").write_text(json.dumps({"status": "unaccepted_mesh_before_optimization", **optimization})+"\n")
            gmsh.option.setNumber("Mesh.HighOrderNumLayers", 6)
            gmsh.option.setNumber("Mesh.HighOrderThresholdMin", 0.1)
            optimization["method"] = "HighOrder localized 6 layers"
            gmsh.model.mesh.optimize("HighOrder", niter=10)
            optimized_quality = np.asarray(gmsh.model.mesh.getElementQualities(initial_element_ids.tolist(), "minSJ"))
            optimization["nonpositive_after"] = int((optimized_quality <= 0).sum())
            optimization["minimum_after"] = float(optimized_quality.min())
        if retain_rigid_hub_carriers:
            volume, carrier_count = _add_fixed_rigid_surface_carriers(gmsh, hub_surface_geometry)
            body_volumes["Hub"] = [(3, volume)]
        gmsh.write(str(output_dir / "registered-separate-contact.msh"))
        node_ids, xyz, _ = gmsh.model.mesh.getNodes()
        node_ids, xyz = np.asarray(node_ids, dtype=np.int64), np.asarray(xyz).reshape(-1, 3)
        if len(node_ids) > max_nodes:
            rejection = {"schema_version": 1, "status": "blocked_mesh_exceeds_node_limit", "source_sha256": manifest["source_sha256"],
                         "nodes": len(node_ids), "max_nodes": max_nodes,
                         "settings": {"thickness_size_mm": thickness_size_mm, "contact_size_mm": contact_size_mm,
                                      "anchor_size_mm": anchor_size_mm, "patch_projected_width_mm": patch_projected_width_mm,
                                      "bore_snout_size_mm": bore_snout_size_mm,
                                      **({"slot_size_mm": slot_size_mm} if slot_size_mm is not None else {})},
                         "elapsed_seconds": time.monotonic()-start, "solver_started": False,
                         "raw_mesh": "registered-separate-contact.msh", "high_order_optimization": optimization,
                         "resource_observations": {"before_generation": initial_memory, "at_rejection": _process_memory_snapshot()},
                         "adaptive_volume_preservation": adaptive_volume_checks}
            (output_dir / "mesh-manifest.json").write_text(json.dumps(rejection, indent=2)+"\n")
            raise ValueError(f"Mesh has {len(node_ids)} nodes, exceeding bounded limit {max_nodes}")
        cell_ids, cells, component_elsets, head_element_sets = [], [], {}, {}
        pipe_partition_volumes = []
        permutation = _tetra_quadratic_ccx_permutation(gmsh) if order == 2 else list(range(4))
        expected_type, width = (11, 10) if order == 2 else (4, 4)
        for name, volumes in body_volumes.items():
            tags_for_component = []
            head_elements = []
            for _, tag in volumes:
                if name == "New Pipe" and radial_partitions:
                    _, local_xyz, _ = gmsh.model.mesh.getNodes(3, tag, includeBoundary=True)
                    local_radii = np.linalg.norm(np.asarray(local_xyz).reshape(-1, 3)[:, :2], axis=1)
                    radial_bounds = [float(local_radii.min()), float(local_radii.max())]
                    interval = int(np.searchsorted(radial_partitions, sum(radial_bounds)/2))
                    expected_bounds = [17.5, *radial_partitions, 19.0][interval:interval+2]
                    # Source edge tolerances reach 0.0074 mm; preserve them.
                    if radial_bounds[0] < expected_bounds[0]-.01 or radial_bounds[1] > expected_bounds[1]+.01:
                        raise ValueError("Material partition crosses its intended radial interval")
                    pipe_partition_volumes.append({"cad_volume_tag": tag, "radial_interval_index": interval+1,
                                                   "nominal_radial_bounds_mm": expected_bounds,
                                                   "mesh_node_radial_bounds_mm": radial_bounds,
                                                   "cad_volume_mm3": gmsh.model.occ.getMass(3, tag)})
                types, tags, connectivity = gmsh.model.mesh.getElements(3, tag)
                if any(t != expected_type for t in types):
                    raise ValueError("Unexpected element family in tetrahedral mesh")
                for ids, nodes in zip(tags, connectivity):
                    cell_ids.extend(ids)
                    tags_for_component.extend(int(i) for i in ids)
                    cells.extend(np.asarray(nodes).reshape(-1, width)[:, permutation])
                    if tag in head_volumes.get(name, []):
                        head_elements.extend(int(i) for i in ids)
            canonical_name = {"New Pipe": "pipe", "Hub": "hub"}.get(name, name.lower().replace("bolt ", "screw_"))
            component_elsets[canonical_name] = tags_for_component
            if head_elements:
                head_element_sets[canonical_name] = set(head_elements)
        boundary_arrays, boundary_info = {}, {}
        for name, surfaces in semantic_surfaces.items():
            triangles = []
            for surface in surfaces:
                types, tags, connectivity = gmsh.model.mesh.getElements(2, surface)
                expected_triangle = 9 if order == 2 else 2
                if any(t != expected_triangle for t in types):
                    raise ValueError("Unexpected boundary element family")
                for nodes in connectivity:
                    triangles.extend(np.asarray(nodes).reshape(-1, 6 if order == 2 else 3))
            key = "boundary_" + name
            boundary_arrays[key] = np.asarray(triangles, dtype=np.int64)
            boundary_info[name] = {"array_key": key, "surface_tags": surfaces,
                                   "triangle_count": len(triangles),
                                   "cad_area_mm2": sum(gmsh.model.occ.getMass(2, s) for s in surfaces)}
        element_ids = np.asarray(cell_ids, dtype=np.int64)
        connectivity = np.asarray(cells, dtype=np.int64)
        qualities = np.asarray(gmsh.model.mesh.getElementQualities(element_ids.tolist(), "minSJ"))
        np.savez_compressed(output_dir / "mesh-arrays-and-boundaries.npz", node_ids=node_ids, nodes=xyz,
                            element_ids=element_ids, tetra10=connectivity if order == 2 else np.empty((0, 10), dtype=np.int64),
                            tetra4=connectivity if order == 1 else np.empty((0, 4), dtype=np.int64), **boundary_arrays)
        theoretical_patch_area = 2 * 19 * math.asin(half_width / 19) * 25
        actual_patch_area = boundary_info["load_patch"]["cad_area_mm2"]
        if abs(actual_patch_area - theoretical_patch_area) > 1e-4:
            raise ValueError("Projected-width cylindrical load-patch area failed analytical check")
        result = {"schema_version": 1, "status": "pilot_mesh_only_not_solver_accepted", "gmsh_version": gmsh.__version__,
                  "source_sha256": manifest["source_sha256"], "registered_manifest": str(Path(registered_manifest).resolve()),
                  "high_order_optimization": optimization,
                  "settings": {"far_size_mm": far_size_mm, "contact_size_mm": contact_size_mm, "order": order,
                               "thickness_size_mm": thickness_size_mm, "hub_size_mm": hub_size_mm,
                               "anchor_size_mm": anchor_size_mm,
                               "bore_snout_size_mm": bore_snout_size_mm,
                               **({"slot_size_mm": slot_size_mm} if slot_size_mm is not None else {}),
                               "enforce_wall_layers": enforce_wall_layers,
                               "max_nodes": max_nodes, "retain_rigid_hub_carriers": retain_rigid_hub_carriers,
                               "threads": threads, "include_hub": include_hub,
                               "patch_projected_width_mm": patch_projected_width_mm, "patch_axial_length_mm": 25},
                  "load_patch": {"side": "-Y", "force_direction": [0, 1, 0], "total_force_N": 1112.0554,
                                 "analytical_curved_area_mm2": theoretical_patch_area,
                                 "constant_specified_direction_traction_MPa": 1112.0554 / theoretical_patch_area},
                  "counts": {"nodes": len(node_ids), "elements": len(element_ids), "displacement_dofs_upper_bound": 3 * len(node_ids)},
                  "mesh_quality": {"min_scaled_jacobian": float(qualities.min()), "nonpositive_elements": int((qualities <= 0).sum()),
                                   "by_component": {name: {"minimum": float(qualities[np.isin(element_ids, ids)].min()),
                                                            "nonpositive": int((qualities[np.isin(element_ids, ids)] <= 0).sum())}
                                                    for name, ids in component_elsets.items()}},
                  "component_element_sets": component_elsets, "boundary_groups": boundary_info,
                  "volume_preservation": volume_checks, "elapsed_seconds": time.monotonic() - start,
                  "adaptive_volume_preservation": adaptive_volume_checks,
                  "sizing_field_scopes": {"contact": "pipe seat cones and complete screw head side including cut",
                                          "anchor": "complete screw shank side and hub anchor holes",
                                          "bore_snout": bore_snout_scope,
                                          **({"slot": slot_scope} if slot_scope is not None else {}),
                                          "transition": "minimum distance fields with 0.3 mm near region; local grading can refine near shoulder"},
                  "contact_nodes_merged": False, "pretension_cuts": screw_cuts,
                  "limitations": ["Mesh preparation alone does not establish convergence", "No assembly solve or convergence claim", "Requires validated explicit paired-cut kinematics and fixed-rigid anchorage"]}
        if not np.isfinite(qualities).all() or float(qualities.min()) < .1:
            result["status"] = "rejected_mesh_quality"
            result["failure"] = "Every exported tetrahedron must have finite scaled Jacobian at least 0.1"
            (output_dir / "mesh-manifest.json").write_text(json.dumps(result, indent=2)+"\n")
            raise ValueError(result["failure"])
        if order == 2:
            from .mesh import SolidMesh
            mesh = SolidMesh(node_ids, xyz, element_ids, connectivity, element_sets=component_elsets,
                             provenance={"source_sha256": manifest["source_sha256"],
                                         "coordinate_frame": "comsol_reference_mm", "patch_width_mm": patch_projected_width_mm,
                                         "bore_snout_size_mm": bore_snout_size_mm,
                                         "bore_snout_sizing_scope": bore_snout_scope,
                                         **({"slot_size_mm": slot_size_mm, "slot_sizing_scope": slot_scope}
                                            if slot_size_mm is not None else {}),
                                         "pretension_axes_reference": {str(i): screw_cuts[f"Bolt {i}"]["inward_axis"] for i in range(1, 4)},
                                         "pretension_head_element_sets": {str(i): sorted(head_element_sets[f"screw_{i}"]) for i in range(1, 4)},
                                         "pretension_implementation": "explicit_paired_cut_required_native_rim_algorithm_rejected",
                                         "cuts_verified": True, "contact_clearances_verified": True,
                                         "mesh_manifest": str(output_dir / "mesh-manifest.json")})
            lookup = mesh.face_lookup()
            layer_continuity = {}
            for name, info in boundary_info.items():
                faces = []
                for triangle in boundary_arrays[info["array_key"]]:
                    candidates = lookup.get(tuple(sorted(int(n) for n in triangle[:3])), [])
                    if name.startswith("pipe_layer_interface_"):
                        if len(candidates) != 2:
                            raise ValueError("Layer interface does not have shared nodes on two adjacent tetrahedra")
                        if any(set(map(int, mesh.face_nodes(*face))) != set(map(int, triangle)) for face in candidates):
                            raise ValueError("Quadratic layer interface has mismatched midside nodes")
                        continue
                    if name.endswith("_cut"):
                        head_ids = head_element_sets[name.removesuffix("_cut")]
                        candidates = [f for f in candidates if f[0] in head_ids]
                    if len(candidates) != 1:
                        raise ValueError(f"Boundary triangle {name} matched {len(candidates)} solid faces")
                    faces.append(candidates[0])
                if name.startswith("pipe_layer_interface_"):
                    layer_continuity[name] = {"triangles_with_two_adjacent_tetrahedra": len(boundary_arrays[info["array_key"]]),
                                              "shared_node_continuity_verified": True,
                                              "all_six_quadratic_face_nodes_shared": True}
                else:
                    mesh.surfaces[name] = sorted(set(faces))
            result["radial_layer_continuity"] = layer_continuity
            result["radial_partition_volumes"] = pipe_partition_volumes
            mesh.node_sets["midpoint"] = mesh.surface_nodes("midpoint")
            mesh.node_sets["hub_fixed"] = sorted({int(n) for e in component_elsets.get("hub", []) for n in mesh.tetra10[mesh._elements[e]]})
            adjacency = {}
            for i in range(1, 4):
                cut_elements = {e for e, _ in mesh.surfaces[f"screw_{i}_cut"]}
                contact_elements = {e for e, _ in mesh.surfaces[f"screw_{i}_cone"]}
                cut_nodes = set(mesh.surface_nodes(f"screw_{i}_cut"))
                contact_elements_with_cut_node = [e for e in contact_elements
                                                  if cut_nodes.intersection(int(n) for n in mesh.tetra10[mesh._elements[e]])]
                adjacency[str(i)] = {"cut_elements_with_contact_faces": len(cut_elements & contact_elements),
                                     "contact_elements_containing_any_cut_node": len(contact_elements_with_cut_node)}
            result["native_pretension_quadratic_adjacency"] = adjacency
            if retain_rigid_hub_carriers:
                all_hub_elements = set(component_elsets["hub"])
                needed_hub_elements = {e for s in ["hub_snout", "hub_hole_1", "hub_hole_2", "hub_hole_3"] for e, _ in mesh.surfaces[s]}
                retained_elements = np.array([e not in all_hub_elements or e in needed_hub_elements for e in element_ids])
                retained_connectivity = connectivity[retained_elements]
                needed_nodes = set(int(n) for n in retained_connectivity.ravel())
                retained_nodes = np.array([n in needed_nodes for n in node_ids])
                component_elsets["hub"] = sorted(needed_hub_elements)
                mesh = SolidMesh(node_ids[retained_nodes], xyz[retained_nodes], element_ids[retained_elements], retained_connectivity,
                                 element_sets=component_elsets, surfaces=mesh.surfaces,
                                 node_sets={"midpoint": mesh.node_sets["midpoint"],
                                            "hub_fixed": sorted(set(mesh.node_sets["hub_fixed"]) & needed_nodes)},
                                 provenance={**mesh.provenance, "hub_representation": "fixed_surface_adjacent_tetra_carriers_only"})
                retained_qualities = qualities[retained_elements]
                result["fixed_rigid_hub_reduction"] = {"original_hub_elements": len(all_hub_elements),
                                                      "retained_contact_carrier_elements": len(needed_hub_elements),
                                                      "generation": "Only exact snout/hole triangle6 surfaces meshed; full hub volume not meshed",
                                                      "physical_assumption": "Every retained hub node fixed; no hub elasticity or stress interpretation"}
                result["retained_mesh_quality"] = {"min_scaled_jacobian": float(retained_qualities.min()),
                                                   "nonpositive_elements": int((retained_qualities <= 0).sum())}
                result["retained_counts"] = {"nodes": len(mesh.node_ids), "elements": len(mesh.element_ids),
                                              "displacement_dofs_upper_bound": 3 * len(mesh.node_ids)}
                mesh.provenance["mesh_quality"] = result["retained_mesh_quality"]
            result["resource_limit_passed"] = len(mesh.node_ids) <= max_nodes
            if not result["resource_limit_passed"]:
                result["status"] = "blocked_mesh_exceeds_node_limit"
            result["tie_projection"] = _tie_projection_diagnostic(mesh)
            mesh.provenance["tie_projection_tolerance_mm"] = result["tie_projection"]["recommended_search_tolerance_mm"]
            contact_pairs = [("pipe_bore", "hub_snout")] + [(f"screw_{i}_cone", f"screw_{i}_seat") for i in range(1, 4)]
            result["separate_contact_nodes"] = {f"{a}__{b}": len(set(mesh.surface_nodes(a)) & set(mesh.surface_nodes(b)))
                                                 for a, b in contact_pairs}
            if any(result["separate_contact_nodes"].values()):
                raise ValueError("Separating contact interfaces share node IDs")
            result["surface_resolution"] = surface_resolution(mesh)
            pipe_cells = [mesh._elements[e] for e in component_elsets["pipe"]]
            pipe_xyz = mesh.points_mm[mesh.connectivity_indices()[pipe_cells, :4]]
            radii = np.linalg.norm(pipe_xyz[:, :, :2], axis=2)
            spans = np.ptp(radii, axis=1)
            result["pipe_wall_resolution"] = {"metric": "exact concentric material partitions plus measured corner-node radial spans" if radial_partitions else "corner-node radial span; descriptive measurement, not a count of swept layers",
                                                "enforced_radial_layers": wall_layers,
                                                "radial_partition_radii_mm": radial_partitions,
                                                "maximum_radial_span_mm": float(spans.max()),
                                                "percentile95_radial_span_mm": float(np.percentile(spans, 95)),
                                                "tetrahedra_touching_both_bore_and_outer_wall": int(((radii.min(axis=1) <= 17.5+1e-4) & (radii.max(axis=1) >= 19-1e-4)).sum()),
                                                "requested_thickness_size_mm": thickness_size_mm}
            if wall_layers >= 2 and result["pipe_wall_resolution"]["tetrahedra_touching_both_bore_and_outer_wall"]:
                raise ValueError("A layered pipe tetrahedron spans the complete wall")
            try:
                mesh.save(output_dir / "structural-mesh.npz")
            except ValueError as exc:
                result["status"] = "rejected_mesh_quality"
                result["failure"] = str(exc)
                (output_dir / "mesh-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
                raise
            mesh.export_vtu(output_dir / "structural-mesh.vtu")
        result["resource_observations"] = {"before_generation": initial_memory, "after_export": _process_memory_snapshot()}
        (output_dir / "mesh-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        (output_dir / "gmsh.log").write_text("\n".join(gmsh.logger.get()) + "\n")
        gmsh.logger.stop()
        gmsh.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("source", type=Path)
    register.add_argument("audit", type=Path)
    register.add_argument("output", type=Path)
    mesh = commands.add_parser("pilot-mesh")
    mesh.add_argument("registered_manifest", type=Path)
    mesh.add_argument("audit", type=Path)
    mesh.add_argument("output", type=Path)
    mesh.add_argument("--far-size", type=float, default=3.0)
    mesh.add_argument("--contact-size", type=float, default=1.0)
    mesh.add_argument("--patch-width", type=float, default=35.0)
    mesh.add_argument("--order", type=int, default=2)
    mesh.add_argument("--thickness-size", type=float, default=1.5)
    mesh.add_argument("--max-nodes", type=int, default=500000)
    mesh.add_argument("--enforce-wall-layers", action="store_true")
    mesh.add_argument("--bore-snout-size", type=float, default=None)
    mesh.add_argument("--slot-size", type=float, default=None)
    args = parser.parse_args()
    if args.command == "register":
        result = prepare_registered_geometry(args.source, args.audit, args.output)
        print(json.dumps({"body_count": len(result["bodies"]), "output": str(args.output)}))
    else:
        result = generate_pilot_mesh(args.registered_manifest, args.audit, args.output,
                                     far_size_mm=args.far_size, contact_size_mm=args.contact_size,
                                     patch_projected_width_mm=args.patch_width, order=args.order, thickness_size_mm=args.thickness_size,
                                     max_nodes=args.max_nodes, enforce_wall_layers=args.enforce_wall_layers,
                                     bore_snout_size_mm=args.bore_snout_size, slot_size_mm=args.slot_size)
        print(json.dumps({k: result[k] for k in ("counts", "mesh_quality", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
