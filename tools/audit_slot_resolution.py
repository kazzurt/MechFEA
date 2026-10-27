"""Read-only original slot topology and axial/circumferential mesh resolution.

Usage: python audit_slot_resolution.py NEW_AUDIT_DIRECTORY [MESH_FOLDER ...]
No source/mesh edits, native meshing, solver calls, or acceptance promotion.
"""
from pathlib import Path
import gc
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from structural.geometry import bounds, describe_face, shape_parts
from structural.mesh import SolidMesh, TET10_FACES

PREP = ROOT / "Dodec FEA/structural-fea-model-preparation"
GEOMETRY = PREP / "registered-geometry/registered-geometry-map.json"
SLOT_GROUPS = {"slot_1": list(range(5, 13)), "slot_2": list(range(13, 21)),
               "slot_3": [*range(21, 28), 29]}
FACE_IDS = [index for group in SLOT_GROUPS.values() for index in group]
PROFILE_EDGES = ((0, 1, 3), (1, 2, 4), (2, 0, 5))
BAND = [-201., -190.]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def distribution(values):
    values = np.asarray(values, dtype=float).ravel()
    if not len(values):
        return {"count": 0, "minimum": None, "median": None, "percentile95": None, "maximum": None}
    return {"count": len(values), "minimum": float(values.min()), "median": float(np.median(values)),
            "percentile95": float(np.percentile(values, 95)), "maximum": float(values.max())}


def edge_measures(xyz):
    """Endpoints in Cartesian coordinates; short angular arc in cylindrical frame."""
    a, b = xyz[:, 0], xyz[:, 1]
    r = (np.linalg.norm(a[:, :2], axis=1)+np.linalg.norm(b[:, :2], axis=1))/2
    theta = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])
    angle = (theta[:, 1]-theta[:, 0]+np.pi) % (2*np.pi)-np.pi
    axial = abs(b[:, 2]-a[:, 2])
    circum = abs(angle)*r
    radial = abs(np.linalg.norm(a[:, :2], axis=1)-np.linalg.norm(b[:, :2], axis=1))
    return {"axial_span_mm": distribution(axial), "circumferential_arc_span_mm": distribution(circum),
            "in_plane_span_mm": distribution(np.hypot(axial, circum)), "radial_span_mm": distribution(radial),
            "chord_length_mm": distribution(np.linalg.norm(b-a, axis=1))}


def source_slot_topology():
    from OCP.TopoDS import TopoDS, TopoDS_Shape
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepTools import BRepTools
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    geometry = json.loads(GEOMETRY.read_text())
    pipe = next(body for body in geometry["bodies"] if body["name"] == "New Pipe")
    brep = Path(pipe["path"])
    shape = TopoDS_Shape()
    assert BRepTools.Read_s(shape, str(brep), BRep_Builder())
    faces = [TopoDS.Face(face) for face in shape_parts(shape, TopAbs_FACE)]
    descriptors = {face["index"]: face for face in pipe["registered_faces"]}
    edges, face_edges, adjacent = [], {}, []
    for index, face in enumerate(faces, 1):
        measured = describe_face(face, index)
        assert np.linalg.norm(np.array(measured["centroid_mm"])-descriptors[index]["centroid_mm"]) < 1e-7
        assert abs(measured["area_mm2"]-descriptors[index]["area_mm2"]) < 1e-7
        face_edges[index] = []
        for edge in shape_parts(face, TopAbs_EDGE):
            found = next((i for i, old in enumerate(edges) if old.IsSame(edge)), None)
            if found is None:
                found = len(edges); edges.append(TopoDS.Edge(edge)); adjacent.append([])
            face_edges[index].append(found+1)
            if index not in adjacent[found]: adjacent[found].append(index)
    slot_edges = sorted({edge for index in FACE_IDS for edge in face_edges[index]})
    edge_records = {}
    for index in slot_edges:
        edge = edges[index-1]
        curve = BRepAdaptor_Curve(edge)
        parameters = np.linspace(curve.FirstParameter(), curve.LastParameter(), 101)
        points = np.array([[p.X(), p.Y(), p.Z()] for p in (curve.Value(float(t)) for t in parameters)])
        prop = GProp_GProps(); BRepGProp.LinearProperties_s(edge, prop)
        edge_records[str(index)] = {"type": curve.GetType().name, "length_mm": float(prop.Mass()),
            "tolerance_mm": float(BRep_Tool.Tolerance_s(edge)), "adjacent_source_face_indices": adjacent[index-1],
            "bounds": bounds(edge), "sampled_pipe_radius_bounds_mm": [float(np.linalg.norm(points[:, :2], axis=1).min()), float(np.linalg.norm(points[:, :2], axis=1).max())],
            "endpoint_midpoint_mm": points[[0, 50, 100]].tolist(),
            "sampled_support_normal_errors_mm": {str(face): float(support_error(points, descriptors[face]).max())
                for face in adjacent[index-1] if face in FACE_IDS},
            "sampled_pipe_radius_error_mm": {str(radius): float(abs(np.linalg.norm(points[:, :2], axis=1)-radius).max())
                for face, radius in ((28, 17.5), (30, 19.)) if face in adjacent[index-1]}}
    record = {"schema_version": 1, "status": "read_only_original_cad_slot_topology", "coordinate_frame": "comsol_reference_mm",
        "source_sha256": geometry["source_sha256"], "source_step_current_sha256": sha(geometry["source"]),
        "inputs_sha256": {str(path): sha(path) for path in (GEOMETRY, brep, Path(__file__))},
        "fixed_slot_region_distance_mm": [190., 201.], "slot_groups_source_face_indices": SLOT_GROUPS,
        "face_index_validation": "Every registered BRep face centroid and area matches the immutable registered map within1e-7",
        "faces": {str(index): {**descriptors[index], "source_edge_indices": face_edges[index]} for index in FACE_IDS},
        "edges": edge_records, "source_unchanged": sha(geometry["source"]) == geometry["source_sha256"]}
    assert record["source_unchanged"]
    return record


def support_error(points, descriptor):
    relative = points-np.array(descriptor["origin_mm"])
    axis = np.array(descriptor["axis"])
    axial = relative@axis
    if descriptor["type"] == "Plane": return abs(axial)
    return abs(np.linalg.norm(relative-axial[..., None]*axis, axis=-1)-descriptor["radius_mm"])


def audit_mesh(folder, cad):
    folder = Path(folder).resolve()
    mesh_path, manifest_path = folder/"structural-mesh.npz", folder/"mesh-manifest.json"
    before = {str(path): sha(path) for path in (mesh_path, manifest_path)}
    mesh = SolidMesh.load(mesh_path)
    manifest = json.loads(manifest_path.read_text())
    conn = mesh.tetra10[[mesh._elements[e] for e in mesh.element_sets["pipe"]]]
    faces = np.stack([conn[:, list(local)] for local in TET10_FACES.values()], axis=1).reshape(-1, 6)
    _, first, counts = np.unique(np.sort(faces[:, :3], axis=1), axis=0, return_index=True, return_counts=True)
    exterior = faces[first[counts == 1]]
    order = np.argsort(mesh.node_ids)
    def xyz(nodes): return mesh.points_mm[order[np.searchsorted(mesh.node_ids[order], nodes)]]
    q = xyz(exterior)
    in_band = (q[:, :, 2].max(axis=1) >= BAND[0]-1e-7) & (q[:, :, 2].min(axis=1) <= BAND[1]+1e-7)
    exterior, q = exterior[in_band], q[in_band]
    radii = np.linalg.norm(q[:, :, :2], axis=2)
    surface_keys, surface_edge_keys = {}, {}
    for name, semantic in (("outer", "pipe_outer_wall"), ("bore", "pipe_bore")):
        named = np.array([mesh.face_nodes(e, f) for e, f in mesh.surfaces[semantic]])
        surface_keys[name] = {tuple(sorted(map(int, nodes[:3]))) for nodes in named}
        surface_edge_keys[name] = {tuple(sorted(map(int, edge))) for edge in named[:, PROFILE_EDGES].reshape(-1, 3)}
    on_outer = np.array([tuple(sorted(map(int, nodes[:3]))) in surface_keys["outer"] for nodes in exterior])
    on_bore = np.array([tuple(sorted(map(int, nodes[:3]))) in surface_keys["bore"] for nodes in exterior])
    slot_nodes, slot_xyz = exterior[~(on_outer | on_bore)], q[~(on_outer | on_bore)]
    errors = np.stack([support_error(slot_xyz, cad["faces"][str(index)]).max(axis=1) for index in FACE_IDS], axis=1)
    # Three coplanar top/bottom slot faces need their physical angular footprint,
    # not an infinite analytic plane, to distinguish original face identities.
    group_names = list(SLOT_GROUPS)
    group_centers = np.array([cad["faces"][str(ids[0])]["centroid_mm"][:2] for ids in SLOT_GROUPS.values()])
    nearest_group = np.linalg.norm(slot_xyz[:, :3, :2].mean(axis=1)[:, None, :]-group_centers[None, :, :], axis=2).argmin(axis=1)
    face_group = np.array([next(i for i, ids in enumerate(SLOT_GROUPS.values()) if face in ids) for face in FACE_IDS])
    errors[nearest_group[:, None] != face_group[None, :]] = np.inf
    best = errors.argmin(axis=1)
    assignments = np.array(FACE_IDS)[best]
    best_error = errors[np.arange(len(errors)), best]
    face_tolerances = {index: max(cad["edges"][str(edge)]["tolerance_mm"]
        for edge in cad["faces"][str(index)]["source_edge_indices"]) for index in FACE_IDS}
    original_tolerance = np.array([face_tolerances[int(index)] for index in assignments])
    # Classification uses the original stored edge tolerances; it never heals
    # an approximate trim or declares these coordinates exact analytic supports.
    if np.any(best_error > original_tolerance+1e-8):
        worst = int(best_error.argmax())
        print(json.dumps({"unresolved_support_error_mm": float(best_error[worst]), "closest_source_face": int(assignments[worst]),
                          "points_mm": slot_xyz[worst].tolist(), "pipe_radii_mm": np.linalg.norm(slot_xyz[worst, :, :2], axis=1).tolist()}), flush=True)
        raise ValueError(f"Unresolved source-face assignment: {best_error.max()} mm")
    per_face = {}
    for index in FACE_IDS:
        selected = assignments == index
        tri, p = slot_nodes[selected], slot_xyz[selected]
        edges = tri[:, PROFILE_EDGES].reshape(-1, 3)
        _, unique = np.unique(np.sort(edges[:, :2], axis=1), axis=0, return_index=True)
        edges = edges[unique]
        ep = xyz(edges)
        er = np.linalg.norm(ep[:, :, :2], axis=2)
        perimeters = {}
        for name, radius in (("outer", 19.), ("bore", 17.5)):
            keep = np.array([tuple(sorted(map(int, edge))) in surface_edge_keys[name] for edge in edges])
            perimeters[name] = {"quadratic_edges": int(keep.sum()), **edge_measures(ep[keep, :2]),
                "maximum_perimeter_node_pipe_radius_error_mm": float(abs(er[keep]-radius).max()) if keep.any() else None}
        per_face[str(index)] = {"source_surface_type": cad["faces"][str(index)]["type"],
            "source_corner_radius_mm": cad["faces"][str(index)].get("radius_mm"),
            "quadratic_triangles": int(selected.sum()), "maximum_six_node_support_error_mm": float(best_error[selected].max()),
            "outer_and_bore_slot_perimeters": perimeters,
            "all_slot_wall_edges_including_radial": {"quadratic_edges": len(edges), **edge_measures(ep[:, :2])}}
    band_metrics = {}
    for name, radius, keep in (("outer", 19., on_outer), ("bore", 17.5, on_bore)):
        tri = exterior[keep]
        edges = tri[:, PROFILE_EDGES].reshape(-1, 3)
        _, unique = np.unique(np.sort(edges[:, :2], axis=1), axis=0, return_index=True)
        edges = edges[unique]; ep = xyz(edges)
        midpoint = ep[:, 2, 2]
        include = (midpoint >= BAND[0]) & (midpoint <= BAND[1])
        band_metrics[name] = {"quadratic_triangles_intersecting_band": int(keep.sum()),
            "edges_with_midside_node_in_band": int(include.sum()), **edge_measures(ep[include, :2])}
    groups = {}
    for group, ids in SLOT_GROUPS.items():
        group_tri = slot_nodes[np.isin(assignments, ids)]
        edges = group_tri[:, PROFILE_EDGES].reshape(-1, 3)
        _, unique = np.unique(np.sort(edges[:, :2], axis=1), axis=0, return_index=True)
        edges = edges[unique]; ep = xyz(edges); er = np.linalg.norm(ep[:, :, :2], axis=2)
        groups[group] = {"quadratic_triangles": len(group_tri), "perimeters": {}}
        for name, radius in (("outer", 19.), ("bore", 17.5)):
            keep = np.array([tuple(sorted(map(int, edge))) in surface_edge_keys[name] for edge in edges])
            groups[group]["perimeters"][name] = {"quadratic_edges": int(keep.sum()), **edge_measures(ep[keep, :2]),
                "maximum_perimeter_node_pipe_radius_error_mm": float(abs(er[keep]-radius).max())}
    record = {"schema_version": 1, "status": "slot_mesh_geometry_diagnostic_only", "mesh_folder": str(folder),
        "inputs_sha256": before, "geometry_revision": mesh.provenance.get("geometry_revision"),
        "settings": manifest["settings"], "nodes": len(mesh.node_ids), "elements": len(mesh.element_ids),
        "wall_layers": manifest["pipe_wall_resolution"]["enforced_radial_layers"],
        "slot_assignment": {"exterior_quadratic_triangles_in_fixed_band": len(exterior), "slot_wall_quadratic_triangles": len(slot_nodes),
            "maximum_six_node_support_error_mm": float(best_error.max()), "source_face_support_tolerance_mm": face_tolerances,
            "rule": "Original angular slot footprint then nearest full-six-node analytic face support; every selected residual must lie within the corresponding original edge tolerance plus1e-8 roundoff",
            "all_24_source_slot_faces_represented": set(assignments) == set(FACE_IDS)},
        "metric_definitions": {"band": "External pipe faces intersecting z[-201,-190]mm; unique edges counted when their midside node lies within band",
            "slot_perimeter": "Unique triangle6 edges shared topologically with named original pipe_outer_wall or pipe_bore faces; radial-through-wall edges excluded. Analytic-radius errors measured independently, never used to filter trim nodes",
            "axial": "absolute endpoint z difference", "circumferential": "short endpoint angular difference times average endpoint pipe radius",
            "in_plane": "hypot(axial,circumferential); never includes radial thickness span",
            "statistics": "unweighted unique-edge distributions, not response convergence or fixed-area-weighted quantiles"},
        "slot_groups": groups, "source_faces": per_face, "pipe_band_outer_and_bore": band_metrics,
        "sources_unchanged": all(sha(Path(path)) == value for path, value in before.items())}
    assert record["sources_unchanged"] and record["slot_assignment"]["all_24_source_slot_faces_represented"]
    return record


def main():
    output = Path(sys.argv[1]).resolve(); output.mkdir(parents=True, exist_ok=False)
    cad = source_slot_topology(); write(output/"original-slot-cad-topology.json", cad)
    for argument in sys.argv[2:]:
        folder = Path(argument).resolve()
        report = audit_mesh(folder, cad)
        report["cad_topology_path"] = str(output/"original-slot-cad-topology.json")
        report["cad_topology_sha256"] = sha(output/"original-slot-cad-topology.json")
        report["recipe_sha256"] = sha(__file__)
        write(output/(folder.name+"-slot-resolution.json"), report)
        print(json.dumps({"case": folder.name, "assignment": report["slot_assignment"], "slot_1_outer": report["slot_groups"]["slot_1"]["perimeters"]["outer"]}), flush=True)
        del report; gc.collect()


if __name__ == "__main__": main()
