"""Read-only source CAD audit and explicitly mapped, deduplicated derived CAD.

OpenCascade is imported only by kernel routines so project/configuration tools do
not require the native CAD runtime. Geometry is normalized to mm on STEP import.
No healing, tolerance inflation, clearance removal, or contact fusion is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "ca9207ebd900aec8013ddad45af0ac59b767afcb5f8ef2a2121909c665622d6f"
PIPE_AXIS_TOWARD_HUB = (-0.951056516295154, 0.309016994374947, 0.0)
EXPECTED_HALF_LENGTH_MM = 233.68


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def norm(v):
    return math.sqrt(dot(v, v))


def axis_distance(origin_a, direction_a, origin_b):
    """Distance from a point on an assumed parallel axis to the first axis."""
    delta = subtract(origin_b, origin_a)
    projection = dot(delta, direction_a)
    return norm([x - projection * y for x, y in zip(delta, direction_a)])


def point_values(p):
    return [float(p.X()), float(p.Y()), float(p.Z())]


def shape_parts(shape, shape_type):
    from OCP.TopExp import TopExp_Explorer
    explorer = TopExp_Explorer(shape, shape_type)
    while explorer.More():
        yield explorer.Current()
        explorer.Next()


def shape_properties(shape, *, surface=False):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    if surface:
        BRepGProp.SurfaceProperties_s(shape, p)
    else:
        BRepGProp.VolumeProperties_s(shape, p)
    inertia = p.MatrixOfInertia()
    return {
        "measure": float(p.Mass()),
        "centroid_mm": point_values(p.CentreOfMass()),
        "inertia": [[float(inertia.Value(i, j)) for j in range(1, 4)] for i in range(1, 4)],
    }


def bounds(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    return {"minimum_mm": point_values(box.CornerMin()), "maximum_mm": point_values(box.CornerMax())}


def geometric_fingerprint(data):
    """Stable numeric signature; selections must still be revalidated on remesh."""
    def rounded(value):
        if isinstance(value, float):
            return round(value, 7)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items() if k not in {"index", "fingerprint"}}
        if isinstance(value, (list, tuple)):
            return [rounded(v) for v in value]
        return value
    raw = json.dumps(rounded(data), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def describe_face(face, index):
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopoDS import TopoDS
    surface = BRepAdaptor_Surface(face, True)
    kind = surface.GetType().name.removeprefix("GeomAbs_")
    properties = shape_properties(face, surface=True)
    result = {"index": index, "type": kind, "area_mm2": properties["measure"],
              "centroid_mm": properties["centroid_mm"], "bounds": bounds(face),
              "orientation": face.Orientation().name}
    primitive = None
    if kind == "Plane":
        primitive = surface.Plane()
    elif kind == "Cylinder":
        primitive = surface.Cylinder()
        result["radius_mm"] = primitive.Radius()
    elif kind == "Cone":
        primitive = surface.Cone()
        result["reference_radius_mm"] = primitive.RefRadius()
        result["semi_angle_radians"] = primitive.SemiAngle()
        result["apex_mm"] = point_values(primitive.Apex())
    elif kind == "Sphere":
        primitive = surface.Sphere()
        result["radius_mm"] = primitive.Radius()
    elif kind == "Torus":
        primitive = surface.Torus()
        result["major_radius_mm"] = primitive.MajorRadius()
        result["minor_radius_mm"] = primitive.MinorRadius()
    if primitive is not None:
        result["origin_mm"] = point_values(primitive.Location())
        direction = primitive.Axis().Direction() if hasattr(primitive, "Axis") else primitive.Position().Direction()
        result["axis"] = point_values(direction)
        vertices = [point_values(BRep_Tool.Pnt_s(TopoDS.Vertex(v)))
                    for v in shape_parts(face, TopAbs_VERTEX)]
        if vertices:
            axial = [dot(subtract(v, result["origin_mm"]), result["axis"]) for v in vertices]
            result["vertex_axial_extent_mm"] = [min(axial), max(axial)]
    result["fingerprint"] = geometric_fingerprint(result)
    return result


def read_step_bodies(path):
    """Return native shapes keyed by STEP source entity names, in mm."""
    import re
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.collections import Sequence_TDF_Label
    from OCP.Interface import Interface_Static
    from OCP.TopAbs import TopAbs_SOLID
    source = Path(path).resolve()
    source_text = source.read_text(errors="replace")
    step_ids = {name: int(entity) for entity, name in re.findall(
        r"#(\d+)\s*=\s*MANIFOLD_SOLID_BREP\s*\(\s*'([^']*)'", source_text)}
    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError(f"STEP import failed: {source}")
    reader.Reader().SetSystemLengthUnit(1.0)
    document = TDocStd_Document(TCollection_ExtendedString("structural-audit"))
    if not reader.Transfer(document):
        raise ValueError("STEP XDE transfer failed")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    labels = Sequence_TDF_Label()
    shape_tool.GetFreeShapes(labels)
    bodies = []
    for i in range(1, labels.Length() + 1):
        label = labels.Value(i)
        attribute = TDataStd_Name()
        if not label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
            raise ValueError("Imported STEP body has no source label")
        name = attribute.Get().ToExtString()
        shape = shape_tool.GetShape_s(label)
        if shape.IsNull():
            raise ValueError(f"No imported shape for STEP solid {name}")
        solids = list(shape_parts(shape, TopAbs_SOLID))
        if len(solids) != 1:
            raise ValueError(f"Expected one solid for {name}, found {len(solids)}")
        bodies.append({"name": name, "source_step_entity": step_ids.get(name), "shape": solids[0]})
    if not bodies:
        raise ValueError("No named MANIFOLD_SOLID_BREP entities transferred")
    return bodies


def describe_body(body):
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    shape = body["shape"]
    properties = shape_properties(shape)
    result = {"name": body["name"], "source_step_entity": body["source_step_entity"],
              "valid": bool(BRepCheck_Analyzer(shape).IsValid()),
              "volume_mm3": properties["measure"], "centroid_mm": properties["centroid_mm"],
              "volume_inertia_mm5": properties["inertia"], "bounds": bounds(shape),
              "faces": [describe_face(TopoDS.Face(f), i)
                        for i, f in enumerate(shape_parts(shape, TopAbs_FACE), 1)]}
    result["fingerprint"] = geometric_fingerprint(result)
    return result


def source_tolerance_audit(source):
    """Expose import topology tolerances; they are not contact penetration."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX, TopAbs_FACE
    from OCP.TopoDS import TopoDS
    bodies = read_step_bodies(source)
    selected_faces = {"New Pipe": [1, 3, 4, 28], "Hub": [2, 225, 226, 227],
                      "Bolt 1": [4, 6], "Bolt 2": [4, 6], "Bolt 3": [4, 6]}
    records = []
    for body in bodies:
        maxima = {label: max(BRep_Tool.Tolerance_s(cast(item)) for item in shape_parts(body["shape"], kind))
                  for label, kind, cast in [("edge", TopAbs_EDGE, TopoDS.Edge),
                                             ("vertex", TopAbs_VERTEX, TopoDS.Vertex), ("face", TopAbs_FACE, TopoDS.Face)]}
        relevant = []
        for i, face in enumerate(shape_parts(body["shape"], TopAbs_FACE), 1):
            if i in selected_faces.get(body["name"], []):
                edges = [BRep_Tool.Tolerance_s(TopoDS.Edge(e)) for e in shape_parts(face, TopAbs_EDGE)]
                relevant.append({"face_index_at_audit": i, "maximum_edge_tolerance_mm": max(edges)})
        records.append({"body": body["name"], "maximum_tolerance_mm": maxima, "contact_anchor_faces": relevant})
    return {"source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "units": "mm", "method": "BRep_Tool imported tolerance metadata; no healing or tolerance modification",
            "bodies": records,
            "interpretation": "Surface supporting equations can be exact while trimmed edge curves carry larger approximation tolerances. These values are not solver penetration and must not be subtracted from measured gap."}


def pair_audit(body_a, body_b, tolerance_mm3=1e-5):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    a, b = body_a["shape"], body_b["shape"]
    operation = BRepAlgoAPI_Common(a, b)
    operation.Build()
    if not operation.IsDone():
        raise ValueError(f"Common failed for {body_a['name']} / {body_b['name']}")
    intersection = max(0.0, shape_properties(operation.Shape())["measure"])
    va, vb = shape_properties(a)["measure"], shape_properties(b)["measure"]
    distance = BRepExtrema_DistShapeShape(a, b)
    distance.Perform()
    if not distance.IsDone():
        raise ValueError("CAD distance operation failed")
    fraction_a, fraction_b = intersection / va, intersection / vb
    duplicate = abs(va - vb) <= max(tolerance_mm3, va * 1e-8) and min(fraction_a, fraction_b) >= 1 - 1e-8
    return {"bodies": [body_a["name"], body_b["name"]], "common_volume_mm3": intersection,
            "common_fraction": [fraction_a, fraction_b], "distance_mm": distance.Value(),
            "classification": "coincident_duplicate" if duplicate else (
                "volumetric_interference" if intersection > tolerance_mm3 else (
                    "touching" if distance.Value() < 1e-6 else "separated"))}


def make_named_selections(descriptions):
    """Discover analytic interfaces without silently subdividing whole cylinders."""
    by_name = {b["name"]: b for b in descriptions}
    pipe, hub = by_name["New Pipe"], by_name["Hub"]
    selection = {}
    def face_ref(body, face, **extra):
        return {"body": body, "face_index_at_audit": face["index"],
                "fingerprint": face["fingerprint"], **extra}
    bore = [f for f in pipe["faces"] if f["type"] == "Cylinder" and abs(f["radius_mm"] - 17.5) < 1e-6]
    outer = [f for f in pipe["faces"] if f["type"] == "Cylinder" and abs(f["radius_mm"] - 19) < 1e-6]
    ends = [f for f in pipe["faces"] if f["type"] == "Plane" and abs(dot(f["axis"], PIPE_AXIS_TOWARD_HUB)) > 1 - 1e-8]
    if len(bore) != 1 or len(outer) != 1 or len(ends) < 2:
        raise ValueError("Pipe analytic selections are ambiguous")
    ends.sort(key=lambda f: dot(f["centroid_mm"], PIPE_AXIS_TOWARD_HUB))
    midpoint, far_end = ends[0], ends[-1]
    expected_end_area = math.pi * (outer[0]["radius_mm"] ** 2 - bore[0]["radius_mm"] ** 2)
    if any(abs(f["area_mm2"] - expected_end_area) > 1e-4 for f in (midpoint, far_end)):
        raise ValueError("Pipe extreme end faces are not the expected annuli")
    half_length = dot(subtract(far_end["centroid_mm"], midpoint["centroid_mm"]), PIPE_AXIS_TOWARD_HUB)
    if abs(half_length - EXPECTED_HALF_LENGTH_MM) > 1e-5:
        raise ValueError(f"Pipe half-length {half_length} mm violates the unit/dimension contract")
    selection["midpoint_symmetry"] = face_ref("New Pipe", midpoint, normal=list(PIPE_AXIS_TOWARD_HUB))
    selection["pipe_bore"] = face_ref("New Pipe", bore[0])
    selection["pipe_outer_wall"] = face_ref("New Pipe", outer[0])
    selection["pipe_hub_side_end"] = face_ref("New Pipe", far_end)
    pipe_axis_origin = bore[0]["origin_mm"]
    snout_faces = [f for f in hub["faces"] if f["type"] == "Cylinder"
                   and abs(dot(f["axis"], PIPE_AXIS_TOWARD_HUB)) > 1 - 1e-8
                   and axis_distance(pipe_axis_origin, PIPE_AXIS_TOWARD_HUB, f["origin_mm"]) < 1e-5]
    selection["hub_snout_candidates"] = [face_ref("Hub", f, radius_mm=f["radius_mm"]) for f in snout_faces]
    cuts = []
    for name in ("Bolt 1", "Bolt 2", "Bolt 3"):
        body = by_name[name]
        shafts = [f for f in body["faces"] if f["type"] == "Cylinder" and abs(f["radius_mm"] - 4) < 1e-6]
        cones = [f for f in body["faces"] if f["type"] == "Cone"]
        if len(shafts) != 1 or len(cones) != 1:
            raise ValueError(f"Unexpected screw primitives for {name}")
        shaft, cone = shafts[0], cones[0]
        matching_holes = [f for f in hub["faces"] if f["type"] == "Cylinder"
                          and abs(f["radius_mm"] - shaft["radius_mm"]) < 1e-6
                          and abs(dot(f["axis"], shaft["axis"])) > 1 - 1e-8
                          and axis_distance(shaft["origin_mm"], shaft["axis"], f["origin_mm"]) < 1e-5]
        matching_seats = [f for f in pipe["faces"] if f["type"] == "Cone"
                          and abs(dot(f["axis"], cone["axis"])) > 1 - 1e-8
                          and axis_distance(cone["origin_mm"], cone["axis"], f["origin_mm"]) < 1e-5]
        key = name.lower().replace(" ", "_")
        selection[key + "_shaft"] = face_ref(name, shaft, axis=shaft["axis"], origin_mm=shaft["origin_mm"])
        selection[key + "_anchor_holes"] = [face_ref("Hub", f) for f in matching_holes]
        selection[key + "_cone"] = face_ref(name, cone)
        selection[key + "_pipe_seat"] = [face_ref("New Pipe", f) for f in matching_seats]
        shaft_range = shaft["vertex_axial_extent_mm"]
        hole_ranges = []
        for hole in matching_holes:
            origin_offset = dot(subtract(hole["origin_mm"], shaft["origin_mm"]), shaft["axis"])
            sense = dot(hole["axis"], shaft["axis"])
            h = sorted(origin_offset + sense * t for t in hole["vertex_axial_extent_mm"])
            hole_ranges.append(h)
        cuts.append({"body": name, "axis": shaft["axis"], "origin_mm": shaft["origin_mm"],
                     "shaft_axial_range_mm": shaft_range, "hub_hole_axial_ranges_mm": hole_ranges,
                     "status": "requires_partition_and_pretension_compatibility_check",
                     "warning": "The whole shaft face is not an anchorage selection; tie only its hub-engaged interval."})
    origin_projection = dot(subtract(midpoint["centroid_mm"], pipe_axis_origin), PIPE_AXIS_TOWARD_HUB)
    midpoint_center = [x + origin_projection * y for x, y in zip(pipe_axis_origin, PIPE_AXIS_TOWARD_HUB)]
    return {"named_selections": selection,
            "dimensions": {"pipe_outer_diameter_mm": 2 * outer[0]["radius_mm"],
                           "pipe_inner_diameter_mm": 2 * bore[0]["radius_mm"],
                           "pipe_wall_mm": outer[0]["radius_mm"] - bore[0]["radius_mm"],
                           "half_length_mm": half_length, "full_length_mm": 2 * half_length},
            "source_frame": {"midpoint_center_mm": midpoint_center,
                             "axis_midpoint_toward_hub": list(PIPE_AXIS_TOWARD_HUB)},
            "pretension_cut_candidates": cuts,
            "load_patch": {"status": "unpartitioned_outer_cylinder", "nominal_dimensions_mm": [35, 25],
                           "width_definition": "projected_straight_line_user_confirmed",
                           "requires": ["registered load clocking"]}}


def audit_step(source, output_dir, *, expected_sha256=SOURCE_SHA256):
    import itertools
    import OCP
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.TopoDS import TopoDS_Compound
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Source hash differs from approved CAD; declare an explicit geometry revision")
    output_dir.mkdir(parents=True, exist_ok=True)
    bodies = read_step_bodies(source)
    descriptions = [describe_body(b) for b in bodies]
    pairs = []
    for a, b in itertools.combinations(bodies, 2):
        pair = pair_audit(a, b)
        pairs.append(pair)
        print(json.dumps(pair), flush=True)
    excluded = {}
    # Prefer semantic Bolt names, only following a full-volume kernel intersection.
    for pair in pairs:
        if pair["classification"] != "coincident_duplicate":
            continue
        a, b = pair["bodies"]
        keep, remove = (a, b) if a.startswith("Bolt") else (b, a)
        if not keep.startswith("Bolt") or not remove.startswith("Part"):
            raise ValueError(f"Unexpected duplicate pair requires review: {a}, {b}")
        excluded[remove] = keep
    retained = [b for b in bodies if b["name"] not in excluded]
    if {b["name"] for b in retained} != {"New Pipe", "Hub", "Bolt 1", "Bolt 2", "Bolt 3"}:
        raise ValueError("Derived body set differs from one pipe, hub and three screws")
    compound, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(compound)
    derived_map = []
    for body in retained:
        filename = body["name"].lower().replace(" ", "-") + ".brep"
        BRepTools.Write_s(body["shape"], str(output_dir / filename))
        builder.Add(compound, body["shape"])
        derived_map.append({"source_name": body["name"], "source_step_entity": body["source_step_entity"],
                            "derived_body_path": filename,
                            "aliases_excluded_as_duplicates": [k for k, v in excluded.items() if v == body["name"]]})
    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    writer.Write(str(output_dir / "deduplicated-source-frame-mm.step"))
    report = {"schema_version": 1, "source": str(source), "source_sha256": digest,
              "source_unit_declaration": "metre", "kernel_import_length_unit": "mm",
              "kernel": {"package": "cadquery-ocp", "version": OCP.__version__},
              "method": "Exact BREP kernel validity, analytic surface, volume intersection and minimum-distance audit",
              "bodies": descriptions, "body_pairs": pairs, "derived_body_map": derived_map,
              "excluded_duplicates": excluded, **make_named_selections(descriptions),
              "source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == digest,
              "geometry_operations": ["unit normalization to mm", "full-overlap duplicate exclusion"],
              "not_performed": ["geometry healing", "contact fusion", "clearance removal", "load patch partition", "pretension cut partition"]}
    report["unresolved_interferences"] = [p for p in pairs if p["classification"] == "volumetric_interference"
                                           and not any(n in excluded for n in p["bodies"])]
    (output_dir / "cad-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_step(args.source, args.output)
    print(json.dumps({"dimensions": report["dimensions"], "source_frame": report["source_frame"],
                      "excluded_duplicates": report["excluded_duplicates"],
                      "unresolved_interferences": report["unresolved_interferences"]}, indent=2))


if __name__ == "__main__":
    main()
