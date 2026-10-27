"""Read-only CAD tessellation for setup review; never used as a solver mesh."""
from pathlib import Path
import json
import numpy as np


def cad_surface_meshes(registered_manifest):
    import pyvista as pv
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Shape
    from OCP.TopLoc import TopLoc_Location
    manifest = json.loads(Path(registered_manifest).read_text(encoding="utf-8"))
    meshes = {}
    for body in manifest["bodies"]:
        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, body["path"], BRep_Builder()):
            raise ValueError(f"Could not read registered body {body['name']}")
        BRepMesh_IncrementalMesh(shape, .15, False, .25, True).Perform()
        points, faces = [], []
        iterator = TopExp_Explorer(shape,TopAbs_FACE)
        while iterator.More():
            face = (getattr(TopoDS,"Face_s",None) or TopoDS.Face)(iterator.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face,location)
            if triangulation is not None:
                offset = len(points)
                transform = location.Transformation()
                for i in range(1,triangulation.NbNodes()+1):
                    p = triangulation.Node(i).Transformed(transform)
                    points.append([p.X(),p.Y(),p.Z()])
                for i in range(1,triangulation.NbTriangles()+1):
                    indices = list(triangulation.Triangle(i).Get())
                    if face.Orientation() == TopAbs_REVERSED:
                        indices.reverse()
                    faces.append([3,*[offset+n-1 for n in indices]])
            iterator.Next()
        if not faces:
            raise ValueError(f"Empty visualization for {body['name']}")
        meshes[body["name"]] = pv.PolyData(np.array(points),np.array(faces))
    return meshes
