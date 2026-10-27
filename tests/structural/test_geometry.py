"""Geometry identity, unit policy and duplicate-detection regressions."""
import hashlib
import importlib.util
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

# The two native distributions carry different OCCT DLL generations. Loading
# Gmsh first is the verified Windows order; production meshing is subprocess
# isolated from GUI CAD tessellation for the same reason.
if importlib.util.find_spec("gmsh"):
    import gmsh

from structural.geometry import SOURCE_SHA256, axis_distance, geometric_fingerprint
from structural.preparation import source_to_reference_transform


class FrameTests(unittest.TestCase):
    def test_reference_transform_is_proper_and_aligns_pipe(self):
        midpoint = np.array([-107.9717113677623, -336.18604013019956, 0])
        rotation, translation = source_to_reference_transform(midpoint)
        self.assertTrue(np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-14))
        self.assertAlmostEqual(np.linalg.det(rotation), 1, places=14)
        self.assertTrue(np.allclose(rotation @ midpoint + translation, 0, atol=1e-12))
        self.assertTrue(np.allclose(rotation @ [-.951056516295154, .309016994374947, 0], [0, 0, -1], atol=1e-14))
        self.assertTrue(np.allclose(rotation.T @ [0, 1, 0], [0, 0, 1]))

    def test_axis_distance_ignores_axial_station(self):
        self.assertAlmostEqual(axis_distance([1, 2, 3], [0, 0, 1], [4, 6, 100]), 5)

    def test_fingerprint_ignores_transient_face_index(self):
        a = {"index": 2, "radius": 4.0, "centroid": [1.0, 2.0, 3.0]}
        self.assertEqual(geometric_fingerprint(a), geometric_fingerprint({**a, "index": 91}))
        self.assertNotEqual(geometric_fingerprint(a), geometric_fingerprint({**a, "radius": 4.01}))


@unittest.skipUnless(importlib.util.find_spec("OCP"), "Isolated CAD runtime required")
class KernelTests(unittest.TestCase):
    def test_duplicate_requires_full_common_volume(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.gp import gp_Pnt
        from structural.geometry import pair_audit
        a = {"name": "a", "shape": BRepPrimAPI_MakeBox(2, 2, 2).Shape()}
        identical = {"name": "b", "shape": BRepPrimAPI_MakeBox(2, 2, 2).Shape()}
        overlap = {"name": "c", "shape": BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), 2, 2, 2).Shape()}
        touch = {"name": "d", "shape": BRepPrimAPI_MakeBox(gp_Pnt(2, 0, 0), 2, 2, 2).Shape()}
        self.assertEqual(pair_audit(a, identical)["classification"], "coincident_duplicate")
        self.assertEqual(pair_audit(a, overlap)["classification"], "volumetric_interference")
        self.assertEqual(pair_audit(a, touch)["classification"], "touching")

    @unittest.skipUnless((Path(__file__).resolve().parents[2] / "Dodec FEA" / "Dodec Hub Conformal and 1.5mm pipe half.step").is_file(), "External source CAD fixture unavailable")
    def test_exact_source_import_units_and_named_interfaces(self):
        from structural.geometry import read_step_bodies, describe_body, make_named_selections
        source = Path(__file__).resolve().parents[2] / "Dodec FEA" / "Dodec Hub Conformal and 1.5mm pipe half.step"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), SOURCE_SHA256)
        bodies = [describe_body(b) for b in read_step_bodies(source)]
        self.assertEqual(len(bodies), 7)
        self.assertTrue(all(b["valid"] for b in bodies))
        selections = make_named_selections(bodies)
        self.assertAlmostEqual(selections["dimensions"]["half_length_mm"], 233.68, places=7)
        self.assertAlmostEqual(selections["dimensions"]["pipe_wall_mm"], 1.5, places=7)
        for i in range(1, 4):
            self.assertEqual(len(selections["named_selections"][f"bolt_{i}_anchor_holes"]), 1)
            self.assertEqual(len(selections["named_selections"][f"bolt_{i}_pipe_seat"]), 1)
            # Identical axial vertex endpoints do not imply identical trimmed
            # engagement: the snout's curved mouth leaves exposed crescents.
            screw = next(b for b in bodies if b["name"] == f"Bolt {i}")
            hub = next(b for b in bodies if b["name"] == "Hub")
            shaft_ref = selections["named_selections"][f"bolt_{i}_shaft"]
            hole_ref = selections["named_selections"][f"bolt_{i}_anchor_holes"][0]
            shaft_area = next(f["area_mm2"] for f in screw["faces"] if f["index"] == shaft_ref["face_index_at_audit"])
            hole_area = next(f["area_mm2"] for f in hub["faces"] if f["index"] == hole_ref["face_index_at_audit"])
            self.assertGreater(shaft_area-hole_area, 5.9)
            self.assertLess(shaft_area-hole_area, 6.1)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), SOURCE_SHA256)


@unittest.skipUnless(importlib.util.find_spec("gmsh"), "Isolated mesh runtime required")
class RigidCarrierTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[2] / "Dodec FEA" / "Dodec Hub Conformal and 1.5mm pipe half.step").is_file(), "External source CAD fixture unavailable")
    def test_source_three_layer_partition_preserves_valid_material(self):
        from OCP.BRepTools import BRepTools
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.gp import gp_Trsf
        from structural.geometry import read_step_bodies
        root = Path(__file__).resolve().parents[2]
        pipe = next(b for b in read_step_bodies(root / "Dodec FEA" / "Dodec Hub Conformal and 1.5mm pipe half.step")
                    if b["name"] == "New Pipe")
        rotation, translation = source_to_reference_transform([-107.9717113677623, -336.18604013019956, -6.159219072802159e-13])
        transform = gp_Trsf()
        transform.SetValues(*(float(v) for row, offset in zip(rotation, translation) for v in [*row, offset]))
        shape = BRepBuilderAPI_Transform(pipe["shape"], transform, True).Shape()
        # Match production's isolation: source CAD work and Gmsh booleans do
        # not share a process that has already run unrelated OCP kernels.
        script = """
from pathlib import Path
import sys
import gmsh
from structural.preparation import _partition_radial_layers, _adaptive_partition_volume_audit
path, folder = Path(sys.argv[1]), Path(sys.argv[2])
gmsh.initialize()
try:
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setNumber('General.NumThreads', 2)
    gmsh.model.add('three-layer-source-material-regression')
    objects = gmsh.model.occ.importShapes(str(path))
    box = gmsh.model.occ.addBox(-17.5, -40, -25, 35, 80, 26)
    fragments, mapping = gmsh.model.occ.fragment(objects, [(3, box)])
    physical_pipe = mapping[0]
    gmsh.model.occ.remove([v for v in fragments if v not in physical_pipe], recursive=True)
    volumes = _partition_radial_layers(gmsh, physical_pipe, [18.0, 18.5])
    gmsh.model.occ.synchronize()
    audit = _adaptive_partition_volume_audit(gmsh, {'New Pipe': volumes},
                                            {'bodies': [{'name': 'New Pipe', 'path': str(path)}]}, folder)
    assert len(volumes) == 9
    assert audit['New Pipe'].get('boolean_removed_volume_mm3', 0) == 0
    assert audit['New Pipe'].get('boolean_added_volume_mm3', 0) == 0
finally:
    gmsh.finalize()
"""
        with tempfile.TemporaryDirectory(prefix="radial-partition-regression-", dir=root / "Dodec FEA") as temporary:
            folder = Path(temporary)
            path = folder / "registered-source-pipe.brep"
            BRepTools.Write_s(shape, str(path))
            result = subprocess.run([sys.executable, "-c", script, str(path), str(folder)],
                                    cwd=root, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_curved_interface_is_preserved_and_outward(self):
        import gmsh
        from structural.preparation import _add_fixed_rigid_surface_carriers, _tetra_quadratic_ccx_permutation
        from structural.mesh import SolidMesh
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("rigid-carrier-regression")
            surface = gmsh.model.addDiscreteEntity(2)
            parameters = [(-.1, 0), (.1, 0), (0, 1), (0, 0), (.05, .5), (-.05, .5)]
            points = np.array([[4*np.cos(a), 4*np.sin(a), z] for a, z in parameters])
            gmsh.model.mesh.addNodes(2, surface, list(range(1, 7)), points.ravel())
            gmsh.model.mesh.addElementsByType(surface, 9, [1], list(range(1, 7)))
            volume, count = _add_fixed_rigid_surface_carriers(gmsh, {surface: {
                "axis": [0, 0, 1], "origin_mm": [0, 0, 0], "orientation": "TopAbs_FORWARD"}})
            nodes, xyz, _ = gmsh.model.mesh.getNodes()
            _, ids, connectivity = gmsh.model.mesh.getElements(3, volume)
            mesh = SolidMesh(nodes, np.asarray(xyz).reshape(-1, 3), ids[0],
                             np.asarray(connectivity[0]).reshape(-1, 10)[:, _tetra_quadratic_ccx_permutation(gmsh)])
            mesh.validate()
            self.assertEqual(count, 1)
            face = mesh.face_points(int(ids[0][0]), 1)
            self.assertEqual(set(mesh.face_nodes(int(ids[0][0]), 1)), set(range(1, 7)))
            self.assertTrue(np.allclose(mesh.points_mm[[mesh._nodes[i] for i in range(1, 7)]], points))
            normal = np.cross(face[1]-face[0], face[2]-face[0])
            self.assertGreater(normal[0], 0)
        finally:
            gmsh.finalize()


if __name__ == "__main__":
    unittest.main()
