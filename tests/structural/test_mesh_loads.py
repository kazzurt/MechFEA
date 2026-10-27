import math
import numpy as np
import pytest
from structural.mesh import SolidMesh, TET10_EDGES, TET10_FACES, verify_gmsh_tet10_order
from structural.loads import consistent_surface_force, projected_patch_area, cload_text


def tetra():
    corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    points = np.vstack((corners, [(corners[a]+corners[b])/2 for a,b in TET10_EDGES]))
    return SolidMesh(np.arange(1,11), points, [1], [np.arange(1,11)], {"body":[1]}, {"load":[(1,1)]})


def test_actual_gmsh_to_solver_edge_mapping():
    import gmsh
    gmsh.initialize()
    try:
        verify_gmsh_tet10_order()
    finally:
        gmsh.finalize()


def test_outward_face_normals_and_quadratic_load_moments():
    mesh = tetra()
    mesh.validate()
    center = mesh.points_mm[:4].mean(0)
    for face in TET10_FACES:
        p = mesh.face_points(1,face)[:3]
        assert np.dot(np.cross(p[1]-p[0],p[2]-p[0]),p.mean(0)-center) > 0
    forces, report = consistent_surface_force(mesh,"load",[0,100,0])
    assert report["area_mm2"] == pytest.approx(.5)
    assert report["integrated_resultant_n"] == pytest.approx([0,100,0])
    assert report["integrated_moment_nmm"] == pytest.approx([0,0,100/3])
    # Constant traction on a flat quadratic triangle has zero corner weights.
    assert all(np.linalg.norm(forces[n]) < 1e-12 for n in (1,2,3))
    assert all(forces[n][1] == pytest.approx(100/3) for n in (5,6,7))


def test_curved_face_force_and_moment_and_reopen(tmp_path):
    mesh = tetra()
    mesh.points_mm[4:7,2] -= [.04,.05,.06]
    _, report = consistent_surface_force(mesh,"load",[3,100,-7])
    assert report["area_mm2"] > .5
    mesh.save(tmp_path/"mesh.npz")
    loaded = SolidMesh.load(tmp_path/"mesh.npz")
    _, again = consistent_surface_force(loaded,"load",[3,100,-7])
    assert again == report


def test_projected_width_is_not_surface_area():
    assert projected_patch_area(19,35,25) > 35*25
    assert projected_patch_area(19,35,25) > projected_patch_area(19,30,25)
    with pytest.raises(ValueError):
        projected_patch_area(19,38,25)


def test_zero_service_resultant_retains_patch_audit_without_empty_cload():
    forces,report=consistent_surface_force(tetra(),"load",[0,0,0])
    assert report["area_mm2"]==pytest.approx(.5)
    assert report["integrated_resultant_n"]==[0.,0.,0.]
    assert report["integrated_moment_nmm"]==[0.,0.,0.]
    assert cload_text(forces)==""


def test_quadratic_inversion_rejected_even_when_corners_are_valid():
    mesh=tetra()
    mesh.points_mm[4]=[0.5,2,0]
    with pytest.raises(ValueError,match="quadratic"):
        mesh.validate()
