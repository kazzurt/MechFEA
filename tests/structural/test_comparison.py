import numpy as np
import pytest

from structural.comparison import (BodyInterpolator, FieldMesh, RigidTransform,
    SECTION_SCREEN_EXTENSION_V1, cached_reference_region_measures, compare_probe_fields, compare_run, compare_scalar, compare_vector,
    section_vector_screens, fit_rigid_transform, integrate_section,
    read_comsol_vtu, symmetric_tensor, von_mises)


def tetra_mesh(points=None, *, translation=0.0, stress=None):
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) if points is None else np.array(points, dtype=float)
    u = p @ np.array([[1, 2, 0], [0, 1, 0], [1, 0, 3]], dtype=float).T + translation
    s = np.tile(np.diag([2., 3., 6.]), (len(p), 1, 1)) if stress is None else stress
    return FieldMesh(p, np.array([[0, 1, 2, 3]]), np.array([2]), u, s)


def test_reference_volume_cache_is_source_recipe_and_metric_bound(tmp_path):
    import json
    points=[[0,0,0],[1,0,0],[0,1,0],[0,0,-1]]
    mesh=tetra_mesh(points)
    source={"sha256":"0123456789abcdef"*4}
    first=cached_reference_region_measures(source,mesh,tmp_path)
    second=cached_reference_region_measures(source,mesh,tmp_path)
    assert first==second and first["regions"]["center"]["volume_mm3"]==pytest.approx(1/6)
    cache=next(tmp_path.glob("*-finite-volume-*.json"))
    data=json.loads(cache.read_text());data["regions"]["center"]["volume_mm3"]=999
    cache.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="provenance"):cached_reference_region_measures(source,mesh,tmp_path)


def test_registration_transforms_stress_and_displacement_objectively():
    r = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    transform = RigidTransform(r, np.array([7., -4, 2]))
    source = np.array([[0., 0, 0], [2, 0, 0], [0, 3, 0], [1, 2, 4]])
    fitted, residual = fit_rigid_transform(source, transform.points(source))
    np.testing.assert_allclose(fitted.rotation, r, atol=1e-14)
    assert residual["max_residual_mm"] < 1e-13
    stress = symmetric_tensor(np.array([100., 20, 30, 4, 5, 6]))
    np.testing.assert_allclose(von_mises(transform.tensors(stress)), von_mises(stress))
    normal = np.array([.3, .4, np.sqrt(.75)])
    np.testing.assert_allclose(transform.tensors(stress) @ transform.vectors(normal), transform.vectors(stress @ normal))
    np.testing.assert_allclose(transform.inverse().points(transform.points(source)), source, atol=1e-14)


def test_registration_rejects_scaling_reflection_and_axis_only_landmarks():
    for r in (np.eye(3) * 1.001, np.diag([1., 1, -1])):
        with pytest.raises(ValueError):
            RigidTransform(r, np.zeros(3))
    with pytest.raises(ValueError, match="Collinear"):
        fit_rigid_transform([[0, 0, 0], [0, 0, 1], [0, 0, 2]], [[0, 0, 0], [0, 0, 1], [0, 0, 2]])


def test_linear_patch_interpolation_and_no_extrapolation():
    mesh = tetra_mesh()
    points = np.array([[.1, .2, .3], [.2, .2, .2], [1., 1, 1]])
    result = BodyInterpolator(mesh, [2]).sample(points)
    np.testing.assert_array_equal(result["valid"], [True, True, False])
    np.testing.assert_allclose(result["displacement_mm"][:2], [[.5, .2, 1.], [.6, .2, .8]])
    assert np.isnan(result["stress_mpa"][2]).all()
    with pytest.raises(ValueError, match="No tetrahedra"):
        BodyInterpolator(mesh, [1])


def test_interpolation_keeps_coincident_body_and_cut_traces_separate():
    a, b = tetra_mesh(), tetra_mesh(translation=10)
    mesh = FieldMesh(np.vstack([a.points_mm, b.points_mm]), [[0, 1, 2, 3], [4, 5, 6, 7]], [3, 4],
                     np.vstack([a.displacement_mm, b.displacement_mm]), np.vstack([a.stress_mpa, b.stress_mpa]))
    point = [[.1, .2, .3]]
    assert BodyInterpolator(mesh, [3]).sample(point)["valid"][0]
    result = BodyInterpolator(mesh, [3, 4]).sample(point)
    assert result["ambiguous_trace"][0] and not result["valid"][0]


def test_section_constant_traction_area_force_moment_and_linear_displacement():
    # Plane z=.25 clips the unit tetrahedron to a right triangle of side .75.
    result = integrate_section(tetra_mesh(), [2], [0, 0, .25], [0, 0, 1], moment_origin_mm=[0, 0, 0])
    area = .75 ** 2 / 2
    assert result["area_mm2"] == pytest.approx(area)
    np.testing.assert_allclose(result["force_N"], [0, 0, 6 * area])
    np.testing.assert_allclose(result["moment_Nmm"], [6 * area * .25, -6 * area * .25, 0])
    np.testing.assert_allclose(result["mean_displacement_mm"], [.75, .25, 1.])


def test_section_linear_stress_moment_uses_quadratic_exact_integration():
    mesh = tetra_mesh()
    mesh.stress_mpa[:, 2, 2] = mesh.points_mm[:, 1]
    result = integrate_section(mesh, [2], [0, 0, 0], [0, 0, 1])
    # Unit right triangle integral(y)=1/6, integral(y^2)=1/12, integral(xy)=1/24.
    np.testing.assert_allclose(result["force_N"], [0, 0, 1/6])
    np.testing.assert_allclose(result["moment_Nmm"], [1/12, -1/24, 0])


def test_zero_reference_uses_predeclared_absolute_tolerance():
    assert compare_scalar(1e-6, 0., relative_tolerance=.05, absolute_tolerance=1e-5)["passed"]
    assert not compare_scalar(.001, 0., relative_tolerance=.05, absolute_tolerance=1e-5)["passed"]
    assert not compare_scalar(float("nan"), 1., relative_tolerance=.05, absolute_tolerance=1e-5)["passed"]


def test_vector_screen_detects_non_y_changes_and_is_rotation_invariant():
    reference=np.array([0.,100.,0.]);candidate=np.array([6.,100.,0.])
    result=compare_vector(candidate,reference,relative_tolerance=.05,absolute_tolerance=1.)
    assert not result["passed"] and result["absolute_difference"]==6. and result["allowed_difference"]==5.
    rotation=np.array([[0.,-1,0],[0,0,1],[-1,0,0]])
    rotated=compare_vector(rotation@candidate,rotation@reference,relative_tolerance=.05,absolute_tolerance=1.)
    assert rotated["passed"]==result["passed"] and rotated["absolute_difference"]==result["absolute_difference"]
    assert compare_vector([0,0,1e-5],[0,0,0],relative_tolerance=.05,absolute_tolerance=1e-5)["passed"]
    assert compare_vector([0,0,float("nan")],[0,0,0],relative_tolerance=.05,absolute_tolerance=1e-5)["passed"] is None


def section_fixture(distance):
    return {"distance_mm":distance,"area_mm2":1.,"mean_displacement_mm":[0.,.5,0.],
            "force_N":[0.,1000.,0.],"moment_Nmm":[0.,10000.,0.],
            "origin_mm":[0.,0.,-distance],"moment_origin_mm":[0.,0.,-distance],"normal":[0.,0.,1.]}


def frozen_tolerances():
    return {"displacement_relative_tolerance":.05,"displacement_absolute_tolerance_mm":1e-5,
            "section_force_relative_tolerance":.05,"section_force_absolute_tolerance_N":1.,
            "finite_region_stress_relative_tolerance":.1,"stress_absolute_tolerance_MPa":.05}


@pytest.mark.parametrize("distance",[.1,25.,35.,75.,125.,175.,210.])
def test_free_span_scope_and_vector_moment_floor(distance):
    import copy
    reference=section_fixture(distance);candidate=copy.deepcopy(reference)
    candidate["mean_displacement_mm"][2]=.03
    candidate["force_N"][0]=60.
    candidate["moment_Nmm"][2]=600.
    rows=section_vector_screens(candidate,reference,frozen_tolerances(),eligible=True)
    assert all(row["observed_within_tolerance"] is False for row in rows)
    free=distance in (35.,75.,125.,175.)
    assert all(row["contributes_to_screen"] is free for row in rows)
    assert all(row["status"]==("eligible" if free else "diagnostic_only") for row in rows)
    assert rows[2]["absolute_tolerance"]==233.68 and rows[2]["relative_tolerance"]==.05
    candidate["moment_Nmm"]=[233.,0,0];reference["moment_Nmm"]=[0,0,0]
    assert section_vector_screens(candidate,reference,frozen_tolerances())[2]["observed_within_tolerance"] is True
    candidate["moment_origin_mm"]=[0,0,0]
    assert section_vector_screens(candidate,reference,frozen_tolerances())[2]["status"]=="missing"


def test_versioned_section_contract_matches_separate_additive_definition():
    import json
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/"structural/contracts/comsol-section-screen-extension-v1.json"
    assert json.loads(path.read_text(encoding="utf-8"))==SECTION_SCREEN_EXTENSION_V1
    assert "after the original assembly solve" in SECTION_SCREEN_EXTENSION_V1["timing"]


@pytest.mark.parametrize("changed_dat",[False,True])
@pytest.mark.parametrize("omit_free_span",[False,True])
@pytest.mark.parametrize("changed_vtu",[False,True])
def test_full_comparison_excludes_loaded_sections_and_rejects_stale_native_sources(tmp_path,monkeypatch,changed_dat,omit_free_span,changed_vtu):
    import copy
    import json
    from types import SimpleNamespace
    from structural import comparison,results,studies
    from structural.project import sha256
    deck=tmp_path/"assembly.inp";deck.write_text("retained deck")
    extensions=(".inp",".assembly.json",".run.json",".dat",".frd",".sta",".stdout.log",".stderr.log")
    for extension in extensions[1:]:deck.with_suffix(extension).write_text("retained raw fixture")
    run=deck.with_suffix(".run.json")
    run.write_text(json.dumps({"status":"completed","returncode":0,"input_path":str(deck),"input_sha256":sha256(deck)}))
    vtu=tmp_path/"component.vtu";vtu.write_text("mock field input")
    names=("hub","pipe","screw_1","screw_2","screw_3")
    candidate={"source_run":str(run),"input_sha256":sha256(deck),"friction_coefficient":0.,
        "source_file_hashes":{extension:sha256(deck.with_suffix(extension)) for extension in extensions},
        "accepted":True,"diagnostic_only":False,"solver_status":"completed","comparison_identity_verified":True,
        "assessment":{"status":"numerically_accepted","gates":[{"name":name,"status":"passed"} for name in results.REQUIRED_RESULT_GATE_NAMES]},
        "stages":{state:{"time":time,"components":{name:{"status":"available","vtu":str(vtu),"vtu_sha256":sha256(vtu)} for name in names}}
                  for state,time in (("preload",1.),("loaded",2.))}}
    report_path=tmp_path/"results.json";report_path.write_text(json.dumps(candidate))
    distances=(.1,35.,75.,125.,210.) if omit_free_span else (.1,35.,75.,125.,175.,210.)
    reference_state={"source":{},"fixed_probes":{"points_mm":[[.1,.1,.1]]},
        "sections":[section_fixture(distance) for distance in distances],"pipe_max_displacement_mm":.5,
        "pipe_regions":{name:{"raw_peak_mises_mpa":0.} for name in ("center","slots","seats")}}
    reference_path=tmp_path/"reference.json"
    reference_path.write_text(json.dumps({"states":{state:reference_state for state in ("preload","loaded")},
                                         "comparison_screen_frozen_before_assembly_solve":frozen_tolerances()}))
    original_reference_sha=sha256(reference_path)
    mesh=tetra_mesh();mesh.displacement_mm[:]=[0,.5,0.];mesh.stress_mpa[:]=0
    monkeypatch.setattr(comparison,"cached_reference_deformable",lambda *args:mesh)
    monkeypatch.setattr(results,"read_component_vtu",lambda *args,**kwargs:SimpleNamespace(domain=2,
                        displacement_mm=mesh.displacement_mm,field_mesh=lambda:mesh))
    class Interpolator:
        def __init__(self,*args):pass
        def sample(self,points):return {"valid":np.ones(len(points),dtype=bool),
            "displacement_mm":np.zeros((len(points),3)),"stress_mpa":np.zeros((len(points),3,3))}
    monkeypatch.setattr(comparison,"BodyInterpolator",Interpolator)
    def section(*args):
        distance=-args[2][2];value=section_fixture(distance)
        if distance in (.1,210.):
            value["force_N"][0]=1e6;value["moment_Nmm"][2]=1e9;value["mean_displacement_mm"][0]=10
        return value
    monkeypatch.setattr(comparison,"integrate_section",section)
    regions={name:{"volume_mean_nodal_von_mises_mpa":1.,"distance_window_mm":[0,1],"volume_mm3":1.,"raw_display_peak_mpa":1.}
             for name in ("center","slots","seats")}
    monkeypatch.setattr(studies,"region_measures",lambda *args:copy.deepcopy(regions))
    monkeypatch.setattr(comparison,"cached_reference_region_measures",lambda *args:{"regions":regions,"recipe_sha256":"fixture","metrics_sha256":"fixture"})
    if changed_dat:deck.with_suffix(".dat").write_text("changed native output")
    if changed_vtu:vtu.write_text("changed derived field after accepted postprocessing")
    report=compare_run(reference_path,report_path,make_plots=False,reference_cache_directory=tmp_path/"cache")
    assert sha256(reference_path)==original_reference_sha
    assert report["candidate_retained_result_source_binding"]["verified"] is not changed_dat
    assert report["candidate_component_field_binding"]["verified"] is not changed_vtu
    assert report["screen_eligible"] is (not changed_dat and not omit_free_span and not changed_vtu)
    assert report["screen_status"]==("diagnostic_only_identity_or_numerical_gate_unmet" if changed_dat or omit_free_span or changed_vtu else "passed")
    for state in report["states"].values():
        assert len(state["screens"])==(14 if omit_free_span else 16)
        assert len(state["section_diagnostics"])==6
        assert all(row["observed_within_tolerance"] is False for row in state["section_diagnostics"])
        assert {row["distance_mm"] for row in state["screens"] if "distance_mm" in row}==({35.,75.,125.} if omit_free_span else {35.,75.,125.,175.})
        if omit_free_span:
            missing=next(row for row in state["screens"] if row["quantity"]=="free_span_section_coverage")
            assert missing["status"]=="missing" and missing["required_station_counts"][175.] == 0


def test_vtu_reader_converts_pa_and_does_not_merge_domains(tmp_path):
    import meshio
    p = tetra_mesh().points_mm
    fields = {f"Displacement_field,_{axis}-component": np.full(4, i * .1) for i, axis in enumerate("XYZ")}
    fields.update({f"Stress_tensor,_{axis}-component": np.full(4, value) for axis, value in
                   zip(("xx", "yy", "zz", "xy", "xz", "yz"), (1e6, 2e6, 3e6, 4e6, 5e6, 6e6))})
    fields["Entity_index"] = np.full(4, 2.)
    path = tmp_path / "reference.vtu"
    meshio.write(path, meshio.Mesh(p, [("tetra", [[0, 1, 2, 3]])], point_data=fields))
    mesh = read_comsol_vtu(path)
    np.testing.assert_allclose(mesh.stress_mpa[0], [[1,4,5],[4,2,6],[5,6,3]])
    np.testing.assert_allclose(mesh.displacement_mm[0], [0,.1,.2])
    assert mesh.provenance["source_units_explicit_in_vtu"] is False
    fields["Entity_index"][0] = 3
    meshio.write(path, meshio.Mesh(p, [("tetra", [[0, 1, 2, 3]])], point_data=fields))
    with pytest.raises(ValueError, match="crosses domain"):
        read_comsol_vtu(path)


def test_matched_comparison_requires_identity_and_tracks_invalid_probes():
    mesh = tetra_mesh()
    args = dict(candidate_domains=[2], reference_domains=[2], points_reference_mm=[[.1,.2,.3], [5,5,5]],
                candidate_to_reference=RigidTransform(np.eye(3), np.zeros(3)))
    with pytest.raises(ValueError, match="identity"):
        compare_probe_fields(mesh, mesh, **args)
    result = compare_probe_fields(mesh, mesh, **args, geometry_identity_verified=True)
    assert not result["screen_eligible"]
    np.testing.assert_allclose(result["stress_tensor_difference_mpa"][0], np.zeros((3,3)))
    assert result["valid"].tolist() == [True, False]


def test_run_comparison_never_promotes_failed_or_absent_fields(tmp_path):
    import json
    reference=tmp_path/"reference.json";candidate=tmp_path/"results.json"
    reference.write_text(json.dumps({"states":{"preload":{},"loaded":{}},"comparison_screen_frozen_before_assembly_solve":{}}),encoding="utf-8")
    candidate.write_text(json.dumps({"solver_status":"failed","accepted":False,"diagnostic_only":True,"stages":{}}),encoding="utf-8")
    report=compare_run(reference,candidate,make_plots=False,reference_cache_directory=tmp_path/"reference-cache")
    assert not report["screen_eligible"]
    assert report["screen_status"]=="diagnostic_only_identity_or_numerical_gate_unmet"
    assert report["states"]["loaded"]["components"]["pipe"]["status"]=="missing"
