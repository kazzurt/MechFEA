from structural.project import StructuralProject
from structural.studies import controlled_cases, numerical_controls_fingerprint, physical_model_fingerprint
from structural.studies import SCREW_CONTACT_SURFACES, _surface_resolution_evidence, assess_existing_runs, assess_mesh_release, assess_study, region_measures, summarize_run
from structural.comparison import FieldMesh
import numpy as np
import pytest
import copy
import json


def test_mesh_cases_preserve_physics_and_contact_cases_change_only_penalty():
    base=StructuralProject()
    mesh=controlled_cases(base,"mesh")
    assert [p.mesh.thickness_size_mm for _,p in mesh]==[.75,.5,.375]
    assert all(p.mesh.contact_size_mm==base.mesh.contact_size_mm and p.mesh.global_size_mm==base.mesh.global_size_mm for _,p in mesh)
    assert len({physical_model_fingerprint(p) for _,p in mesh})==1
    assert [p.contacts.normal_stiffness_mpa_per_mm for _,p in controlled_cases(base,"contact")]==[50000,100000,200000]
    for _,p in controlled_cases(base,"contact"):
        assert p.mesh==base.mesh
        assert p.loads==base.loads
        assert p.source_sha256==base.source_sha256


def test_clipped_region_volume_and_tensor_quadrature_match_independent_integral():
    from scipy.integrate import quad
    # Axial clipping at d=25 removes the tip above zeta=.5 of this tetra.
    points=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,-50]])
    stress=np.zeros((4,3,3));stress[:,0,0]=1
    stress[:,0,1]=points[:,0];stress[:,1,0]=points[:,0]
    mesh=FieldMesh(points,np.array([[0,1,2,3]]),np.array([2]),np.zeros((4,3)),stress)
    result=region_measures(mesh)["center"]
    volume=50/6*(1-.5**3)
    # VM=sqrt(1+3*x*x). Analytically eliminate y,z; integrate only x.
    lower=quad(lambda x:np.sqrt(1+3*x*x)*(.5*(1-x)-.125),0,.5,epsabs=1e-12)[0]
    upper=quad(lambda x:np.sqrt(1+3*x*x)*.5*(1-x)**2,.5,1,epsabs=1e-12)[0]
    expected=50*(lower+upper)/volume
    assert result["volume_mm3"]==pytest.approx(volume,rel=1e-12)
    # A four-point rule is degree-two exact; the square-root invariant is not
    # polynomial. This coarse clipped-cell fixture allows 0.1% quadrature error.
    assert result["volume_mean_nodal_von_mises_mpa"]==pytest.approx(expected,rel=1e-3)
    assert result["raw_display_peak_mpa"]==pytest.approx(2.)
    assert "four-point" in result["integration"]


def test_roundoff_only_plane_slivers_are_counted_and_bounded():
    points=np.array([[0.,0,-25+1e-11],[1,0,-25-1e-11],[0,1,-25-2e-11],[0,0,-26]])
    stress=np.tile(np.diag([1.,0,0]),(4,1,1))
    mesh=FieldMesh(points,np.array([[0,1,2,3]]),np.array([2]),np.zeros((4,3)),stress)
    result=region_measures(mesh)["center"]
    assert result["volume_mm3"]==0
    assert result["degenerate_boundary_cells_excluded"]==1
    assert result["excluded_volume_upper_bound_mm3"]<1e-10


def _study_summary(index, *, scale=1.):
    from structural.results import REQUIRED_RESULT_GATE_NAMES
    state={"max_displacement_mm":.5*scale,"bolt_force_n":[100*scale,110*scale,120*scale],
           "contact_force_n":{"seat":[0,100*scale,0]},"section_force_n":{"35.0":[0,1000*scale,0]},
           "section_moment_nmm":{"35.0":[100000*scale,0,0]},
           "regions":{name:{"volume_mean_nodal_von_mises_mpa":10*scale,"raw_display_peak_mpa":100*10**index,
                            "raw_peak_location_mm":[1,2,-210]} for name in ("center","slots","seats")}}
    layers=index+2
    return {"numerically_accepted":True,"project_binding_verified":True,"mesh_binding_verified":True,
            "retained_result_source_binding_verified":True,
            "retained_component_field_binding_verified":True,
            "single_run_acceptance":{"accepted":True,"diagnostic_only":False,"solver_status":"completed",
                "assessment":{"status":"numerically_accepted","gates":[{"name":name,"status":"passed"} for name in REQUIRED_RESULT_GATE_NAMES]}},
            "geometry_revision":None,"geometry_revision_binding_verified":True,
            "numerical_controls_fingerprint":"numerical controls","step_controls_binding_verified":True,
            "metric_definition_sha256":"definition","solver_sha256":"solver","physical_model_fingerprint":"physics",
            "controlled_fingerprints":{kind:"controlled" for kind in ("contact","friction","load","preload","patch")},
            "mesh_fingerprint":f"mesh{index}","thickness_size_mm":[.75,.5,.375][index],"mesh_settings_verified":True,
            "contact_size_mm":.5,"far_size_mm":3.,"anchor_size_mm":.5,
            "bore_snout_size_mm":None,"effective_bore_snout_size_mm":3.,"bore_snout_settings_verified":True,
            "surface_resolution":{"verified":True,"surfaces":{name:{"quadratic_triangles":100,
                "corner_edge_length_mm":{"median":.5,"percentile95":.6}} for name in SCREW_CONTACT_SURFACES}},
            "bore_snout_area_resolution":{"verified":True,"weight_definition":"fixed 36-point quadratic area rule",
                 "snout_axial_bounds_mm":[-233.88,-202.88]},
            "radial_layer_continuity_verified":True,
            "normal_stiffness_mpa_per_mm":[50000.,100000.,200000.][index],"friction_coefficient":0.,
            "wall_resolution":{"enforced_radial_layers":layers,"radial_partition_radii_mm":np.linspace(17.5,19,layers+1)[1:-1].tolist(),
                               "tetrahedra_touching_both_bore_and_outer_wall":0},
            "states":{"preload":copy.deepcopy(state),"loaded":copy.deepcopy(state)}}


def test_mesh_aggregate_requires_every_gate_and_reports_unresolved_growing_peaks():
    summaries=[_study_summary(0,scale=1.3),_study_summary(1,scale=1.01),_study_summary(2)]
    report=assess_study(summaries,"mesh")
    assert report["status"]=="passed" and report["numerical_convergence_accepted"]
    assert report["design_adequacy"]=="not_assessed" and report["raw_peak_status"].startswith("unresolved")
    assert report["raw_peak_behavior"][-1]["regions"]["seats"]["peak_mpa"]==10000
    summaries[0]["numerically_accepted"]=False
    assert assess_study(summaries,"mesh")["status"]=="unaccepted"


def test_nonzero_friction_study_cannot_reuse_unqualified_accepted_flags():
    cases=[_study_summary(0),_study_summary(0)]
    cases[1]["friction_coefficient"]=.2
    report=assess_study(cases,"friction")
    assert report["status"]=="unaccepted"
    gate=next(row for row in report["gates"] if row["name"]=="all_cases_numerically_accepted")
    assert gate["status"]=="failed"


def test_nominal_size_or_missing_region_cannot_pass_mesh_study():
    summaries=[_study_summary(i) for i in range(3)]
    summaries[1]["wall_resolution"]["tetrahedra_touching_both_bore_and_outer_wall"]=1
    report=assess_study(summaries,"mesh")
    assert next(g for g in report["gates"] if g["name"]=="enforced_wall_refinement")["status"]=="failed"
    summaries[1]["wall_resolution"]["tetrahedra_touching_both_bore_and_outer_wall"]=0
    summaries[2]["states"]["loaded"]["regions"]["seats"]["volume_mean_nodal_von_mises_mpa"]=None
    report=assess_study(summaries,"mesh")
    assert report["status"]=="unaccepted" and any(g["status"]=="missing" for g in report["gates"])
    assert assess_study(summaries[:2],"mesh")["status"]=="incomplete"


def test_wall_refinement_does_not_claim_spatial_contact_convergence():
    wall=[_study_summary(i) for i in range(3)]
    wall_report=assess_study(wall,"mesh")
    assert wall_report["status"]=="passed" and wall_report["scope"]=="through_wall_refinement_only"
    assert not wall_report["release_mesh_accepted"]
    contact=[copy.deepcopy(wall[-1]) for _ in range(3)]
    for index,size in ((1,.375),(2,.25)):
        contact[index].update(contact_size_mm=size,mesh_fingerprint=f"contact-{index}")
        for value in contact[index]["surface_resolution"]["surfaces"].values():
            value.update(quadratic_triangles=100*(index+1),corner_edge_length_mm={"median":size,"percentile95":size*1.2})
    contact_report=assess_study(contact,"contact_mesh")
    assert contact_report["status"]=="passed"
    combined=assess_mesh_release(wall_report,contact_report)
    assert not combined["release_mesh_accepted"]
    assert combined["unassessed_contact_surfaces"]==["hub_snout","pipe_bore"]
    assert contact_report["scope"].startswith("screw_seat_and_head")
    incomplete=copy.deepcopy(contact_report);incomplete["gates"]=[]
    assert not assess_mesh_release(wall_report,incomplete)["release_mesh_accepted"]
    arbitrary=copy.deepcopy(contact_report);arbitrary["gates"]=[{"name":"arbitrary","status":"passed"}]
    assert "Independent spatial contact refinement has not passed" in assess_mesh_release(wall_report,arbitrary)["issues"]
    contact_report["cases"][0]["mesh_fingerprint"]="no-common-case"
    assert not assess_mesh_release(wall_report,contact_report)["release_mesh_accepted"]
    wall[2]["contact_size_mm"]=.25
    assert next(g for g in assess_study(wall,"mesh")["gates"] if g["name"]=="isolated_spatial_refinement")["status"]=="failed"


def test_nominal_contact_size_without_measured_surface_refinement_cannot_pass():
    contact=[_study_summary(2),_study_summary(2)]
    contact[1].update(contact_size_mm=.25,mesh_fingerprint="contact-fine")
    report=assess_study(contact,"contact_mesh")
    assert next(g for g in report["gates"] if g["name"]=="measured_contact_surface_refinement")["status"]=="failed"
    contact[1]["surface_resolution"]["verified"]=False
    report=assess_study(contact,"contact_mesh")
    assert next(g for g in report["gates"] if g["name"]=="measured_contact_surface_refinement")["status"]=="missing"


def _combined_release_fixture():
    wall=[_study_summary(i) for i in range(3)]
    seat=[copy.deepcopy(wall[-1]) for _ in range(3)]
    for index,size in ((1,.375),(2,.25)):
        seat[index].update(contact_size_mm=size,mesh_fingerprint=f"seat-{index}")
        for row in seat[index]["surface_resolution"]["surfaces"].values():
            row.update(quadratic_triangles=100*(index+1),corner_edge_length_mm={"median":size,"percentile95":size*1.2})
    bore=[]
    for index,size in enumerate((3.,2.,1.5)):
        row=copy.deepcopy(wall[-1]);row.update(effective_bore_snout_size_mm=size,bore_snout_size_mm=None if index==0 else size)
        if index:row["mesh_fingerprint"]=f"bore-{size}"
        row["bore_snout_area_resolution"]={"verified":True,"weight_definition":"fixed 36-point quadratic area rule",
            "snout_axial_bounds_mm":[-233.88,-202.88],"surfaces":{name:{"quadratic_triangles_with_positive_selected_area":100*(index+1),
            "area_weighted_corner_edge_length_mm":{"median":size*.8,"percentile95":size}} for name in ("pipe_bore_snout_band","hub_snout")}}
        bore.append(row)
    candidate=copy.deepcopy(seat[-1]);candidate.update(mesh_fingerprint="combined-candidate",bore_snout_size_mm=1.5,effective_bore_snout_size_mm=1.5,
        bore_snout_area_resolution=copy.deepcopy(bore[-1]["bore_snout_area_resolution"]))
    return wall,seat,bore,candidate


def test_explicit_combined_candidate_is_separate_and_requires_every_binding_and_response():
    wall,seat,bore,candidate=_combined_release_fixture()
    reports=[assess_study(rows,kind) for rows,kind in ((wall,"mesh"),(seat,"contact_mesh"),(bore,"bore_contact_mesh"))]
    assert not assess_mesh_release(*reports)["release_mesh_accepted"]
    accepted=assess_mesh_release(*reports,candidate)
    assert accepted["legacy_spatial_scope_accepted"] and not accepted["release_mesh_accepted"]
    assert accepted["scope_version"]==2 and len(accepted["unassessed_slot_traces"])==30
    assert len(accepted["combined_candidate_assessment"]["changes"])==3*2*8
    assert [len(report["cases"]) for report in reports]==[3,3,3]
    for key,value in (("geometry_revision_binding_verified",False),("mesh_settings_verified",False),
                      ("retained_result_source_binding_verified",False),
                      ("retained_component_field_binding_verified",False),
                      ("friction_coefficient",.2),
                      ("contact_size_mm",.5),("physical_model_fingerprint","other physics")):
        bad=copy.deepcopy(candidate);bad[key]=value
        assert not assess_mesh_release(*reports,bad)["legacy_spatial_scope_accepted"]
    bad=copy.deepcopy(candidate);bad["single_run_acceptance"]["assessment"]["gates"].pop()
    assert not assess_mesh_release(*reports,bad)["legacy_spatial_scope_accepted"]
    for key,value in (("weight_definition","different sampling rule"),("snout_axial_bounds_mm",[-220,-219]),
                      ("weight_definition",None),("snout_axial_bounds_mm",[-220,-220])):
        bad=copy.deepcopy(candidate);bad["bore_snout_area_resolution"][key]=value
        assert not assess_mesh_release(*reports,bad)["legacy_spatial_scope_accepted"]
    bad=copy.deepcopy(candidate);bad["bore_snout_area_resolution"]["verified"]=False
    assert not assess_mesh_release(*reports,bad)["legacy_spatial_scope_accepted"]
    bad=copy.deepcopy(candidate);bad["states"]["preload"]["regions"]["seats"]["volume_mean_nodal_von_mises_mpa"]=10.51
    failed=assess_mesh_release(*reports,bad)
    assert not failed["legacy_spatial_scope_accepted"]
    assert any(g["name"].startswith("vs_wall/preload/") and g["status"]=="failed" for g in failed["combined_candidate_assessment"]["gates"])
    bad=copy.deepcopy(candidate);bad["states"]["loaded"]["bolt_force_n"]=None
    assert not assess_mesh_release(*reports,bad)["legacy_spatial_scope_accepted"]


def test_combined_candidate_must_agree_with_each_isolated_family_not_just_wall():
    wall,seat,bore,candidate=_combined_release_fixture()
    for row in bore[1:]:
        for state in row["states"].values():state["max_displacement_mm"]*=.97
    reports=[assess_study(rows,kind) for rows,kind in ((wall,"mesh"),(seat,"contact_mesh"),(bore,"bore_contact_mesh"))]
    assert all(report["status"]=="passed" for report in reports)
    report=assess_mesh_release(*reports,candidate)
    assert report["independent_spatial_studies_accepted"] and not report["combined_finest_configuration_verified"]
    gates={g["name"]:g["status"] for g in report["combined_candidate_assessment"]["gates"]}
    assert gates["vs_bore_contact/loaded/max_displacement_mm"]=="failed"
    assert gates["vs_wall/loaded/max_displacement_mm"]=="passed"


@pytest.mark.parametrize("kind,index",[("contact_mesh",1),("bore_contact_mesh",2)])
def test_every_contact_spatial_family_requires_three_measured_levels(kind,index):
    rows=_combined_release_fixture()[index]
    assert assess_study(rows,kind)["status"]=="passed"
    two=assess_study([rows[0],rows[-1]],kind)
    assert two["status"]=="incomplete" and not two["numerical_convergence_accepted"]
    assert next(g for g in two["gates"] if g["name"]=="case_count")["limit"]==3


def test_bore_contact_family_is_separate_and_combined_release_remains_unverified():
    base=StructuralProject();base.mesh.thickness_size_mm=.375;base.mesh.contact_size_mm=.5;base.mesh.global_size_mm=3.
    variants=controlled_cases(base,"bore_contact_mesh")
    assert [project.mesh.bore_snout_size_mm for _,project in variants]==[None,2.,1.5]
    assert all(project.mesh.contact_size_mm==.5 and project.mesh.thickness_size_mm==.375 for _,project in variants)
    wall=[_study_summary(i) for i in range(3)]
    contact=[copy.deepcopy(wall[-1]) for _ in range(3)]
    for index,size in ((1,.375),(2,.25)):
        contact[index].update(contact_size_mm=size,mesh_fingerprint=f"screw-{index}")
        for value in contact[index]["surface_resolution"]["surfaces"].values():
            value.update(quadratic_triangles=100*(index+1),corner_edge_length_mm={"median":size,"percentile95":size*1.2})
    bore=[]
    for index,size in enumerate((3.,2.,1.5)):
        row=copy.deepcopy(wall[-1]);row.update(effective_bore_snout_size_mm=size,bore_snout_size_mm=None if index==0 else size)
        if index:row["mesh_fingerprint"]=f"bore-{size}"
        # Raw count-weighted median can rise when coarse faces are subdivided;
        # fixed-surface area weighting remains the refinement measure.
        row["bore_snout_area_resolution"]={"verified":True,"weight_definition":"fixed 36-point quadratic area rule",
            "snout_axial_bounds_mm":[-233.88,-202.88],"surfaces":{name:{"quadratic_triangles_with_positive_selected_area":100*(index+1),
            "corner_edge_length_mm":{"median":1.+index,"percentile95":3.+index},
            "area_weighted_corner_edge_length_mm":{"median":size*.8,"percentile95":size}} for name in ("pipe_bore_snout_band","hub_snout")}}
        bore.append(row)
    wall_report=assess_study(wall,"mesh");contact_report=assess_study(contact,"contact_mesh")
    bore_report=assess_study(bore,"bore_contact_mesh")
    assert bore_report["status"]=="passed"
    altered=copy.deepcopy(bore);altered[1]["bore_snout_area_resolution"]["snout_axial_bounds_mm"]=[-220,-219]
    assert assess_study(altered,"bore_contact_mesh")["status"]=="unaccepted"
    combined=assess_mesh_release(wall_report,contact_report,bore_report)
    assert combined["independent_spatial_studies_accepted"]
    assert combined["unassessed_contact_surfaces"]==[]
    assert not combined["release_mesh_accepted"] and not combined["combined_finest_configuration_verified"]
    bore[2]["contact_size_mm"]=.25
    assert assess_study(bore,"bore_contact_mesh")["status"]=="unaccepted"
    bore[2]["contact_size_mm"]=.5
    bore[1]["bore_snout_area_resolution"]["verified"]=False
    assert next(g for g in assess_study(bore,"bore_contact_mesh")["gates"] if g["name"]=="measured_bore_snout_refinement")["status"]=="missing"


def test_surface_resolution_sidecar_requires_both_immutable_source_hashes(tmp_path):
    from structural.project import sha256
    manifest=tmp_path/"mesh-manifest.json";manifest.write_text("{}")
    mesh=tmp_path/"structural-mesh.npz";mesh.write_bytes(b"mesh fixture")
    sidecar=tmp_path/"surface-resolution-verification.json"
    data={"mesh_sha256":sha256(mesh),"mesh_manifest_sha256":sha256(manifest),
          "surface_resolution":{"screw_1_seat":{"quadratic_triangles":100}}}
    sidecar.write_text(json.dumps(data))
    assert _surface_resolution_evidence(manifest,{})["verified"]
    mesh.write_bytes(b"changed mesh")
    assert not _surface_resolution_evidence(manifest,{})["verified"]


def test_derived_geometry_cannot_reuse_parent_surface_or_continuity_metrics(tmp_path):
    from types import SimpleNamespace
    from structural.project import sha256
    from structural.studies import _radial_layer_continuity_evidence
    manifest_path=tmp_path/"mesh-manifest.json";mesh_path=tmp_path/"structural-mesh.npz"
    mesh_path.write_bytes(b"original mesh")
    manifest={"pipe_wall_resolution":{"enforced_radial_layers":2},
              "radial_layer_continuity":{"pipe_layer_interface_1":{"triangles_with_two_adjacent_tetrahedra":4,
                                                                      "shared_node_continuity_verified":True}}}
    manifest_path.write_text(json.dumps(manifest))
    interfaces={"pipe_layer_interface_1":{"quadratic_triangles_verified":4,"all_six_face_nodes_shared":True}}
    sidecar=tmp_path/"quadratic-layer-continuity-verification.json"
    sidecar.write_text(json.dumps(interfaces))
    mesh=SimpleNamespace(surfaces={})
    # Count-only original manifest entries do not hide its later six-node proof.
    assert _radial_layer_continuity_evidence(manifest_path,manifest,mesh)["verified"]
    manifest["geometry_revision"]={"schema_version":1,"kind":"derived_seat_trim_intersection","definition_sha256":"a"*64}
    manifest["surface_resolution"]={"screw_1_seat":{"quadratic_triangles":100}}
    manifest_path.write_text(json.dumps(manifest))
    assert not _surface_resolution_evidence(manifest_path,manifest)["verified"]
    assert not _radial_layer_continuity_evidence(manifest_path,manifest,mesh)["verified"]
    sidecar.write_text(json.dumps({"mesh_sha256":sha256(mesh_path),"mesh_manifest_sha256":sha256(manifest_path),"interfaces":interfaces}))
    assert _radial_layer_continuity_evidence(manifest_path,manifest,mesh)["verified"]
    mesh_path.write_bytes(b"new coordinates")
    assert not _radial_layer_continuity_evidence(manifest_path,manifest,mesh)["verified"]


def test_contact_study_checks_all_penalties_vectors_and_identical_mesh():
    summaries=[_study_summary(i,scale=1+.005*i) for i in range(3)]
    for summary in summaries:summary["mesh_fingerprint"]="same mesh"
    assert assess_study(summaries,"contact")["status"]=="passed"
    summaries[0]["states"]["loaded"]["contact_force_n"]["seat"]=[100,0,0]
    report=assess_study(summaries,"contact")
    assert report["status"]=="unaccepted"
    assert any(g["name"].endswith("loaded/contact_force_n") and g["status"]=="failed" for g in report["gates"])
    summaries[0]["mesh_fingerprint"]="different mesh"
    assert next(g for g in assess_study(summaries,"contact")["gates"] if g["name"]=="identical_mesh")["status"]=="failed"


def test_physical_sensitivity_has_no_convergence_acceptance_or_response_cap():
    summaries=[_study_summary(i,scale=1.+i) for i in range(3)]
    for i,summary in enumerate(summaries):
        summary.update(mesh_fingerprint="same mesh",friction_coefficient=.1*i)
        if i:
            proof={"verified":True,"friction_coefficient":.1*i,"coulomb_absolute_tolerance_mpa":1e-5,"coulomb_relative_tolerance":1e-6}
            summary["single_run_acceptance"].update(friction_coefficient=.1*i,
                friction_qualification_evidence={"verified":True,"friction_coefficient":.1*i,"qualification":proof})
            summary["single_run_acceptance"]["assessment"]["gates"].append({"name":"local_coulomb_bound","status":"passed"})
    report=assess_study(summaries,"friction")
    assert report["status"]=="sensitivity_completed"
    assert not report["numerical_convergence_accepted"]
    assert all(change["limit"] is None for change in report["changes"])


def test_controlled_sweeps_change_only_intended_parameter():
    base=StructuralProject()
    for kind in ("friction","load","preload"):
        cases=controlled_cases(base,kind)
        assert len({physical_model_fingerprint(project,exclude_parameter=kind) for _,project in cases})==1
        assert all(project.mesh==base.mesh for _,project in cases)
    base.contacts.friction_coefficient=.1
    with pytest.raises(ValueError,match="frictionless"):controlled_cases(base,"contact")


def test_increment_schedule_is_controlled_independently_of_physics_and_watchdog():
    original=StructuralProject();changed=copy.deepcopy(original)
    changed.solver["timeout_s"]=10800
    changed.solver["numerical_control_fixture"]={"path":"retained-proof.json","sha256":"proof"}
    changed.solver["friction_qualification_manifest"]={"path":"friction-proof.json","sha256":"proof"}
    assert numerical_controls_fingerprint(original)==numerical_controls_fingerprint(changed)
    changed.solver["step_controls"]=copy.deepcopy(original.static_step_controls())
    changed.solver["step_controls"]["preload"]["maximum_increment"]=.25
    assert physical_model_fingerprint(original)==physical_model_fingerprint(changed)
    assert numerical_controls_fingerprint(original)!=numerical_controls_fingerprint(changed)
    cases=[_study_summary(i) for i in range(3)]
    cases[-1]["numerical_controls_fingerprint"]="changed schedule"
    report=assess_study(cases,"mesh")
    assert report["status"]=="unaccepted"
    assert next(g for g in report["gates"] if g["name"]=="fixed_numerical_controls")["status"]=="failed"


def test_geometry_revision_is_physical_identity_in_every_controlled_study():
    original=StructuralProject();derived=copy.deepcopy(original)
    derived.geometry_revision={"schema_version":1,"kind":"derived_seat_trim_intersection","definition_sha256":"a"*64}
    for parameter in (None,"contact","friction","load","preload","patch"):
        assert physical_model_fingerprint(original,exclude_parameter=parameter)!=physical_model_fingerprint(derived,exclude_parameter=parameter)
    assert all(project.geometry_revision==derived.geometry_revision for _,project in controlled_cases(derived,"mesh"))
    cases=[_study_summary(i) for i in range(3)]
    # Even an imported summary with a copied physical fingerprint cannot hide
    # an explicitly different derived geometry revision.
    cases[-1]["geometry_revision"]=derived.geometry_revision
    report=assess_study(cases,"mesh")
    assert next(g for g in report["gates"] if g["name"]=="unchanged_geometry_revision")["status"]=="failed"
    cases[-1]["geometry_revision"]=None
    del cases[-1]["geometry_revision_binding_verified"]
    assert next(g for g in assess_study(cases,"mesh")["gates"] if g["name"]=="geometry_revision_binding")["status"]=="failed"


def test_mesh_fingerprint_shared_api_excludes_provenance_but_keeps_geometry_and_selections():
    from structural.mesh import SolidMesh,mesh_fingerprint
    from structural.studies import mesh_fingerprint as legacy_import
    assert legacy_import is mesh_fingerprint
    mesh=SolidMesh([1,2],[[0,0,0],[1,0,0]],[],np.empty((0,10),dtype=int),node_sets={"anchor":[1]})
    original=mesh_fingerprint(mesh)
    mesh.provenance["audit"]={"output_mesh_fingerprint":original}
    assert mesh_fingerprint(mesh)==original
    mesh.points_mm[1,0]+=.006
    assert mesh_fingerprint(mesh)!=original
    mesh.points_mm[1,0]=1
    mesh.node_sets["anchor"]=[2]
    assert mesh_fingerprint(mesh)!=original


def test_revision_solver_transform_rejects_coordinate_and_head_connectivity_changes():
    from structural.mesh import SolidMesh,TET10_EDGES
    from structural.backends.explicit_pretension import explicit_pretension
    from structural.studies import _verify_pretension_mesh_transform
    points=[];lookup={};cells=[]
    def node(point):
        key=tuple(point)
        if key not in lookup:lookup[key]=len(points)+1;points.append(point)
        return lookup[key]
    for corners in ([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,0,0],[0,1,0],[1,0,0],[0,0,-1]]):
        corners=np.asarray(corners,dtype=float)
        cells.append([node(point) for point in corners]+[node((corners[a]+corners[b])/2) for a,b in TET10_EDGES])
    prepared=SolidMesh(np.arange(1,len(points)+1),points,[1,2],cells,surfaces={"screw_1_cut":[(2,1)]},
                       provenance={"pretension_head_element_sets":{"1":[2]}})
    solver,_,mapping=explicit_pretension(prepared,"screw_1_cut",[2],[0,0,1],100)
    metadata={"pretension_formulation":"explicit paired-node axial jump with scalar work-conjugate force",
              "pretension_cut_node_maps":{"1":{str(a):b for a,b in mapping.items()}}}
    assert _verify_pretension_mesh_transform(prepared,solver,metadata)["verified"]
    altered=copy.deepcopy(solver);altered.points_mm[-1,0]+=.001
    with pytest.raises(ValueError,match="differs"):_verify_pretension_mesh_transform(prepared,altered,metadata)
    altered=copy.deepcopy(solver);altered.tetra10[1]=prepared.tetra10[1]
    with pytest.raises(ValueError,match="differs"):_verify_pretension_mesh_transform(prepared,altered,metadata)


def test_existing_missing_runs_remain_failed_in_retained_assessment(tmp_path):
    report=assess_existing_runs([tmp_path/"absent.json"]*3,"mesh",tmp_path/"retained-assessment")
    assert report["status"]=="unaccepted" and len(report["cases"])==3
    assert all(row["error"] for row in report["cases"])
    assert (tmp_path/"retained-assessment/study-assessment.json").is_file()


def test_study_commands_are_fresh_children_and_failed_case_logs_are_retained(tmp_path,monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    from structural.studies import run_study
    calls=[]
    def child(arguments,**kwargs):
        calls.append(arguments)
        assert arguments[:3]==[sys.executable,"-m","structural"]
        assert kwargs["creationflags"]==getattr(subprocess,"CREATE_NO_WINDOW",0)
        assert "start_new_session" not in kwargs and not kwargs.get("shell",False)
        kwargs["stdout"].write("retained child output\n")
        kwargs["stderr"].write("retained child diagnostic\n")
        output=Path(arguments[arguments.index("--output")+1])
        if arguments[3]=="mesh":
            output.mkdir();(output/"structural-mesh.npz").write_bytes(b"child mesh")
            return subprocess.CompletedProcess(arguments,0)
        return subprocess.CompletedProcess(arguments,7)
    monkeypatch.setattr(subprocess,"run",child)
    report=run_study(StructuralProject(),"mesh",tmp_path/"study")
    assert len(calls)==6 and [args[3] for args in calls]==["mesh","solve"]*3
    assert report["status"]=="unaccepted"
    assert all(case["status"]=="failed" and case["exit_code"]==7 for case in report["cases"])
    for case in report["cases"]:
        directory=tmp_path/"study"/case["name"]
        saved=json.loads((directory/"solve.command.json").read_text())
        assert saved["status"]=="failed" and saved["exit_code"]==7
        assert (directory/"solve.stdout.log").read_text()=="retained child output\n"
        assert (directory/"solve.stderr.log").read_text()=="retained child diagnostic\n"


def test_hash_linked_mesh_metadata_erratum_does_not_modify_frozen_inputs(tmp_path):
    from structural.mesh import SolidMesh
    from structural.project import sha256
    project=StructuralProject();project.mesh.thickness_size_mm=1.5;project.mesh.contact_size_mm=1.;project.mesh.global_size_mm=3.
    project_path=tmp_path/"project.json";project.save(project_path)
    corners=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    points=np.vstack([corners,*[(corners[a]+corners[b])/2 for a,b in ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))]])
    mesh_manifest=tmp_path/"mesh-manifest.json"
    mesh_manifest.write_text(json.dumps({"settings":{"thickness_size_mm":1.5,"contact_size_mm":.5,"far_size_mm":3.,"order":2}}))
    mesh=SolidMesh(np.arange(1,11),points,[1],[list(range(1,11))],element_sets={"pipe":[1]},provenance={"mesh_manifest":str(mesh_manifest)})
    mesh.save(tmp_path/"structural-mesh.npz");mesh.save(tmp_path/"solver-mesh.npz")
    deck=tmp_path/"model.inp";deck.write_text("*HEADING\nmetadata fixture\n")
    deck.with_suffix(".assembly.json").write_text(json.dumps({"project_fingerprint":project.fingerprint(),"solver_mesh":"solver-mesh.npz"}))
    run=deck.with_suffix(".run.json");run.write_text(json.dumps({"input_path":str(deck),"executable_sha256":"solver"}))
    results=tmp_path/"structural-results/results.json";results.parent.mkdir()
    results.write_text(json.dumps({"source_run":str(run),"solver_status":"failed","stages":{},"pretension":{}}))
    before=sha256(project_path)
    assert summarize_run(results)["mesh_settings_verified"] is False
    actual=project.to_dict()["mesh"].copy();actual["contact_size_mm"]=.5
    correction={"original_project_sha256":before,"input_deck_sha256":sha256(deck),
                "original_mesh_sha256":sha256(tmp_path/"structural-mesh.npz"),"requested_mesh":project.to_dict()["mesh"],"actual_mesh":actual,
                "verified_preparation":{"mesh_settings_binding":{"manifest":{"sha256":sha256(mesh_manifest)}}}}
    (tmp_path/"provenance-correction.json").write_text(json.dumps(correction))
    summary=summarize_run(results)
    assert summary["mesh_settings_verified"] and summary["contact_size_mm"]==.5
    assert summary["mesh_settings_correction"]["verified"]
    assert sha256(project_path)==before and not summary["numerically_accepted"]
    relocated=tmp_path/"separate-reviewed-reports/fresh-results/results.json"
    relocated.parent.mkdir(parents=True);relocated.write_text(results.read_text())
    reviewed=summarize_run(relocated)
    assert reviewed["mesh_settings_verified"]
    assert reviewed["source_files"]["project"]["path"]==str(project_path.resolve())
    deck.write_text(deck.read_text()+"** changed\n")
    assert summarize_run(results)["mesh_settings_verified"] is False


def _slot_evidence(level):
    from structural.studies import SLOT_MEASUREMENT_DEFINITION,SLOT_TRACE_FACES
    return {"verified":True,"definition":copy.deepcopy(SLOT_MEASUREMENT_DEFINITION),"cad_topology_sha256":"a"*64,
            "traces":{name:{"source_face_indices":faces,"quadratic_edges":2**(level+1),"integrated_arc_length_mm":10.,
                 "arc_length_weighted_in_plane_span_mm":{"median":2./(level+1),"percentile95":3./(level+1)},
                 "axial_span_mm":{"median":0.},"in_plane_span_mm":{"median":99.+level}}
                 for name,faces in SLOT_TRACE_FACES.items()}}


def _slot_family(baseline):
    rows=[]
    for level,target in enumerate((None,1.5,.75)):
        row=copy.deepcopy(baseline)
        row.update(slot_size_mm=target,effective_slot_size_mm=target or 3.,slot_settings_verified=True,slot_resolution=_slot_evidence(level))
        if level:row["mesh_fingerprint"]=f"slot-{target}"
        rows.append(row)
    return rows


def test_slot_family_preserves_other_controls_and_requires_all_30_weighted_traces():
    base=StructuralProject();base.mesh.thickness_size_mm=.375;base.mesh.contact_size_mm=.5;base.mesh.global_size_mm=3.
    cases=controlled_cases(base,"slot_mesh")
    assert [p.mesh.slot_size_mm for _,p in cases]==[None,1.5,.75]
    assert all(p.mesh.thickness_size_mm==.375 and p.mesh.contact_size_mm==.5 and p.mesh.bore_snout_size_mm is None for _,p in cases)
    rows=_slot_family(_study_summary(2))
    report=assess_study(rows,"slot_mesh")
    assert report["status"]=="passed" and len(report["slot_trace_coverage"])==30
    assert not report["release_mesh_accepted"] and report["controlled_contact_surfaces"]==[]
    assert assess_study(rows[:2],"slot_mesh")["status"]=="incomplete"
    # Every R2 trace matters, even if the six full perimeters visibly refine.
    for trace in rows[-1]["slot_resolution"]["traces"]:
        bad=copy.deepcopy(rows);bad[-1]["slot_resolution"]["traces"][trace]=copy.deepcopy(rows[-2]["slot_resolution"]["traces"][trace])
        assert not assess_study(bad,"slot_mesh")["numerical_convergence_accepted"]
    for key,value in (("definition",{}),("cad_topology_sha256","b"*64),("verified",False)):
        bad=copy.deepcopy(rows);bad[-1]["slot_resolution"][key]=value
        assert not assess_study(bad,"slot_mesh")["numerical_convergence_accepted"]
    bad=copy.deepcopy(rows);bad[-1]["slot_resolution"]["traces"].pop("slot_3_corner_4_bore")
    assert not assess_study(bad,"slot_mesh")["numerical_convergence_accepted"]
    for key,value in (("contact_size_mm",.25),("effective_bore_snout_size_mm",1.5),("slot_settings_verified",False)):
        bad=copy.deepcopy(rows);bad[-1][key]=value
        assert not assess_study(bad,"slot_mesh")["numerical_convergence_accepted"]
    # Existing wall studies cannot silently add a local slot control.
    wall=[_study_summary(i) for i in range(3)];wall[-1]["slot_size_mm"]=.75
    assert not assess_study(wall,"mesh")["numerical_convergence_accepted"]


def test_slot_sidecar_binds_current_coordinates_manifest_topology_and_geometry_revision(tmp_path):
    from structural.project import sha256
    from structural.studies import _slot_resolution_evidence,_measured_slot_refinement
    manifest_data={"source_sha256":"c"*64}
    manifest=tmp_path/"mesh-manifest.json";manifest.write_text(json.dumps(manifest_data))
    mesh=tmp_path/"structural-mesh.npz";mesh.write_bytes(b"current coordinates")
    topology=tmp_path/"topology.json";topology.write_text(json.dumps({"source_sha256":"c"*64,"source_step_current_sha256":"c"*64}))
    sidecar=tmp_path/"slot-resolution-verification.json"
    evidence=_slot_evidence(0)
    evidence.update(mesh_sha256=sha256(mesh),mesh_manifest_sha256=sha256(manifest),cad_topology_path=topology.name,
                    cad_topology_sha256=sha256(topology),geometry_revision=None,
                    all_24_source_slot_faces_represented=True,sources_unchanged=True)
    sidecar.write_text(json.dumps(evidence))
    assert _slot_resolution_evidence(manifest,manifest_data)["verified"]
    for path in (mesh,manifest,topology):
        original=path.read_bytes();path.write_bytes(original+b" ")
        assert not _slot_resolution_evidence(manifest,manifest_data)["verified"]
        path.write_bytes(original)
    assert not _slot_resolution_evidence(manifest,{**manifest_data,"geometry_revision":{"kind":"other"}})["verified"]
    assert _measured_slot_refinement([{"slot_resolution":_slot_resolution_evidence(manifest,manifest_data)}]) is True
    for key in ("all_24_source_slot_faces_represented","sources_unchanged"):
        changed=copy.deepcopy(evidence);changed[key]=False;sidecar.write_text(json.dumps(changed))
        assert not _slot_resolution_evidence(manifest,manifest_data)["verified"]
    topology.write_text(json.dumps({"source_sha256":"d"*64}));evidence["cad_topology_sha256"]=sha256(topology)
    sidecar.write_text(json.dumps(evidence))
    assert not _slot_resolution_evidence(manifest,manifest_data)["verified"]


def test_full_spatial_release_requires_independent_slot_family_and_four_candidate_comparisons():
    wall,seat,bore,candidate=_combined_release_fixture()
    wall[-1]["slot_resolution"]=_slot_evidence(0)
    slot=_slot_family(wall[-1])
    reports=[assess_study(rows,kind) for rows,kind in ((wall,"mesh"),(seat,"contact_mesh"),(bore,"bore_contact_mesh"))]
    slot_report=assess_study(slot,"slot_mesh")
    candidate.update(slot_size_mm=.75,effective_slot_size_mm=.75,slot_settings_verified=True,slot_resolution=_slot_evidence(2))
    report=assess_mesh_release(*reports,candidate,slot_mesh_study=slot_report)
    assert report["release_mesh_accepted"] and report["full_local_spatial_release_accepted"]
    assert report["all_four_independent_spatial_studies_accepted"] and len(report["controlled_slot_traces"])==30
    assert len(report["combined_candidate_assessment"]["changes"])==4*2*8
    assert [len(r["cases"]) for r in [*reports,slot_report]]==[3,3,3,3]
    assert not assess_mesh_release(*reports,candidate)["release_mesh_accepted"]
    for key,value in (("slot_size_mm",None),("effective_slot_size_mm",3.),("slot_settings_verified",False)):
        bad=copy.deepcopy(candidate);bad[key]=value
        assert not assess_mesh_release(*reports,bad,slot_mesh_study=slot_report)["release_mesh_accepted"]
    bad=copy.deepcopy(candidate);bad["slot_resolution"]["traces"]["slot_2_corner_3_bore"]["quadratic_edges"]=2
    assert not assess_mesh_release(*reports,bad,slot_mesh_study=slot_report)["release_mesh_accepted"]
    # Passing the other three families cannot conceal a slot-dependent response.
    for row in slot[1:]:
        for state in row["states"].values():state["max_displacement_mm"]*=.97
    slot_report=assess_study(slot,"slot_mesh");assert slot_report["status"]=="passed"
    report=assess_mesh_release(*reports,candidate,slot_mesh_study=slot_report)
    assert not report["release_mesh_accepted"]
    assert next(g for g in report["combined_candidate_assessment"]["gates"] if g["name"]=="vs_slot/loaded/max_displacement_mm")["status"]=="failed"


def test_optional_contact_count_control_cannot_mix_or_lack_actual_deck_binding():
    original=StructuralProject();changed=copy.deepcopy(original)
    changed.solver["contact_count_relative_tolerance"]=None
    assert numerical_controls_fingerprint(original)==numerical_controls_fingerprint(changed)
    changed.solver["contact_count_relative_tolerance"]=1e-8
    assert numerical_controls_fingerprint(original)!=numerical_controls_fingerprint(changed)
    rows=[_study_summary(i) for i in range(3)]
    for row in rows:row.update(contact_count_relative_tolerance=1e-8,contact_controls_binding_verified=True)
    assert assess_study(rows,"mesh")["status"]=="passed"
    del rows[-1]["contact_controls_binding_verified"]
    assert next(g for g in assess_study(rows,"mesh")["gates"] if g["name"]=="contact_controls_binding")["status"]=="failed"
