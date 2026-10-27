import json
from pathlib import Path
import numpy as np
import pytest

from structural.backends.calculix import parse_dat, parse_frd
from structural.mesh import SolidMesh
from structural.results import (component_result, contact_diagnostics, contact_pair_statistics,
    REQUIRED_RESULT_GATE_NAMES, assembly_mpc_support_transfers, assembly_support_dofs, nodal_dat_values, physical_equilibrium, pretension_history, postprocess_assembly, read_component_vtu, report_is_accepted, result_display_field)


def test_equilibrium_projects_symmetry_and_uses_current_moment():
    # The loaded symmetry node has RF_y=CLOAD_y, which is not a support reaction.
    p={1:np.array([0.,0,0]),2:np.array([1.,0,0])}
    u={1:np.zeros(3),2:np.array([0.,.5,0])}
    rf={1:np.array([-2.,-3,0]),2:np.array([2.,3,0])}
    result=physical_equilibrium(p,u,rf,support_dofs={1:[1,2,3],2:[3]},
        applied_nodal_forces={2:np.array([2.,3,0])},characteristic_force_n=10,characteristic_length_mm=2)
    np.testing.assert_allclose(result["force_residual_N"],[0,0,0])
    np.testing.assert_allclose(result["moment_residual_undeformed_Nmm"],[0,0,3])
    np.testing.assert_allclose(result["moment_residual_current_Nmm"],[0,0,2])
    assert result["force_imbalance_fraction"]==0
    assert result["moment_imbalance_fraction"]==pytest.approx(.1)


def test_missing_reaction_does_not_pass_equilibrium():
    result=physical_equilibrium({1:[0,0,0]},{1:[0,0,0]}, {}, support_dofs={1:[1]},
        applied_nodal_forces={},characteristic_force_n=100,characteristic_length_mm=200)
    assert result["status"]=="missing" and result["force_imbalance_fraction"] is None
    assert result["missing"]["support_rf"]=={"count":1,"first_node_ids":[1]}


def test_first_real_dat_table_after_step_preamble_is_available(tmp_path):
    path=tmp_path/"first-table.dat"
    path.write_text(" S T E P  1\n INCREMENT  1\n\n displacements (vx,vy,vz) for set ALL_NODES and time  0.1000000E+01\n\n         1  1.000000E-03 2.000000E-03 3.000000E-03\n",encoding="utf-8")
    blocks=parse_dat(path)
    np.testing.assert_allclose(nodal_dat_values(blocks,1.,"displacements","ALL_NODES")[1],[.001,.002,.003])


def test_fixed_master_anchorage_reactions_enter_physical_equilibrium():
    metadata={"hub_fixed_nodes":[1],"midpoint_nodes":[3],
              "rigid_anchorage_nodes":{"screw_1":[2]},"anchorage_enforcement":"fixed_master_elimination"}
    supports=assembly_support_dofs(metadata)
    assert supports=={1:[1,2,3],2:[1,2,3],3:[3]}
    points={1:[0,0,-250],2:[0,0,-217.88],3:[0,1,-217.88]}
    u={node:np.zeros(3) for node in points}
    result=physical_equilibrium(points,u,{1:[0,0,0],2:[0,-100,0],3:[0,100,0]},
        support_dofs=supports,applied_nodal_forces={3:[0,100,0]},
        characteristic_force_n=100,characteristic_length_mm=233.68)
    np.testing.assert_allclose(result["support_resultant_N"],[0,-100,0])
    np.testing.assert_allclose(result["force_residual_N"],[0,0,0])
    np.testing.assert_allclose(result["moment_residual_current_Nmm"],[0,0,0])
    metadata["anchorage_enforcement"]="unverified"
    with pytest.raises(ValueError,match="fixed-master"):assembly_support_dofs(metadata)


def test_explicit_cut_dependent_rf_condenses_to_existing_anchor_without_q_force():
    metadata={"pretension_formulation":"explicit paired-node axial jump with scalar work-conjugate force",
              "pretension_cut_node_maps":{"1":{"1":2,"4":5}}}
    transfers=assembly_mpc_support_transfers(metadata,{1:[1,2,3]})
    assert transfers=={2:1}  # Free originals are not made into physical supports.
    points={1:[1,0,0],2:[1,0,0],3:[1,0,0]}
    u={1:[0,0,0],2:[.2,0,0],3:[0,0,0]}
    rf={1:[0,-6,0],2:[0,-4,0],3:[0,10,0],99:[1000,0,0]}
    kwargs=dict(support_dofs={1:[1,2,3]},applied_nodal_forces={3:[0,10,0]},
                characteristic_force_n=10,characteristic_length_mm=2,mpc_support_transfers=transfers)
    result=physical_equilibrium(points,u,rf,**kwargs)
    assert result["force_imbalance_fraction"]==0
    np.testing.assert_allclose(result["moment_residual_current_Nmm"],[0,0,0])
    assert result["uncondensed_support_balance"]["force_residual_N"]==[0,4,0]
    assert not result["mpc_reaction_condensation"]["generalized_reference_force_added"]
    # Transpose-Jacobian transfer preserves virtual work for original-node motion.
    delta=np.array([.03,.04,.05])
    assert np.dot(np.array(rf[1])+rf[2],delta)==pytest.approx(np.dot(rf[1],delta)+np.dot(rf[2],delta))
    del rf[2]
    missing=physical_equilibrium(points,u,rf,**kwargs)
    assert missing["status"]=="missing" and missing["force_imbalance_fraction"] is None
    assert missing["missing"]["dependent_mpc_rf"]["first_node_ids"]==[2]


def _node_block(kind,time,value):
    return {"header":kind,"time":time,"set":"PRETENSION_BOLT_1","rows":[[101,value,0,0]]}


def test_pretension_retention_uses_direct_scalar_and_all_service_increments():
    blocks=[_node_block(kind,time,value) for kind,time,value in
        [("displacements",1.,.02),("forces",1.,100.),("displacements",1.5,.0201),
         ("forces",1.5,130.),("displacements",2.,.02),("forces",2.,150.)]]
    result=pretension_history(blocks,[{"node_set":"PRETENSION_BOLT_1","reference_node":101,"target_n":100.}])
    assert result["max_preload_relative_error"]==0
    assert result["max_tightening_drift_mm"]==pytest.approx(.0001)
    assert result["screws"]["PRETENSION_BOLT_1"]["force_change_N"]==50
    result=pretension_history(blocks[:-1],[{"node_set":"PRETENSION_BOLT_1","reference_node":101,"target_n":100.}])
    assert result["max_tightening_drift_mm"] is None


def test_zero_preload_uses_declared_absolute_force_floor():
    blocks=[_node_block(kind,time,value) for kind,time,value in
            [("displacements",1.,0),("forces",1.,.005),("displacements",2.,0),("forces",2.,30.)]]
    result=pretension_history(blocks,[{"node_set":"PRETENSION_BOLT_1","reference_node":101,"target_n":0.}])
    screw=result["screws"]["PRETENSION_BOLT_1"]
    assert screw["preload_error_normalizer_N"]==1.
    assert screw["preload_force_tolerance_N"]==.01
    assert result["max_preload_relative_error"]==.005
    assert screw["force_change_N"]==pytest.approx(29.995)


def _contact_blocks(gaps=(-.001,),pressures=(100.,)):
    blocks=[{"header":"relative contact displacement", "time":2.,"set":None,
             "rows":[[10,1,gap,0,0] for gap in gaps]},
            {"header":"contact stress","time":2.,"set":None,
             "rows":[[10,1,pressure,0,0] for pressure in pressures]}]
    for name in ("","normal "):
        blocks.extend([{"header":f"statistics for slave set SEAT, master set CONE and time 2 total {name}surface force", "time":2.,"set":None,"rows":[[0,100,0,1,2,3]]},
                       {"header":"area, normal force (+ = tension) and shear force (size)","time":2.,"set":None,"rows":[[1,-100,0]]}])
    return blocks


def test_contact_sign_and_pair_force_are_not_confused():
    blocks=_contact_blocks()
    stats=contact_pair_statistics(blocks,2.)
    assert stats[("SEAT","CONE")]["CFN"]["signed_normal_force_N"]==-100
    result=contact_diagnostics(blocks,2.,[{"slave":"SEAT","master":"CONE"}],{"seat":[(10,1)]},
                               sign_verified=True,physical_gap_verified=True)
    assert result["max_physical_penetration_mm"]==.001
    assert result["compression_only_contact"] is True
    assert result["open_interface_force_fraction"]==0
    assert result["samples"][0]["raw_CDIS_normal"]==-.001
    unknown=contact_diagnostics(blocks,2.,[{"slave":"SEAT","master":"CONE"}],{"seat":[(10,1)]})
    assert unknown["max_physical_penetration_mm"] is None
    assert unknown["compression_only_contact"] is None


def test_zero_area_contact_preserves_undefined_normal_but_identifies_inactive_pair():
    blocks=_contact_blocks((),())
    for force_index,area_index in ((2,3),(4,5)):
        blocks[force_index]["rows"]=[[0.,0.,0.,0.,0.,0.]]
        blocks[area_index]["rows"]=[[0.,float("nan"),float("nan")]]
    arguments=dict(sign_verified=True,physical_gap_verified=True)
    result=contact_diagnostics(blocks,2.,[{"slave":"SEAT","master":"CONE"}],{"seat":[(10,1)]},**arguments)
    assert result["status"]=="available" and result["compression_only_contact"] is True
    assert result["max_physical_penetration_mm"]==0 and result["open_interface_force_fraction"]==0
    pair=result["pairs"][0]
    assert pair["zero_area_inactive_pair"] and pair["gap_range_mm"] is None
    assert np.isnan(pair["statistics"]["CFN"]["signed_normal_force_N"])
    # Undefined scalar force without complete zero-area evidence stays missing.
    blocks[5]["rows"]=[[1.,float("nan"),float("nan")]]
    result=contact_diagnostics(blocks,2.,[{"slave":"SEAT","master":"CONE"}],{"seat":[(10,1)]},**arguments)
    assert result["status"]=="missing" and result["compression_only_contact"] is None
    assert result["missing_reasons"]
    blocks[5]["rows"]=[[0.,float("nan"),float("nan")]]
    blocks[2]["rows"][0][1]=1.
    result=contact_diagnostics(blocks,2.,[{"slave":"SEAT","master":"CONE"}],{"seat":[(10,1)]},**arguments)
    assert result["status"]=="missing"  # No arbitrary nonzero vector becomes zero.


def test_mixed_open_contact_tail_force_is_missing_without_quadrature_weights():
    result=contact_diagnostics(_contact_blocks((-.001,.001),(100.,.01)),2.,
        [{"slave":"SEAT","master":"CONE"}],{"SEAT":[(10,1)]},sign_verified=True,physical_gap_verified=True)
    assert result["open_interface_force_fraction"] is None


def test_open_tangential_traction_is_included_even_when_normal_pressure_is_zero():
    blocks=_contact_blocks((.001,),(0.,))
    blocks[1]["rows"][0][3]=100.
    blocks[3]["rows"][0]=[1,0,100.]
    blocks[5]["rows"][0]=[1,0,0.]
    result=contact_diagnostics(blocks,2.,[{"slave":"SEAT","master":"CONE"}],{"SEAT":[(10,1)]},
                               sign_verified=True,physical_gap_verified=True,characteristic_force_n=100.)
    assert result["open_interface_force_fraction"]==1.
    assert result["pairs"][0]["max_open_traction_magnitude_MPa"]==100.
    # A zero integrated resultant does not erase nonzero local shear evidence.
    blocks[2]["rows"][0][:3]=[0,0,0];blocks[3]["rows"][0]=[1,0,0]
    result=contact_diagnostics(blocks,2.,[{"slave":"SEAT","master":"CONE"}],{"SEAT":[(10,1)]},
                               sign_verified=True,physical_gap_verified=True,characteristic_force_n=100.)
    assert result["open_interface_force_fraction"] is None


def _tetra_fixture():
    corners=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    points=np.vstack([corners,*[(corners[a]+corners[b])/2 for a,b in ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))]])
    mesh=SolidMesh(np.arange(1,11),points,[1],[list(range(1,11))],element_sets={"pipe":[1]})
    native_ids=list(range(11,21)) # Simulate native-cut duplicated solver nodes.
    frd={"nodes":dict(zip(native_ids,points)),"elements":{1:{"type":6,"nodes":native_ids}},
         "fields":[{"name":"DISP","time":2.,"components":["D1","D2","D3"],"values":{i:list(p*.01) for i,p in zip(native_ids,points)}},
                   {"name":"STRESS","time":2.,"components":["SXX","SYY","SZZ","SXY","SYZ","SZX"],
                    "values":{i:[1,2,3,4,5,6] for i in native_ids}}]}
    return mesh,frd


def test_tetra10_export_preserves_native_cut_ids_stress_order_and_volume(tmp_path):
    import meshio
    mesh,frd=_tetra_fixture()
    component=component_result(mesh,frd,2.,"pipe")
    assert component.node_ids.tolist()==list(range(11,21))
    np.testing.assert_allclose(component.stress_mpa[0],[[1,4,6],[4,2,5],[6,5,3]])
    linear=component.field_mesh();q=linear.points_mm[linear.tetrahedra]
    volume=np.abs(np.linalg.det(q[:,1:]-q[:,:1])).sum()/6
    assert len(linear.tetrahedra)==8 and volume==pytest.approx(1/6)
    path=tmp_path/"pipe.vtu";component.export_vtu(path)
    reread=meshio.read(path)
    assert reread.cells[0].type=="tetra10"
    np.testing.assert_array_equal(reread.point_data["solver_node_id"],np.arange(11,21))
    restored=read_component_vtu(path,"pipe",time=2.)
    np.testing.assert_allclose(restored.stress_mpa,component.stress_mpa)
    np.testing.assert_array_equal(restored.node_ids,component.node_ids)


def test_failed_assembly_outputs_remain_diagnostic_and_missing(tmp_path):
    mesh,_=_tetra_fixture()
    path=tmp_path/"failed.inp";path.write_text("*HEADING\nfailed test",encoding="utf-8")
    path.with_suffix(".run.json").write_text(json.dumps({"status":"failed"}),encoding="utf-8")
    path.with_suffix(".frd").write_text("    2C\n -1         1broken\n",encoding="utf-8")
    metadata={"pretension":[{"node_set":"PRETENSION_BOLT_1","reference_node":101,"target_n":100.}],
              "hub_fixed_nodes":[],"midpoint_nodes":[1,2,3],"applied_nodal_forces":{"1":[0,100,0]},
              "contact_pairs":[{"slave":"SEAT","master":"CONE"}],"expected_step_times":[1,1]}
    path.with_suffix(".assembly.json").write_text(json.dumps(metadata),encoding="utf-8")
    result=postprocess_assembly(mesh,path)
    assert result["diagnostic_only"] and not result["accepted"]
    assert result["metrics"]["force_imbalance_fraction"] is None
    assert result["metrics"]["no_final_artificial_support"] is None
    assert len(result["reader_errors"])==1
    assert set(result["implementation_provenance"])=={"results","comparison","backend_calculix","validation"}
    assert all(len(row["source_sha256_at_import"])==64 and len(row["loaded_function_code_sha256"])==64 and row["python_version"] for row in result["implementation_provenance"].values())
    assert (tmp_path/"structural-results/results.json").is_file()


def test_postprocessor_pins_actual_exported_component_bytes(tmp_path,monkeypatch):
    from structural import results
    from structural.project import sha256
    mesh,frd=_tetra_fixture()
    path=tmp_path/"fields.inp";path.write_text("*HEADING\nexport fixture")
    path.with_suffix(".frd").write_text("reader fixture")
    path.with_suffix(".run.json").write_text(json.dumps({"status":"failed"}))
    path.with_suffix(".assembly.json").write_text(json.dumps({"pretension":[],"hub_fixed_nodes":[],"midpoint_nodes":[],
        "applied_nodal_forces":{},"contact_pairs":[],"expected_step_times":[1,1]}))
    monkeypatch.setattr(results,"parse_frd",lambda *args:frd)
    report=postprocess_assembly(mesh,path)
    field=report["stages"]["loaded"]["components"]["pipe"]
    assert field["status"]=="available"
    assert field["vtu_sha256"]==sha256(tmp_path/"structural-results"/field["vtu"])


def test_old_completed_run_with_omitted_tie_is_independently_rejected(tmp_path):
    mesh,_=_tetra_fixture()
    path=tmp_path/"omitted_tie.inp";path.write_text("*HEADING\nretained old run",encoding="utf-8")
    path.with_suffix(".run.json").write_text(json.dumps({"status":"completed"}),encoding="utf-8")
    path.with_suffix(".stdout.log").write_text("*WARNING in gentiedmpc: no tied MPC generated for node 3\n",encoding="utf-8")
    path.with_suffix(".stderr.log").write_text("",encoding="utf-8")
    metadata={"pretension":[],"hub_fixed_nodes":[],"midpoint_nodes":[],"applied_nodal_forces":{},
              "contact_pairs":[],"expected_step_times":[1,1]}
    path.with_suffix(".assembly.json").write_text(json.dumps(metadata),encoding="utf-8")
    report=postprocess_assembly(mesh,path)
    gate=next(g for g in report["assessment"]["gates"] if g["name"]=="solver_connection_warnings")
    assert gate["status"]=="failed"
    assert report["critical_solver_warnings"] and not report["accepted"]


def test_retained_frictional_run_without_qualification_gets_missing_seventeenth_gate(tmp_path):
    mesh,_=_tetra_fixture()
    path=tmp_path/"unqualified.inp";path.write_text("*FRICTION\n.2,10000\n",encoding="utf-8")
    path.with_suffix(".run.json").write_text(json.dumps({"status":"completed"}),encoding="utf-8")
    metadata={"pretension":[],"hub_fixed_nodes":[],"midpoint_nodes":[],"applied_nodal_forces":{},
              "contact_pairs":[],"expected_step_times":[1,1]}
    path.with_suffix(".assembly.json").write_text(json.dumps(metadata),encoding="utf-8")
    report=postprocess_assembly(mesh,path)
    assert len(report["assessment"]["gates"])==17
    gate=next(g for g in report["assessment"]["gates"] if g["name"]=="local_coulomb_bound")
    assert gate["status"]=="missing" and not report["accepted"]
    assert report["friction_required"] and report["friction_coefficient"]==.2


def test_actual_pinned_preload_contact_fixture_reader():
    root=Path(__file__).resolve().parents[2]
    source=root/"Dodec FEA/structural-fea-benchmark-runs/fixtures/verified-hex20-and-preloaded-contact/preload_contact/preload_contact.dat"
    if not source.is_file():pytest.skip("Optional retained solver evidence absent")
    blocks=parse_dat(source)
    result=pretension_history(blocks,[{"node_set":"PRETENSION","reference_node":1241,"target_n":100.}])
    assert result["max_preload_relative_error"]==0
    assert result["max_tightening_drift_mm"]==0
    stats=contact_pair_statistics(blocks,2.)[("SEAT","GRIPTOP")]
    assert stats["CF"]["force_on_slave_N"][0]==pytest.approx(55.65077)
    assert stats["CFN"]["signed_normal_force_N"]==pytest.approx(-55.65078)


def test_viewer_scalar_is_magnitude_and_rigid_hub_stress_is_hidden():
    field=result_display_field({"displacement_mm":np.array([[3.,4,0]])},"displacement")
    np.testing.assert_allclose(field["values"],[5.])
    assert field["name"]=="displacement_magnitude_mm"
    field=result_display_field({"physical_stress_valid":np.array([0]),"von_mises_MPa_nodal_display":np.array([np.nan])},"mises")
    assert field["name"] is None and "hub" in field["note"]


def test_consumers_do_not_accept_empty_gates_or_completion_alone():
    report={"accepted":True,"diagnostic_only":False,"solver_status":"completed",
            "assessment":{"status":"numerically_accepted","gates":[]}}
    assert not report_is_accepted(report)
    report["assessment"]["gates"]=[{"name":"unrelated","status":"passed"}]
    assert not report_is_accepted(report)
    report["assessment"]["gates"]=[{"name":name,"status":"passed"} for name in REQUIRED_RESULT_GATE_NAMES]
    assert report_is_accepted(report)
    report["assessment"]["gates"].append(report["assessment"]["gates"][0].copy())
    assert not report_is_accepted(report)
    report["assessment"]["gates"].pop()
    report["assessment"]["gates"][0]["status"]="missing"
    assert not report_is_accepted(report)


def test_actual_c3d10_native_order_and_display_stress_provenance():
    root=Path(__file__).resolve().parents[2]
    folder=root/"Dodec FEA/structural-fea-benchmark-runs/fixtures/results-c3d10-affine-patch-20261018T051346Z-0b51f3"
    if not folder.is_dir():pytest.skip("Optional native tetra10 result evidence absent")
    mesh=SolidMesh.load(folder/"affine-patch-mesh.npz")
    frd=parse_frd(folder/"affine_patch.frd")
    assert frd["elements"][1]["type"]==6
    component=component_result(mesh,frd,1.,"pipe",solver_status="completed")
    np.testing.assert_allclose(component.displacement_mm,component.points_mm*np.array([.001,-.00025,-.00025]))
    np.testing.assert_allclose(component.stress_mpa[:,0,0],199.998)
    raw=[row for block in parse_dat(folder/"affine_patch.dat") if block["header"].startswith("stresses") for row in block["rows"]]
    np.testing.assert_allclose([row[2] for row in raw],np.full(4,200.))
