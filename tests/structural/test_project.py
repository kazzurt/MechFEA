import json
import pytest

from structural.project import StructuralProject
from structural.validation import assess_assembly, assess_mesh_study


def test_saved_project_reproduces_fingerprint_and_physics(tmp_path):
    project = StructuralProject()
    project.loads.screw_preloads_n = [90, 100, 110]
    project.contacts.friction_coefficient = .15
    project.save(tmp_path / "test.json")
    loaded = StructuralProject.load(tmp_path / "test.json")
    assert loaded.fingerprint() == project.fingerprint()
    assert loaded.loads.screw_preloads_n == [90, 100, 110]
    assert loaded.half_length_mm == 233.68
    assert loaded.readiness_issues()


def test_optional_bore_resolution_preserves_legacy_serialization(tmp_path):
    import hashlib
    from dataclasses import asdict
    project=StructuralProject()
    legacy=asdict(project);del legacy['mesh']['bore_snout_size_mm'];del legacy['mesh']['slot_size_mm'];del legacy['geometry_revision']
    expected=hashlib.sha256(json.dumps(legacy,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    path=tmp_path/'legacy-project.json';path.write_text(json.dumps(legacy))
    loaded=StructuralProject.load(path)
    assert loaded.mesh.bore_snout_size_mm is None
    assert loaded.to_dict()==legacy and loaded.fingerprint()==expected
    loaded.mesh.bore_snout_size_mm=1.5
    loaded.save(path)
    assert StructuralProject.load(path).mesh.bore_snout_size_mm==1.5
    assert loaded.fingerprint()!=expected


def test_derived_geometry_identity_survives_save_without_changing_source(tmp_path):
    from structural.studies import physical_model_fingerprint
    project=StructuralProject()
    original=project.fingerprint();original_physics=physical_model_fingerprint(project)
    project.geometry_revision={"schema_version":1,"kind":"derived_seat_trim_intersection","definition_sha256":"a"*64}
    project.save(tmp_path/"derived-project.json")
    loaded=StructuralProject.load(tmp_path/"derived-project.json")
    assert loaded.geometry_revision==project.geometry_revision
    assert loaded.source_sha256==StructuralProject().source_sha256
    assert loaded.fingerprint()!=original
    assert physical_model_fingerprint(loaded)!=original_physics
    assert any("geometry revision" in issue for issue in loaded.readiness_issues())
    loaded.preparation["geometry_revision_verification"]={"verified":True,"geometry_revision":loaded.geometry_revision.copy()}
    assert not any("geometry revision" in issue for issue in loaded.readiness_issues())
    loaded.geometry_revision["definition_sha256"]="unidentified"
    with pytest.raises(ValueError,match="Derived geometry"):
        loaded.validate()


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf'),True])
def test_invalid_bore_resolution_rejected(value):
    project=StructuralProject();project.mesh.bore_snout_size_mm=value
    with pytest.raises(ValueError,match='bore/snout'):
        project.validate()


@pytest.mark.parametrize("mutation", [
    lambda p: setattr(p, "half_length_mm", 467.36),
    lambda p: setattr(p.loads, "screw_preloads_n", [50, 50]),
    lambda p: setattr(p.loads, "lock_tightening", False),
    lambda p: setattr(p.loads, "direction_reference", [0, 2, 0]),
    lambda p: p.units.update(stress="Pa"),
    lambda p: p.contacts.final_artificial_supports.append("temporary spring"),
    lambda p: setattr(p.pipe, "youngs_modulus_mpa", float("nan")),
    lambda p: p.solver.update(timeout_s=float("inf")),
    lambda p: p.solver.update(threads=0),
    lambda p: p.solver.update(step_controls={"preload":{"initial_increment":2}}),
])
def test_bad_physical_configuration_rejected(mutation):
    p = StructuralProject()
    mutation(p)
    with pytest.raises(ValueError):
        p.validate()


def test_empty_diagnostics_and_failed_solver_cannot_be_accepted():
    assert assess_assembly({}, "completed")["status"] == "unaccepted"
    metrics = dict(model_preparation_verified=True, force_imbalance_fraction=0,
        moment_imbalance_fraction=0, max_preload_relative_error=0, max_tightening_drift_mm=0,
        screw_forces_free_to_redistribute=True, no_final_artificial_support=True,
        max_physical_penetration_mm=0, open_interface_force_fraction=0, compression_only_contact=True)
    assert assess_assembly(metrics, "failed")["status"] == "unaccepted"
    assert assess_assembly(metrics, "completed")["status"] == "numerically_accepted"
    metrics["max_physical_penetration_mm"] = float("nan")
    assert assess_assembly(metrics, "completed")["status"] == "unaccepted"


@pytest.mark.parametrize("key,value",[("patch_width_convention","arc"),("patch_center_angle_deg",90.),("patch_axial_length_mm",50.)])
def test_unsupported_footprint_cannot_be_prepared(key,value):
    project=StructuralProject()
    setattr(project.loads,key,value)
    assert any("patch" in issue for issue in project.readiness_issues())


def test_study_does_not_pass_with_changed_physics_or_missing_metrics():
    runs = [dict(acceptance_status="numerically_accepted", physical_model_fingerprint="same", thickness_size_mm=h) for h in (.75, .5, .375)]
    assert assess_mesh_study(runs)["status"] == "unaccepted"
    for run in runs:
        run.update(max_displacement_mm=.49, bolt_force_n=[100, 110, 120], contact_force_n=[1000, 200], seat_region_stress_mpa=100, slot_region_stress_mpa=200)
    assert assess_mesh_study(runs)["status"] == "passed"
    runs[-1]["physical_model_fingerprint"] = "different"
    assert assess_mesh_study(runs)["status"] == "unaccepted"
