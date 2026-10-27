"""Assembly preparation must reject poor-positive native high-order meshes."""
import json
from types import SimpleNamespace

import pytest

from structural.assembly import bind_mesh_evidence
from structural.project import StructuralProject, sha256


def binding_inputs(tmp_path, monkeypatch, quality):
    project = StructuralProject()
    path = tmp_path / "mesh-manifest.json"
    path.write_text(json.dumps({"source_sha256":project.source_sha256,
        "counts":{"nodes":10,"elements":1}, "mesh_quality":quality,
        "settings":{"far_size_mm":3., "contact_size_mm":.5,
                    "thickness_size_mm":.75,"order":2,"threads":2}}))
    mesh = SimpleNamespace(node_ids=list(range(1,11)),element_ids=[1],element_sets={},surfaces={},
        provenance={"source_sha256":project.source_sha256,"coordinate_frame":"comsol_reference_mm",
                    "mesh_manifest":str(path),"cuts_verified":True,"contact_clearances_verified":True})
    # Isolate binding policy; backend proof and geometric selection checks have
    # their own numerical/structural tests and are not bypassed in production.
    monkeypatch.setattr("structural.backends.feasibility.verify_backend_feasibility",lambda *a,**k:{"verified":True})
    monkeypatch.setattr("structural.assembly.required_mesh_selections",lambda mesh:None)
    return project,mesh,path


@pytest.mark.parametrize("minimum",[7.769110072231727e-5,.099999,-.1,None,float("nan"),float("inf"),True])
def test_binding_rejects_poor_positive_missing_or_nonfinite_native_quality(tmp_path,monkeypatch,minimum):
    project,mesh,_=binding_inputs(tmp_path,monkeypatch,{"min_scaled_jacobian":minimum,"nonpositive_elements":0})
    with pytest.raises(ValueError,match="minimum scaled Jacobian >= 0.1"):
        bind_mesh_evidence(project,mesh,backend_fixture_report={})
    assert "mesh_quality_binding" not in project.preparation


def test_binding_rejects_inconsistent_nonpositive_count(tmp_path,monkeypatch):
    project,mesh,_=binding_inputs(tmp_path,monkeypatch,{"min_scaled_jacobian":.2,"nonpositive_elements":1})
    with pytest.raises(ValueError,match="zero nonpositive"):
        bind_mesh_evidence(project,mesh,backend_fixture_report={})


@pytest.mark.parametrize("minimum",[.1,.10265138031642053])
def test_binding_retains_quality_evidence_and_actual_mesh_settings(tmp_path,monkeypatch,minimum):
    project,mesh,path=binding_inputs(tmp_path,monkeypatch,{"min_scaled_jacobian":minimum,"nonpositive_elements":0})
    bound=bind_mesh_evidence(project,mesh,backend_fixture_report={})
    quality=bound.preparation["mesh_quality_binding"]
    assert quality["passed"] and quality["minimum_scaled_jacobian"]==minimum
    assert quality["manifest"]["sha256"]==sha256(path)
    assert bound.mesh.contact_size_mm==.5
    assert bound.preparation["mesh_settings_binding"]["changes"]["contact_size_mm"]=={"requested":.75,"actual":.5}
    assert project.mesh.contact_size_mm==.75
    assert 'slot_size_mm' not in bound.preparation['mesh_settings_binding']['requested']
    assert 'slot_size_mm' not in bound.preparation['mesh_settings_binding']['actual']


@pytest.mark.parametrize('actual',[None,.75,1.5])
def test_binding_restores_actual_optional_slot_setting(tmp_path,monkeypatch,actual):
    project,mesh,path=binding_inputs(tmp_path,monkeypatch,{"min_scaled_jacobian":.2,"nonpositive_elements":0})
    project.mesh.slot_size_mm=2.
    manifest=json.loads(path.read_text())
    if actual is not None: manifest['settings']['slot_size_mm']=actual
    path.write_text(json.dumps(manifest))
    bound=bind_mesh_evidence(project,mesh,backend_fixture_report={})
    assert bound.mesh.slot_size_mm==actual
    assert project.mesh.slot_size_mm==2.
    assert bound.preparation['mesh_settings_binding']['changes']['slot_size_mm']=={'requested':2.,'actual':actual}
