"""Loading a mesh preserves pending inputs and commits only verified candidates."""
import copy
import json
from types import SimpleNamespace

import pytest

from structural.project import StructuralProject


class ValueField:
    def __init__(self,value):self.current=value
    def value(self):return self.current
    def setValue(self,value):self.current=value


class TextField:
    def __init__(self,text):self.current=text
    def text(self):return self.current
    def setText(self,text):self.current=text
    def setPlainText(self,text):self.current=text


class Selection:
    def __init__(self):self.items=["All components","old_surface"];self.blocked=False
    def blockSignals(self,blocked):self.blocked=blocked
    def clear(self):self.items=[]
    def addItem(self,item):self.items.append(item)
    def addItems(self,items):self.items.extend(items)


@pytest.fixture
def mesh_window(tmp_path,monkeypatch):
    from structural import gui
    project=StructuralProject()
    fields={key:ValueField(value) for key,value in {
        "pipe_e":205000.,"pipe_nu":.29,"screw_e":201000.,"screw_nu":.291,
        "force":1112.0554,"friction":0.,"stiffness":100000.,"far":4.,"thickness":.75,"contact":.75,"bore":0.,"slot":.75,
        "preload_1":125.,"preload_2":85.,"preload_3":110.}.items()}
    events=[];messages=[]
    window=SimpleNamespace(project=project,fields=fields,patch=SimpleNamespace(currentIndex=lambda:0),
        direction=TextField("0, 1, 0"),node_limit=ValueField(500000),
        mesh=object(),mesh_path=tmp_path/"old-verified-mesh.npz",mesh_summary=TextField("Old verified mesh"),
        selection=Selection(),message=lambda exc:messages.append(str(exc)),
        refresh_readiness=lambda:events.append("readiness"),show_mesh=lambda:events.append("render"))
    window.apply_fields=lambda:gui.StructuralWindow.apply_fields(window)
    def fill_fields():
        events.append("fill")
        for index,value in enumerate(window.project.loads.screw_preloads_n,1):
            window.fields[f"preload_{index}"].setValue(value)
        window.fields["slot"].setValue(window.project.mesh.slot_size_mm or 0.)
    window.fill_fields=fill_fields
    evidence=tmp_path/"structural-fea-validation-deliverables/backend-feasibility.json"
    evidence.parent.mkdir();evidence.write_text(json.dumps({"evidence":"fixture placeholder"}))
    monkeypatch.setattr(gui,"DODEC",tmp_path)
    candidate=SimpleNamespace(node_ids=[1,2,3],element_ids=[1],surfaces={"new_surface":[(1,1)]},
                              provenance={"patch_width_mm":35.})
    monkeypatch.setattr("structural.mesh.SolidMesh.load",lambda path:candidate)
    return window,candidate,events,messages


def test_loading_mesh_preserves_unsaved_preload_edits(mesh_window,tmp_path,monkeypatch):
    from structural import gui
    window,candidate,events,messages=mesh_window
    assert window.project.loads.screw_preloads_n==[100.,100.,100.]
    def bind(project,mesh,*,backend_fixture_report):
        assert project.loads.screw_preloads_n==[125.,85.,110.]
        assert project.mesh.slot_size_mm==.75
        assert mesh is candidate
        bound=copy.deepcopy(project);bound.preparation["bound_candidate"]=True
        bound.mesh.slot_size_mm=None  # Adopt the actual loaded mesh's optional target.
        return bound
    monkeypatch.setattr("structural.assembly.bind_mesh_evidence",bind)
    selected=tmp_path/"new-verified-mesh.npz"
    gui.StructuralWindow.load_mesh(window,selected)
    assert window.mesh is candidate and window.mesh_path==selected
    assert window.project.preparation["bound_candidate"]
    assert window.project.loads.screw_preloads_n==[125.,85.,110.]
    assert window.project.mesh.slot_size_mm is None and window.fields["slot"].value()==0.
    assert [window.fields[f"preload_{i}"].value() for i in (1,2,3)]==[125.,85.,110.]
    assert window.selection.items==["All components","new_surface"]
    assert not window.selection.blocked and events==["fill","readiness","render"]
    assert not messages


def test_rejected_candidate_keeps_previous_verified_mesh_and_pending_edits(mesh_window,tmp_path,monkeypatch):
    from structural import gui
    window,candidate,events,messages=mesh_window
    old_mesh,old_path,old_project=window.mesh,window.mesh_path,window.project
    def reject(project,mesh,*,backend_fixture_report):
        assert project.loads.screw_preloads_n==[125.,85.,110.]
        assert project.mesh.slot_size_mm==.75
        assert mesh is candidate
        raise ValueError("Assembly mesh minimum scaled Jacobian is below 0.1")
    monkeypatch.setattr("structural.assembly.bind_mesh_evidence",reject)
    gui.StructuralWindow.load_mesh(window,tmp_path/"rejected-poor-positive-mesh.npz")
    assert window.mesh is old_mesh and window.mesh_path==old_path and window.project is old_project
    assert window.project.loads.screw_preloads_n==[125.,85.,110.]
    assert window.project.mesh.slot_size_mm==.75 and window.fields["slot"].value()==.75
    assert window.mesh_summary.current=="Old verified mesh"
    assert window.selection.items==["All components","old_surface"] and not window.selection.blocked
    assert events==["readiness"]
    assert messages==["Assembly mesh minimum scaled Jacobian is below 0.1"]
