import importlib.util
import json
from pathlib import Path

import pytest

from structural.project import sha256


@pytest.fixture
def recipe():
    path=Path(__file__).resolve().parents[2]/"tools/assess_spatial_release.py"
    spec=importlib.util.spec_from_file_location("spatial_release_recipe",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def retained_sources(tmp_path):
    sources={}
    for name in ("results","project","mesh","run"):
        path=tmp_path/(name+".fixture");path.write_text(name)
        sources[name]={"path":str(path),"sha256":sha256(path)}
    return sources


def test_imported_study_requires_unchanged_retained_sources(recipe,tmp_path):
    sources=retained_sources(tmp_path);report={"cases":[{"source_files":sources}]}
    assert recipe.retained_records(report)==[{key:sources[key]["path"] for key in ("results","project","mesh")}]
    Path(sources["mesh"]["path"]).write_text("changed coordinates")
    with pytest.raises(ValueError,match="no longer matches"):recipe.retained_records(report)
    with pytest.raises(ValueError,match="requires retained"):recipe.retained_records({"cases":[{"numerically_accepted":True}]})


@pytest.mark.parametrize("mutate",[False,True])
@pytest.mark.parametrize("with_slot",[False,True])
def test_recipe_reassesses_each_family_and_candidate_without_solving(recipe,tmp_path,monkeypatch,mutate,with_slot):
    sources=retained_sources(tmp_path);paths=[];calls=[]
    kinds=["mesh","contact_mesh","bore_contact_mesh"]+(["slot_mesh"] if with_slot else [])
    for kind in kinds:
        path=tmp_path/(kind+".json");path.write_text(json.dumps({"study":kind,"cases":[{"source_files":sources}]}));paths.append(path)
    def assess(records,kind):
        assert records==[{key:sources[key]["path"] for key in ("results","project","mesh")}]
        calls.append(kind);return {"study":kind,"status":"passed","fresh":True}
    def summarize(path):
        assert Path(path)==Path(sources["results"]["path"])
        calls.append("candidate");return {"source_files":sources,"fresh":True}
    def release(wall,seat,bore,candidate,*,slot_mesh_study=None):
        assert all(row["fresh"] for row in (wall,seat,bore,candidate))
        assert (slot_mesh_study is not None)==with_slot
        if with_slot:assert slot_mesh_study["fresh"] and slot_mesh_study["study"]=="slot_mesh"
        if mutate:Path(sources["run"]["path"]).write_text("changed run")
        return {"release_mesh_accepted":False,"independent_spatial_studies_accepted":True,
                "combined_finest_configuration_verified":False,"issues":["candidate unaccepted"]}
    monkeypatch.setattr(recipe,"assess_existing_runs",assess)
    monkeypatch.setattr(recipe,"summarize_run",summarize)
    monkeypatch.setattr(recipe,"assess_mesh_release",release)
    output=tmp_path/"fresh-spatial-review"
    args=[*map(str,paths[:3]),"--candidate-results",sources["results"]["path"],"--output",str(output)]
    if with_slot:args.extend(["--slot-study",str(paths[3])])
    if mutate:
        with pytest.raises(ValueError,match="changed while reassessing"):recipe.main(args)
        assert not (output/"spatial-release-assessment.json").exists()
    else:assert recipe.main(args)==2
    receipt=json.loads((output/"assessment-manifest.json").read_text())
    assert receipt["solver_invoked"] is False
    assert receipt["status"]==("failed" if mutate else "completed")
    assert calls==[*kinds,"candidate"]
    with pytest.raises(FileExistsError):recipe.main(args)
