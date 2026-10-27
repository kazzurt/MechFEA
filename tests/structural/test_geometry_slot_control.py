import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from structural.project import StructuralProject, ROOT


def test_optional_slot_round_trip_and_legacy_identity(tmp_path):
    project = StructuralProject()
    baseline = project.fingerprint()
    assert 'slot_size_mm' not in project.to_dict()['mesh']
    project.mesh.slot_size_mm = .75
    path = tmp_path/'slot-project.json'
    project.save(path)
    loaded = StructuralProject.load(path)
    assert loaded.mesh.slot_size_mm == .75
    assert loaded.fingerprint() == project.fingerprint() != baseline
    loaded.mesh.slot_size_mm = None
    assert loaded.fingerprint() == baseline


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),True,'small'])
def test_invalid_slot_target(value):
    project = StructuralProject()
    project.mesh.slot_size_mm = value
    with pytest.raises(ValueError,match='slot'):
        project.validate()


def test_quadratic_arc_weights_and_nonuniform_distribution():
    folder = ROOT/'tools'
    sys.path.insert(0,str(folder))
    try:
        spec = importlib.util.spec_from_file_location('slot_weighted_recipe',folder/'audit_slot_weighted_resolution.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(folder))
    straight = np.array([[[0,0,0],[2,0,0],[1,0,0]]],float)
    assert module.arc_lengths(straight)[0] == pytest.approx(2)
    curved = straight.copy(); curved[0,2,1] = .5
    assert module.arc_lengths(curved)[0] > 2
    assert module.arc_lengths(curved[:,[1,0,2]])[0] == pytest.approx(module.arc_lengths(curved)[0])
    # Many small edges cannot dominate a long physical part of the trace merely by count.
    values = np.array([1.,1.,1.,4.]); weights = np.array([1.,1.,1.,12.])
    assert module.weighted_distribution(values,weights)['median'] > np.median(values)
    assert len(module.CORNER_FACES) == 3
    assert sum(len(v)*2 for v in module.CORNER_FACES.values())+6 == 30
