import numpy as np
import pytest
from structural.mesh import SolidMesh, TET10_EDGES
from structural.backends.explicit_pretension import explicit_pretension


def shoulder_like_mesh():
    coordinates={}
    nodes=[]
    cells=[]
    def node(point):
        key=tuple(point)
        if key not in coordinates:
            coordinates[key]=len(nodes)+1
            nodes.append(point)
        return coordinates[key]
    for q in ([(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
              [(0,0,0),(0,1,0),(1,0,0),(0,0,-1)],
              [(0,0,0),(1,0,0),(1,-1,-.4),(0,-1,-1)]):
        q=np.asarray(q,dtype=float)
        if np.linalg.det(np.stack((q[1]-q[0],q[2]-q[0],q[3]-q[0]),axis=-1))<0:
            q[[1,2]]=q[[2,1]]
        cells.append([node(p) for p in q]+[node((q[a]+q[b])/2) for a,b in TET10_EDGES])
    mesh=SolidMesh(np.arange(1,len(nodes)+1),nodes,[1,2,3],cells,element_sets={"bolt":[1,2,3]},surfaces={"cut":[(2,1)]})
    mesh.validate()
    return mesh


def test_explicit_cut_replaces_edge_only_head_neighbours_and_preserves_shaft():
    mesh=shoulder_like_mesh()
    original=mesh.tetra10.copy()
    reference=int(mesh.node_ids.max())+1
    reserved=[reference,reference+1,reference+2]
    new,equations,mapping=explicit_pretension(mesh,"cut",[2,3],[0,0,1],reference,reserved_node_ids=reserved)
    assert min(mapping.values())>max(reserved)
    assert np.array_equal(mesh.tetra10,original)
    assert np.array_equal(new.tetra10[0],original[0])
    # Element 3 has no cut face; its shared cut-edge nodes must still duplicate.
    for eid in (2,3):
        assert not set(mapping).intersection(new.tetra10[new._elements[eid]])
    assert set(mapping.values()).intersection(new.tetra10[2])
    for old,clone in mapping.items():
        assert np.array_equal(new.points_mm[new._nodes[clone]],mesh.points_mm[mesh._nodes[old]])
    assert equations.count("*EQUATION")==3*len(mapping)
    assert f",{reference},1,-1" in equations
    new.validate()


def test_cut_requires_nodes_on_both_sides():
    mesh=shoulder_like_mesh()
    with pytest.raises(ValueError,match="anchorage-side"):
        explicit_pretension(mesh,"cut",[1,2,3],[0,0,1],100)
