"""Read-only, hash-bound fixed-trace slot measurements on current mesh coordinates.

Usage: python audit_slot_weighted_resolution.py MESH_FOLDER [MESH_FOLDER ...]
Each folder receives a new slot-resolution-verification.json; existing evidence
is never replaced. The original CAD topology audit remains a separate input.
"""
from pathlib import Path
import json
import sys
import numpy as np
import audit_slot_resolution as original

from structural.mesh import SolidMesh, TET10_FACES

CAD_PATH = original.PREP/'slot-spatial-resolution-final-verification/original-slot-cad-topology.json'
CORNER_FACES = {'slot_1': [6,8,10,12], 'slot_2': [14,16,18,20], 'slot_3': [22,24,26,29]}
DEFINITION = {
    'schema_version': 1,
    'coordinate_frame': 'comsol_reference_mm',
    'fixed_axial_band_mm': original.BAND,
    'source_face_groups': original.SLOT_GROUPS,
    'corner_source_face_order': CORNER_FACES,
    'perimeter_rule': 'Unique quadratic edges shared topologically by original slot-wall fragments and named pipe_outer_wall or pipe_bore. Radial thickness edges excluded.',
    'in_plane_span_rule': 'hypot(abs(endpoint z difference), abs(short wrapped endpoint azimuth difference)*mean(endpoint pipe radius))',
    'weight_definition': 'Each unique quadratic edge receives its physical arc length integrated with8 positive Gauss-Legendre points on[-1,1]. Weighted quantiles interpolate the centered cumulative arc-length distribution after sorting in-plane spans.',
    'support_classification': 'Original angular slot footprint then nearest full-six-node original analytic support; residual bounded by original adjacent edge tolerance plus1e-8 mm. No slot projection or healing.',
    'required_trace_count': 30,
}


def weighted_distribution(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Trace spans and strictly positive arc weights must be finite and nonempty')
    order = np.argsort(values, kind='stable')
    values, weights = values[order], weights[order]
    cumulative = (np.cumsum(weights)-weights/2)/weights.sum()
    return {'minimum': float(values.min()), 'median': float(np.interp(.5,cumulative,values)),
            'percentile95': float(np.interp(.95,cumulative,values)), 'maximum': float(values.max())}


def arc_lengths(points):
    """points[..., endpoint A, endpoint B, midside, xyz]."""
    xi, weights = np.polynomial.legendre.leggauss(8)
    derivative = np.stack((xi-.5,xi+.5,-2*xi),axis=1)
    tangents = np.einsum('gn,eni->egi',derivative,points)
    return np.linalg.norm(tangents,axis=2)@weights


def trace_metrics(edges, xyz, source_faces, cad, radius):
    points = xyz(edges)
    lengths = arc_lengths(points)
    a,b = points[:,0],points[:,1]
    azimuth = np.arctan2(points[:,:,1],points[:,:,0])
    dtheta = (azimuth[:,1]-azimuth[:,0]+np.pi)%(2*np.pi)-np.pi
    circum = abs(dtheta)*(np.linalg.norm(a[:,:2],axis=1)+np.linalg.norm(b[:,:2],axis=1))/2
    spans = np.hypot(abs(a[:,2]-b[:,2]),circum)
    errors = np.stack([original.support_error(points,cad['faces'][str(i)]) for i in source_faces])
    return {'source_face_indices': source_faces, 'quadratic_edges': len(edges),
            'integrated_arc_length_mm': float(lengths.sum()),
            'arc_length_weighted_in_plane_span_mm': weighted_distribution(spans,lengths),
            **original.edge_measures(points),
            'maximum_perimeter_node_pipe_radius_error_mm': float(abs(np.linalg.norm(points[:,:,:2],axis=2)-radius).max()),
            'maximum_nearest_source_support_error_mm': float(errors.min(axis=0).max())}


def audit(folder):
    folder = Path(folder).resolve()
    output = folder/'slot-resolution-verification.json'
    if output.exists(): raise FileExistsError(output)
    mesh_path,manifest_path = folder/'structural-mesh.npz',folder/'mesh-manifest.json'
    inputs = [mesh_path,manifest_path,CAD_PATH,original.GEOMETRY,Path(original.__file__),Path(__file__)]
    before = {str(path.resolve()):original.sha(path) for path in inputs}
    cad = json.loads(CAD_PATH.read_text())
    assert cad['inputs_sha256'][str(original.GEOMETRY)] == original.sha(original.GEOMETRY)
    mesh = SolidMesh.load(mesh_path)
    assert mesh.provenance['source_sha256'] == cad['source_sha256']
    conn = mesh.tetra10[[mesh._elements[e] for e in mesh.element_sets['pipe']]]
    faces = np.stack([conn[:,list(local)] for local in TET10_FACES.values()],axis=1).reshape(-1,6)
    _,first,counts = np.unique(np.sort(faces[:,:3],axis=1),axis=0,return_index=True,return_counts=True)
    exterior = faces[first[counts==1]]
    order = np.argsort(mesh.node_ids)
    def xyz(nodes): return mesh.points_mm[order[np.searchsorted(mesh.node_ids[order],nodes)]]
    q = xyz(exterior)
    keep = (q[:,:,2].max(axis=1)>=original.BAND[0]-1e-7)&(q[:,:,2].min(axis=1)<=original.BAND[1]+1e-7)
    exterior,q = exterior[keep],q[keep]
    named_keys,perimeter_keys = set(),{}
    for name,semantic in (('outer','pipe_outer_wall'),('bore','pipe_bore')):
        tri = np.array([mesh.face_nodes(e,f) for e,f in mesh.surfaces[semantic]])
        named_keys.update(tuple(sorted(map(int,row[:3]))) for row in tri)
        perimeter_keys[name] = {tuple(sorted(map(int,e))) for e in tri[:,original.PROFILE_EDGES].reshape(-1,3)}
    selected = np.array([tuple(sorted(map(int,row[:3]))) not in named_keys for row in exterior])
    slot_nodes,slot_xyz = exterior[selected],q[selected]
    centers = np.array([cad['faces'][str(ids[0])]['centroid_mm'][:2] for ids in original.SLOT_GROUPS.values()])
    nearest_group = np.linalg.norm(slot_xyz[:,:3,:2].mean(axis=1)[:,None,:]-centers[None,:,:],axis=2).argmin(axis=1)
    face_group = np.array([next(i for i,ids in enumerate(original.SLOT_GROUPS.values()) if face in ids) for face in original.FACE_IDS])
    errors = np.stack([original.support_error(slot_xyz,cad['faces'][str(i)]).max(axis=1) for i in original.FACE_IDS],axis=1)
    errors[nearest_group[:,None]!=face_group[None,:]] = np.inf
    assignments = np.array(original.FACE_IDS)[errors.argmin(axis=1)]
    tolerances = {i:max(cad['edges'][str(edge)]['tolerance_mm'] for edge in cad['faces'][str(i)]['source_edge_indices']) for i in original.FACE_IDS}
    if np.any(errors.min(axis=1)>np.array([tolerances[int(i)] for i in assignments])+1e-8):
        raise ValueError('Unresolved original slot support assignment')
    if set(assignments)!=set(original.FACE_IDS): raise ValueError('All24 original slot faces must be represented')
    traces = {}
    for group,ids in original.SLOT_GROUPS.items():
        for suffix,source_faces in [('',ids)]+[(f'_corner_{i}',[face]) for i,face in enumerate(CORNER_FACES[group],1)]:
            edges = slot_nodes[np.isin(assignments,source_faces)][:,original.PROFILE_EDGES].reshape(-1,3)
            _,unique = np.unique(np.sort(edges[:,:2],axis=1),axis=0,return_index=True)
            edges = edges[unique]
            for side,radius in (('outer',19.),('bore',17.5)):
                keep = np.array([tuple(sorted(map(int,e))) in perimeter_keys[side] for e in edges])
                traces[group+suffix+'_'+side] = trace_metrics(edges[keep],xyz,source_faces,cad,radius)
    assert len(traces)==30
    unchanged = all(original.sha(path)==digest for path,digest in before.items())
    assert unchanged
    record = {'schema_version':1,'status':'read_only_geometry_diagnostic_not_response_convergence',
              'mesh_sha256':original.sha(mesh_path),'mesh_manifest_sha256':original.sha(manifest_path),
              'cad_topology_path':str(CAD_PATH.resolve()),'cad_topology_sha256':original.sha(CAD_PATH),
              'geometry_revision':mesh.provenance.get('geometry_revision'),
              'definition':DEFINITION,'traces':traces,'all_24_source_slot_faces_represented':True,
              'recipe_path':str(Path(__file__).resolve()),'recipe_sha256':original.sha(__file__),
              'inputs_sha256':before,'sources_unchanged':unchanged}
    original.write(output,record)
    print(json.dumps({'case':folder.name,'output':str(output),'traces':len(traces),
                      'slot_1_outer':traces['slot_1_outer']}),flush=True)
    return record


if __name__=='__main__':
    for arg in sys.argv[1:]: audit(arg)
