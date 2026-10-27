"""A repaired mesh cannot inherit unrelated or stale geometry approval."""
import copy
import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from structural.geometry_revision import verify_mesh_geometry_revision
from structural.mesh import SolidMesh, TET10_EDGES, mesh_fingerprint
from structural.project import StructuralProject


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def revision_fixture(tmp_path):
    project = StructuralProject()
    # A small curved tetrahedron with one face on the cone rho=z.
    vertices = np.array([[17.7*np.cos(-.015), 17.7*np.sin(-.015), 17.7],
                         [17.7*np.cos(.015), 17.7*np.sin(.015), 17.7],
                         [18., 0., 18.], [17.5, 0., 17.7]])
    if np.linalg.det((vertices[1:]-vertices[0]).T) < 0:
        vertices[[0,1]] = vertices[[1,0]]
    points = list(vertices)
    for a,b in TET10_EDGES:
        point = (vertices[a]+vertices[b])/2
        if a < 3 and b < 3:
            point[:2] *= point[2]/np.linalg.norm(point[:2])
        points.append(point)
    points = np.array(points)
    parent_points = points.copy()
    # Move outward along the true final cone normal so the inverse correction
    # is the stationary nearest-point solution, not an arbitrary on-cone point.
    normal = np.r_[points[4,:2]/np.linalg.norm(points[4,:2])/np.sqrt(2), -1/np.sqrt(2)]
    parent_points[4] += .001*normal
    parent = SolidMesh(np.arange(1,11), parent_points, [1], [np.arange(1,11)],
                       element_sets={"pipe":[1]},
                       surfaces={f"screw_{i}_seat":[(1,1)] for i in (1,2,3)},
                       provenance={"source_sha256":project.source_sha256})
    parent_path = tmp_path/'original-mesh.npz'; parent.save(parent_path)
    mesh = copy.deepcopy(parent); mesh.points_mm = points
    geometry_path = write_json(tmp_path/'registered-geometry.json', {
        "source_sha256":project.source_sha256,
        "bodies":[{"name":f"Bolt {i}","registered_faces":[{"type":"Cone","apex_mm":[0,0,0],
                    "axis":[0,0,1],"semi_angle_radians":np.pi/4}]} for i in (1,2,3)]})
    parent_manifest = write_json(tmp_path/'parent-manifest.json', {
        "source_sha256":project.source_sha256,"pipe_wall_resolution":{"radial_partition_radii_mm":[]},
        "settings":{"contact_size_mm":.5}})
    parent.provenance['mesh_manifest']=str(parent_manifest);parent.save(parent_path)
    boundaries = tmp_path/'original-boundaries.npz'
    np.savez(boundaries,node_ids=parent.node_ids,nodes=parent.points_mm,
             boundary_pipe_bore=np.array([],dtype=int),boundary_pipe_outer_wall=np.array([],dtype=int))
    definition_path = write_json(tmp_path/'definition.json', {
        "schema_version":1,"kind":"derived_seat_trim_intersection","source_sha256":project.source_sha256,
        "registered_geometry_sha256":sha(geometry_path),"method":"Original analytic supports, unchanged global connectivity",
        "max_node_shift_mm":.01,"max_source_support_error_mm":1e-8})
    project.geometry_revision={"schema_version":1,"kind":"derived_seat_trim_intersection",
                               "definition_sha256":sha(definition_path)}
    csv_path=tmp_path/'changed-nodes.csv'
    row={"node_id":5,"seat":"screw_1","additional_support":""}
    for prefix, vector in (("original",parent_points[4]),("corrected",points[4]),("shift",points[4]-parent_points[4])):
        row.update({f"{prefix}_{axis}_mm":value for axis,value in zip('xyz',vector)})
    row['shift_distance_mm']=float(np.linalg.norm(points[4]-parent_points[4]))
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    audit={"schema_version":1,"geometry_revision":copy.deepcopy(project.geometry_revision),
           "output_mesh_fingerprint":mesh_fingerprint(mesh),"changed_nodes":1,
           "maximum_node_shift_mm":row['shift_distance_mm'],"max_source_support_error_mm":0.,
           "mesh_quality":{"min_scaled_jacobian":.4,"nonpositive_elements":0},
           "shared_node_continuity":{"verified":True,"disconnected_shared_node_pairs":0}}
    for name,path in {"definition":definition_path,"parent_mesh":parent_path,"changed_nodes_csv":csv_path,
                      "registered_geometry":geometry_path,"parent_manifest":parent_manifest,"boundary_arrays":boundaries}.items():
        audit[name+'_path']=str(path);audit[name+'_sha256']=sha(path)
    audit_path=tmp_path/'revision-audit.json'
    manifest={"geometry_revision":copy.deepcopy(project.geometry_revision),"mesh_quality":copy.deepcopy(audit['mesh_quality']),
              "settings":{"contact_size_mm":.5},"pipe_wall_resolution":{"radial_partition_radii_mm":[]}}
    mesh.provenance['geometry_revision']=copy.deepcopy(project.geometry_revision)
    def seal():
        write_json(audit_path,audit)
        receipt={"audit_path":str(audit_path),"audit_sha256":sha(audit_path)}
        mesh.provenance['geometry_revision_evidence']=receipt
        manifest['geometry_revision_evidence']=copy.deepcopy(receipt)
    seal()
    return project,mesh,manifest,audit,seal


def test_legacy_binding_has_no_native_dependency_and_rejects_orphan_metadata():
    project=StructuralProject();mesh=SimpleNamespace(provenance={})
    assert verify_mesh_geometry_revision(project,mesh,{})['kind']=='legacy_original'
    mesh.provenance['geometry_revision_evidence']={}
    with pytest.raises(ValueError,match='orphan'):
        verify_mesh_geometry_revision(project,mesh,{})


def test_repaired_binding_recomputes_coordinates_topology_and_original_supports(tmp_path):
    project,mesh,manifest,_,_=revision_fixture(tmp_path)
    receipt=verify_mesh_geometry_revision(project,mesh,manifest)
    assert receipt['verified'] and receipt['output_mesh_fingerprint']==mesh_fingerprint(mesh)
    assert receipt['checks']['changed_nodes']==1
    assert receipt['checks']['maximum_source_support_error_mm']<1e-12
    assert receipt['checks']['maximum_node_shift_mm']==pytest.approx(.001)


@pytest.mark.parametrize('participant',['project','mesh','manifest'])
def test_binding_rejects_missing_or_mismatched_revision_identity(tmp_path,participant):
    project,mesh,manifest,_,_=revision_fixture(tmp_path)
    if participant=='project':project.geometry_revision=None
    elif participant=='mesh':mesh.provenance['geometry_revision']['definition_sha256']='a'*64
    else:manifest.pop('geometry_revision')
    with pytest.raises(ValueError,match='explicitly equal'):
        verify_mesh_geometry_revision(project,mesh,manifest)


def test_binding_rejects_coordinate_change_with_stale_quality_audit(tmp_path):
    project,mesh,manifest,_,_=revision_fixture(tmp_path)
    mesh.points_mm[5,0]+=.0001
    with pytest.raises(ValueError,match='stale audit'):
        verify_mesh_geometry_revision(project,mesh,manifest)


@pytest.mark.parametrize('artifact',['definition','parent_mesh','changed_nodes_csv','registered_geometry','parent_manifest','boundary_arrays'])
def test_binding_rejects_tampered_referenced_artifact(tmp_path,artifact):
    project,mesh,manifest,audit,_=revision_fixture(tmp_path)
    from pathlib import Path
    with Path(audit[artifact+'_path']).open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='SHA mismatch'):
        verify_mesh_geometry_revision(project,mesh,manifest)


@pytest.mark.parametrize('mutation,match',[
    ('selection','selections changed'),('continuity','continuity'),('quality','Jacobian'),
    ('shift','maximum shift audit differs'),('csv_membership','cylinder membership'),
])
def test_rehashed_receipt_cannot_bypass_independent_checks(tmp_path,mutation,match):
    project,mesh,manifest,audit,seal=revision_fixture(tmp_path)
    if mutation=='selection':
        mesh.node_sets['extra']=[1];audit['output_mesh_fingerprint']=mesh_fingerprint(mesh)
    elif mutation=='continuity':audit['shared_node_continuity']['verified']=False
    elif mutation=='quality':
        audit['mesh_quality']['min_scaled_jacobian']=.099
        manifest['mesh_quality']=copy.deepcopy(audit['mesh_quality'])
    elif mutation=='shift':audit['maximum_node_shift_mm']=.0001
    else:
        from pathlib import Path
        path=Path(audit['changed_nodes_csv_path'])
        text=path.read_text().replace('5,screw_1,,','5,screw_1,pipe_bore,')
        path.write_text(text);audit['changed_nodes_csv_sha256']=sha(path)
    seal()
    with pytest.raises(ValueError,match=match):verify_mesh_geometry_revision(project,mesh,manifest)


def test_rehashed_new_coordinates_cannot_bypass_analytic_support_check(tmp_path):
    project,mesh,manifest,audit,seal=revision_fixture(tmp_path)
    # Adjust one changed node, update its CSV and semantic fingerprint honestly,
    # but leave it off the immutable original cone. The source check must fail.
    mesh.points_mm[4,0]+=.0001
    from pathlib import Path
    path=Path(audit['changed_nodes_csv_path'])
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    row=rows[0];row['corrected_x_mm']=mesh.points_mm[4,0]
    row['shift_x_mm']=float(row['corrected_x_mm'])-float(row['original_x_mm'])
    row['shift_distance_mm']=float(np.linalg.norm([float(row[f'shift_{axis}_mm']) for axis in 'xyz']))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
    audit['changed_nodes_csv_sha256']=sha(path);audit['maximum_node_shift_mm']=row['shift_distance_mm']
    audit['output_mesh_fingerprint']=mesh_fingerprint(mesh);seal()
    with pytest.raises(ValueError,match='analytic support bounds'):
        verify_mesh_geometry_revision(project,mesh,manifest)


def test_support_correct_tangential_shift_still_fails_nearest_point_stationarity(tmp_path):
    project,mesh,manifest,audit,seal=revision_fixture(tmp_path)
    theta=1e-5;point=mesh.points_mm[4].copy()
    mesh.points_mm[4,:2]=[np.cos(theta)*point[0]-np.sin(theta)*point[1],
                          np.sin(theta)*point[0]+np.cos(theta)*point[1]]
    from pathlib import Path
    path=Path(audit['changed_nodes_csv_path'])
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    row=rows[0]
    for axis,value in zip('xyz',mesh.points_mm[4]):
        row[f'corrected_{axis}_mm']=value
        row[f'shift_{axis}_mm']=value-float(row[f'original_{axis}_mm'])
    row['shift_distance_mm']=float(np.linalg.norm([row[f'shift_{axis}_mm'] for axis in 'xyz']))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
    audit['changed_nodes_csv_sha256']=sha(path);audit['maximum_node_shift_mm']=row['shift_distance_mm']
    audit['output_mesh_fingerprint']=mesh_fingerprint(mesh);seal()
    with pytest.raises(ValueError,match='stationarity'):
        verify_mesh_geometry_revision(project,mesh,manifest)


@pytest.mark.parametrize('key',['settings','pipe_wall_resolution'])
def test_repaired_mesh_cannot_relabel_parent_refinement(tmp_path,key):
    project,mesh,manifest,_,_=revision_fixture(tmp_path)
    manifest[key]['invented_refinement']=.1
    with pytest.raises(ValueError,match='parent refinement'):
        verify_mesh_geometry_revision(project,mesh,manifest)
