"""Controlled numerical study cases and finite-volume physical region metrics.

Cases never modify CAD dimensions/materials to reproduce COMSOL peaks. Mesh and
contact parameters vary in separate studies. Failed cases remain in summaries.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from .project import StructuralProject, write_json
from .comparison import FieldMesh, file_provenance, integrate_section, von_mises
from .mesh import mesh_fingerprint  # Re-export preserves the original studies API.


def physical_model_fingerprint(project: StructuralProject, *, exclude_parameter: str | None = None) -> str:
    data=project.to_dict()
    for key in ("name","mesh","solver","preparation","selections"):
        data.pop(key,None)
    # Registration evidence file paths are provenance, not model physics.
    data["registration"]={k:v for k,v in data["registration"].items() if k in ("rotation","translation_mm")}
    data.pop("source_path",None);data.pop("reference",None)
    if exclude_parameter:
        paths={"contact":("contacts","normal_stiffness_mpa_per_mm"),"friction":("contacts","friction_coefficient"),
               "load":("loads","service_force_n"),"preload":("loads","screw_preloads_n"),"patch":("loads","patch_width_mm")}
        if exclude_parameter not in paths:raise ValueError("Unknown controlled physical parameter")
        group,key=paths[exclude_parameter];data[group].pop(key)
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def numerical_controls_fingerprint(project: StructuralProject) -> str:
    """Separate numerical schedules from physics and wall-clock/provenance metadata."""
    controls=copy.deepcopy(project.solver)
    for key in ("timeout_s","numerical_control_fixture","friction_qualification_manifest"):
        controls.pop(key,None)
    if controls.get("contact_count_relative_tolerance") is None:
        controls.pop("contact_count_relative_tolerance",None)
    controls["step_controls"]=project.static_step_controls()
    controls.setdefault("variant","serial");controls.setdefault("threads",1)
    return hashlib.sha256(json.dumps(controls,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def controlled_cases(base: StructuralProject, study: str, *, values=None) -> list[tuple[str,StructuralProject]]:
    if study=="mesh":
        variants=[("mesh-0750",.75),("mesh-0500",.5),("mesh-0375",.375)]
    elif study=="contact_mesh":
        variants=[("contact-mesh-coarse",base.mesh.contact_size_mm*1.5),("contact-mesh-baseline",base.mesh.contact_size_mm),("contact-mesh-fine",base.mesh.contact_size_mm*.5)]
    elif study=="bore_contact_mesh":
        variants=[("bore-contact-background",None),("bore-contact-2000",2.),("bore-contact-1500",1.5)]
    elif study=="slot_mesh":
        variants=[("slot-background",None),("slot-1500",1.5),("slot-0750",.75)]
    elif study=="contact":
        variants=[("contact-half",.5),("contact-baseline",1.),("contact-double",2.)]
    elif study=="patch":
        variants=[("patch-35mm",35.),("patch-30mm",30.)]
    elif study=="friction":variants=[("friction-000",0.),("friction-010",.1),("friction-020",.2)]
    elif study in ("load","preload"):
        variants=[(study+"-half",.5),(study+"-baseline",1.),(study+"-one-and-half",1.5)]
    else:raise ValueError("Study must be mesh, contact_mesh, bore_contact_mesh, slot_mesh, contact, friction, load, preload or patch")
    if values is not None:
        variants=[(f"{study}-{i+1:02d}",None if value is None and study in ("bore_contact_mesh","slot_mesh") else float(value)) for i,value in enumerate(values)]
    if study=="contact" and base.contacts.friction_coefficient:
        raise ValueError("Contact penalty study requires frictionless baseline: the current deck couples tangential slope to normal penalty")
    if study in ("load","preload") and any(value<=0 for _,value in variants):
        raise ValueError("Load/preload sensitivity uses positive scale factors")
    result=[]
    for name,value in variants:
        project=copy.deepcopy(base)
        project.name=base.name+" / "+name
        if study=="mesh":
            project.mesh.thickness_size_mm=value
        elif study=="contact_mesh":project.mesh.contact_size_mm=value
        elif study=="bore_contact_mesh":project.mesh.bore_snout_size_mm=value
        elif study=="slot_mesh":project.mesh.slot_size_mm=value
        elif study=="contact":project.contacts.normal_stiffness_mpa_per_mm=base.contacts.normal_stiffness_mpa_per_mm*value
        elif study=="friction":project.contacts.friction_coefficient=value
        elif study=="load":project.loads.service_force_n=base.loads.service_force_n*value
        elif study=="preload":project.loads.screw_preloads_n=[force*value for force in base.loads.screw_preloads_n]
        else:
            project.loads.patch_width_mm=value
            project.registration["comparison_classification"]="revised_model" if value==35 else "matched_partition_diagnostic"
        project.validate()
        result.append((name,project))
    return result


def region_measures(pipe: FieldMesh) -> dict:
    """Fixed physical volume windows; no mesh-node percentile or peak trimming.

    Uses linear-tet exported/display stress interpolation and clips each tet to
    axial planes exactly before integrating. Raw integration-point measures are
    separate and must be retained; this metric tracks region response, not a
    material strength allowable.
    """
    from scipy.spatial import ConvexHull, QhullError
    points=pipe.points_mm[pipe.tetrahedra]
    nodal_vm=von_mises(pipe.stress_mpa)
    gauss=np.full((4,4),.1381966011250105);np.fill_diagonal(gauss,.5854101966249685)
    results={}
    # Uniform physical windows match reference d=-Z, mm.
    for name,lo,hi in (("center",0.,25.),("slots",190.,201.),("seats",208.,226.)):
        volume_total=0.;integral=0.;raw_peak=0.;raw_peak_location=None
        degenerate_boundary_cells=0;degenerate_volume_bound=0.
        distance=-points[:,:,2]
        candidates=np.flatnonzero((distance.max(1)>=lo)&(distance.min(1)<=hi))
        full=candidates[(distance[candidates].min(1)>=lo)&(distance[candidates].max(1)<=hi)]
        if len(full):
            q=points[full]
            volumes=np.abs(np.linalg.det(np.stack((q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]),axis=-1)))/6
            stress_gp=np.einsum("gi,eijk->egjk",gauss,pipe.stress_mpa[pipe.tetrahedra[full]])
            volume_total=float(volumes.sum())
            integral=float(np.dot(volumes,von_mises(stress_gp).mean(1)))
            raw_peak=float(nodal_vm[pipe.tetrahedra[full]].max())
            raw_peak_location=pipe.points_mm[pipe.tetrahedra[full].ravel()[nodal_vm[pipe.tetrahedra[full]].argmax()]].tolist()
        boundary=np.setdiff1d(candidates,full,assume_unique=True)
        for cell in boundary:
            q=points[cell];d=distance[cell];ids=pipe.tetrahedra[cell]
            weights=[]
            for i in range(4):
                if lo-1e-10<=d[i]<=hi+1e-10:weights.append(np.eye(4)[i])
                for j in range(i+1,4):
                    for plane in (lo,hi):
                        if (d[i]-plane)*(d[j]-plane)<-1e-16:
                            fraction=(plane-d[i])/(d[j]-d[i]);w=np.zeros(4);w[i]=1-fraction;w[j]=fraction;weights.append(w)
            if len(weights)<4:continue
            weights=np.unique(np.round(weights,13),axis=0)
            xyz=weights@q
            if len(xyz)<4:continue
            centered=xyz-xyz.mean(0)
            singular=np.linalg.svd(centered,compute_uv=False)
            # Exact plane-coincident export vertices differ at floating-point
            # precision. Bound, count and disclose the negligible discarded
            # sliver instead of perturbing geometry to make a positive volume.
            if singular[-1]<=1e-10:
                degenerate_boundary_cells+=1
                degenerate_volume_bound+=float(8*np.prod(singular))
                continue
            try:hull=ConvexHull(centered)
            except QhullError:
                raise
            center=xyz.mean(0);center_weight=weights.mean(0)
            stress=pipe.stress_mpa[ids]
            # Interpolate stress tensor before computing invariant at each vertex.
            vm=von_mises(np.einsum("ni,ijk->njk",weights,stress))
            if raw_peak_location is None or vm.max()>raw_peak:
                raw_peak=float(vm.max());raw_peak_location=xyz[vm.argmax()].tolist()
            for tri in hull.simplices:
                tetra=np.vstack((center,xyz[tri]))
                volume=abs(np.linalg.det((tetra[1:]-tetra[0]).T))/6
                volume_total+=volume
                tetra_weights=np.vstack((center_weight,weights[tri]))
                stress_gp=np.einsum("gi,ijk->gjk",gauss@tetra_weights,stress)
                integral+=volume*float(von_mises(stress_gp).mean())
        results[name]={"distance_window_mm":[lo,hi],"volume_mm3":volume_total,
                       "volume_mean_nodal_von_mises_mpa":integral/volume_total if volume_total else None,
                       "raw_display_peak_mpa":raw_peak if volume_total else None,
                       "raw_peak_location_mm":raw_peak_location,
                       "degenerate_boundary_cells_excluded":degenerate_boundary_cells,
                       "excluded_volume_upper_bound_mm3":degenerate_volume_bound,
                       "clipping_coordinate_tolerance_mm":1e-10,
                       "stress_location":pipe.provenance.get("stress_location","exported_nodal_tensor_samples"),
                       "integration":"four-point tetrahedral quadrature of von Mises after tensor interpolation; clipped boundary tetrahedra use the same rule",
                       "purpose":"mesh convergence of average local response, not local strength"}
    return results


STUDY_METRIC_DEFINITION = {
    "version":2,"states":["preload","loaded"],
    "sections_distance_from_midpoint_mm":[35.,75.,125.,175.],
    "region_distance_windows_mm":{"center":[0.,25.],"slots":[190.,201.],"seats":[208.,226.]},
    "region_stress":"volume mean of von Mises of interpolated nodal tensor, four-point tetrahedral quadrature",
    "field_representation":"C3D10 nodal display fields sampled by eight linear tetrahedra; raw integration-point files remain separate",
    "clipping_coordinate_tolerance_mm":1e-10,
    "mesh_resolution":"at least three meshes with increasing enforced radial layers≥2, no full-wall tetrahedra, and conformal six-node internal faces",
    "contact_surface_resolution":{"screw_seat_head":"measured median/p95 corner-edge lengths and quadratic face counts",
        "bore_snout":"median/p95 corner-edge lengths weighted by fixed physical surface area from36-point positive quadratic-face quadrature; selected positive-area face counts"},
    "changes":"absolute scalar or Euclidean vector difference / max(norm of comparison reference, physical absolute floor)",
    "relative_limits":{"global_integrated":.02,"finite_region_stress":.05},
    "normalization_floors":{"displacement_mm":.001,"force_n":1.,"moment_nmm":233.68,"stress_mpa":.1},
    "raw_peaks":"report values and positions without acceptance threshold; local adequacy remains unresolved",
}

SCREW_CONTACT_SURFACES=tuple(f"screw_{i}_{side}" for i in (1,2,3) for side in ("seat","cone"))
ALL_CONTACT_SURFACES=(*SCREW_CONTACT_SURFACES,"pipe_bore","hub_snout")
SLOT_SOURCE_FACES={1:list(range(5,13)),2:list(range(13,21)),3:[21,22,23,24,25,26,27,29]}
SLOT_CORNER_FACES={1:[6,8,10,12],2:[14,16,18,20],3:[22,24,26,29]}
SLOT_TRACE_FACES={**{f"slot_{i}_{side}":faces for i,faces in SLOT_SOURCE_FACES.items() for side in ("outer","bore")},
                  **{f"slot_{i}_corner_{corner}_{side}":[face] for i,faces in SLOT_CORNER_FACES.items()
                     for corner,face in enumerate(faces,1) for side in ("outer","bore")}}
SLOT_TRACES=tuple(SLOT_TRACE_FACES)
SLOT_MEASUREMENT_DEFINITION={
    "schema_version":1,"coordinate_frame":"comsol_reference_mm","fixed_axial_band_mm":[-201.,-190.],
    "source_face_groups":{f"slot_{i}":faces for i,faces in SLOT_SOURCE_FACES.items()},
    "corner_source_face_order":{f"slot_{i}":faces for i,faces in SLOT_CORNER_FACES.items()},
    "perimeter_rule":"Unique quadratic edges shared topologically by original slot-wall fragments and named pipe_outer_wall or pipe_bore. Radial thickness edges excluded.",
    "in_plane_span_rule":"hypot(abs(endpoint z difference), abs(short wrapped endpoint azimuth difference)*mean(endpoint pipe radius))",
    "weight_definition":"Each unique quadratic edge receives its physical arc length integrated with8 positive Gauss-Legendre points on[-1,1]. Weighted quantiles interpolate the centered cumulative arc-length distribution after sorting in-plane spans.",
    "support_classification":"Original angular slot footprint then nearest full-six-node original analytic support; residual bounded by original adjacent edge tolerance plus1e-8 mm. No slot projection or healing.",
    "required_trace_count":30,
}


def _surface_resolution_evidence(manifest_path, manifest):
    """Read generated statistics or a sidecar bound to both original mesh files."""
    # A derived manifest may inherit the parent's descriptive entries. Require
    # a new, hash-linked coordinate measurement instead of promoting those.
    values={} if manifest.get("geometry_revision") else manifest.get("surface_resolution",{})
    source=manifest_path
    verified=bool(values)
    if not values:
        source=manifest_path.parent/"surface-resolution-verification.json"
        if source.is_file():
            evidence=json.loads(source.read_text(encoding="utf-8"))
            original_mesh=manifest_path.parent/"structural-mesh.npz"
            verified=(original_mesh.is_file()
                      and evidence.get("mesh_sha256")==file_provenance(original_mesh)["sha256"]
                      and evidence.get("mesh_manifest_sha256")==file_provenance(manifest_path)["sha256"])
            values=evidence.get("surface_resolution",{})
    return {"verified":bool(verified and values),"surfaces":values,
            "source":file_provenance(source) if source.is_file() else None,
            # The present distance field controls the seats and screw heads.
            # Incidental refinement elsewhere does not extend that study scope.
            "controlled_surfaces":list(SCREW_CONTACT_SURFACES),
            "uncontrolled_contact_surfaces":["pipe_bore","hub_snout"],
            "field_scopes":manifest.get("sizing_field_scopes",{})}


def _measured_contact_refinement(summaries, surfaces,*,edge_metric="corner_edge_length_mm",count_metric="quadratic_triangles"):
    """Require measured refinement of every declared surface, not only a label."""
    if not summaries or not surfaces:return None
    if not all(row.get("surface_resolution",{}).get("verified") is True for row in summaries):return None
    for name in surfaces:
        rows=[row["surface_resolution"].get("surfaces",{}).get(name,{}) for row in summaries]
        if not all(isinstance(row,dict) for row in rows):return None
        for metric in ("median","percentile95"):
            values=[row.get(edge_metric,{}).get(metric) for row in rows]
            if not all(isinstance(value,(int,float)) and np.isfinite(value) and value>0 for value in values):return None
            if not all(a>b for a,b in zip(values,values[1:])):return False
        counts=[row.get(count_metric) for row in rows]
        if not all(isinstance(value,int) and value>0 for value in counts):return None
        if not all(a<b for a,b in zip(counts,counts[1:])):return False
    return True


def _bore_snout_resolution_evidence(manifest_path,manifest=None):
    source=manifest_path.parent/"bore-snout-resolution-verification.json"
    if not source.is_file():return {"verified":False,"status":"missing"}
    evidence=json.loads(source.read_text(encoding="utf-8"))
    mesh=manifest_path.parent/"structural-mesh.npz"
    verified=(mesh.is_file() and evidence.get("mesh_sha256")==file_provenance(mesh)["sha256"]
              and evidence.get("mesh_manifest_sha256")==file_provenance(manifest_path)["sha256"])
    return {"verified":verified,"source":file_provenance(source),
            "status":"geometry_diagnostic_only","paired_local_target_mm":(manifest or {}).get("settings",{}).get("bore_snout_size_mm"),
            "sample_rule":evidence.get("sample_rule"),"linear_subtriangle_scope":evidence.get("linear_subtriangle_scope"),
            "surfaces":{name:evidence.get(name) for name in ("pipe_bore_full","pipe_bore_snout_band_intersecting_faces","hub_snout")}}


def _bore_snout_area_resolution_evidence(manifest_path):
    source=manifest_path.parent/"bore-snout-area-weighted-resolution-verification.json"
    if not source.is_file():return {"verified":False,"status":"missing"}
    evidence=json.loads(source.read_text(encoding="utf-8"))
    mesh=manifest_path.parent/"structural-mesh.npz"
    verified=(mesh.is_file() and evidence.get("mesh_sha256")==file_provenance(mesh)["sha256"]
              and evidence.get("mesh_manifest_sha256")==file_provenance(manifest_path)["sha256"])
    return {"verified":verified,"source":file_provenance(source),"status":"geometry_diagnostic_only",
            "weight_definition":evidence.get("weight_definition"),"snout_axial_bounds_mm":evidence.get("snout_axial_bounds_mm"),
            "surfaces":{name:evidence.get(name) for name in ("pipe_bore_snout_band","hub_snout")}}


def _same_bore_measurement_definition(summaries):
    """Require the same identified area rule and fixed physical axial band."""
    definitions=[];bands=[]
    for row in summaries:
        evidence=row.get("bore_snout_area_resolution",{})
        rule=evidence.get("weight_definition");band=evidence.get("snout_axial_bounds_mm")
        if not isinstance(rule,str) or not rule.strip() or not isinstance(band,(list,tuple)) or len(band)!=2:return None
        if not all(isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) for value in band) or band[0]>=band[1]:return None
        definitions.append(rule);bands.append(band)
    return bool(definitions) and len(set(definitions))==1 and all(np.allclose(band,bands[0],rtol=0,atol=1e-9) for band in bands)


def _slot_resolution_evidence(manifest_path,manifest):
    """Read current-coordinate measurements, bound to mesh, manifest and CAD topology."""
    source=manifest_path.parent/"slot-resolution-verification.json"
    if not source.is_file():return {"verified":False,"status":"missing"}
    evidence=json.loads(source.read_text(encoding="utf-8"))
    mesh=manifest_path.parent/"structural-mesh.npz"
    topology=Path(evidence.get("cad_topology_path", ""))
    if not topology.is_absolute():topology=(source.parent/topology).resolve()
    cad=json.loads(topology.read_text(encoding="utf-8")) if topology.is_file() else {}
    source_hash=manifest.get("source_sha256")
    topology_bound=(isinstance(source_hash,str) and len(source_hash)==64 and cad.get("source_sha256")==source_hash
                    and cad.get("source_step_current_sha256",source_hash)==source_hash)
    verified=(mesh.is_file() and evidence.get("mesh_sha256")==file_provenance(mesh)["sha256"]
              and evidence.get("mesh_manifest_sha256")==file_provenance(manifest_path)["sha256"]
              and topology.is_file() and evidence.get("cad_topology_sha256")==file_provenance(topology)["sha256"]
              and evidence.get("geometry_revision")==manifest.get("geometry_revision") and topology_bound
              and evidence.get("all_24_source_slot_faces_represented") is True and evidence.get("sources_unchanged") is True)
    return {"verified":bool(verified),"source":file_provenance(source),
            "cad_topology":file_provenance(topology) if topology.is_file() else None,
            "cad_topology_sha256":evidence.get("cad_topology_sha256"),
            "topology_source_binding_verified":topology_bound,
            "all_24_source_slot_faces_represented":evidence.get("all_24_source_slot_faces_represented"),
            "sources_unchanged":evidence.get("sources_unchanged"),
            "definition":evidence.get("definition"),"traces":evidence.get("traces",{}),
            "status":"spatial_measurement_only"}


def _same_slot_measurement_definition(summaries):
    """Keep all six perimeters and 24 corner traces on the approved physical rule."""
    hashes=[]
    for row in summaries:
        evidence=row.get("slot_resolution",{})
        if evidence.get("definition") is None:return None
        if evidence["definition"]!=SLOT_MEASUREMENT_DEFINITION:return False
        digest=evidence.get("cad_topology_sha256")
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):return None
        hashes.append(digest)
        traces=evidence.get("traces",{})
        if not isinstance(traces,dict) or set(traces)!=set(SLOT_TRACES):return None
        for name,faces in SLOT_TRACE_FACES.items():
            trace=traces[name]
            if not isinstance(trace,dict) or trace.get("source_face_indices")!=faces:return False
            length=trace.get("integrated_arc_length_mm")
            if not isinstance(length,(int,float)) or isinstance(length,bool) or not np.isfinite(length) or length<=0:return None
    return bool(hashes) and len(set(hashes))==1


def _measured_slot_refinement(summaries):
    if not summaries or not all(row.get("slot_resolution",{}).get("verified") is True for row in summaries):return None
    scope=_same_slot_measurement_definition(summaries)
    if scope is not True:return scope
    measurements=[{"surface_resolution":{"verified":True,"surfaces":row["slot_resolution"]["traces"]}} for row in summaries]
    return _measured_contact_refinement(measurements,SLOT_TRACES,
        edge_metric="arc_length_weighted_in_plane_span_mm",count_metric="quadratic_edges")


def _effective_slot_size(summary):
    # Older retained summaries predate the optional local field. A missing
    # declaration means the unchanged background target, never slot convergence.
    return summary.get("effective_slot_size_mm",summary.get("slot_size_mm") or summary.get("far_size_mm"))


def _contact_controls_bound(summary):
    if "contact_controls_binding_verified" in summary or summary.get("contact_count_relative_tolerance") is not None:
        return summary.get("contact_controls_binding_verified") is True
    return True  # Legacy summaries declare no optional control; no new positive claim.


def _definition_fingerprint():
    return hashlib.sha256(json.dumps(STUDY_METRIC_DEFINITION,sort_keys=True).encode()).hexdigest()


def _radial_layer_continuity_evidence(manifest_path, manifest, mesh):
    count=manifest.get("pipe_wall_resolution",{}).get("enforced_radial_layers")
    interfaces=manifest.get("radial_layer_continuity",{})
    source=manifest_path
    def complete(values):
        return (isinstance(count,int) and count>=2 and len(values)==count-1
            and all(row.get("all_six_quadratic_face_nodes_shared",row.get("all_six_face_nodes_shared")) is True
                and row.get("triangles_with_two_adjacent_tetrahedra",row.get("quadratic_triangles_verified",0))>0
                for row in values.values()))
    bound=True
    # Original two-layer manifests predate the six-node audit. Their nonempty
    # count-only entries must not prevent consumption of that retained proof.
    if manifest.get("geometry_revision") or not complete(interfaces):
        source=manifest_path.parent/"quadratic-layer-continuity-verification.json"
        bound=False
        if source.is_file():
            evidence=json.loads(source.read_text(encoding="utf-8"))
            if "interfaces" in evidence:
                interfaces=evidence["interfaces"]
                original_mesh=manifest_path.parent/"structural-mesh.npz"
                bound=(original_mesh.is_file() and evidence.get("mesh_sha256")==file_provenance(original_mesh)["sha256"]
                       and evidence.get("mesh_manifest_sha256")==file_provenance(manifest_path)["sha256"])
            elif not manifest.get("geometry_revision"):
                interfaces=evidence;bound=True
    return {"verified":bool(bound and complete(interfaces) and not any("layer_interface" in name for name in mesh.surfaces)),
            "source":file_provenance(source) if source.is_file() else None,"interfaces":interfaces,
            "derived_geometry_requires_hash_linked_evidence":bool(manifest.get("geometry_revision"))}


def _verify_pretension_mesh_transform(prepared, solver, metadata):
    """Check the declared cut duplication without treating it as a geometry edit."""
    from .mesh import SolidMesh
    if metadata.get("pretension_formulation")!="explicit paired-node axial jump with scalar work-conjugate force":
        raise ValueError("Derived geometry requires an identified solver-mesh transformation")
    maps=metadata.get("pretension_cut_node_maps")
    if not isinstance(maps,dict) or not maps:
        raise ValueError("Missing prepared-to-solver cut-node maps")
    nodes=list(map(int,prepared.node_ids));points=[prepared.points_mm]
    connectivity=prepared.tetra10.copy();node_sets=copy.deepcopy(prepared.node_sets)
    originals=set(nodes);clones=set()
    for screw,raw in maps.items():
        mapping={int(original):int(clone) for original,clone in raw.items()}
        surface=f"screw_{screw}_cut"
        if set(mapping)!=set(prepared.surface_nodes(surface)):
            raise ValueError("Declared cut map does not cover exactly the prepared cut nodes")
        if len(set(mapping.values()))!=len(mapping) or (set(mapping.values()) & (originals|clones)):
            raise ValueError("Duplicate or reused head-side node ID")
        head=prepared.provenance.get("pretension_head_element_sets",{}).get(str(screw))
        if not head or not set(head).issubset(prepared._elements):
            raise ValueError("Missing prepared head-side element selection")
        for element in head:
            index=prepared._elements[int(element)]
            connectivity[index]=[mapping.get(int(node),int(node)) for node in connectivity[index]]
        nodes.extend(mapping.values());clones.update(mapping.values())
        points.append(prepared.points_mm[[prepared._nodes[node] for node in mapping]])
        node_sets[surface+"_head_nodes"]=list(mapping.values())
        node_sets[surface+"_shaft_nodes"]=list(mapping)
    expected=SolidMesh(nodes,np.vstack(points),prepared.element_ids,connectivity,
                       prepared.element_sets,prepared.surfaces,node_sets)
    if mesh_fingerprint(expected)!=mesh_fingerprint(solver):
        raise ValueError("Solver mesh differs from the declared cut duplication of the prepared geometry")
    return {"verified":True,"cut_count":len(maps),"duplicated_node_count":len(clones),
            "prepared_mesh_fingerprint":mesh_fingerprint(prepared),"solver_mesh_fingerprint":mesh_fingerprint(solver)}


def _geometry_revision_evidence(project, mesh, mesh_manifest, metadata, deck):
    """Fail closed on repaired geometry while preserving absent legacy revisions."""
    from .geometry_revision import verify_mesh_geometry_revision
    from .mesh import SolidMesh
    revision=getattr(project,"geometry_revision",None)
    result={"verified":False,"geometry_revision":revision}
    try:
        if any(value!=revision for value in (mesh.provenance.get("geometry_revision"),
                                             mesh_manifest.get("geometry_revision"),metadata.get("geometry_revision"))):
            raise ValueError("Project, solver mesh, generation manifest and assembly geometry revisions differ")
        if revision is None:
            result.update(verify_mesh_geometry_revision(project,mesh,mesh_manifest))
        else:
            prepared_path=Path(deck).parent/"structural-mesh.npz"
            prepared=SolidMesh.load(prepared_path)
            if prepared.provenance.get("geometry_revision")!=revision:
                raise ValueError("Retained prepared mesh has a different geometry revision")
            if metadata.get("prepared_mesh_fingerprint")!=mesh_fingerprint(prepared):
                raise ValueError("Prepared mesh fingerprint differs from assembly metadata")
            if metadata.get("solver_mesh_fingerprint")!=mesh_fingerprint(mesh):
                raise ValueError("Solver mesh fingerprint differs from assembly metadata")
            receipt=verify_mesh_geometry_revision(project,prepared,mesh_manifest)
            frozen_receipt=metadata.get("geometry_revision_verification",{})
            if frozen_receipt.get("verified") is not True or frozen_receipt.get("geometry_revision")!=revision:
                raise ValueError("Assembly did not record a verified geometry revision")
            transform=_verify_pretension_mesh_transform(prepared,mesh,metadata)
            result.update(receipt,prepared_mesh=file_provenance(prepared_path),solver_transform=transform)
        if result.get("verified") is not True:
            raise ValueError("Geometry revision verifier did not verify the retained evidence")
    except (OSError,ValueError,KeyError,TypeError) as exc:
        result.update(verified=False,error=str(exc))
    return result


def summarize_run(results_path, *, project_path=None, mesh_path=None):
    """Read retained run evidence into fixed study metrics; never solve or accept it.

    Defaults locate the frozen project and solver mesh beside the retained run
    manifest. Reviewed reports may live elsewhere; explicit paths still permit
    manually prepared meshes/runs.
    """
    from .mesh import SolidMesh
    from .backends import contact_controls
    from .provenance import function_code_sha256
    from .results import friction_configuration_evidence, read_component_vtu, report_is_accepted, retained_result_source_binding, retained_component_field_binding
    results_path=Path(results_path).resolve()
    report=json.loads(results_path.read_text(encoding="utf-8"))
    result_binding=retained_result_source_binding(report,results_path)
    field_binding=retained_component_field_binding(report,results_path)
    run_source=Path(report["source_run"])
    if not run_source.is_absolute():run_source=results_path.parent/run_source
    run_source=run_source.resolve()
    run_dir=run_source.parent
    project_path=Path(project_path).resolve() if project_path else run_dir/"project.json"
    mesh_path=Path(mesh_path).resolve() if mesh_path else run_dir/"solver-mesh.npz"
    project=StructuralProject.load(project_path)
    mesh=SolidMesh.load(mesh_path)
    run=json.loads(run_source.read_text(encoding="utf-8"))
    deck=Path(run["input_path"])
    metadata=json.loads(deck.with_suffix(".assembly.json").read_text(encoding="utf-8"))
    recorded_mesh=deck.parent/metadata.get("solver_mesh","solver-mesh.npz")
    mesh_binding=mesh_path==recorded_mesh.resolve() or (recorded_mesh.is_file() and mesh_fingerprint(mesh)==mesh_fingerprint(SolidMesh.load(recorded_mesh)))
    provenance={name:file_provenance(path) for name,path in (("results",results_path),("project",project_path),("mesh",mesh_path),("run",run_source))}
    provenance.update({"native"+extension:row for extension,row in result_binding["files"].items()})
    provenance.update({"field/"+name:row for name,row in field_binding["files"].items()})
    summary={"schema_version":1,"name":project.name,"source_files":provenance,
             "numerically_accepted":report_is_accepted(report) and result_binding["verified"] and field_binding["verified"],"solver_status":report.get("solver_status"),
             "retained_result_source_binding_verified":result_binding["verified"],"retained_result_source_binding":result_binding,
             "retained_component_field_binding_verified":field_binding["verified"],"retained_component_field_binding":field_binding,
             "single_run_acceptance":{key:copy.deepcopy(report.get(key)) for key in
                 ("accepted","diagnostic_only","solver_status","assessment")},
             "unaccepted_gates":[gate for gate in report.get("assessment",{}).get("gates",[]) if gate.get("status")!="passed"],
             "project_binding_verified":metadata.get("project_fingerprint")==project.fingerprint(),
             "mesh_binding_verified":mesh_binding,
             "geometry_revision":getattr(project,"geometry_revision",None),"geometry_revision_binding_verified":None,
             "numerical_controls_fingerprint":numerical_controls_fingerprint(project),
             "step_controls":project.static_step_controls(),
             "step_controls_binding_verified":metadata.get("step_controls")==project.static_step_controls(),
             "physical_model_fingerprint":physical_model_fingerprint(project),
             "controlled_fingerprints":{kind:physical_model_fingerprint(project,exclude_parameter=kind) for kind in ("contact","friction","load","preload","patch")},
             "metric_definition_sha256":_definition_fingerprint(),"mesh_fingerprint":mesh_fingerprint(mesh),
             "solver_sha256":run.get("executable_sha256"),"nodes":len(mesh.node_ids),"elements":len(mesh.element_ids),
             "thickness_size_mm":project.mesh.thickness_size_mm,"contact_size_mm":project.mesh.contact_size_mm,
             "bore_snout_size_mm":getattr(project.mesh,"bore_snout_size_mm",None),
             "effective_bore_snout_size_mm":getattr(project.mesh,"bore_snout_size_mm",None) or project.mesh.global_size_mm,
             "slot_size_mm":getattr(project.mesh,"slot_size_mm",None),
             "effective_slot_size_mm":getattr(project.mesh,"slot_size_mm",None) or project.mesh.global_size_mm,
             "far_size_mm":project.mesh.global_size_mm,
             "normal_stiffness_mpa_per_mm":project.contacts.normal_stiffness_mpa_per_mm,
             "friction_coefficient":project.contacts.friction_coefficient,
             "service_force_n":project.loads.service_force_n,"screw_preloads_n":project.loads.screw_preloads_n,
             "patch_width_mm":project.loads.patch_width_mm,"states":{},"limitations":[]}
    controls=contact_controls.verify_contact_control_binding(project,deck.read_text(encoding="utf-8"),metadata)
    controls["implementation_provenance"]={**contact_controls.IMPLEMENTATION_PROVENANCE,
        "loaded_function_code_sha256":function_code_sha256(contact_controls.verify_contact_control_binding)}
    summary["contact_count_relative_tolerance"]=project.contact_count_relative_tolerance()
    summary["contact_controls_binding"]=controls
    summary["contact_controls_binding_verified"]=controls["verified"]
    if project.contacts.friction_coefficient>0:
        friction=friction_configuration_evidence(deck,metadata,run)
        summary["friction_qualification_evidence"]=friction
        summary["single_run_acceptance"].update({key:copy.deepcopy(report.get(key)) for key in
            ("friction_required","friction_coefficient","friction_qualification_evidence")})
        summary["numerically_accepted"]=(summary["numerically_accepted"] and isinstance(friction,dict)
            and friction.get("verified") is True and report.get("friction_coefficient")==project.contacts.friction_coefficient)
        if not summary["numerically_accepted"]:summary["limitations"].append("Nonzero friction lacks current complete numerical/qualification evidence")
    manifest_path=Path(mesh.provenance.get("mesh_manifest",""))
    summary["mesh_settings_verified"]=None
    if manifest_path.is_file():
        mesh_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["geometry_revision_evidence"]=_geometry_revision_evidence(project,mesh,mesh_manifest,metadata,deck)
        summary["geometry_revision_binding_verified"]=summary["geometry_revision_evidence"]["verified"]
        settings=mesh_manifest.get("settings",{})
        summary["wall_resolution"]=mesh_manifest.get("pipe_wall_resolution",{})
        summary["anchor_size_mm"]=settings.get("anchor_size_mm")
        summary["surface_resolution"]=_surface_resolution_evidence(manifest_path,mesh_manifest)
        summary["bore_snout_resolution"]=_bore_snout_resolution_evidence(manifest_path,mesh_manifest)
        summary["bore_snout_area_resolution"]=_bore_snout_area_resolution_evidence(manifest_path)
        summary["slot_resolution"]=_slot_resolution_evidence(manifest_path,mesh_manifest)
        for key in ("source","cad_topology"):
            evidence=summary["slot_resolution"].get(key)
            if evidence:summary["source_files"]["slot_resolution/"+key]=evidence
        summary["radial_layer_continuity_evidence"]=_radial_layer_continuity_evidence(manifest_path,mesh_manifest,mesh)
        summary["radial_layer_continuity_verified"]=summary["radial_layer_continuity_evidence"]["verified"]
        checks={"thickness_size_mm":project.mesh.thickness_size_mm,"contact_size_mm":project.mesh.contact_size_mm,
                "far_size_mm":project.mesh.global_size_mm,"order":project.mesh.element_order}
        summary["mesh_settings_verified"]=all(key in settings and np.isclose(settings[key],value,rtol=0,atol=1e-12) for key,value in checks.items())
        declared_bore=getattr(project.mesh,"bore_snout_size_mm",None)
        recorded_bore=settings.get("bore_snout_size_mm")
        bore_matches=(recorded_bore is None if declared_bore is None else isinstance(recorded_bore,(int,float)) and np.isclose(recorded_bore,declared_bore,rtol=0,atol=1e-12))
        summary["mesh_settings_verified"]=bool(summary["mesh_settings_verified"] and bore_matches)
        summary["bore_snout_settings_verified"]=bool(bore_matches)
        declared_slot=getattr(project.mesh,"slot_size_mm",None)
        recorded_slot=settings.get("slot_size_mm")
        slot_matches=(recorded_slot is None if declared_slot is None else isinstance(recorded_slot,(int,float)) and not isinstance(recorded_slot,bool) and np.isclose(recorded_slot,declared_slot,rtol=0,atol=1e-12))
        summary["mesh_settings_verified"]=bool(summary["mesh_settings_verified"] and slot_matches)
        summary["slot_settings_verified"]=bool(slot_matches)
        summary["mesh_settings_evidence"]={"source":file_provenance(manifest_path),"recorded":{key:settings.get(key) for key in checks},"declared":checks}
        correction_path=deck.parent/"provenance-correction.json"
        if summary["mesh_settings_verified"] is False and correction_path.is_file():
            correction=json.loads(correction_path.read_text(encoding="utf-8"))
            original_mesh=deck.parent/"structural-mesh.npz"
            actual=correction.get("actual_mesh",{})
            corrected_checks={"thickness_size_mm":actual.get("thickness_size_mm"),"contact_size_mm":actual.get("contact_size_mm"),
                              "far_size_mm":actual.get("global_size_mm"),"order":actual.get("element_order")}
            manifest_evidence=correction.get("verified_preparation",{}).get("mesh_settings_binding",{}).get("manifest",{})
            verified=(correction.get("original_project_sha256")==provenance["project"]["sha256"]
                      and correction.get("input_deck_sha256")==file_provenance(deck)["sha256"]
                      and original_mesh.is_file() and correction.get("original_mesh_sha256")==file_provenance(original_mesh)["sha256"]
                      and correction.get("requested_mesh")==project.to_dict()["mesh"]
                      and manifest_evidence.get("sha256")==summary["mesh_settings_evidence"]["source"]["sha256"]
                      and bore_matches and slot_matches
                      and all(value is not None and key in settings and np.isclose(settings[key],value,rtol=0,atol=1e-12) for key,value in corrected_checks.items()))
            summary["mesh_settings_correction"]={"source":file_provenance(correction_path),"verified":bool(verified),"reason":correction.get("reason")}
            if verified:
                summary["mesh_settings_verified"]=True
                summary["thickness_size_mm"]=actual["thickness_size_mm"]
                summary["contact_size_mm"]=actual["contact_size_mm"]
                summary["far_size_mm"]=actual["global_size_mm"]
    for state in STUDY_METRIC_DEFINITION["states"]:
        stage=report.get("stages",{}).get(state,{})
        values={"max_displacement_mm":None,"bolt_force_n":None,"contact_force_n":None,
                "section_force_n":None,"section_moment_nmm":None,"regions":None}
        bolts=report.get("pretension",{}).get("screws",{})
        bolt_values=[]
        for index in (1,2,3):
            rows=bolts.get(f"PRETENSION_BOLT_{index}",{}).get("history",[])
            row=next((row for row in rows if stage.get("time") is not None and np.isclose(row["time"],stage["time"],rtol=1e-8,atol=1e-10)),None)
            bolt_values.append(None if row is None else row.get("direct_cut_force_N"))
        if all(value is not None for value in bolt_values):values["bolt_force_n"]=bolt_values
        contact={f"{pair['slave']}__{pair['master']}":pair.get("statistics",{}).get("CF",{}).get("force_on_slave_N")
                 for pair in stage.get("contact",{}).get("pairs",[])}
        expected_pairs={"PIPE_BORE__HUB_SNOUT",*[f"SCREW_{i}_SEAT__SCREW_{i}_CONE" for i in (1,2,3)]}
        if set(contact)==expected_pairs and all(value is not None for value in contact.values()):values["contact_force_n"]=contact
        item=stage.get("components",{}).get("pipe",{})
        if item.get("status")=="available" and item.get("vtu"):
            source=Path(item["vtu"])
            if not source.is_absolute():source=results_path.parent/source
            try:
                component=read_component_vtu(source,"pipe",time=stage["time"],solver_status=report.get("solver_status","unknown"))
                sampled=component.field_mesh()
                values["max_displacement_mm"]=float(np.linalg.norm(component.displacement_mm,axis=1).max())
                values["regions"]=region_measures(sampled)
                values["section_force_n"]={};values["section_moment_nmm"]={}
                for distance in STUDY_METRIC_DEFINITION["sections_distance_from_midpoint_mm"]:
                    section=integrate_section(sampled,[component.domain],[0,0,-distance],[0,0,1])
                    if section["area_mm2"]<=0:raise ValueError(f"Missing physical section at {distance} mm")
                    values["section_force_n"][str(distance)]=section["force_N"]
                    values["section_moment_nmm"][str(distance)]=section["moment_Nmm"]
            except (ValueError,KeyError,OSError) as exc:
                summary["limitations"].append(f"{state} pipe metrics unavailable: {exc}")
        summary["states"][state]=values
    return summary


def _maximum_change(a,b,floor,*,vector=False):
    """Compare corresponding physical entries; a missing entry stays missing."""
    if a is None or b is None:return None
    if isinstance(a,dict) and isinstance(b,dict):
        if not a or a.keys()!=b.keys():return None
        changes=[_maximum_change(a[key],b[key],floor,vector=vector) for key in a]
        return None if any(value is None for value in changes) else max(changes)
    try:a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    except (ValueError,TypeError):return None
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():return None
    if vector:
        if a.shape!=(3,):return None
        return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),floor))
    return float(np.max(abs(a-b)/np.maximum(abs(b),floor)))


def _response_changes(a,b,case,reference_case):
    """One frozen both-state response recipe for isolated and combined checks."""
    changes=[]
    definitions=[("max_displacement_mm",.001,.02,False),("bolt_force_n",1.,.02,False),("contact_force_n",1.,.02,True),
                 ("section_force_n",1.,.02,True),("section_moment_nmm",233.68,.02,True)]
    for state in STUDY_METRIC_DEFINITION["states"]:
        left=a.get("states",{}).get(state,{});right=b.get("states",{}).get(state,{})
        for metric,floor,limit,vector in definitions:
            changes.append({"case":case,"reference_case":reference_case,"state":state,"metric":metric,
                "relative_change":_maximum_change(left.get(metric),right.get(metric),floor,vector=vector),"floor":floor,"limit":limit})
        for region in ("center","slots","seats"):
            av=(left.get("regions") or {}).get(region,{}).get("volume_mean_nodal_von_mises_mpa")
            bv=(right.get("regions") or {}).get(region,{}).get("volume_mean_nodal_von_mises_mpa")
            changes.append({"case":case,"reference_case":reference_case,"state":state,"metric":f"{region}_finite_volume_stress_mpa",
                "relative_change":_maximum_change(av,bv,.1),"floor":.1,"limit":.05})
    return changes


def assess_study(summaries, study, *, baseline_index=None):
    """Aggregate immutable metric summaries; physical sweeps are diagnostic only.

    Mesh comparisons use the last two ordered coarse-to-fine cases. Contact
    stiffness compares every case to the middle stiffness (or explicit baseline),
    requires one unchanged semantic mesh and a frictionless contact law.
    """
    if study not in ("mesh","contact_mesh","bore_contact_mesh","slot_mesh","contact","friction","load","preload","patch"):raise ValueError("Unsupported study")
    summaries=list(summaries);spatial=study in ("mesh","contact_mesh","bore_contact_mesh","slot_mesh");numerical=spatial or study=="contact"
    gates=[]
    def gate(name,passed,detail,measured=None,limit=True):
        gates.append({"name":name,"status":"passed" if passed is True else "missing" if passed is None else "failed",
                      "measured":passed if measured is None else measured,"limit":limit,"detail":detail})
    required=3 if study in ("mesh","contact_mesh","bore_contact_mesh","contact","slot_mesh") else 2
    gate("case_count",len(summaries)>=required,f"At least {required} retained cases",len(summaries),required)
    from .results import report_is_accepted
    def accepted_case(row):
        coefficient=row.get("friction_coefficient",0)
        if not isinstance(coefficient,(int,float)) or isinstance(coefficient,bool) or not np.isfinite(coefficient) or coefficient<0:return False
        if coefficient>0:
            acceptance=row.get("single_run_acceptance",{})
            if acceptance.get("friction_coefficient")!=coefficient or not report_is_accepted(acceptance):return False
        return row.get("numerically_accepted") is True
    gate("all_cases_numerically_accepted",bool(summaries) and all(accepted_case(row) for row in summaries),"Every requested case must pass every independent assembly gate, including qualified local Coulomb checks for nonzero friction")
    gate("retained_result_source_binding",bool(summaries) and all(row.get("retained_result_source_binding_verified") is True for row in summaries),
         "All eight report-bound native files must remain unchanged, with the exact deck/run path and completed native status")
    gate("retained_component_field_binding",bool(summaries) and all(row.get("retained_component_field_binding_verified") is True for row in summaries),
         "Every component VTU in both final states must match the hash pinned during result export")
    gate("frozen_project_binding",bool(summaries) and all(row.get("project_binding_verified") is True for row in summaries),"Each frozen project fingerprint must match its actual assembly metadata")
    gate("solver_mesh_binding",bool(summaries) and all(row.get("mesh_binding_verified") is True for row in summaries),"Study mesh must match the actual solver mesh recorded by the assembly")
    gate("geometry_revision_binding",bool(summaries) and all(row.get("geometry_revision_binding_verified") is True for row in summaries),
         "Derived geometry requires verified definition, prepared-mesh audit and declared solver cut transformation; missing evidence cannot inherit the source STEP identity")
    revisions={json.dumps(row.get("geometry_revision"),sort_keys=True,separators=(",",":")) for row in summaries}
    gate("unchanged_geometry_revision",bool(revisions) and len(revisions)==1,
         "Original and derived geometry, or different repair definitions, cannot be mixed in any numerical or controlled physical study")
    gate("step_controls_binding",bool(summaries) and all(row.get("step_controls_binding_verified") is True for row in summaries),"Frozen numerical step controls must match actual assembly metadata")
    if any("contact_controls_binding_verified" in row or row.get("contact_count_relative_tolerance") is not None for row in summaries):
        gate("contact_controls_binding",bool(summaries) and all(_contact_controls_bound(row) for row in summaries),
             "Each explicit contact-count control must match frozen project, metadata and both actual step cards; undeclared controls fail")
    for name,key in (("metric_definitions","metric_definition_sha256"),("pinned_solver_build","solver_sha256"),("fixed_numerical_controls","numerical_controls_fingerprint")):
        values={row.get(key) for row in summaries}
        gate(name,bool(values) and len(values)==1 and None not in values,"All cases require the same identified definition/build")
    physical={row.get("physical_model_fingerprint") if spatial else row.get("controlled_fingerprints",{}).get(study) for row in summaries}
    gate("controlled_physical_model",bool(physical) and len(physical)==1 and None not in physical,"Only the declared study parameter may vary; source geometry, registration and other physics remain fixed")
    if spatial:
        size_key={"mesh":"thickness_size_mm","contact_mesh":"contact_size_mm","bore_contact_mesh":"effective_bore_snout_size_mm","slot_mesh":"effective_slot_size_mm"}[study]
        sizes=[_effective_slot_size(row) if study=="slot_mesh" else row.get(size_key) for row in summaries]
        ordered=all(isinstance(value,(int,float)) and np.isfinite(value) and value>0 for value in sizes) and all(a>b for a,b in zip(sizes,sizes[1:]))
        gate("coarse_to_fine_order",bool(sizes) and ordered,"Actual recorded size of the declared refinement parameter must strictly decrease")
        fixed_sizes={"mesh":("contact_size_mm","far_size_mm","anchor_size_mm","effective_bore_snout_size_mm"),
                     "contact_mesh":("thickness_size_mm","far_size_mm","anchor_size_mm","effective_bore_snout_size_mm"),
                     "bore_contact_mesh":("thickness_size_mm","contact_size_mm","far_size_mm","anchor_size_mm"),
                     "slot_mesh":("thickness_size_mm","contact_size_mm","far_size_mm","anchor_size_mm","effective_bore_snout_size_mm")}[study]
        gate("isolated_spatial_refinement",all(len({row.get(key) for row in summaries})==1 and summaries[0].get(key) is not None for key in fixed_sizes) if summaries else None,
             "Through-wall and contact-surface spatial refinement are separate studies; other mesh sizes stay fixed")
        if study!="slot_mesh":
            slot_sizes=[_effective_slot_size(row) for row in summaries]
            gate("fixed_slot_resolution_target",bool(slot_sizes) and None not in slot_sizes and len(set(slot_sizes))==1,
                 "Wall/contact/bore studies hold the optional slot target fixed; an absent target means the recorded background size")
        meshes=[row.get("mesh_fingerprint") for row in summaries]
        gate("distinct_meshes",None not in meshes and len(set(meshes))==len(meshes),"Reusing one mesh with different labels cannot establish refinement")
        gate("recorded_mesh_sizes",all(row.get("mesh_settings_verified") is True for row in summaries) if summaries else None,"Declared refinement must agree with original mesh-generation evidence")
        layers=[];resolved=[]
        for row in summaries:
            wall=row.get("wall_resolution",{});count=wall.get("enforced_radial_layers")
            radii=wall.get("radial_partition_radii_mm",[])
            valid=(isinstance(count,int) and count>=2 and len(radii)==count-1
                   and all(isinstance(value,(int,float)) and 17.5<value<19. for value in radii)
                   and all(a<b for a,b in zip(radii,radii[1:]))
                   and wall.get("tetrahedra_touching_both_bore_and_outer_wall")==0)
            resolved.append(valid);layers.append(count)
        layer_control=all(a<b for a,b in zip(layers,layers[1:])) if study=="mesh" and all(resolved) else len(set(layers))==1
        gate("enforced_wall_refinement",bool(resolved) and all(resolved) and layer_control,
             "Each mesh requires≥2 enforced radial layers and zero full-wall tetrahedra; layers increase only in the wall study and remain fixed during contact refinement")
        gate("conformal_internal_radial_interfaces",bool(summaries) and all(row.get("radial_layer_continuity_verified") is True for row in summaries),
             "Every internal interface triangle must share all six quadratic face nodes between adjacent solids and must not become a contact/tie surface")
        if study=="contact_mesh":
            gate("measured_contact_surface_refinement",_measured_contact_refinement(summaries,SCREW_CONTACT_SURFACES),
                 "All three screw seats and cones must have decreasing measured median/p95 corner-edge lengths and increasing triangle counts; bore/snout are outside this controlled field")
        elif study=="bore_contact_mesh":
            gate("fixed_bore_measurement_definition",_same_bore_measurement_definition(summaries),
                 "All area-weighted bore/snout measurements require the same identified rule and finite fixed axial band (1e-9 mm coordinate roundoff)")
            measured=[{"surface_resolution":row.get("bore_snout_area_resolution",{})} for row in summaries]
            gate("measured_bore_snout_refinement",_measured_contact_refinement(measured,("pipe_bore_snout_band","hub_snout"),
                 edge_metric="area_weighted_corner_edge_length_mm",count_metric="quadratic_triangles_with_positive_selected_area"),
                 "Both fixed physical surfaces require decreasing area-weighted median/p95 edge lengths and increasing positive-area triangle counts; raw triangle-count weighting changes under nonuniform subdivision")
            gate("paired_bore_snout_settings",bool(summaries) and all(row.get("bore_snout_settings_verified") is True for row in summaries)
                 and all(row.get("bore_snout_size_mm") is not None for row in summaries[1:]),
                 "The baseline may use the background far-field target; each refined case must apply the recorded local target to both bore and snout")
        elif study=="slot_mesh":
            gate("fixed_slot_measurement_definition",_same_slot_measurement_definition(summaries),
                 "All 30 traces require one hash-bound original topology, approved arc-length weighting and unchanged physical/source-face scope")
            gate("measured_slot_perimeter_refinement",_measured_slot_refinement(summaries),
                 "Each of six full perimeters and 24 corner traces must decrease weighted median/p95 span and increase quadratic-edge count; wall triangle counts are insufficient")
            gate("slot_settings",bool(summaries) and all(row.get("slot_settings_verified") is True for row in summaries)
                 and all(row.get("slot_size_mm") is not None for row in summaries[1:]),
                 "Baseline may retain the background target; each refined case must apply its recorded independent slot target")
        comparisons=[(len(summaries)-2,len(summaries)-1)] if len(summaries)>=2 else []
    else:
        if study!="patch":
            meshes={row.get("mesh_fingerprint") for row in summaries}
            gate("identical_mesh",bool(meshes) and len(meshes)==1 and None not in meshes,"Contact/material-load sensitivities must use one identical mesh and selections")
        key={"contact":"normal_stiffness_mpa_per_mm","friction":"friction_coefficient","load":"service_force_n","preload":"screw_preloads_n","patch":"patch_width_mm"}[study]
        parameters=[row.get(key) for row in summaries]
        gate("distinct_control_values",all(value is not None for value in parameters) and len({json.dumps(value,sort_keys=True) for value in parameters})==len(parameters),"Requested controlled parameter must actually differ")
        if study=="contact":gate("frictionless_penalty_study",all(row.get("friction_coefficient")==0 for row in summaries),"Current friction slope depends on penalty; a frictional penalty sweep is not isolated")
        if baseline_index is None:baseline_index=0 if study in ("friction","patch") else len(summaries)//2
        if summaries and not 0<=baseline_index<len(summaries):raise ValueError("Baseline index outside run list")
        if study=="contact" and summaries:
            finite=all(isinstance(value,(int,float)) and np.isfinite(value) and value>0 for value in parameters)
            gate("penalty_brackets_baseline",finite and min(parameters)<parameters[baseline_index]<max(parameters),"Contact sensitivity must include lower and higher penalty than its baseline")
        comparisons=[(index,baseline_index) for index in range(len(summaries)) if index!=baseline_index]
    changes=[]
    for left,right in comparisons:
        for entry in _response_changes(summaries[left],summaries[right],left,right):
            change,limit=entry["relative_change"],entry["limit"]
            if numerical:
                metric=entry["metric"].removesuffix("_mpa")
                gate(f"{left}_vs_{right}/{entry['state']}/{metric}",None if change is None else change<=limit,
                     "Fixed physical response and unchanged tensor quadrature",change,limit)
            else:entry["limit"]=None
            changes.append(entry)
    if not numerical:
        gate("response_metrics_available",bool(changes) and all(change["relative_change"] is not None for change in changes),"Physical sensitivities report response changes without a numerical-convergence threshold")
    passed=bool(gates) and all(item["status"]=="passed" for item in gates)
    status=("passed" if numerical else "sensitivity_completed") if passed else ("incomplete" if len(summaries)<required else "unaccepted")
    peaks=[{"case":index,"state":state,"regions":{name:{"peak_mpa":region.get("raw_display_peak_mpa"),"position_mm":region.get("raw_peak_location_mm")} for name,region in (values.get("regions") or {}).items()}}
           for index,row in enumerate(summaries) for state,values in row.get("states",{}).items()]
    return {"study":study,"status":status,"numerical_convergence_accepted":passed and numerical,
            "scope":{"mesh":"through_wall_refinement_only","contact_mesh":"screw_seat_and_head_surface_refinement_at_fixed_wall_layers","bore_contact_mesh":"paired_bore_and_snout_refinement_at_fixed_wall_and_screw_contact_sizes","slot_mesh":"six_slot_perimeters_and_24_corner_traces_at_fixed_wall_and_contact_sizes","contact":"contact_penalty_sensitivity"}.get(study,"physical_parameter_sensitivity"),
            "slot_trace_coverage":list(SLOT_TRACES) if study=="slot_mesh" else [],
            "slot_spatial_convergence":"passed" if study=="slot_mesh" and passed else "unaccepted" if study=="slot_mesh" else "not_assessed",
            "slot_measurement_definition":SLOT_MEASUREMENT_DEFINITION if study=="slot_mesh" else None,
            "controlled_contact_surfaces":list(SCREW_CONTACT_SURFACES) if study=="contact_mesh" else ["pipe_bore","hub_snout"] if study=="bore_contact_mesh" else [],
            "unassessed_contact_surfaces":["pipe_bore","hub_snout"] if study=="contact_mesh" else list(SCREW_CONTACT_SURFACES) if study=="bore_contact_mesh" else list(ALL_CONTACT_SURFACES),
            "release_mesh_accepted":False,
            "controlled_contact_surface_mesh_convergence":"passed" if study in ("contact_mesh","bore_contact_mesh") and passed else "not_assessed" if study not in ("contact_mesh","bore_contact_mesh") else "unaccepted",
            "design_adequacy":"not_assessed","metric_definition":STUDY_METRIC_DEFINITION,"gates":gates,"changes":changes,
            "raw_peak_status":"unresolved_local_concentration_or_singularity; no pass/fail threshold",
            "raw_peak_behavior":peaks,"cases":summaries,
            "interpretation":"Finite-volume convergence does not establish local strength or resolve a growing contact-edge peak"}


def assess_mesh_release(wall_study,contact_mesh_study,bore_contact_mesh_study=None,combined_candidate=None,*,slot_mesh_study=None):
    """Assess isolated families and disclose whether one combined mesh was verified.

    Existing positional arguments retain their wall/contact scope. Version 2
    full local spatial release additionally requires the keyword-only slot
    study. A separate candidate is checked against each finest isolated case.
    """
    wall=wall_study.get("cases",[]);contact=contact_mesh_study.get("cases",[])
    bore=(bore_contact_mesh_study or {}).get("cases",[])
    slot=(slot_mesh_study or {}).get("cases",[])
    issues=[]
    def accepted(report,kind):
        gates=report.get("gates",[])
        if not (report.get("study")==kind and report.get("status")=="passed" and report.get("numerical_convergence_accepted") is True
                and isinstance(gates,list) and gates and all(isinstance(gate,dict) and gate.get("status")=="passed" for gate in gates)):
            return False
        # Imported status flags cannot replace the current required assessment.
        try:recomputed=assess_study(report.get("cases",[]),kind)
        except (TypeError,ValueError,KeyError):return False
        return recomputed["numerical_convergence_accepted"] is True and {gate["name"] for gate in recomputed["gates"]}=={gate.get("name") for gate in gates} and len(gates)==len(recomputed["gates"])
    if not accepted(wall_study,"mesh"):issues.append("Three-level controlled wall study has not passed")
    contact_accepted=accepted(contact_mesh_study,"contact_mesh")
    if not contact_accepted:issues.append("Independent spatial contact refinement has not passed")
    # Derive the supported coverage from the implemented controlled study,
    # never from an editable imported report's coverage claim.
    covered=set(SCREW_CONTACT_SURFACES) if contact_accepted else set()
    bore_accepted=accepted(bore_contact_mesh_study or {},"bore_contact_mesh")
    if bore_accepted:covered.update(("pipe_bore","hub_snout"))
    else:issues.append("Independent paired bore/snout spatial refinement has not passed")
    uncovered=sorted(set(ALL_CONTACT_SURFACES)-covered)
    if uncovered:issues.append("Spatial contact refinement does not cover: "+", ".join(uncovered))
    if not wall or not contact or wall[-1].get("mesh_fingerprint") not in {row.get("mesh_fingerprint") for row in contact}:issues.append("Contact refinement does not include the finest wall-study mesh")
    if wall and any(row.get("thickness_size_mm")!=wall[-1].get("thickness_size_mm") for row in contact):issues.append("Contact refinement must retain the finest enforced wall resolution")
    if bore and (not wall or wall[-1].get("mesh_fingerprint") not in {row.get("mesh_fingerprint") for row in bore}):issues.append("Bore/snout refinement does not include the finest wall-study mesh")
    if wall and any(row.get("thickness_size_mm")!=wall[-1].get("thickness_size_mm") for row in bore):issues.append("Bore/snout refinement must retain the finest enforced wall resolution")
    for key in ("physical_model_fingerprint","solver_sha256","metric_definition_sha256","numerical_controls_fingerprint"):
        values={row.get(key) for row in wall+contact+bore}
        if len(values)!=1 or None in values:issues.append(f"Missing or inconsistent {key}")
    legacy_independent_passed=not issues
    legacy_combined=bool(contact and bore and contact[-1].get("mesh_fingerprint")==bore[-1].get("mesh_fingerprint"))
    if combined_candidate is not None:
        legacy_combined=_assess_combined_candidate(combined_candidate,wall,contact,bore)["accepted"]
    legacy_scope_accepted=legacy_independent_passed and legacy_combined
    slot_accepted=accepted(slot_mesh_study or {},"slot_mesh")
    if not slot_accepted:issues.append("Independent slot perimeter/corner spatial refinement has not passed; full local spatial release is incomplete")
    if slot and (not wall or wall[-1].get("mesh_fingerprint") not in {row.get("mesh_fingerprint") for row in slot}):
        issues.append("Slot refinement does not include the finest wall-study mesh")
    if wall and any(row.get("thickness_size_mm")!=wall[-1].get("thickness_size_mm") for row in slot):
        issues.append("Slot refinement must retain the finest enforced wall resolution")
    if slot:
        for key in ("physical_model_fingerprint","solver_sha256","metric_definition_sha256","numerical_controls_fingerprint"):
            values={row.get(key) for row in wall+contact+bore+slot}
            if len(values)!=1 or None in values:issues.append(f"Slot family has missing or inconsistent {key}")
    independent_passed=not issues
    combined_assessment=None
    combined=bool(contact and bore and slot and len({rows[-1].get("mesh_fingerprint") for rows in (contact,bore,slot)})==1)
    if combined_candidate is not None:
        combined_assessment=_assess_combined_candidate(combined_candidate,wall,contact,bore,slot=slot if slot_mesh_study is not None else None)
        combined=independent_passed and combined_assessment["accepted"]
        if not combined:issues.append("Explicit combined candidate has not passed all identity, resolution and both-state response checks")
    elif not combined:issues.append("The finest screw-contact, bore/snout and slot cases do not identify one verified combined release mesh")
    return {"scope_version":2,"status":"passed" if not issues else "unaccepted","release_mesh_accepted":not issues,
            "full_local_spatial_release_accepted":not issues,"issues":issues,
            "legacy_spatial_scope_accepted":legacy_scope_accepted,
            "legacy_scope":"version1 wall and screw/bore contact spatial refinement; excludes independent slot refinement",
            "controlled_contact_surfaces":sorted(covered),"unassessed_contact_surfaces":uncovered,
            "controlled_slot_traces":list(SLOT_TRACES) if slot_accepted else [],
            "unassessed_slot_traces":[] if slot_accepted else list(SLOT_TRACES),
            "independent_spatial_studies_accepted":legacy_independent_passed,
            "all_four_independent_spatial_studies_accepted":independent_passed,"combined_finest_configuration_verified":combined,
            "combined_candidate_assessment":combined_assessment,
            "design_adequacy":"not_assessed","scope":"wall, screw contact, bore/snout and slot local spatial refinement; contact penalty sensitivity and whole-phase release remain separate",
            "raw_peak_status":"unresolved_local_concentration_or_singularity"}


def _assess_combined_candidate(candidate,wall,contact,bore,*,slot=None):
    from .results import report_is_accepted
    gates=[];changes=[]
    def gate(name,passed,detail,measured=None,limit=True):
        gates.append({"name":name,"status":"passed" if passed is True else "missing" if passed is None else "failed",
                      "detail":detail,"measured":passed if measured is None else measured,"limit":limit})
    single=candidate.get("single_run_acceptance",{})
    coefficient=candidate.get("friction_coefficient",0)
    friction_matches=(isinstance(coefficient,(int,float)) and not isinstance(coefficient,bool) and np.isfinite(coefficient)
                      and coefficient>=0 and (coefficient==0 or single.get("friction_coefficient")==coefficient))
    gate("single_run_acceptance",candidate.get("numerically_accepted") is True and friction_matches and report_is_accepted(single),
         "Fresh candidate evidence must contain every mandatory single-run numerical gate exactly once")
    for key in ("retained_result_source_binding_verified","retained_component_field_binding_verified","project_binding_verified","mesh_binding_verified","geometry_revision_binding_verified",
                "step_controls_binding_verified","mesh_settings_verified","bore_snout_settings_verified","radial_layer_continuity_verified"):
        gate(key,candidate.get(key) is True,"Candidate must independently verify retained run, geometry and resolution evidence")
    if "contact_controls_binding_verified" in candidate or candidate.get("contact_count_relative_tolerance") is not None:
        gate("contact_controls_binding_verified",_contact_controls_bound(candidate),"Candidate contact-count setting is bound to project, metadata and both step cards")
    families={"wall":wall,"screw_contact":contact,"bore_contact":bore}
    if slot is not None:families["slot"]=slot
    gate("four_isolated_families" if slot is not None else "three_isolated_families",all(families.values()),"A combined candidate cannot replace an absent isolated family")
    if all(families.values()):
        for key in ("physical_model_fingerprint","solver_sha256","metric_definition_sha256","numerical_controls_fingerprint"):
            values={row.get(key) for rows in families.values() for row in rows}
            gate("same_"+key,len(values)==1 and None not in values and candidate.get(key) in values,
                 "Combined candidate retains the exact physical model, solver, controls and metric definitions")
        gate("same_geometry_revision",all(candidate.get("geometry_revision")==row.get("geometry_revision") for rows in families.values() for row in rows),
             "No original/derived geometry or repair-definition mixing")
        finest_wall=wall[-1];resolution=candidate.get("wall_resolution",{})
        expected=finest_wall.get("wall_resolution",{})
        gate("finest_wall_resolution",resolution.get("enforced_radial_layers")==expected.get("enforced_radial_layers")
             and resolution.get("radial_partition_radii_mm")==expected.get("radial_partition_radii_mm")
             and resolution.get("tetrahedra_touching_both_bore_and_outer_wall")==0,
             "Same finest enforced layers and conformal partitions; no full-wall tetrahedra")
        targets={"thickness_size_mm":finest_wall.get("thickness_size_mm"),"far_size_mm":finest_wall.get("far_size_mm"),
                 "anchor_size_mm":finest_wall.get("anchor_size_mm"),"contact_size_mm":contact[-1].get("contact_size_mm"),
                 "effective_bore_snout_size_mm":bore[-1].get("effective_bore_snout_size_mm")}
        gate("combined_finest_targets",all(value is not None and candidate.get(key)==value for key,value in targets.items())
             and candidate.get("bore_snout_size_mm")==bore[-1].get("bore_snout_size_mm"),
             "Use finest screw/bore targets together, holding wall, far field and anchorage fixed",
             {key:candidate.get(key) for key in targets},targets)
        expected_slot=_effective_slot_size(slot[-1] if slot is not None else finest_wall)
        gate("combined_slot_target",expected_slot is not None and _effective_slot_size(candidate)==expected_slot,
             "Selected finest slot target joins the combined mesh; legacy wall/contact scope holds slot target fixed",
             _effective_slot_size(candidate),expected_slot)
        gate("fixed_combined_bore_measurement_definition",_same_bore_measurement_definition([finest_wall,*bore,candidate]),
             "Candidate and all bore-family/common-baseline measurements use one identified area rule and physical band")
        # A coupled mesh need not reproduce the exact node-count-weighted
        # statistics of either isolated finest mesh. It must demonstrate actual
        # refinement of both interfaces against their common wall baseline.
        baseline_fingerprint=finest_wall.get("mesh_fingerprint")
        seat_base=next((row for row in contact if row.get("mesh_fingerprint")==baseline_fingerprint),None)
        bore_base=next((row for row in bore if row.get("mesh_fingerprint")==baseline_fingerprint),None)
        gate("measured_combined_screw_surface_refinement",
             _measured_contact_refinement([seat_base,candidate],SCREW_CONTACT_SURFACES) if seat_base else None,
             "Hash-bound candidate seat/cone measurements must improve the common wall baseline")
        measurements=[{"surface_resolution":row.get("bore_snout_area_resolution",{})} for row in (bore_base,candidate)] if bore_base else []
        gate("measured_combined_bore_surface_refinement",
             _measured_contact_refinement(measurements,("pipe_bore_snout_band","hub_snout"),edge_metric="area_weighted_corner_edge_length_mm",
                                          count_metric="quadratic_triangles_with_positive_selected_area") if bore_base else None,
             "Hash-bound area-weighted candidate bore/snout measurements must improve the common wall baseline")
        if slot is not None:
            gate("slot_settings_verified",candidate.get("slot_settings_verified") is True
                 and candidate.get("slot_size_mm")==slot[-1].get("slot_size_mm"),
                 "Candidate local slot target must match its actual prepared mesh and the isolated finest target")
            gate("fixed_combined_slot_measurement_definition",_same_slot_measurement_definition([finest_wall,*slot,candidate]),
                 "Candidate and slot family retain all 30 trace identities, topology and the approved weighting rule")
            slot_base=next((row for row in slot if row.get("mesh_fingerprint")==baseline_fingerprint),None)
            gate("measured_combined_slot_refinement",_measured_slot_refinement([slot_base,candidate]) if slot_base else None,
                 "Every candidate slot perimeter/corner trace must improve its common wall baseline")
        for name,rows in families.items():
            for change in _response_changes(candidate,rows[-1],"combined_candidate",name+"_finest"):
                measured,limit=change["relative_change"],change["limit"]
                gate(f"vs_{name}/{change['state']}/{change['metric']}",None if measured is None else measured<=limit,
                     "Both saved states compared independently to this family's finest case",measured,limit)
                changes.append(change)
    return {"accepted":bool(gates) and all(row["status"]=="passed" for row in gates),"gates":gates,"changes":changes,
            "candidate":candidate,"scope":"Combined spatial configuration confirmation; contact-penalty qualification remains separate"}


def assess_existing_runs(runs, study, output_dir=None, *, baseline_index=None):
    """Assess ordered records: {'results': path, 'project': path?, 'mesh': path?}.

    A bare results.json path is also accepted. Failures stay in the report and
    prevent acceptance. Existing output directories are never overwritten.
    """
    summaries=[]
    for item in runs:
        record={"results":item} if isinstance(item,(str,Path)) else item
        try:summary=summarize_run(record["results"],project_path=record.get("project"),mesh_path=record.get("mesh"))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            summary={"name":str(record.get("results","missing results")),"numerically_accepted":False,"error":str(exc),"states":{}}
        summaries.append(summary)
    report=assess_study(summaries,study,baseline_index=baseline_index)
    if output_dir is not None:
        output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
        write_json(output/"study-assessment.json",report)
        import csv
        if report["changes"]:
            with (output/"controlled-response-changes.csv").open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(report["changes"][0]));writer.writeheader();writer.writerows(report["changes"])
    return report


def write_study_projects(base: StructuralProject, study: str, output: Path) -> dict:
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for name,project in controlled_cases(base,study):
        project.save(output/(name+".json"))
        cases.append({"name":name,"project":name+".json","physical_model_fingerprint":physical_model_fingerprint(project),"status":"not_run"})
    manifest={"study":study,"cases":cases,"status":"incomplete","definition":"one numerical/footprint study at a time; fixed source geometry and materials"}
    write_json(output/"study.json",manifest)
    return manifest


def run_study(base: StructuralProject, study: str, output: Path, *, existing_mesh: Path | None = None) -> dict:
    """Execute controlled cases serially, persisting each failure before continuing.

    Contact studies reuse exactly one mesh. Mesh studies hold penalty/friction
    fixed. The orchestrator never promotes a completed but unaccepted case.
    """
    import subprocess
    import sys
    output=Path(output).resolve()
    existing_mesh=Path(existing_mesh).resolve() if existing_mesh is not None else None
    if study in ("contact","friction","load","preload") and existing_mesh is None:
        raise ValueError("Contact/friction/load/preload sensitivity requires an existing identical mesh")
    if study in ("mesh","contact_mesh","bore_contact_mesh","slot_mesh","patch") and existing_mesh is not None:
        raise ValueError("Mesh/patch studies require a separately generated mesh for each case")
    manifest=write_study_projects(base,study,output)

    def command(case_dir,name,arguments):
        # Isolate Gmsh/OCP/solver/result-reader lifetimes between commands. Keep
        # the process in the same parent tree so GUI cancellation reaches it.
        args=[sys.executable,"-m","structural",*map(str,arguments)]
        stdout=case_dir/(name+".stdout.log");stderr=case_dir/(name+".stderr.log")
        evidence={"arguments":args,"cwd":str(Path(__file__).resolve().parent.parent),
                  "stdout":str(stdout),"stderr":str(stderr),"status":"running"}
        record=case_dir/(name+".command.json")
        write_json(record,evidence)
        try:
            with stdout.open("w",encoding="utf-8") as out,stderr.open("w",encoding="utf-8") as err:
                completed=subprocess.run(args,cwd=evidence["cwd"],stdout=out,stderr=err,
                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),check=False)
            evidence.update(status="completed" if completed.returncode==0 else "failed",exit_code=completed.returncode)
            return completed.returncode
        except BaseException as exc:
            evidence.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=str(exc))
            raise
        finally:write_json(record,evidence)

    for case in manifest["cases"]:
        case_dir=output/case["name"]
        case_dir.mkdir()
        project_path=output/case["project"]
        try:
            mesh=existing_mesh
            if mesh is None:
                mesh_dir=case_dir/"mesh-preparation"
                mesh_status=command(case_dir,"mesh",["mesh",project_path,"--output",mesh_dir])
                if mesh_status:
                    raise ValueError(f"Mesh preparation failed with status {mesh_status}")
                mesh=mesh_dir/"structural-mesh.npz"
            solve_dir=case_dir/"assembly-solve"
            code=command(case_dir,"solve",["solve",project_path,mesh,"--output",solve_dir])
            results_file=solve_dir/"structural-results/results.json"
            if results_file.exists():
                results=json.loads(results_file.read_text(encoding="utf-8"))
                case.update(status=results["assessment"]["status"],solver_status=results["solver_status"],results=str(results_file),
                            frozen_project=str(solve_dir/"project.json"),solver_mesh=str(solve_dir/"solver-mesh.npz"),metrics=results["metrics"])
            else:
                case.update(status="failed",exit_code=code,results=None)
        except Exception as exc:
            case.update(status="failed",error=str(exc))
        write_json(output/"study.json",manifest)
        print(json.dumps({"case":case["name"],"status":case["status"]}),flush=True)
    assessment=assess_existing_runs([{"results":case.get("results"),"project":case.get("frozen_project"),"mesh":case.get("solver_mesh")} for case in manifest["cases"]],
                                    study,output/"convergence-assessment")
    manifest["status"]=assessment["status"]
    manifest["numerical_convergence_accepted"]=assessment["numerical_convergence_accepted"]
    manifest["assessment"]="convergence-assessment/study-assessment.json"
    write_json(output/"study.json",manifest)
    return manifest
