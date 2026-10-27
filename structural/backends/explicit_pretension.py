"""Equivalent shoulder pretension with explicitly duplicated head-side nodes.

For a shoulder cut terminating on an exposed annulus, every head-side element
touching a cut node (including edge-only neighbours) must use the duplicate.
The scalar reference coordinate is the signed tightening jump. Its conjugate
load is total section force. Stage B fixes this coordinate, not the bolt force.
"""
from __future__ import annotations
from pathlib import Path
import json


def explicit_pretension(mesh, cut_surface: str, head_element_ids, axis, reference_node: int, *, reserved_node_ids=()):
    """Return ``(new_mesh, equation_text, node_map)`` for one physical cut.

    ``head_element_ids`` MUST include all physical head-side cells, including
    shoulder annulus cells which touch a cut node only at an edge or vertex.
    Axis points outward from the selected head-side cut toward the anchorage;
    positive scalar force therefore creates tensile section force. Reserve all
    reference-node IDs before the first call when processing multiple bolts.
    Reference nodes are emitted by the assembly writer, not added to solid mesh.
    """
    import copy
    import numpy as np
    vector=np.asarray(axis,dtype=float)
    if vector.shape!=(3,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1.,atol=1e-8):
        raise ValueError("Pretension axis must be a finite unit vector")
    if reference_node in mesh._nodes:
        raise ValueError("Pretension reference node must not belong to the physical mesh")
    head=set(map(int,head_element_ids))
    if not head or not head.issubset(mesh._elements):
        raise ValueError("Pretension requires the complete, valid head-side element selection")
    cut_nodes=mesh.surface_nodes(cut_surface)
    if not cut_nodes:
        raise ValueError("Pretension cut is empty")
    cutset=set(cut_nodes)
    used_head={int(n) for e in head for n in mesh.tetra10[mesh._elements[e]]}
    used_other={int(n) for e,row in zip(mesh.element_ids,mesh.tetra10) if int(e) not in head for n in row}
    if not cutset.issubset(used_head & used_other):
        raise ValueError("Each cut node must connect a head-side and an anchorage-side element")
    start=max([int(mesh.node_ids.max()),int(reference_node),*map(int,reserved_node_ids)])+1
    mapping={n:start+i for i,n in enumerate(cut_nodes)}
    result=copy.deepcopy(mesh)
    result.node_ids=np.concatenate((mesh.node_ids,np.asarray(list(mapping.values()),dtype=np.int64)))
    result.points_mm=np.vstack((mesh.points_mm,mesh.points_mm[[mesh._nodes[n] for n in cut_nodes]]))
    for e in head:
        idx=mesh._elements[e]
        result.tetra10[idx]=[mapping.get(int(n),int(n)) for n in mesh.tetra10[idx]]
    result.node_sets[cut_surface+"_head_nodes"]=list(mapping.values())
    result.node_sets[cut_surface+"_shaft_nodes"]=list(mapping.keys())
    lines=[]
    for original,clone in mapping.items():
        for dof,component in enumerate(vector,1):
            terms=f"{clone},{dof},1.,{original},{dof},-1."
            if abs(component)>1e-14:
                lines += ["*EQUATION","3",terms+f",{reference_node},1,{-component:.15g}"]
            else:
                lines += ["*EQUATION","2",terms]
    result.provenance.setdefault("explicit_pretension_cuts",[]).append({"cut_surface":cut_surface,"reference_node":reference_node,
             "axis":vector.tolist(),"head_element_count":len(head),"node_map":{str(k):v for k,v in mapping.items()},
             "formulation":"u_head - u_shaft = axis * scalar_tightening; scalar force work-conjugate to jump"})
    result.__post_init__()
    return result,"\n".join(lines)+"\n",mapping


def convert_shoulder_fixture(native_deck: str | Path, destination: str | Path):
    """Exact-dimension diagnostic deck conversion; fixture-local X cut only."""
    native_deck=Path(native_deck)
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    raw=native_deck.read_text().splitlines()
    source=[]
    expected=0
    pending=[]
    for line in raw:
        if line.startswith("*"):
            if pending:
                raise ValueError("Incomplete element connectivity")
            expected=int(line.split("TYPE=C3D")[-1].split(",")[0]) if line.startswith("*ELEMENT,TYPE=C3D") else 0
            source.append(line)
        elif expected:
            pending.extend(v.strip() for v in line.split(",") if v.strip())
            if len(pending)==expected+1:
                source.append(",".join(pending))
                pending=[]
            elif len(pending)>expected+1:
                raise ValueError("Unexpected element connectivity length")
        else:
            source.append(line)
    nodes={}
    elements={}
    groups={}
    mode=None
    group=None
    cut=[]
    reference=None
    for line in source:
        if line.startswith("*"):
            mode="node" if line=="*NODE" or line.startswith("*NODE,") else "element" if line.startswith("*ELEMENT,") else "cut" if line=="*SURFACE,NAME=CUT" else None
            if mode=="element":
                group=line.split("ELSET=")[-1]
                groups.setdefault(group,[])
            if line.startswith("*PRE-TENSION SECTION"):
                reference=int(line.split("NODE=")[-1])
        elif mode=="node":
            tokens=line.split(",")
            nodes[int(tokens[0])]=tuple(map(float,tokens[1:4]))
        elif mode=="element":
            tokens=list(map(int,line.split(",")))
            elements[tokens[0]]=tokens[1:]
            groups[group].append(tokens[0])
        elif mode=="cut":
            e,f=line.split(",")
            cut.append((int(e),int(f[1:])))
    if reference is None:
        raise ValueError("Native pretension definition not found")
    cut_nodes={n for e,f in cut for n in elements[e] if abs(nodes[n][0])<1e-7}
    duplicates={n:max(nodes)+1+i for i,n in enumerate(sorted(cut_nodes))}
    head_elements={e for e in groups["BOLT"] if sum(nodes[n][0] for n in elements[e][:4])/4<0}
    # This includes shoulder annulus cells touching only the cut perimeter.
    for e in head_elements:
        elements[e]=[duplicates.get(n,n) for n in elements[e]]
    result=[]
    mode=None
    skip_normal=False
    inserted=False
    for line in source:
        if skip_normal:
            skip_normal=False
            continue
        if line.startswith("*PRE-TENSION SECTION"):
            skip_normal=True
            continue
        if line.startswith("*"):
            if not inserted and line.startswith("*ELEMENT"):
                result += ["*NODE,NSET=EXPLICIT_CUT_HEAD"]+[f"{duplicates[n]},"+",".join(f"{v:.12g}" for v in nodes[n]) for n in sorted(cut_nodes)]
                for n,clone in duplicates.items():
                    result += ["*EQUATION","3",f"{clone},1,1.,{n},1,-1.,{reference},1,-1.","*EQUATION","2",f"{clone},2,1.,{n},2,-1.","*EQUATION","2",f"{clone},3,1.,{n},3,-1."]
                inserted=True
            mode="element" if line.startswith("*ELEMENT,") else None
            result.append(line)
        elif mode=="element":
            eid=int(line.split(",")[0])
            connectivity=elements[eid]
            if len(connectivity)>15:
                result += [f"{eid},"+",".join(map(str,connectivity[:15]))+",",",".join(map(str,connectivity[15:]))]
            else:
                result.append(f"{eid},"+",".join(map(str,connectivity)))
        else:
            result.append(line)
    output=destination/"short_screw.inp"
    output.write_text("\n".join(result)+"\n",encoding="ascii")
    manifest=json.loads((native_deck.parent/"mesh-manifest.json").read_text())
    manifest.update(pretension_formulation="explicit paired-node displacement jump; scalar reference conjugate force",
                    head_side_cut_nodes=len(duplicates),head_side_elements=len(head_elements),
                    duplicates=duplicates,source_native_deck=str(native_deck.resolve()))
    (destination/"mesh-manifest.json").write_text(json.dumps(manifest,indent=2))
    return output
