"""Diagnosed native pretension feasibility for the CAD screw shoulder.

The screw is reconstructed from kernel-verified analytic dimensions. A full
annular seat is a deliberately simple fixture, not the curved/slotted pipe.
Full 6 mm shank-cylinder anchorage is tied to a fixed rigid foundation. The
pretension disk matches the shoulder plane/radius in the reference split.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from .calculix import CalculixRunner, parse_dat


def build_deck(output: str | Path, size=0.22, order=2, quarter=False, rim_refinement=False):
    import gmsh
    import numpy as np
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal",0)
        gmsh.option.setNumber("General.NumThreads",2)
        gmsh.model.add("shoulder_native_pretension")
        o=gmsh.model.occ
        radius_head=5.27833777980134
        radius_shoulder=4.5503973112689
        head=o.addCylinder(-5,0,0,3,0,0,radius_head)
        cone=o.addCone(-2,0,0,2,0,0,radius_head,radius_shoulder)
        head_cone=o.fuse([(3,head)],[(3,cone)])[0]
        shaft=o.addCylinder(0,0,0,6,0,0,4)
        core=o.addCylinder(-.7,0,0,.7,0,0,4)
        bolt_parts=o.fragment(head_cone,[(3,shaft),(3,core)])[0]
        bolt_volumes=[tag for dim,tag in bolt_parts if dim==3]
        outer_hub=o.addCylinder(0,0,0,6,0,0,7)
        inner_hub=o.addCylinder(0,0,0,6,0,0,4)
        hub=o.cut([(3,outer_hub)],[(3,inner_hub)])[0]
        # A separate, elastically deformable, full annular seat; fixture support
        # fixes only its outer cylinder. The original curved pipe is not modeled.
        outer_seat=o.addCylinder(-1.5,0,0,1.5,0,0,7)
        inner_seat=o.addCone(-1.5,0,0,1.5,0,0,radius_shoulder+1.5*math.tan(math.radians(20)),radius_shoulder)
        seat=o.cut([(3,outer_seat)],[(3,inner_seat)])[0]
        fraction=.25 if quarter else 1.
        if quarter:
            box=lambda:o.addBox(-6,0,0,13,8,8)
            bolt_parts=o.intersect(bolt_parts,[(3,box())])[0]
            bolt_parts=o.fragment(bolt_parts,[])[0]
            bolt_volumes=[tag for dim,tag in bolt_parts if dim==3]
            hub=o.intersect(hub,[(3,box())])[0]
            seat=o.intersect(seat,[(3,box())])[0]
        o.synchronize()
        groups={"BOLT":bolt_volumes,"HUB":[t for d,t in hub],"SEAT":[t for d,t in seat]}
        faces={}
        for group,volumes in groups.items():
            faces[group]=sorted({tag for vol in volumes for dim,tag in gmsh.model.getBoundary([(3,vol)],False,False) if dim==2})
        def bounds(tag): return gmsh.model.getBoundingBox(2,tag)
        cut=[f for f in faces["BOLT"] if abs(bounds(f)[0])<1e-5 and abs(bounds(f)[3])<1e-5 and abs(o.getMass(2,f)-math.pi*16*fraction)<1e-5]
        anchor=[f for f in faces["BOLT"] if gmsh.model.getType(2,f)=="Cylinder" and bounds(f)[0]>-1e-5 and bounds(f)[3]>5.99]
        hubhole=[f for f in faces["HUB"] if gmsh.model.getType(2,f)=="Cylinder" and abs(o.getMass(2,f)-2*math.pi*4*6*fraction)<1e-5]
        cone_faces=[f for f in faces["BOLT"] if gmsh.model.getType(2,f)=="Cone"]
        seatcontact=[f for f in faces["SEAT"] if gmsh.model.getType(2,f)=="Cone"]
        seatfixed=[f for f in faces["SEAT"] if gmsh.model.getType(2,f)=="Cylinder"]
        headload=[f for f in faces["BOLT"] if abs(bounds(f)[0]+5)<1e-5 and abs(bounds(f)[3]+5)<1e-5]
        assert len(cut)==len(anchor)==len(hubhole)==len(seatcontact)==1,(cut,anchor,hubhole,seatcontact)
        for name, vols in groups.items():
            pg=gmsh.model.addPhysicalGroup(3,vols)
            gmsh.model.setPhysicalName(3,pg,name)
        dist=gmsh.model.mesh.field.add("Distance")
        if rim_refinement:
            curves=sorted({tag for f in cut for dim,tag in gmsh.model.getBoundary([(2,f)],False,False) if dim==1 and gmsh.model.getType(1,tag)=="Circle"})
            gmsh.model.mesh.field.setNumbers(dist,"CurvesList",curves)
            gmsh.model.mesh.field.setNumber(dist,"Sampling",100)
        else:
            gmsh.model.mesh.field.setNumbers(dist,"SurfacesList",cut)
        field=gmsh.model.mesh.field.add("Threshold")
        for key,val in {"InField":dist,"SizeMin":size,"SizeMax":1.1,"DistMin":.2 if rim_refinement else .8,"DistMax":.7 if rim_refinement else 1.8}.items():
            gmsh.model.mesh.field.setNumber(field,key,val)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        gmsh.option.setNumber("Mesh.MeshSizeMin",size)
        gmsh.option.setNumber("Mesh.MeshSizeMax",1.1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",20)
        gmsh.option.setNumber("Mesh.Algorithm3D",1)
        gmsh.model.mesh.generate(3)
        if order==2:
            gmsh.model.mesh.setOrder(2)
            gmsh.model.mesh.optimize("HighOrderFastCurving")
        gmsh.write(str(output/"shoulder-screw-fixture.msh"))
        ntags,xyz,_=gmsh.model.mesh.getNodes()
        nodes={int(n):x for n,x in zip(ntags,np.asarray(xyz).reshape(-1,3))}
        elements={}
        egroup={}
        for group,volumes in groups.items():
            egroup[group]=[]
            for vol in volumes:
                types,tags,connectivities=gmsh.model.mesh.getElements(3,vol)
                for typ,etags,conn in zip(types,tags,connectivities):
                    count=10 if typ==11 else 4
                    assert typ in (4,11),typ
                    data=np.asarray(conn).reshape(-1,count)
                    if count==10:
                        data=data[:,[0,1,2,3,4,5,6,7,9,8]]
                    for eid,row in zip(etags,data):
                        elements[int(eid)]=list(map(int,row))
                        egroup[group].append(int(eid))
        face_conn={1:(0,1,2),2:(0,3,1),3:(1,3,2),4:(2,3,0)}
        by_tri={}
        for eid,conn in elements.items():
            for face,loc in face_conn.items():
                by_tri.setdefault(tuple(sorted(conn[i] for i in loc)),[]).append((eid,face))
        def surface_faces(face_ids,group=None):
            records=[]
            for f in face_ids:
                types,tags,connectivities=gmsh.model.mesh.getElements(2,f)
                for typ,etags,conn in zip(types,tags,connectivities):
                    width={2:3,9:6}[int(typ)]
                    for row in np.asarray(conn).reshape(-1,width):
                        records += [entry for entry in by_tri[tuple(sorted(map(int,row[:3])))] if group is None or entry[0] in egroup[group]]
            return sorted(set(records))
        selections={"ANCHOR":surface_faces(anchor,"BOLT"),"HUBHOLE":surface_faces(hubhole,"HUB"),"CONE":surface_faces(cone_faces,"BOLT"),"SEATCONTACT":surface_faces(seatcontact,"SEAT")}
        cut_options=surface_faces(cut,"BOLT")
        # Choose the cone/head-side cut face. Its outward normal is +X, hence
        # positive reference force stretches the bolt and seats the head.
        selections["CUT"]=[(e,f) for e,f in cut_options if np.mean([nodes[n][0] for n in elements[e][:4]])<0]
        assert selections["CUT"]
        cut_nodes={n for e,f in selections["CUT"] for n in elements[e] if abs(nodes[n][0])<1e-6}
        contact_elements={e for e,f in selections["CONE"]}
        forbidden=sorted(e for e in contact_elements if cut_nodes.intersection(elements[e]))
        if order==2 and forbidden:
            raise ValueError(f"Quadratic native cut/contact adjacency remains: {len(forbidden)} elements")
        def surface_nodes(face_ids):
            return sorted({int(n) for f in face_ids for n in gmsh.model.mesh.getNodes(2,f,includeBoundary=True)[0]})
        fixed_nodes=surface_nodes(seatfixed)
        lines=["*HEADING","Shoulder-cut native feasibility; reconstructed screw, annular fixture seat; N-mm-MPa","*NODE,NSET=NALL"]
        lines += [f"{n},"+",".join(f"{x:.12g}" for x in p) for n,p in nodes.items()]
        ref=max(nodes)+1
        lines += ["*NODE,NSET=PRETENSION",f"{ref},0,0,0"]
        for group,eids in egroup.items():
            lines += [f"*ELEMENT,TYPE=C3D{10 if order==2 else 4},ELSET={group}"]
            lines += [f"{e},"+",".join(map(str,elements[e])) for e in eids]
            lines += [f"*NSET,NSET=N{group}"]
            ids=sorted({n for e in eids for n in elements[e]})
            lines += [",".join(map(str,ids[i:i+16])) for i in range(0,len(ids),16)]
        lines += ["*NSET,NSET=SEATFIX"]+[",".join(map(str,fixed_nodes[i:i+16])) for i in range(0,len(fixed_nodes),16)]
        lines += ["*BOUNDARY","NHUB,1,3","SEATFIX,1,3","*MATERIAL,NAME=SCREW","*ELASTIC","201000,.291","*SOLID SECTION,ELSET=BOLT,MATERIAL=SCREW","*MATERIAL,NAME=STEEL","*ELASTIC","205000,.29","*SOLID SECTION,ELSET=SEAT,MATERIAL=STEEL","*SOLID SECTION,ELSET=HUB,MATERIAL=STEEL"]
        if quarter:
            already=set(fixed_nodes)|set(surface_nodes(anchor))|{n for e in egroup["HUB"] for n in elements[e]}
            lines += ["*BOUNDARY"]
            for n,p in nodes.items():
                if n not in already:
                    for axis in (1,2):
                        if abs(p[axis])<1e-8:
                            lines += [f"{n},{axis+1},{axis+1}"]
        for name,records in selections.items():
            lines += [f"*SURFACE,NAME={name}"]+[f"{e},S{f}" for e,f in records]
        lines += ["*TIE,NAME=FULL_CYLINDER,POSITION TOLERANCE=0.005,ADJUST=NO","ANCHOR,HUBHOLE",f"*PRE-TENSION SECTION,SURFACE=CUT,NODE={ref}","1,0,0","*CONTACT PAIR,INTERACTION=SEAT,TYPE=SURFACE TO SURFACE","CONE,SEATCONTACT","*SURFACE INTERACTION,NAME=SEAT","*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR","1.e7"]
        out=["*NODE FILE","U,RF","*EL FILE","S","*NODE PRINT,NSET=PRETENSION","U,RF","*NODE PRINT,NSET=NHUB,TOTALS=ONLY","RF","*NODE PRINT,NSET=SEATFIX,TOTALS=ONLY","RF","*CONTACT FILE","CDIS,CSTR","*CONTACT PRINT","CDIS,CSTR","*CONTACT PRINT,SLAVE=CONE,MASTER=SEATCONTACT","CF,CFN"]
        increments="1.,1.,1.e-6,1." if rim_refinement else ".1,1.,1.e-6,.2"
        lines += ["*STEP,NLGEOM,INC=100","*STATIC",increments,"*CLOAD",f"{ref},1,{100*fraction}"]+out+["*END STEP"]
        # Distributed pressure on the outer head end is a separating load -X.
        head_area=math.pi*radius_head**2
        lines += ["*STEP,NLGEOM,INC=100","*STATIC",increments,"*CLOAD,OP=NEW","*BOUNDARY,FIXED",f"{ref},1,1","*DLOAD"]
        lines += [f"{e},P{f},{-50/head_area:.12g}" for e,f in surface_faces(headload,"BOLT")]
        lines += out+["*END STEP"]
        inp=output/"short_screw.inp"
        inp.write_text("\n".join(lines)+"\n",encoding="ascii")
        manifest={"purpose":"backend feasibility; not actual assembly or geometry-equivalent COMSOL result","symmetry_fraction":fraction,"order":order,"local_size_mm":size,"nodes":len(nodes),"elements":len(elements),"element_groups":{g:len(es) for g,es in egroup.items()},"cut_area_mm2":math.pi*16*fraction,"cut_contact_radial_clearance_mm":radius_shoulder-4,"quadratic_cut_adjacent_contact_elements":forbidden,"cut_nodes":len(cut_nodes),"reference_node":ref,"shank_length_mm":6,"full_cylinder_tied":True,"surfaces":{k:len(v) for k,v in selections.items()},"contact": "full annular compliant seat surrogate, outer cylinder fixed"}
        (output/"mesh-manifest.json").write_text(json.dumps(manifest,indent=2))
        return inp
    finally:
        gmsh.finalize()


def evaluate_short_screw(inp):
    dat=parse_dat(Path(inp).with_suffix(".dat"))
    fraction=json.loads((Path(inp).parent/"mesh-manifest.json").read_text())["symmetry_fraction"]
    u={b["time"]:b["rows"][0][1] for b in dat if b["set"]=="PRETENSION" and "displacements (" in b["header"]}
    force={b["time"]:b["rows"][0][1] for b in dat if b["set"]=="PRETENSION" and b["header"].startswith("forces")}
    cf={b["time"]:b["rows"][0][:3] for b in dat if "total surface force" in b["header"]}
    pen=max([-r[-3] for b in dat if "relative contact displacement" in b["header"] for r in b["rows"]]+[0])
    metrics={"adjustment_preload_mm":u.get(1.),"adjustment_service_mm":u.get(2.),"preload_force_N":force.get(1.),"service_force_N":force.get(2.),"preload_contact_force_N":cf.get(1.),"service_contact_force_N":cf.get(2.),"max_penetration_mm":pen}
    if all(t in u and t in force and t in cf for t in (1.,2.)):
        checks={"target_preload":abs(force[1.]/(100*fraction)-1)<.01,"tightening_retained":abs(u[2.]-u[1.])<1e-8,"bolt_force_redistribution":abs(force[2.]-force[1.])>fraction,"preload_contact_resultant":abs(abs(cf[1.][0])-100*fraction)<fraction,"penetration":pen<.0075}
        metrics["service_axial_imbalance_N"]=force[2.]-abs(cf[2.][0])-50*fraction
        checks["service_axial_equilibrium"]=abs(metrics["service_axial_imbalance_N"])<.005*100*fraction
        log=Path(inp).with_suffix(".stdout.log").read_text()
        checks["complete_cylindrical_tie"]="no tied MPC" not in log and "no tied constraint" not in log
    else:
        checks={"complete_results":False}
    return {"accepted_fixture":all(checks.values()),"checks":checks,"metrics":metrics}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--size",type=float,default=.22)
    parser.add_argument("--order",type=int,choices=[1,2],default=2)
    parser.add_argument("--quarter",action="store_true")
    parser.add_argument("--rim-refinement",action="store_true")
    args=parser.parse_args()
    inp=build_deck(args.output,args.size,args.order,args.quarter,args.rim_refinement)
    result=CalculixRunner().run(inp,expected_steps=2,timeout_s=300)
    check=evaluate_short_screw(inp) if result.completed else {"accepted_fixture":False,"reason":result.errors}
    record={"run":result.to_dict(),"verification":check}
    (Path(args.output)/"verification.json").write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
