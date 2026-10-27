"""Reproducible N-mm-MPa elasticity, contact and native pretension fixtures.

Run: python -m structural.fixtures --output <new-directory>
Each run directory is immutable. Thresholds are analytical, not fitted to data.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from .backends.calculix import CalculixRunner, parse_dat, parse_frd


@dataclass
class HexMesh:
    nodes: dict[int, tuple[float, float, float]]
    elements: dict[int, list[int]]
    indices: dict[tuple[int, int, int], int]
    counts: tuple[int, int, int]

    def node_plane(self, axis: int, value: float) -> list[int]:
        return [n for n, xyz in self.nodes.items() if abs(xyz[axis] - value) < 1e-9]


def hex20_block(lengths=(100., 10., 10.), counts=(4, 2, 2), origin=(0., 0., 0.), node_offset=0, element_offset=0):
    nx, ny, nz = counts
    nodes, elements, indices, lattice = {}, {}, {}, {}
    corners = ((0,0,0),(2,0,0),(2,2,0),(0,2,0),(0,0,2),(2,0,2),(2,2,2),(0,2,2))
    edges = ((1,0,0),(2,1,0),(1,2,0),(0,1,0),(1,0,2),(2,1,2),(1,2,2),(0,1,2),(0,0,1),(2,0,1),(2,2,1),(0,2,1))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                conn = []
                for dx, dy, dz in corners + edges:
                    key = (2*i+dx, 2*j+dy, 2*k+dz)
                    if key not in lattice:
                        nid = node_offset + len(nodes) + 1
                        lattice[key] = nid
                        nodes[nid] = tuple(origin[a] + key[a] * lengths[a] / (2 * counts[a]) for a in range(3))
                    conn.append(lattice[key])
                eid = element_offset + len(elements) + 1
                elements[eid] = conn
                indices[(i,j,k)] = eid
    return HexMesh(nodes, elements, indices, counts)


def mesh_text(mesh: HexMesh, name="ALL") -> str:
    lines = [f"*NODE,NSET=N{name}"]
    lines += [f"{n}," + ",".join(f"{x:.12g}" for x in xyz) for n, xyz in mesh.nodes.items()]
    lines += [f"*ELEMENT,TYPE=C3D20,ELSET=E{name}"]
    for eid, conn in mesh.elements.items():
        lines += [f"{eid}," + ",".join(map(str, conn[:15])) + ",", ",".join(map(str, conn[15:]))]
    return "\n".join(lines) + "\n"


def nset(name, nodes):
    return f"*NSET,NSET={name}\n" + "\n".join(",".join(map(str, nodes[i:i+16])) for i in range(0, len(nodes), 16)) + "\n"


def material(name="STEEL", modulus=200000., poisson=0., elset="EALL"):
    return f"*MATERIAL,NAME={name}\n*ELASTIC\n{modulus},{poisson}\n*SOLID SECTION,ELSET={elset},MATERIAL={name}\n"


def output_requests():
    return "*NODE FILE\nU,RF\n*EL FILE\nS\n*NODE PRINT,NSET=NALL\nU,RF\n*EL PRINT,ELSET=EALL\nS\n"


def end_forces(mesh, axis, traction):
    """Consistent 8-node face integration (3x3 Gauss); force = integral N t dA."""
    import numpy as np
    # Hex20 x+ face ordered corner cycle in (y,z), then its edge mids.
    if axis != 0:
        raise ValueError("This fixture integration presently supports x+ faces only")
    gx, gw = np.polynomial.legendre.leggauss(3)
    forces = {}
    nx, ny, nz = mesh.counts
    local = [1,2,6,5,9,18,13,17]
    for j in range(ny):
        for k in range(nz):
            ids = [mesh.elements[mesh.indices[(nx-1,j,k)]][idx] for idx in local]
            xyz = np.array([mesh.nodes[n] for n in ids])
            area = (xyz[1,1]-xyz[0,1]) * (xyz[3,2]-xyz[0,2])
            for a, wa in zip(gx, gw):
                for b, wb in zip(gx, gw):
                    n = np.array([-(1-a)*(1-b)*(1+a+b), -(1+a)*(1-b)*(1-a+b),
                                  -(1+a)*(1+b)*(1-a-b), -(1-a)*(1+b)*(1+a-b),
                                  2*(1-a*a)*(1-b), 2*(1+a)*(1-b*b),
                                  2*(1-a*a)*(1+b), 2*(1-a)*(1-b*b)]) / 4
                    load = np.asarray(traction(n @ xyz)) * wa * wb * area / 4
                    for nid, shape in zip(ids,n):
                        forces[nid] = forces.get(nid,np.zeros(3)) + shape*load
    return "*CLOAD\n" + "".join(f"{nid},{d+1},{v:.12g}\n" for nid, vals in forces.items() for d,v in enumerate(vals) if abs(v)>1e-12)


def tension_deck():
    mesh=hex20_block()
    text="*HEADING\nAnalytic uniaxial tension; N-mm-MPa\n"+mesh_text(mesh)
    text+=nset("ROOT",mesh.node_plane(0,0))+nset("Y0",mesh.node_plane(1,0))+nset("Z0",mesh.node_plane(2,0))
    text+="*BOUNDARY\nROOT,1,1\nY0,2,2\nZ0,3,3\n"+material(modulus=205000,poisson=.29)
    text+="*STEP\n*STATIC\n"+end_forces(mesh,0,lambda xyz:(10,0,0))+output_requests()+"*END STEP\n"
    return text, {"expected_tip_ux_mm":1000*100/(205000*100),"expected_sxx_MPa":10.,"relative_tolerance":.01}


def bending_deck():
    mesh=hex20_block(origin=(0,-5,-5))
    text="*HEADING\nAnalytic pure bending moment; nu=0; N-mm-MPa\n"+mesh_text(mesh)
    text+=nset("ROOT",mesh.node_plane(0,0))+"*BOUNDARY\nROOT,1,3\n"+material()
    text+="*STEP\n*STATIC\n"+end_forces(mesh,0,lambda xyz:(-1000*xyz[1]/(10*10**3/12),0,0))+output_requests()+"*END STEP\n"
    return text,{"expected_tip_uy_mm":1000*100**2/(2*200000*(10*10**3/12)),"relative_tolerance":.02}


def contact_deck(stiffness=1e7):
    lower=hex20_block((10,10,10),(2,2,2))
    upper=hex20_block((10,10,10),(2,2,2),(0,0,10),len(lower.nodes),len(lower.elements))
    text="*HEADING\nTwo elastic blocks in frictionless compression; N-mm-MPa\n"+mesh_text(lower,"LOW")+mesh_text(upper,"UP")
    text+="*NSET,NSET=NALL\nNLOW,NUP\n*ELSET,ELSET=EALL\nELOW,EUP\n"
    text+=nset("FIX",lower.node_plane(2,0))+nset("TOP",upper.node_plane(2,20))
    text+="*BOUNDARY\nFIX,1,3\nNALL,1,2\n"+material()
    text+="*SURFACE,NAME=MASTER\n"+"".join(f"{e},S2\n" for (i,j,k),e in lower.indices.items() if k==1)
    text+="*SURFACE,NAME=SLAVE\n"+"".join(f"{e},S1\n" for (i,j,k),e in upper.indices.items() if k==0)
    text+="*CONTACT PAIR,INTERACTION=FRICTIONLESS,TYPE=SURFACE TO SURFACE\nSLAVE,MASTER\n*SURFACE INTERACTION,NAME=FRICTIONLESS\n*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR\n"+f"{stiffness}\n"
    text+="*STEP,NLGEOM,INC=100\n*STATIC\n.1,1.,1.e-6,.2\n*DLOAD\n"+"".join(f"{e},P2,10.\n" for (i,j,k),e in upper.indices.items() if k==1)
    text+=output_requests()+"*CONTACT FILE\nCDIS,CSTR\n*CONTACT PRINT\nCDIS,CSTR\n*CONTACT PRINT,SLAVE=SLAVE,MASTER=MASTER\nCF,CFN\n*END STEP\n"
    return text,{"expected_contact_force_N":1000.,"expected_pressure_MPa":10.,"relative_tolerance":.02,"stiffness_MPa_per_mm":stiffness,"gap_convention":"Pinned CalculiX 2.23 face-to-face fixture: DAT CDIS negative normal = penetration; CSTR positive = compression. This differs from manual prose and is verified against nodal geometry."}


def preload_lock_deck():
    mesh=hex20_block((20,10,10),(4,2,2))
    reference=max(mesh.nodes)+1
    text="*HEADING\nNative solid pretension and fixed-adjustment force redistribution; N-mm-MPa\n"+mesh_text(mesh)
    text+=f"*NODE\n{reference},0,0,0\n"+nset("PRETENSION",[reference])+nset("ROOT",mesh.node_plane(0,0))+nset("TIP",mesh.node_plane(0,20))
    text+="*BOUNDARY\nROOT,1,1\nTIP,1,1\nNALL,2,3\n"+material()
    text+="*SURFACE,NAME=CUT\n"+"".join(f"{eid},S4\n" for (i,j,k),eid in mesh.indices.items() if i==1)
    text+=f"*PRE-TENSION SECTION,SURFACE=CUT,NODE={reference}\n1,0,0\n"
    text+="*STEP,NLGEOM\n*STATIC\n.2,1.,1.e-6,.2\n*CLOAD\n"+f"{reference},1,100.\n"+output_requests()+"*NODE PRINT,NSET=PRETENSION\nU,RF\n*END STEP\n"
    text+="*STEP,NLGEOM\n*STATIC\n.2,1.,1.e-6,.2\n*CLOAD,OP=NEW\n*BOUNDARY,FIXED\n"+f"{reference},1,1\n*BOUNDARY\nTIP,1,1,.00005\n"+output_requests()+"*NODE PRINT,NSET=PRETENSION\nU,RF\n*END STEP\n"
    return text,{"reference_node":reference,"expected_preload_N":100.,"expected_service_force_N":150.,"expected_adjustment_mm":.0001,"relative_tolerance":.01,"tightening_absolute_tolerance_mm":1e-10,"service":"Prescribed 0.00005 mm end extension; EA/L = 1e6 N/mm. Fixture support is physical end grip."}


def preload_contact_deck(service_load=50.):
    """Quarter symmetric headed rod clamps an annular grip; separate tie/contact.

    The rod and grip are deformable. The rigid foundation is represented by
    fixed solid nodes. A 100 N quarter-fixture preload is locked, then a 50 N
    external separating force acts on the head. Force partition is emergent.
    """
    def crop(mesh, keep):
        indices={idx:eid for idx,eid in mesh.indices.items() if keep(*idx)}
        elements={eid:mesh.elements[eid] for eid in indices.values()}
        used={n for conn in elements.values() for n in conn}
        return HexMesh({n:xyz for n,xyz in mesh.nodes.items() if n in used},elements,indices,mesh.counts)
    bolt=crop(hex20_block((20,3,3),(10,3,3)),lambda i,j,k:i==9 or (j==0 and k==0))
    grip=crop(hex20_block((18,3,3),(9,3,3),node_offset=max(bolt.nodes),element_offset=max(bolt.elements)),lambda i,j,k:not(j==0 and k==0))
    hub=hex20_block((2,3,3),(1,3,3),(-2,0,0),max(grip.nodes),max(grip.elements))
    ref=max(hub.nodes)+1
    text="*HEADING\nNative cut, fixed adjustment, tied foundation and separating contact joint; N-mm-MPa\n"
    text+=mesh_text(bolt,"BOLT")+mesh_text(grip,"GRIP")+mesh_text(hub,"HUB")
    text+=f"*NODE,NSET=PRETENSION\n{ref},0,0,0\n*NSET,NSET=NALL\nNBOLT,NGRIP,NHUB\n*ELSET,ELSET=EALL\nEBOLT,EGRIP,EHUB\n"
    text+=nset("Y0",[n for m in (bolt,grip) for n in m.node_plane(1,0) if m.nodes[n][0]>0])+nset("Z0",[n for m in (bolt,grip) for n in m.node_plane(2,0) if m.nodes[n][0]>0])
    text+="*BOUNDARY\nNHUB,1,3\nY0,2,2\nZ0,3,3\n"+material()
    for name,mesh,condition,face in (("FOUNDATION",hub,lambda i,j,k:True,"S4"),("ANCHOR",bolt,lambda i,j,k:i==0,"S6"),("GRIPBASE",grip,lambda i,j,k:i==0,"S6"),("CUT",bolt,lambda i,j,k:i==4,"S4"),("SEAT",bolt,lambda i,j,k:i==9 and not(j==0 and k==0),"S6"),("GRIPTOP",grip,lambda i,j,k:i==8,"S4")):
        text+=f"*SURFACE,NAME={name}\n"+"".join(f"{eid},{face}\n" for (i,j,k),eid in mesh.indices.items() if condition(i,j,k))
    text+="*TIE,NAME=ANCHORAGE,POSITION TOLERANCE=0.000001\nANCHOR,FOUNDATION\n*TIE,NAME=GRIP_SUPPORT,POSITION TOLERANCE=0.000001\nGRIPBASE,FOUNDATION\n"
    text+=f"*PRE-TENSION SECTION,SURFACE=CUT,NODE={ref}\n1,0,0\n"
    text+="*CONTACT PAIR,INTERACTION=CONTACT,TYPE=SURFACE TO SURFACE\nSEAT,GRIPTOP\n*SURFACE INTERACTION,NAME=CONTACT\n*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR\n1.e7\n"
    out=output_requests()+"*NODE PRINT,NSET=PRETENSION\nU,RF\n*CONTACT FILE\nCDIS,CSTR\n*CONTACT PRINT\nCDIS,CSTR\n*CONTACT PRINT,SLAVE=SEAT,MASTER=GRIPTOP\nCF,CFN\n"
    text+=f"*STEP,NLGEOM,INC=100\n*STATIC\n.1,1.,1.e-6,.2\n*CLOAD\n{ref},1,100.\n"+out+"*END STEP\n"
    text+=f"*STEP,NLGEOM,INC=100\n*STATIC\n.1,1.,1.e-6,.2\n*CLOAD,OP=NEW\n*BOUNDARY,FIXED\n{ref},1,1\n"
    text+=end_forces(bolt,0,lambda xyz:(service_load/9,0,0))+out+"*END STEP\n"
    return text,{"reference_node":ref,"expected_preload_N":100.,"service_separating_load_N":service_load,"relative_tolerance":.01,"tightening_absolute_tolerance_mm":1e-10,"quarter_fixture":True,"native_cut_contact_separation_mm":8.0,"expect_open":service_load>150}


def evaluate(name, result, target):
    if not result.completed:
        return {"accepted":False,"reason":"Solver did not complete all requested steps", "run":result.to_dict()}
    dat=parse_dat(Path(result.input_path).with_suffix(".dat"))
    frd=parse_frd(Path(result.input_path).with_suffix(".frd"))
    fields=frd["fields"]
    displacements=[f for f in fields if f["name"]=="DISP"]
    metrics={}
    checks={}
    if name in ("tension","bending"):
        last=displacements[-1]["values"]
        tip=[n for n,xyz in frd["nodes"].items() if abs(xyz[0]-100)<1e-6]
        component=0 if name=="tension" else 1
        value=sum(last[n][component] for n in tip)/len(tip)
        key="expected_tip_ux_mm" if name=="tension" else "expected_tip_uy_mm"
        err=abs(value/target[key]-1)
        metrics={"tip_displacement_mm":value,"relative_displacement_error":err}
        checks["displacement"]=err<=target["relative_tolerance"]
        if name=="tension":
            stress=[r[2] for b in dat if b["header"].lower().startswith("stresses") for r in b["rows"]]
            serr=max(abs(s/10-1) for s in stress)
            metrics["max_relative_stress_error"]=serr
            checks["stress"]=serr<=.01
    elif name.startswith("contact"):
        end=[b for b in dat if b["time"] is not None and abs(b["time"]-1)<1e-7]
        pressure=[r[-3] for b in end if "contact stress" in b["header"].lower() for r in b["rows"]]
        overlap=[r[-3] for b in end if "relative contact displacement" in b["header"].lower() for r in b["rows"]]
        force_blocks=[b for b in end if "total surface force" in b["header"].lower()]
        force=abs(force_blocks[-1]["rows"][0][2]) if force_blocks else None
        physical_overlap=max(0.,-min(overlap)) if overlap else None
        contact_nodes=[(n,xyz) for n,xyz in frd["nodes"].items() if abs(xyz[2]-10)<1e-8]
        by_xy={}
        for n,xyz in contact_nodes:
            by_xy.setdefault(tuple(xyz[:2]),[]).append(n)
        final_u=displacements[-1]["values"]
        geometric_gap=sum(final_u[max(ids)][2]-final_u[min(ids)][2] for ids in by_xy.values())/len(by_xy)
        metrics={"contact_force_N":force,"pressure_min_MPa":min(pressure) if pressure else None,"pressure_max_MPa":max(pressure) if pressure else None,"max_overlap_mm":physical_overlap,"raw_cdis_normal_min_mm":min(overlap) if overlap else None,"mean_geometric_gap_mm":geometric_gap}
        checks={"contact_force":force is not None and abs(force/1000-1)<=.02,"pressure":bool(pressure) and max(abs(p/10-1) for p in pressure)<=.02,"contact_outputs":bool(pressure and overlap),"penetration":physical_overlap is not None and physical_overlap<=.0075}
        checks["gap_sign_verified"]=abs(geometric_gap+physical_overlap)<1e-8
    elif name=="preload_lock":
        adjustments={b["time"]:b["rows"][0][1] for b in dat if b["set"]=="PRETENSION" and b["header"].startswith("displacements")}
        stresses={tm:[r[2] for b in dat if b["time"]==tm and b["header"].startswith("stresses") for r in b["rows"]] for tm in (1.,2.)}
        force0=sum(stresses[1.])/len(stresses[1.])*100
        force1=sum(stresses[2.])/len(stresses[2.])*100
        delta=max(abs(v-adjustments[1.]) for t,v in adjustments.items() if t>=1.)
        metrics={"preload_force_N":force0,"service_force_N":force1,"stage_a_adjustment_mm":adjustments[1.],"max_locked_adjustment_change_mm":delta}
        checks={"preload":abs(force0/100-1)<=.01,"redistribution":abs(force1/150-1)<=.01,"locked":delta<=1e-10}
    elif name.startswith("preload_contact"):
        adjustments={b["time"]:b["rows"][0][1] for b in dat if b["set"]=="PRETENSION" and b["header"].startswith("displacements")}
        forces={b["time"]:b["rows"][0][1] for b in dat if b["set"]=="PRETENSION" and b["header"].startswith("forces")}
        contacts={b["time"]:abs(b["rows"][0][0]) for b in dat if "total surface force" in b["header"]}
        delta=max(abs(v-adjustments[1.]) for t,v in adjustments.items() if t>=1.)
        penetration=max(-r[-3] for b in dat if "relative contact displacement" in b["header"] for r in b["rows"])
        metrics={"preload_force_N":forces[1.],"preload_contact_N":contacts[1.],"service_force_N":forces[2.],"service_contact_N":contacts[2.],"force_partition_residual_N":forces[2.]-contacts[2.]-target["service_separating_load_N"],"stage_a_adjustment_mm":adjustments[1.],"max_locked_adjustment_change_mm":delta,"max_overlap_mm":penetration}
        checks={"preload":abs(forces[1.]/100-1)<=.01,"preload_contact":abs(contacts[1.]/100-1)<=.01,"redistribution":forces[2.]>100 and contacts[2.]<100,"service_equilibrium":abs(metrics["force_partition_residual_N"])<=.5,"locked":delta<=1e-10,"penetration":0<=penetration<=.0075}
        if target["expect_open"]:
            checks["open_interface_has_zero_force"]=contacts[2.]<=.005
    return {"accepted":bool(checks) and all(checks.values()),"checks":checks,"metrics":metrics,"targets":target,"run":result.to_dict()}


def run_fixtures(output, executable=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    runner=CalculixRunner(executable)
    summary={}
    cases={"tension":tension_deck,"bending":bending_deck,"contact":contact_deck,"contact_stiffer":lambda:contact_deck(2e7),"preload_lock":preload_lock_deck,"preload_contact":preload_contact_deck,"preload_contact_open":lambda:preload_contact_deck(200.)}
    for name, factory in cases.items():
        case=output/name
        case.mkdir()
        deck,target=factory()
        inp=case/(name+".inp")
        inp.write_text(deck,encoding="ascii")
        (case/"analytical-targets.json").write_text(json.dumps(target,indent=2))
        result=runner.run(inp,expected_steps=2 if name.startswith("preload") else 1,timeout_s=120)
        try:
            summary[name]=evaluate(name,result,target)
        except Exception as exc:
            summary[name]={"accepted":False,"evaluation_error":str(exc),"run":result.to_dict()}
        (case/"verification.json").write_text(json.dumps(summary[name],indent=2))
        print(name,json.dumps(summary[name]),flush=True)
    (output/"fixture-summary.json").write_text(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--executable")
    args=parser.parse_args()
    summary=run_fixtures(args.output,args.executable)
    raise SystemExit(0 if all(item["accepted"] for item in summary.values()) else 1)
