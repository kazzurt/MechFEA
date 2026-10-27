"""CalculiX assembly results with explicit field locations and missing evidence.

Nodal FRD stress is display data. DAT integration-point stress remains separate.
RF is total external nodal force (reaction plus CLOAD/DLOAD), so equilibrium
uses constrained DOFs only and subtracts physical loads on those DOFs first.
Pretension scalar reference-node RF is a generalized cut force, excluded from
physical assembly force/moment sums. Failed output remains diagnostic only.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .backends.calculix import critical_log_issues, parse_dat, parse_frd, parse_sta, sha256
from .comparison import FieldMesh, symmetric_tensor, von_mises
from .mesh import SolidMesh
from .validation import assess_assembly
from .backends import calculix as calculix_module
from .backends import friction_qualification as friction_module
from . import comparison as comparison_module, validation as validation_module
from .provenance import capture_module_source, module_code_sha256

IMPLEMENTATION_PROVENANCE=capture_module_source(__file__,__name__)
FRICTION_IMPLEMENTATION_PROVENANCE=capture_module_source(friction_module.__file__,friction_module.__name__)

RESULT_OUTPUT_GATE_DETAILS={
    "required_component_fields":"Both final states require separate component displacement and stress fields (fixed hub stress excluded)",
    "run_input_provenance":"Run manifest must identify the exact retained input deck",
    "raw_integration_point_stress":"Retain raw stress separately from extrapolated nodal display fields",
    "solver_connection_warnings":"Retained solver logs must contain no omitted tied MPC or connection warnings, independent of solver completion status",
    "result_readers":"Malformed or truncated files cannot yield accepted results",
}
REQUIRED_RESULT_GATE_NAMES=frozenset(gate["name"] for gate in assess_assembly({},"unknown")["gates"])|frozenset(RESULT_OUTPUT_GATE_DETAILS)


# Verified geometrically against a zero-initial-gap face-to-face contact fixture.
# The bundled 2.23 manual prose claims the opposite CDIS sign. Do not generalize.
VERIFIED_CONTACT_BUILD_SHA256 = "27a9426a1ee2f72c38fbe892e0ffaf38e14867628ffd0235770261b6f3ec9c5f"
VERIFIED_CONTACT_BUILDS = {
    VERIFIED_CONTACT_BUILD_SHA256: "Serial2.23 zero-initial-gap block: geometric gap agrees with negative CDIS normal",
    "62349096f3b76cef2f0e094d03f747187016683a233ec7f07a363c7089804876":
        "MT2.23 four threads: all8 printed CDIS/CSTR/CF tables exactly equal serial explicit-shoulder fixture; verification short-screw-explicit-jump-multithread4",
}
COMPONENT_DOMAINS = {"hub": 1, "pipe": 2, "screw_1": 3, "screw_2": 4, "screw_3": 5}
TET10_SUBTETS = np.array([[0,4,6,7], [4,1,5,8], [6,5,2,9], [7,8,9,3],
                        [4,6,7,9], [4,6,5,9], [4,5,8,9], [4,8,7,9]])


def _same_time(a, b):
    return a is not None and bool(np.isclose(a, b, rtol=1e-8, atol=1e-10))


def nodal_dat_values(blocks, time, kind, node_set=None):
    """Return ID→vector without silently summing repeated output selections."""
    result = {}
    for block in blocks:
        if not _same_time(block["time"], time) or not block["header"].startswith(kind):
            continue
        if node_set is not None and (block["set"] or "").upper() != node_set.upper():
            continue
        for row in block["rows"]:
            if len(row) != 4:
                continue
            identifier = int(row[0]); value = np.asarray(row[1:])
            if identifier in result and not np.allclose(result[identifier], value, atol=1e-9, rtol=1e-7):
                raise ValueError(f"Conflicting {kind} output for node {identifier} at time {time}")
            result[identifier] = value
    return result


def physical_equilibrium(points_by_id, displacements, external_rf, *, support_dofs,
                         applied_nodal_forces, characteristic_force_n, characteristic_length_mm,
                         moment_origin_mm=(0.,0.,0.), mpc_support_transfers=None):
    """Compute physical support+load resultants, using x+u for NLGEOM moments.

    ``support_dofs`` maps node IDs to 1-based constrained translational DOFs.
    Inputs must already exclude scalar pretension nodes and internal pair forces.
    Missing support or displacement data makes equilibrium unavailable.
    """
    if characteristic_force_n <= 0 or characteristic_length_mm <= 0:
        raise ValueError("Positive physical equilibrium normalizers are required")
    if mpc_support_transfers:
        arguments=dict(support_dofs=support_dofs,applied_nodal_forces=applied_nodal_forces,
                       characteristic_force_n=characteristic_force_n,characteristic_length_mm=characteristic_length_mm,
                       moment_origin_mm=moment_origin_mm)
        raw=physical_equilibrium(points_by_id,displacements,external_rf,**arguments)
        transfers={int(dependent):int(anchor) for dependent,anchor in mpc_support_transfers.items()}
        if any(anchor not in support_dofs or dependent in support_dofs for dependent,anchor in transfers.items()):
            raise ValueError("Pretension reaction condensation requires a dependent node outside supports and its original physical support node")
        missing=sorted(set(transfers)-set(external_rf))
        nonfinite=sorted(node for node in transfers if node in external_rf and not np.isfinite(external_rf[node]).all())
        if missing or nonfinite:
            return {"status":"missing","missing":{"dependent_mpc_rf":{"count":len(missing),"first_node_ids":missing[:20]},
                    "nonfinite_dependent_mpc_rf":{"count":len(nonfinite),"first_node_ids":nonfinite[:20]}},
                    "force_imbalance_fraction":None,"moment_imbalance_fraction":None,"uncondensed_support_balance":raw}
        condensed={node:np.asarray(value,dtype=float).copy() for node,value in external_rf.items() if node in support_dofs}
        for dependent,anchor in transfers.items():
            if anchor in condensed:
                condensed[anchor]+=np.asarray(external_rf[dependent])-np.asarray(applied_nodal_forces.get(dependent,np.zeros(3)))
        result=physical_equilibrium(points_by_id,displacements,condensed,**arguments)
        result["uncondensed_support_balance"]=raw
        result["mpc_reaction_condensation"]={"dependent_to_original_support":transfers,
            "dependent_node_count":len(transfers),"translation_jacobian":"identity: u_head = u_original + axis*q",
            "moment_location":"original physical support position x_original+u_original",
            "generalized_reference_force_added":False,"additional_physical_supports":False}
        if result["status"]=="available":
            result["reaction_definition"]="RF minus physical applied loads, with dependent pretension-node RF condensed by transpose MPC Jacobian onto existing constrained original nodes"
            if raw["status"]=="available":
                result["mpc_reaction_condensation"]["force_correction_N"]=(np.array(result["force_residual_N"])-raw["force_residual_N"]).tolist()
                result["mpc_reaction_condensation"]["moment_correction_current_Nmm"]=(np.array(result["moment_residual_current_Nmm"])-raw["moment_residual_current_Nmm"]).tolist()
        return result
    required = set(support_dofs) | set(applied_nodal_forces)
    missing = {"coordinates": sorted(required-set(points_by_id)),
               "displacements": sorted(required-set(displacements)),
               "support_rf": sorted(set(support_dofs)-set(external_rf))}
    if any(missing.values()):
        return {"status": "missing", "missing": {key:{"count":len(ids),"first_node_ids":ids[:20]} for key,ids in missing.items()},
                "force_imbalance_fraction": None, "moment_imbalance_fraction": None}
    origin = np.asarray(moment_origin_mm, dtype=float)
    reactions = {}
    for node, dofs in support_dofs.items():
        indices = np.array(sorted(set(dofs)), dtype=int)-1
        if np.any((indices < 0) | (indices > 2)):
            raise ValueError("Only translational support DOFs 1,2,3 are supported")
        force = np.zeros(3)
        force[indices] = np.asarray(external_rf[node])[indices] - np.asarray(applied_nodal_forces.get(node, np.zeros(3)))[indices]
        reactions[node] = force
    support_force = np.sum(list(reactions.values()), axis=0) if reactions else np.zeros(3)
    applied_force = np.sum(list(applied_nodal_forces.values()), axis=0) if applied_nodal_forces else np.zeros(3)
    current_moment, original_moment = np.zeros(3), np.zeros(3)
    for node in required:
        force = reactions.get(node, np.zeros(3)) + np.asarray(applied_nodal_forces.get(node, np.zeros(3)))
        r = np.asarray(points_by_id[node])-origin
        current_moment += np.cross(r + np.asarray(displacements[node]), force)
        original_moment += np.cross(r, force)
    residual = support_force+applied_force
    return {"status": "available", "support_resultant_N": support_force.tolist(),
            "applied_resultant_N": applied_force.tolist(), "force_residual_N": residual.tolist(),
            "moment_residual_current_Nmm": current_moment.tolist(),
            "moment_residual_undeformed_Nmm": original_moment.tolist(),
            "moment_configuration": "current_position_x_plus_u_for_NLGEOM", "moment_origin_mm": origin.tolist(),
            "force_imbalance_fraction": float(np.linalg.norm(residual)/characteristic_force_n),
            "moment_imbalance_fraction": float(np.linalg.norm(current_moment)/(characteristic_force_n*characteristic_length_mm)),
            "characteristic_force_N": characteristic_force_n, "characteristic_length_mm": characteristic_length_mm,
            "reaction_definition": "RF minus physical applied loads, restricted to constrained DOFs"}


def assembly_support_dofs(metadata):
    """Account for prescribed physical supports, including eliminated fixed ties.

    A shaft node tied to an entirely fixed rigid master has u=0. When the deck
    eliminates that master equation explicitly, its reaction is located on the
    original shaft node and must enter the same external equilibrium sum.
    """
    supports={int(node):[1,2,3] for node in metadata["hub_fixed_nodes"]}
    anchors=metadata.get("rigid_anchorage_nodes",{})
    if anchors and metadata.get("anchorage_enforcement")!="fixed_master_elimination":
        raise ValueError("Rigid anchorage reactions require explicit fixed-master elimination metadata")
    for nodes in anchors.values():
        for node in nodes:supports[int(node)]=[1,2,3]
    for node in metadata["midpoint_nodes"]:
        supports[int(node)]=sorted(set(supports.get(int(node),[]))|{3})
    return supports


def assembly_mpc_support_transfers(metadata,support_dofs):
    """Select only explicit cut dependencies whose originals are physical supports."""
    maps=metadata.get("pretension_cut_node_maps",{})
    if not maps:return {}
    if metadata.get("pretension_formulation")!="explicit paired-node axial jump with scalar work-conjugate force":
        raise ValueError("Unsupported pretension MPC reaction-condensation formulation")
    result={}
    for mapping in maps.values():
        for original,dependent in mapping.items():
            original=int(original);dependent=int(dependent)
            if original not in support_dofs:continue
            if dependent in result:raise ValueError("Duplicate dependent pretension node in support condensation")
            result[dependent]=original
    return result


def pretension_history(blocks, pretension, preload_time=1., final_time=2.):
    result, preload_errors, drifts = {}, [], []
    for item in pretension:
        name, node = item["node_set"], int(item["reference_node"])
        times = sorted({b["time"] for b in blocks if b["time"] is not None and (b["set"] or "").upper() == name.upper()})
        rows = []
        for time in times:
            u = nodal_dat_values(blocks, time, "displacements", name).get(node)
            f = nodal_dat_values(blocks, time, "forces", name).get(node)
            rows.append({"time": time, "adjustment_mm": None if u is None else float(u[0]),
                         "direct_cut_force_N": None if f is None else float(f[0])})
        preload = next((row for row in rows if _same_time(row["time"], preload_time)), None)
        loaded = next((row for row in rows if _same_time(row["time"], final_time)), None)
        complete = bool(preload and loaded and all(row["adjustment_mm"] is not None and row["direct_cut_force_N"] is not None for row in rows))
        error, drift = None, None
        target=float(item["target_n"])
        if not np.isfinite(target) or target<0:raise ValueError("Preload target must be finite and nonnegative")
        force_normalizer=max(target,1.)
        if complete:
            error = abs(preload["direct_cut_force_N"]-target)/force_normalizer
            drift = max(abs(row["adjustment_mm"]-preload["adjustment_mm"]) for row in rows if row["time"] >= preload_time-1e-10)
            preload_errors.append(error); drifts.append(drift)
        result[name] = {"reference_node": node, "status": "available" if complete else "missing",
                        "history": rows, "preload_relative_error": error, "max_locked_adjustment_drift_mm": drift,
                        "target_force_N":target,"preload_error_normalizer_N":force_normalizer,
                        "preload_force_tolerance_N":.01*force_normalizer,
                        "force_change_N": loaded["direct_cut_force_N"]-preload["direct_cut_force_N"] if complete else None,
                        "force_location": "direct_pretension_generalized_DOF_1_RF",
                        "drift_scope":"saved_service_samples_only",
                        "saved_service_sample_count":sum(row["time"]>preload_time+1e-10 for row in rows)}
    complete = bool(pretension) and len(preload_errors) == len(pretension)
    return {"screws": result, "max_preload_relative_error": max(preload_errors) if complete else None,
            "max_tightening_drift_mm": max(drifts) if complete else None}


def contact_pair_statistics(blocks, time):
    pairs, current, kind = {}, None, None
    for block in blocks:
        if not _same_time(block["time"], time):
            continue
        header = block["header"]
        match = re.search(r"statistics for slave set (\S+), master set (\S+).*total (normal |shear )?surface force", header)
        if match:
            current = (match.group(1).upper(), match.group(2).upper())
            kind = {None:"CF", "normal ":"CFN", "shear ":"CFS"}[match.group(3)]
            row = block["rows"][0]
            if len(row) != 6:
                raise ValueError("Unexpected CalculiX pair force/moment row")
            pairs.setdefault(current, {})[kind] = {"force_on_slave_N": row[:3], "moment_about_origin_Nmm": row[3:]}
        elif current is not None and header.startswith("area,"):
            row = block["rows"][0]
            if len(row) == 3:
                pairs[current][kind].update(contact_area_mm2=row[0], signed_normal_force_N=row[1], shear_force_magnitude_N=row[2])
        elif current is not None and "center of gravity" in header and len(block["rows"][0]) == 6:
            pairs[current][kind]["centroid_and_mean_normal_raw"] = block["rows"][0]
    return pairs


def contact_diagnostics(blocks, time, contact_pairs, surfaces, *, sign_verified=False,
                        physical_gap_verified=False, characteristic_force_n=1112.0554,
                        pressure_tolerance_mpa=1e-8, gap_tolerance_mm=1e-9):
    """Preserve active quadrature samples; never infer absent sample gaps.

    Pinned CCX2.23: DAT CDIS normal is signed gap (negative penetration),
    CSTR normal is pressure (positive compression), CFN scalar normal force is
    positive tension. Pair force vectors are actual global forces on slaves.
    """
    dis, stress = [], []
    for block in blocks:
        if _same_time(block["time"], time):
            if block["header"].startswith("relative contact displacement"):
                dis.extend(block["rows"])
            elif block["header"].startswith("contact stress"):
                stress.extend(block["rows"])
    paired = len(dis) == len(stress) and all(len(a) == 5 and len(b) == 5 and a[:2] == b[:2] for a,b in zip(dis,stress))
    statistics = contact_pair_statistics(blocks, time)
    face_pair = {}
    surface_lookup = {key.upper(): value for key,value in surfaces.items()}
    for pair in contact_pairs:
        key = (pair["slave"].upper(), pair["master"].upper())
        for face in surface_lookup.get(key[0], []):
            if tuple(face) in face_pair:
                raise ValueError("Slave face belongs to more than one diagnostic pair")
            face_pair[tuple(face)] = key
    samples, ordinal = [], {}
    if paired:
        for a,b in zip(dis,stress):
            face = (int(a[0]),int(a[1])); ordinal[face] = ordinal.get(face,0)+1
            pair = face_pair.get(face)
            samples.append({"slave_element":face[0], "slave_face":face[1], "ordinal_on_face":ordinal[face],
                            "slave_surface": pair[0] if pair else None, "master_surface": pair[1] if pair else None,
                            "raw_CDIS_normal":a[2], "gap_mm":a[2] if sign_verified else None,
                            "slip_local_1_mm":a[3], "slip_local_2_mm":a[4], "pressure_MPa":b[2],
                            "shear_local_1_MPa":b[3], "shear_local_2_MPa":b[4]})
    unknown = sum(row["slave_surface"] is None for row in samples)
    pair_results, all_complete, compression_ok, maximum_penetration, open_force_bound = [], paired and unknown == 0, True, 0., 0.
    for pair in contact_pairs:
        key = (pair["slave"].upper(), pair["master"].upper())
        stats = statistics.get(key,{})
        rows = [row for row in samples if (row["slave_surface"],row["master_surface"]) == key]
        complete = "CF" in stats and "CFN" in stats
        all_complete &= complete
        entry = {"slave":key[0], "master":key[1], "statistics":stats, "active_samples":len(rows),
                 "max_penetration_mm":None,"gap_range_mm":None,"pressure_range_MPa":None,
                 "open_interface_force_upper_bound_N":None,"missing_reasons":[]}
        if not complete:entry["missing_reasons"].append("Required CF and/or CFN pair table absent")
        if complete and paired and sign_verified:
            normal_force = stats["CFN"].get("signed_normal_force_N")
            inactive=(not rows and all(stats[kind].get("contact_area_mm2")==0.
                and np.isfinite(stats[kind]["force_on_slave_N"]).all() and np.all(np.asarray(stats[kind]["force_on_slave_N"])==0.)
                and np.isfinite(stats[kind]["moment_about_origin_Nmm"]).all() and np.all(np.asarray(stats[kind]["moment_about_origin_Nmm"])==0.)
                for kind in ("CF","CFN")))
            entry["zero_area_inactive_pair"]=inactive
            if inactive:
                # Native zero-area output has undefined mean normal and scalar
                # projections (NaN), while its global CF/CFN vectors are zero.
                # Preserve those undefined scalars; this exception requires all
                # independent inactivity evidence and never fabricates a gap.
                entry.update(max_penetration_mm=0.,open_interface_force_upper_bound_N=0.,
                             gap_status="No active contact samples; physical open gap is not exported",
                             scalar_normal_projection_status="undefined_mean_normal_on_zero_area")
            elif normal_force is None or not np.isfinite(normal_force):
                all_complete=False;entry["missing_reasons"].append("Signed normal projection is unavailable without complete zero-area inactivity evidence")
            else:compression_ok &= normal_force <= 1e-8 * characteristic_force_n
            if rows:
                gaps = np.array([row["gap_mm"] for row in rows]); pressures = np.array([row["pressure_MPa"] for row in rows])
                traction=np.array([[row["pressure_MPa"],row["shear_local_1_MPa"],row["shear_local_2_MPa"]] for row in rows])
                if not np.isfinite(gaps).all() or not np.isfinite(traction).all():
                    all_complete=False;entry["missing_reasons"].append("Nonfinite active contact gap or traction")
                    pair_results.append(entry);continue
                entry["gap_range_mm"]=[float(gaps.min()),float(gaps.max())]
                entry["pressure_range_MPa"]=[float(pressures.min()),float(pressures.max())]
                penetration = float(max(0.,-gaps.min()))
                entry["max_penetration_mm"] = penetration
                maximum_penetration = max(maximum_penetration,penetration)
                compression_ok &= bool(pressures.min() >= -pressure_tolerance_mpa)
                traction_magnitude=np.linalg.norm(traction,axis=1)
                open_active = (gaps > gap_tolerance_mm) & (traction_magnitude > pressure_tolerance_mpa)
                entry["max_open_traction_magnitude_MPa"]=float(traction_magnitude[gaps>gap_tolerance_mm].max()) if np.any(gaps>gap_tolerance_mm) else 0.
                if not open_active.any():
                    entry["open_interface_force_upper_bound_N"] = 0.
                elif np.all(gaps > gap_tolerance_mm):
                    cf=stats["CF"]
                    shear=cf.get("shear_force_magnitude_N");normal=cf.get("signed_normal_force_N")
                    if shear is not None and normal is not None and np.isfinite([shear,normal]).all():
                        bound=max(float(np.linalg.norm(cf["force_on_slave_N"])),abs(normal)+abs(shear))
                        # Nonzero local tractions with a vanishing net resultant
                        # may cancel. DAT lacks weights to quantify that case.
                        if bound>1e-8*characteristic_force_n:
                            entry["open_interface_force_upper_bound_N"] = bound
                # Mixed open/closed quadrature requires integration weights that DAT does not export.
            elif not inactive:
                all_complete = False
                entry["missing_reasons"].append("No active samples and incomplete zero-area/zero-force inactivity evidence")
            value = entry["open_interface_force_upper_bound_N"]
            open_force_bound = None if value is None or open_force_bound is None else open_force_bound+value
        pair_results.append(entry)
    usable = all_complete and sign_verified and bool(contact_pairs)
    return {"time":time, "status":"available" if usable else "missing", "pairs":pair_results, "samples":samples,
            "paired_active_samples":paired, "unmapped_active_samples":unknown,
            "max_physical_penetration_mm":maximum_penetration if usable and physical_gap_verified else None,
            "compression_only_contact":bool(compression_ok) if usable else None,
            "open_interface_force_fraction":open_force_bound/characteristic_force_n if usable and open_force_bound is not None else None,
            "sign_verified":sign_verified, "physical_gap_verified":physical_gap_verified,
            "missing_reasons":(["CDIS/CSTR active rows do not pair"] if not paired else [])
                +([f"{unknown} active rows do not map to a declared contact pair"] if unknown else [])
                +(["Contact sign convention unverified"] if not sign_verified else [])
                +(["Physical gap/initial adjustment evidence unverified"] if not physical_gap_verified else [])
                +[f"{entry['slave']} / {entry['master']}: {reason}" for entry in pair_results for reason in entry["missing_reasons"]],
            "gap_convention":"Pinned2.23 face-to-face DAT CDIS normal: negative penetration; CSTR positive compression; CFN scalar positive tension",
            "limitations":["Inactive surface points are absent; no gap values fabricated for them.",
                           "Integration-point ordinal is solver table order within a slave face, not a permanent spatial probe.",
                           "Open-force checks include normal and both tangential tractions.",
                           "Mixed open/closed traction tails, or locally nonzero tractions with cancelling resultants, cannot be integrated without unavailable quadrature weights."]}


def friction_configuration_evidence(input_path,metadata,run):
    """Identify friction from the actual deck; incomplete proof cannot enable it.

    Legacy frictionless decks remain unchanged. A nonzero, malformed or
    inconsistent friction declaration always requires the additional gate.
    """
    path=Path(input_path);declarations={};cards=[];normal_stiffness=[];errors=[]
    lines=[line.strip() for line in path.read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("**")]
    for index,line in enumerate(lines):
        if line.upper().startswith("*SURFACE BEHAVIOR,") and "PRESSURE-OVERCLOSURE=LINEAR" in line.upper():
            try:normal_stiffness.append(float(lines[index+1].split(",")[0].replace("D","E")))
            except (IndexError,ValueError):errors.append("Malformed native normal contact stiffness")
        if line.upper().split(",")[0]=="*FRICTION":
            try:
                values=[float(value.replace("D","E")) for value in lines[index+1].split(",")]
                if len(values)<2 or not np.isfinite(values[:2]).all() or values[0]<0 or values[1]<=0:raise ValueError()
                cards.append(values[:2])
            except (IndexError,ValueError):errors.append("Malformed native FRICTION coefficient/stick-slope data")
    if cards:declarations["input_deck"]=cards[0][0]
    if "friction_coefficient" in metadata:declarations["assembly_metadata"]=metadata["friction_coefficient"]
    project_path=path.parent/"project.json"
    if project_path.is_file():
        try:declarations["frozen_project"]=json.loads(project_path.read_text(encoding="utf-8"))["contacts"]["friction_coefficient"]
        except (KeyError,TypeError,ValueError):errors.append("Frozen project friction declaration unavailable")
    required=bool(errors) or any(value!=0 for value in declarations.values()) or any(mu!=0 for mu,slope in cards)
    if not required:return None
    valid=bool(declarations) and all(isinstance(value,(int,float)) and not isinstance(value,bool)
        and np.isfinite(value) and value>=0 for value in declarations.values())
    coefficient=next(iter(declarations.values())) if valid and len(set(declarations.values()))==1 else None
    evidence={"required":True,"verified":False,"friction_coefficient":coefficient,"declarations":declarations,
              "qualification":None,"missing_reasons":errors}
    try:
        if errors:raise ValueError("Incomplete friction declarations prevent qualification")
        if coefficient is None or coefficient<=0:raise ValueError("Friction declarations are invalid or inconsistent")
        if not cards or any(mu!=coefficient for mu,slope in cards):raise ValueError("Actual deck lacks the declared uniform friction law")
        from .project import StructuralProject
        project=StructuralProject.load(project_path)
        proof=friction_module.verify_friction_qualification(project,executable=run.get("executable"))
        if not isinstance(proof,dict) or proof.get("verified") is not True:raise ValueError("Native friction qualification unavailable")
        if (proof.get("coulomb_absolute_tolerance_mpa")!=friction_module.COULOMB_ABSOLUTE_TOLERANCE_MPA
                or proof.get("coulomb_relative_tolerance")!=friction_module.COULOMB_RELATIVE_TOLERANCE):
            raise ValueError("Qualification uses an unsupported local Coulomb precision policy")
        expected={"friction_coefficient":coefficient,"normal_contact_stiffness_mpa_per_mm":project.contacts.normal_stiffness_mpa_per_mm,
                  "friction_stick_slope_mpa_per_mm":project.contacts.normal_stiffness_mpa_per_mm/10}
        if any(proof.get(key)!=value or metadata.get(key)!=value for key,value in expected.items()):
            raise ValueError("Qualified friction settings do not match frozen project and assembly metadata")
        if any(slope!=expected["friction_stick_slope_mpa_per_mm"] for mu,slope in cards):raise ValueError("Actual deck stick slope differs from qualified setting")
        if not normal_stiffness or any(value!=expected["normal_contact_stiffness_mpa_per_mm"] for value in normal_stiffness):
            raise ValueError("Actual deck normal contact stiffness differs from qualified setting")
        if (proof.get("selected_executable_sha256")!=run.get("executable_sha256") or proof.get("threads")!=run.get("threads")
                or proof.get("version")!=run.get("solver_version")):raise ValueError("Actual run build/configuration differs from qualification")
        if metadata.get("friction_qualification_verification")!=proof:raise ValueError("Assembly lacks the same verified friction qualification")
        evidence.update(verified=True,qualification=proof)
    except (OSError,ValueError,KeyError,TypeError) as exc:evidence["missing_reasons"].append(str(exc))
    return evidence


def local_coulomb_diagnostics(contact,coefficient,qualification):
    """Check the complete local traction norm with fixed DAT precision bounds."""
    absolute=friction_module.COULOMB_ABSOLUTE_TOLERANCE_MPA
    relative=friction_module.COULOMB_RELATIVE_TOLERANCE
    result={"status":"missing","passed":None,"friction_coefficient":coefficient,
            "absolute_tolerance_mpa":absolute,"relative_tolerance":relative,"sample_count":0,
            "maximum_excess_mpa":None,"maximum_excess_over_tolerance":None,"missing_reasons":[]}
    if not isinstance(qualification,dict) or qualification.get("verified") is not True:
        result["missing_reasons"].append("Native friction history qualification is unavailable or unaccepted");return result
    if (not isinstance(coefficient,(int,float)) or isinstance(coefficient,bool) or not np.isfinite(coefficient) or coefficient<=0
            or qualification.get("friction_coefficient")!=coefficient
            or qualification.get("coulomb_absolute_tolerance_mpa")!=absolute
            or qualification.get("coulomb_relative_tolerance")!=relative):
        result["missing_reasons"].append("Qualification coefficient or immutable printed-precision tolerances do not match");return result
    if (contact.get("status")!="available" or contact.get("paired_active_samples") is not True
            or contact.get("unmapped_active_samples")!=0 or not contact.get("pairs")):
        result["missing_reasons"].append("Complete paired contact samples and declared pair coverage are required");return result
    rows=contact.get("samples",[])
    if any(pair.get("active_samples",0)==0 and pair.get("zero_area_inactive_pair") is not True for pair in contact["pairs"]):
        result["missing_reasons"].append("Absent active samples lack independently proven zero-area inactivity");return result
    try:values=np.array([[row[key] for key in ("pressure_MPa","shear_local_1_MPa","shear_local_2_MPa","gap_mm")] for row in rows],dtype=float).reshape(-1,4)
    except (KeyError,ValueError,TypeError):values=np.full((1,4),np.nan)
    if not np.isfinite(values).all() or len(rows)!=sum(pair.get("active_samples",0) for pair in contact["pairs"]):
        result["missing_reasons"].append("All paired normal/tangential tractions and gaps must be finite and complete");return result
    shear=np.linalg.norm(values[:,1:3],axis=1);pressure=values[:,0]
    excess=shear-coefficient*np.maximum(pressure,0.)
    tolerance=absolute+relative*np.maximum(coefficient*np.abs(pressure),shear)
    passed=bool(np.all(excess<=tolerance))
    result.update(status="passed" if passed else "failed",passed=passed,sample_count=len(rows),
        maximum_excess_mpa=float(excess.max()) if len(rows) else 0.,
        maximum_excess_over_tolerance=float(np.max(excess/tolerance)) if len(rows) else 0.,
        definition="Euclidean norm of both local shear components; compression cap mu*max(p,0); saved states only")
    return result


@dataclass
class ComponentResult:
    name: str
    time: float
    node_ids: np.ndarray
    points_mm: np.ndarray
    element_ids: np.ndarray
    tetra10: np.ndarray  # zero-based local connectivity, preserving native cut nodes
    displacement_mm: np.ndarray
    stress_mpa: np.ndarray
    domain: int
    solver_status: str

    def field_mesh(self):
        cells = self.tetra10[:,TET10_SUBTETS].reshape(-1,4).copy()
        q = self.points_mm[cells]
        det = np.linalg.det(q[:,1:]-q[:,:1])
        inverted = det < 0
        cells[inverted,1], cells[inverted,2] = cells[inverted,2].copy(), cells[inverted,1].copy()
        if np.any(abs(det) < 1e-14):
            raise ValueError("Degenerate linearized quadratic result tetrahedron")
        return FieldMesh(self.points_mm,cells,np.full(len(cells),self.domain),self.displacement_mm,self.stress_mpa,
                         provenance={"component":self.name,"time":self.time,"solver_status":self.solver_status,
                                     "stress_location":"extrapolated_nodal_display",
                                     "geometry_representation":"quadratic_tetra10_subdivided_into_eight_linear_tets",
                                     "comparison_limitation":"piecewise linear sample of quadratic solver fields; not native integration-point stress"})

    def export_vtu(self,path):
        import meshio
        data={"solver_node_id":self.node_ids,"displacement_mm":self.displacement_mm,
              "displacement_magnitude_mm":np.linalg.norm(self.displacement_mm,axis=1),
              "stress_tensor_MPa_nodal_display":self.stress_mpa.reshape(-1,9),
              "von_mises_MPa_nodal_display":von_mises(self.stress_mpa),
              "physical_stress_valid":np.full(len(self.node_ids),int(self.name!="hub"))}
        meshio.write(path,meshio.Mesh(self.points_mm,[("tetra10",self.tetra10)],point_data=data,
                      cell_data={"solver_element_id":[self.element_ids],"component_domain":[np.full(len(self.element_ids),self.domain)]}))


def component_result(mesh:SolidMesh,frd,time,component,*,solver_status="unknown"):
    if component not in COMPONENT_DOMAINS or not mesh.element_sets.get(component):
        raise ValueError("An explicit known physical component element set is required")
    fields={row["name"]:row for row in frd["fields"] if _same_time(row["time"],time)}
    if "DISP" not in fields or "STRESS" not in fields:
        raise ValueError("Complete displacement and stress frame is unavailable")
    stress_fields=fields["STRESS"]["components"]
    order=[stress_fields.index(name) for name in ("SXX","SYY","SZZ","SXY","SZX","SYZ")]
    element_ids=np.asarray(mesh.element_sets[component],dtype=int)
    connections=[]; precise_coordinates={}
    for eid in element_ids:
        if int(eid) not in frd["elements"]:
            raise ValueError(f"FRD missing original element {eid}")
        nodes=frd["elements"][int(eid)]["nodes"]
        if len(nodes)!=10:
            raise ValueError("Assembly exporter requires tetra10 FRD element topology")
        original=mesh.tetra10[mesh._elements[int(eid)]]
        for node,source in zip(nodes,original):
            coordinate=mesh.points_mm[mesh._nodes[int(source)]]
            if node in precise_coordinates and not np.allclose(precise_coordinates[node],coordinate,atol=1e-9,rtol=0):
                raise ValueError("FRD/native pretension connectivity does not preserve local C3D10 ordering")
            # Native cut duplicate IDs inherit full-precision original local-node coordinates.
            precise_coordinates[node]=coordinate
        connections.append(nodes)
    used,inverse=np.unique(connections,return_inverse=True)
    p=np.array([precise_coordinates[int(node)] for node in used])
    # FRD is rounded; verify topology-derived coordinates against it at display precision.
    for node,coordinate in zip(used,p):
        exported=frd["nodes"].get(int(node))
        if exported is None or not np.allclose(exported,coordinate,atol=.001,rtol=1e-5):
            raise ValueError("FRD coordinates disagree with source/local element map")
    u=np.array([fields["DISP"]["values"].get(int(node),[np.nan]*3)[:3] for node in used])
    stress=np.array([fields["STRESS"]["values"].get(int(node),[np.nan]*6) for node in used])
    s=symmetric_tensor(stress[:,order])
    if component=="hub":s[:]=np.nan
    if not np.isfinite(u).all() or (component!="hub" and not np.isfinite(s).all()):
        raise ValueError("FRD frame has missing or nonfinite required component fields")
    return ComponentResult(component,time,used,p,element_ids,inverse.reshape(-1,10),u,s,
                           COMPONENT_DOMAINS[component],solver_status)


def read_component_vtu(path,component,*,time,solver_status="unknown"):
    """Reopen an exported component without reconstructing/welding its topology."""
    import meshio
    grid=meshio.read(path)
    if component not in COMPONENT_DOMAINS or any(block.type!="tetra10" for block in grid.cells):
        raise ValueError("Expected a known physical component with quadratic tetrahedra")
    connections=np.vstack([block.data for block in grid.cells])
    domains=np.concatenate(grid.cell_data["component_domain"])
    if not np.all(domains==COMPONENT_DOMAINS[component]):
        raise ValueError("Result VTU component domain differs from report")
    return ComponentResult(component,float(time),np.asarray(grid.point_data["solver_node_id"]),
             np.asarray(grid.points),np.concatenate(grid.cell_data["solver_element_id"]),connections,
             np.asarray(grid.point_data["displacement_mm"]),
             np.asarray(grid.point_data["stress_tensor_MPa_nodal_display"]).reshape(-1,3,3),
             COMPONENT_DOMAINS[component],solver_status)


def _write_csv(path,rows):
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _json_clean(value):
    if isinstance(value,np.ndarray):return _json_clean(value.tolist())
    if isinstance(value,np.generic):return _json_clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):_json_clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_json_clean(v) for v in value]
    return value


def report_is_accepted(report):
    """Require every mandatory unique gate; imported flags cannot replace evidence."""
    if not isinstance(report,dict):return False
    assessment=report.get("assessment",{})
    if not isinstance(assessment,dict):return False
    gates=assessment.get("gates",[])
    if not isinstance(gates,list) or not gates or any(not isinstance(gate,dict) or not isinstance(gate.get("name"),str) for gate in gates):return False
    names=[gate["name"] for gate in gates]
    coefficient=report.get("friction_coefficient",0)
    # Older reports predate the explicit coefficient field. A retained
    # nonzero-friction project cannot inherit their sixteen frictionless gates.
    if "friction_coefficient" not in report and report.get("source_run"):
        try:
            source=Path(report["source_run"])
            project=json.loads((source.parent/"project.json").read_text(encoding="utf-8"))
            coefficient=project["contacts"]["friction_coefficient"]
        except (OSError,ValueError,TypeError,KeyError):return False
    if not isinstance(coefficient,(int,float)) or isinstance(coefficient,bool) or not np.isfinite(coefficient) or coefficient<0:return False
    required=REQUIRED_RESULT_GATE_NAMES
    if coefficient>0 or report.get("friction_required") is True:
        evidence=report.get("friction_qualification_evidence",{})
        if not isinstance(evidence,dict) or coefficient<=0 or evidence.get("verified") is not True or evidence.get("friction_coefficient")!=coefficient:return False
        proof=evidence.get("qualification",{})
        if (not isinstance(proof,dict) or proof.get("verified") is not True or proof.get("friction_coefficient")!=coefficient
                or proof.get("coulomb_absolute_tolerance_mpa")!=friction_module.COULOMB_ABSOLUTE_TOLERANCE_MPA
                or proof.get("coulomb_relative_tolerance")!=friction_module.COULOMB_RELATIVE_TOLERANCE):return False
        required=required|{"local_coulomb_bound"}
    return (report.get("accepted") is True and report.get("diagnostic_only") is False
            and report.get("solver_status")=="completed" and assessment.get("status")=="numerically_accepted"
            and len(set(names))==len(names) and required.issubset(names)
            and all(gate.get("status")=="passed" for gate in gates))


def retained_result_source_binding(report,results_path=None):
    """Bind a completed reviewed report to all eight retained native sources."""
    required=(".inp",".assembly.json",".run.json",".dat",".frd",".sta",".stdout.log",".stderr.log")
    result={"verified":False,"required_source_extensions":list(required),"files":{},"failures":[]}
    try:
        source=Path(report["source_run"])
        if not source.is_absolute():
            if results_path is None:raise ValueError("Relative source_run requires the results report path")
            source=Path(results_path).resolve().parent/source
        source=source.resolve();run=json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(run,dict):raise ValueError("Native run manifest is not an object")
        deck=Path(run["input_path"])
        if not deck.is_absolute():deck=source.parent/deck
        deck=deck.resolve()
        result.update(source_run=str(source),input_path=str(deck))
        if source!=deck.with_suffix(".run.json"):result["failures"].append("Report source_run is not the retained deck-derived run manifest path")
        if run.get("status")!="completed" or run.get("returncode")!=0:result["failures"].append("Current native run is not completed with return code zero")
        expected=report.get("source_file_hashes",{})
        if not isinstance(expected,dict):raise ValueError("Report native source hashes are not an object")
        for extension in required:
            path=deck.with_suffix(extension)
            if not path.is_file():result["failures"].append(f"Missing retained native source {extension}");continue
            digest=sha256(path);result["files"][extension]={"path":str(path),"sha256":digest}
            if expected.get(extension)!=digest:result["failures"].append(f"Missing or changed report-bound native source {extension}")
        digest=result["files"].get(".inp",{}).get("sha256")
        if not digest or report.get("input_sha256")!=digest or run.get("input_sha256")!=digest:
            result["failures"].append("Report, native run and current input deck hashes differ or are missing")
        result["verified"]=not result["failures"]
    except (OSError,ValueError,TypeError,KeyError) as exc:result["failures"].append(str(exc))
    return result


def retained_component_field_binding(report,results_path):
    """Check both exported frames against hashes pinned by their result report."""
    result={"verified":False,"files":{},"failures":[]}
    for state in ("preload","loaded"):
        components=report.get("stages",{}).get(state,{}).get("components",{})
        for name in COMPONENT_DOMAINS:
            key=state+"/"+name;entry=components.get(name,{})
            if entry.get("status")!="available" or not entry.get("vtu") or not entry.get("vtu_sha256"):
                result["failures"].append(f"Missing report-bound component field {key}");continue
            try:
                path=Path(entry["vtu"])
                if not path.is_absolute():path=Path(results_path).resolve().parent/path
                path=path.resolve();digest=sha256(path)
                result["files"][key]={"path":str(path),"sha256":digest}
                if digest!=entry["vtu_sha256"]:result["failures"].append(f"Changed report-bound component field {key}")
            except (OSError,TypeError,ValueError) as exc:result["failures"].append(f"{key}: {exc}")
    result["verified"]=not result["failures"]
    return result


def result_display_field(point_data,kind):
    """Choose explicit scalar units; never render an arbitrary vector component."""
    if kind in ("ux","uy","uz"):
        name=f"displacement_{kind[-1]}_mm";label=f"U{kind[-1]} [mm] · reference frame"
        if "displacement_mm" not in point_data:return {"name":None,"values":None,"label":label,"note":"Displacement field unavailable"}
        u=np.asarray(point_data["displacement_mm"])
        if u.ndim!=2 or u.shape[1]!=3:raise ValueError("Displacement field must have three components")
        values=u[:,"xyz".index(kind[-1])]
    elif kind=="displacement":
        name="displacement_magnitude_mm"
        if name in point_data: values=np.asarray(point_data[name])
        elif "displacement_mm" in point_data:
            u=np.asarray(point_data["displacement_mm"])
            if u.ndim!=2 or u.shape[1]!=3:raise ValueError("Displacement field must have three components")
            values=np.linalg.norm(u,axis=1)
        else:return {"name":None,"values":None,"label":"Displacement [mm]","note":"Displacement field unavailable"}
        label="Displacement magnitude [mm]"
    elif kind in ("mises","principal_max","principal_min","sxx","syy","szz","sxy","sxz","syz"):
        name="von_mises_MPa_nodal_display" if kind=="mises" else kind+"_MPa_nodal_display"
        title={"mises":"von Mises","principal_max":"Maximum principal stress","principal_min":"Minimum principal stress"}.get(kind,kind.upper())
        label=title+" [MPa] · nodal display"
        if "physical_stress_valid" in point_data and not np.any(point_data["physical_stress_valid"]):
            return {"name":None,"values":None,"label":label,"note":"Fixed rigid hub stress is not a physical result"}
        if kind=="mises":
            if name not in point_data:return {"name":None,"values":None,"label":label,"note":"Nodal display stress unavailable"}
            values=np.asarray(point_data[name])
        else:
            tensor=point_data.get("stress_tensor_MPa_nodal_display")
            if tensor is None:return {"name":None,"values":None,"label":label,"note":"Nodal stress tensor unavailable"}
            tensor=np.asarray(tensor).reshape(-1,3,3)
            finite=np.isfinite(tensor).all(axis=(1,2))
            values=np.full(len(tensor),np.nan)
            if kind.startswith("principal"):
                values[finite]=np.linalg.eigvalsh(tensor[finite])[:,2 if kind=="principal_max" else 0]
            else:
                values[finite]=tensor[finite,"xyz".index(kind[1]),"xyz".index(kind[2])]
    else:raise ValueError("Unsupported result scalar")
    if values.ndim!=1:raise ValueError("Requested display field is not scalar")
    if not np.isfinite(values).any():return {"name":None,"values":None,"label":label,"note":"No finite values in this field"}
    return {"name":name,"values":values,"label":label,"note":""}


def postprocess_assembly(mesh:SolidMesh,input_path,output_dir=None):
    """Process immutable actual run files; no solve/completion inference from fields."""
    path=Path(input_path).resolve()
    metadata=json.loads(path.with_suffix(".assembly.json").read_text(encoding="utf-8"))
    run=json.loads(path.with_suffix(".run.json").read_text(encoding="utf-8"))
    friction=friction_configuration_evidence(path,metadata,run)
    output=Path(output_dir) if output_dir is not None else path.parent/"structural-results"
    output.mkdir(parents=True,exist_ok=False)
    reader_errors=[]
    try:blocks=parse_dat(path.with_suffix(".dat")) if path.with_suffix(".dat").exists() else []
    except (OSError,ValueError,IndexError) as exc:
        blocks=[];reader_errors.append({"file":str(path.with_suffix(".dat")),"error":str(exc)})
    try:frd=parse_frd(path.with_suffix(".frd")) if path.with_suffix(".frd").exists() else {"fields":[],"elements":{},"nodes":{}}
    except (OSError,ValueError,IndexError) as exc:
        frd={"fields":[],"elements":{},"nodes":{}};reader_errors.append({"file":str(path.with_suffix(".frd")),"error":str(exc)})
    times=np.cumsum(metadata.get("expected_step_times",[1.,1.]))
    if len(times)!=2:raise ValueError("Assembly postprocessor expects preload and service steps")
    pretension=pretension_history(blocks,metadata["pretension"],times[0],times[1])
    supports=assembly_support_dofs(metadata)
    mpc_transfers=assembly_mpc_support_transfers(metadata,supports)
    points={int(n):p for n,p in zip(mesh.node_ids,mesh.points_mm)}
    element_component={int(e):name for name in COMPONENT_DOMAINS for e in mesh.element_sets.get(name,[])}
    applied={int(n):np.array(force) for n,force in metadata["applied_nodal_forces"].items()}
    characteristic=max(float(np.linalg.norm(np.sum(list(applied.values()),axis=0))) if applied else 0.,1.)
    sign_verified=run.get("executable_sha256") in VERIFIED_CONTACT_BUILDS
    gap_verified=metadata.get("contact_initial_adjustment")=="none" and metadata.get("contact_clearances_verified") is True
    log_paths=[path.with_suffix(ext) for ext in (".stdout.log",".stderr.log")]
    logs_available=all(log.is_file() for log in log_paths)
    critical_warnings=critical_log_issues("\n".join(log.read_text(errors="replace") for log in log_paths if log.is_file()))
    for warning in run.get("critical_warnings",[]):
        if warning not in critical_warnings:critical_warnings.append(warning)
    result={"schema_version":1,"solver_status":run.get("status","unknown"),"diagnostic_only":run.get("status")!="completed",
            "implementation_provenance":{"results":{**IMPLEMENTATION_PROVENANCE,"loaded_function_code_sha256":module_code_sha256(globals())},
                **{name:{**module.IMPLEMENTATION_PROVENANCE,"loaded_function_code_sha256":module_code_sha256(vars(module))}
                   for name,module in (("comparison",comparison_module),("backend_calculix",calculix_module),("validation",validation_module))}},
            "source_run":str(path.with_suffix(".run.json")),"input_sha256":sha256(path),"pretension":pretension,"stages":{},
            "reader_errors":reader_errors,"critical_solver_warnings":critical_warnings,
            "support_accounting":{"anchorage_enforcement":metadata.get("anchorage_enforcement"),
                                  "rigid_anchorage_node_counts":{name:len(set(nodes)) for name,nodes in metadata.get("rigid_anchorage_nodes",{}).items()},
                                  "physical_support_node_count":len(supports),"dependent_cut_reaction_transfers":mpc_transfers},
            "source_file_hashes":{ext:sha256(path.with_suffix(ext)) for ext in (".inp",".assembly.json",".run.json",".dat",".frd",".sta",".stdout.log",".stderr.log") if path.with_suffix(ext).is_file()},
            "contact_sign_evidence":VERIFIED_CONTACT_BUILDS.get(run.get("executable_sha256")),
            "increments":parse_sta(path.with_suffix(".sta")),"component_domains":COMPONENT_DOMAINS,
            "comparison_classification":metadata.get("comparison_classification"),
            "geometry_revision":metadata.get("geometry_revision"),
            "geometry_revision_verification":metadata.get("geometry_revision_verification"),
            "numerical_step_controls":metadata.get("step_controls"),
            "limitations":["FRD stresses are extrapolated nodal display values; raw DAT stresses are exported separately.",
                           "No cross-component node welding or stress averaging is performed.",
                           "Hub displacement is shown, but fixed-hub stress is physically undefined.",
                           "Tightening drift is checked only at saved DAT times; endpoint-only output cannot establish intermediate-increment retention."]}
    if friction is not None:
        result.update(friction_required=True,friction_coefficient=friction["friction_coefficient"],friction_qualification_evidence=friction)
        result["implementation_provenance"]["friction_qualification"]={**FRICTION_IMPLEMENTATION_PROVENANCE,
            "loaded_function_code_sha256":module_code_sha256(vars(friction_module))}
    for stage,time in zip(("preload","loaded"),times):
        u=nodal_dat_values(blocks,time,"displacements","ALL_NODES")
        rf=nodal_dat_values(blocks,time,"forces","ALL_NODES")
        equilibrium=physical_equilibrium(points,u,rf,support_dofs=supports,
                     applied_nodal_forces={} if stage=="preload" else applied,
                     characteristic_force_n=characteristic,characteristic_length_mm=233.68,mpc_support_transfers=mpc_transfers)
        contact=contact_diagnostics(blocks,time,metadata["contact_pairs"],mesh.surfaces,sign_verified=sign_verified,
                     physical_gap_verified=gap_verified,characteristic_force_n=characteristic)
        if friction is not None:
            contact["local_coulomb"]=local_coulomb_diagnostics(contact,friction["friction_coefficient"],
                friction["qualification"] if friction["verified"] else None)
        _write_csv(output/f"{stage}-active-contact-samples.csv",contact.pop("samples"))
        stage_result={"time":float(time),"equilibrium":equilibrium,"contact":contact,"components":{}}
        for name in COMPONENT_DOMAINS:
            try:
                component=component_result(mesh,frd,time,name,solver_status=result["solver_status"])
                target=output/f"{stage}-{name}.vtu";component.export_vtu(target)
                stage_result["components"][name]={"vtu":target.name,"vtu_sha256":sha256(target),"status":"available",
                    "max_displacement_mm":float(np.linalg.norm(component.displacement_mm,axis=1).max()),
                    "raw_nodal_peak_von_mises_MPa":float(np.max(von_mises(component.stress_mpa))) if name!="hub" else None,
                    "stress_location":"extrapolated_nodal_display"}
            except (ValueError,KeyError,IndexError) as exc:
                stage_result["components"][name]={"status":"missing","reason":str(exc)}
        raw=[]
        for block in blocks:
            if _same_time(block["time"],time) and block["header"].startswith("stresses"):
                for row in block["rows"]:
                    if len(row)==8:
                        entry=dict(zip(("element","integration_point","sxx_MPa","syy_MPa","szz_MPa","sxy_MPa","sxz_MPa","syz_MPa"),row))
                        entry["element"]=int(entry["element"]);entry["integration_point"]=int(entry["integration_point"])
                        entry["component"]=element_component.get(entry["element"],"unknown")
                        entry["physical_stress_valid"]=int(entry["component"] not in ("hub","unknown"))
                        raw.append(entry)
        _write_csv(output/f"{stage}-raw-integration-point-stress.csv",raw)
        stage_result["raw_stress_integration_point_rows"]=len(raw)
        gp_by_element={}
        for row in raw:gp_by_element.setdefault(row["element"],set()).add(row["integration_point"])
        stage_result["raw_stress_complete_c3d10"]=bool(raw) and all(gp_by_element.get(int(e))=={1,2,3,4} for e in mesh.element_ids) and all(np.isfinite([row[key] for key in ("sxx_MPa","syy_MPa","szz_MPa","sxy_MPa","sxz_MPa","syz_MPa")]).all() for row in raw)
        result["stages"][stage]=stage_result
    loaded=result["stages"]["loaded"]
    metrics={"model_preparation_verified":metadata.get("model_preparation_verified"),
             "force_imbalance_fraction":max((s["equilibrium"]["force_imbalance_fraction"] for s in result["stages"].values()),default=None) if all(s["equilibrium"]["force_imbalance_fraction"] is not None for s in result["stages"].values()) else None,
             "moment_imbalance_fraction":max((s["equilibrium"]["moment_imbalance_fraction"] for s in result["stages"].values()),default=None) if all(s["equilibrium"]["moment_imbalance_fraction"] is not None for s in result["stages"].values()) else None,
             "max_preload_relative_error":pretension["max_preload_relative_error"],"max_tightening_drift_mm":pretension["max_tightening_drift_mm"],
             "screw_forces_free_to_redistribute":metadata.get("screw_forces_free_to_redistribute"),
             "no_final_artificial_support":metadata["final_artificial_supports"]==[] if "final_artificial_supports" in metadata else None,
             "max_physical_penetration_mm":None,"open_interface_force_fraction":None,"compression_only_contact":None}
    for key in ("max_physical_penetration_mm","open_interface_force_fraction","compression_only_contact"):
        values=[s["contact"][key] for s in result["stages"].values()]
        metrics[key]=None if any(value is None for value in values) else (all(values) if key=="compression_only_contact" else max(values))
    if friction is not None:
        values=[stage["contact"]["local_coulomb"]["passed"] for stage in result["stages"].values()]
        metrics.update(friction_required=True,friction_coefficient=friction["friction_coefficient"],
                       local_coulomb_bound=None if any(value is None for value in values) else all(values))
    result["metrics"]=metrics;result["assessment"]=assess_assembly(metrics,result["solver_status"])
    available=all(s["components"][name]["status"]=="available" for s in result["stages"].values() for name in COMPONENT_DOMAINS)
    result["assessment"]["gates"].append({"name":"required_component_fields","status":"passed" if available else "missing",
                                         "measured":True if available else None,"limit":True,
                                         "detail":RESULT_OUTPUT_GATE_DETAILS["required_component_fields"]})
    if not available:result["assessment"]["status"]="unaccepted"
    provenance_ok=run.get("input_sha256")==result["input_sha256"] if run.get("input_sha256") else None
    raw_available=True if all(s["raw_stress_complete_c3d10"] for s in result["stages"].values()) else None
    for name,passed in [("run_input_provenance",provenance_ok),("raw_integration_point_stress",raw_available),
                        ("solver_connection_warnings",False if critical_warnings else True if logs_available else None),("result_readers",not reader_errors)]:
        result["assessment"]["gates"].append({"name":name,"status":"passed" if passed is True else "missing" if passed is None else "failed",
                                             "measured":passed,"limit":True,"detail":RESULT_OUTPUT_GATE_DETAILS[name]})
        if passed is not True:result["assessment"]["status"]="unaccepted"
    result["accepted"]=result["assessment"]["status"]=="numerically_accepted"
    result["diagnostic_only"]=not result["accepted"]
    (output/"results.json").write_text(json.dumps(_json_clean(result),indent=2,allow_nan=False),encoding="utf-8")
    return result
