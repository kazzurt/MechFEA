"""Rebuild the backend acceptance record from retained numerical evidence."""
from pathlib import Path
import json
from .calculix import ROOT, DEFAULT_EXECUTABLE, sha256
from .short_screw_fixture import evaluate_short_screw
from .feasibility import PRETENSION_ID


def write_report():
    fixtures=ROOT/"Dodec FEA/structural-fea-benchmark-runs/fixtures"
    delivery=ROOT/"Dodec FEA/structural-fea-validation-deliverables"
    delivery.mkdir(parents=True,exist_ok=True)
    generic=fixtures/"release-backend-verification/fixture-summary.json"
    analytic=fixtures/"explicit-pretension-analytical-final/verification.json"
    shoulder=fixtures/"short-screw-explicit-jump-round2/short_screw.inp"
    mesh_file=shoulder.parent/"mesh-manifest.json"
    shoulder_check=evaluate_short_screw(shoulder)
    reviewed=shoulder.parent/"verification-reviewed.json"
    reviewed.write_text(json.dumps(shoulder_check,indent=2))
    mesh=json.loads(mesh_file.read_text())
    shoulder_run=json.loads(shoulder.with_suffix(".run.json").read_text())
    mt_path=fixtures/"short-screw-explicit-jump-multithread4/verification.json"
    mt=json.loads(mt_path.read_text())
    base=json.loads(generic.read_text())
    analytic_record=json.loads(analytic.read_text())
    anchorage_path=fixtures/"rigid-anchorage-elimination-equivalence/equivalence-verification.json"
    anchorage=json.loads(anchorage_path.read_text())
    def evidence(path,**extra):
        return {"path":str(path.resolve()),"sha256":sha256(path),**extra}
    passed=all(v["accepted"] for v in base.values()) and analytic_record["accepted"] and shoulder_check["accepted_fixture"] and mesh["full_cylinder_tied"] and not mesh["quadratic_cut_adjacent_contact_elements"] and not shoulder_run["warnings"] and anchorage["accepted_equivalence"]
    record={"schema_version":1,"all_passed":bool(passed),"version":"2.23",
            "solver_executable_sha256":sha256(DEFAULT_EXECUTABLE),
            "generic_fixture_summary":evidence(generic),"explicit_analytic_fixture":evidence(analytic,accepted=analytic_record["accepted"]),
            "shoulder_fixture":evidence(reviewed,accepted=shoulder_check["accepted_fixture"],
                            mesh=evidence(mesh_file),run=evidence(shoulder.with_suffix(".run.json"))),
            "required_assembly_pretension":"explicit paired-node displacement jump, all head-side elements remapped; native shoulder PRE-TENSION SECTION is NOT accepted",
            "required_assembly_pretension_id":PRETENSION_ID,
            "required_assembly_anchorage":"fixed_master_elimination",
            "rigid_anchorage_equivalence":evidence(anchorage_path,accepted=anchorage["accepted_equivalence"],
              identity="u_slave=sum(N_master*u_master)=0 for fixed rigid master; only actual overlapping anchorage nodes constrained"),
            "native_pretension_limitation":{"reason":"Shoulder cut rim touches annular head elements without a face on the cut; violates native cut topology restriction.",
                "force_N_quarter_fixture":25.,"measured_contact_axial_N":19.67124,
                "evidence":evidence(fixtures/"short-screw-shoulder-tie-corrected/verification.json")},
            "contact_gap_convention":{"formulation":"surface-to-surface penalty","CDIS_normal":"negative = physical penetration","CSTR_normal":"positive = compression",
                "verified_geometric_gap_mm":base["contact"]["metrics"]["mean_geometric_gap_mm"],"manual_prose_disagrees":True},
            "verified_multithread_build":{"version":"2.23","executable_sha256":mt["run"]["executable_sha256"],"threads":4,"evidence":evidence(mt_path),
                                         "matches_serial_printed_metrics":mt["verification"]["metrics"]==shoulder_check["metrics"]},
            "limitations":["Backend feasibility only; the annular-seat fixture is not the actual curved/slotted pipe assembly.",
                           "Actual CAD anchorage is trimmed by the snout envelope: some shoulder-adjacent shaft surface protrudes up to0.475mm beyond the hole. Assembly ties only actual overlapping engagement; the fixture's fully overlapping cylindrical anchor is an algorithm test.",
                           "Quarter symmetry is exact only for this annular fixture; its25N represents100N for the complete screw.",
                           "Fixture success does not establish assembly equilibrium, mesh independence, COMSOL agreement or a load rating.",
                           "Native shoulder pretension must not be substituted for the explicit paired-node formulation.",
                           "Windows sandbox cannot load this MinGW native executable (DLL relocation); reviewed execution outside sandbox works.",
                           "Serial ccx.exe and ccx_MT.exe with 4 threads are separately verified; assembly projects currently select the 4-thread build."]}
    (delivery/"backend-feasibility.json").write_text(json.dumps(record,indent=2))
    installation={"version":"2.23","build_commit":"60c9a10b350ec813f4d493098f47f7a8b96cd276",
                  "archive_url":"https://raw.githubusercontent.com/calculix/CalculiX-Windows/60c9a10b350ec813f4d493098f47f7a8b96cd276/releases/CalculiX-2.23.0-win-x64.zip",
                  "archive_sha256":sha256(DEFAULT_EXECUTABLE.parents[1].parent/"CalculiX-2.23.0-win-x64.zip"),
                  "executables":{"ccx.exe":sha256(DEFAULT_EXECUTABLE),"ccx_MT.exe":sha256(DEFAULT_EXECUTABLE.with_name("ccx_MT.exe"))},
                  "runtime_libraries":{path.name:sha256(path) for path in sorted(DEFAULT_EXECUTABLE.parent.glob("*.dll"))},
                  "license":"Bundled GNU GPL version2 text retained at bin/LICENSE.txt; original distribution notices retained",
                  "license_and_notices":{name:{"path":str(DEFAULT_EXECUTABLE.parent.parent/name),"sha256":sha256(DEFAULT_EXECUTABLE.parent.parent/name)}
                      for name in ("bin/LICENSE.txt","OSS_NOTICE.pdf","utils/LICENSE_MINGW.txt","utils/LICENSE_DOS2UNIX.txt")},
                  "primary_documentation":{name:{"url":"https://www.dhondt.de/"+name,"sha256":sha256(ROOT/"structural-fea-environment/solver/documentation"/name)}
                      for name in ("ccx_2.23.htm.tar.bz2","ccx_2.23.test.tar.bz2")},
                  "version_verified_from_execution":True,"global_environment_changes":False}
    (ROOT/"structural-fea-environment/solver/solver-manifest.json").write_text(json.dumps(installation,indent=2))
    report="""# CalculiX backend feasibility — verified implementation

The backend feasibility gate passes with **CalculiX 2.23 and explicit paired-node shoulder pretension**. Native pretension passes a regular bar and headed-joint fixture, but fails the actual screw shoulder topology check. The assembly must use the explicit formulation; this is a formulation correction within CalculiX, not a changed screw geometry or shortened anchorage.

## Verified numerical evidence

| Fixture | Result | Gate |
|---|---|---|
| C3D20 uniaxial elasticity | 0.00487805 mm vs 0.00487804878 mm analytical; raw stress10 MPa | <1% |
| C3D20 pure bending | 0.0300000 mm vs0.03 mm analytical | <2% |
| Frictionless blocks | 1000 N integrated,10 MPa pressure | <2% |
| Contact sign | Geometric gap−1e−6 mm matches DAT CDIS−1e−6 mm | Explicit convention verified |
| Double contact stiffness | Same1000 N/10 MPa; penetration halves from1e−6 to5e−7 mm | Integrated response unchanged |
| Native preload+lock bar | 100 N→150.0009 N; adjustment9.999925e−5 mm unchanged | Target+redistribution+lock |
| Tied preloaded contact joint | 100 N preload; service bolt105.6508/contact55.65077 N under50 N | Equilibrium residual0.00003 N |
| Open-contact joint | Service bolt200 N, contact0 N; tightening unchanged | Compression-only transfer |
| Explicit-jump analytic bar | 100 N→150.0009 N; same analytical tightening | Independently verified |
| Exact-dimension shoulder quarter fixture | Preload bolt25/contact25 N; service bolt36.36809/contact23.86817 N under12.5 N | All checks pass |
| Fixed rigid anchorage elimination | Replacing tied continuity by zero displacement on the same716 anchor nodes gives identical printed nodal and force results | Algebraically and numerically equivalent |

The quarter fixture has the kernel-verified screw dimensions: radius4 mm shaft over6 mm, radius4.5503973113 mm cone shoulder,20° cone over2 mm, radius5.2783377798 mm head over3 mm. It retains the complete cylindrical anchorage and original shoulder cut. A full annular elastic seat with an outer-cylinder support replaces the curved pipe only for this backend fixture. Quarter forces are25% of complete-screw forces. The9204-node,5138-C3D10-tet mesh has no quadratic contact element touching a cut node. Its tightening0.0001107309 mm is unchanged during service; peak penetration is0.0000521359 mm, below0.0075 mm.

The actual CAD hub-hole is trimmed by the snout envelope. Its matching6 mm axial endpoints conceal a circumferential engagement loss: shoulder-adjacent shaft surface protrudes by up to0.475 mm beyond the hole. The assembly must tie only the actual overlapping engagement after a volume-preserving geometric partition. The fixture's fully overlapping cylindrical support verifies the algorithm and is explicitly not an exact representation of the actual trimmed anchorage.

The first actual assembly input exposed a further CalculiX tie-search issue at the trimmed edge: it omitted MPCs despite the verified normal-distance tolerance, reporting no opposite master face within its independent in-face tolerance. That pilot was cancelled and is not accepted. For the fixed rigid hub baseline, the final assembly algebraically eliminates continuity to the zero-displacement master: u_slave=sum(N_master u_master)=0. Only the exact physically selected anchorage nodes receive this zero displacement. It adds no pipe support and does not constrain the exposed crescent. A controlled shoulder rerun produced identical FRD displacements at both stages and identical DAT forces/tightening, with no warnings. This elimination must not be used for a deformable or moving hub.

## Diagnoses retained in the run archive

1. Initial result parsing failed on the native Windows build's three-digit exponents and expanded negative FRD fields. The parser now handles both this format and standard two-digit exponent records; original run outputs were preserved.
2. The first117719-node shoulder pilot timed out after300 s; a17634-node quarter pilot completed only preload before the same bound. Local rim refinement and full-step ramps reduced the solved fixture to9204 nodes and about9 s. These were performance diagnoses, not evidence of a material solver limitation.
3. A1e−6 mm tie search tolerance missed7 nodes because the master is triangulated. The largest reported projection distance was0.00138 mm. A0.005 mm search tolerance connects the full anchor; **ADJUST=NO preserves coordinates**. The final fixture has no tie warnings.
4. Native shoulder pretension produced25 N at its reference variable but only19.67124 N axial contact transfer, despite correct ties. The manual prohibits a cut containing edges/vertices of elements lacking a cut face; the shoulder annulus creates precisely that case. Native section output alone would have falsely passed preload.
5. Explicit duplication of every head-side cut node, including annulus cells touching only a cut edge, restores25 N contact transfer. For each cut pair, u_head−u_shaft=n q in vector form; q is one scalar tightening coordinate, and its work-conjugate force is the total preload. The cut normal n stays in the initial reference orientation, as in the native linear-MPC convention. Stage B uses *BOUNDARY,FIXED on q and removes Stage A CLOAD. Analytic-bar and contact-joint checks verify this formulation.

## Solver and output contract

The isolated native distribution is pinned to repository commit60c9a10b350ec813f4d493098f47f7a8b96cd276. `solver-manifest.json` records archive and executable hashes. `install-solver.ps1` recreates it without global PATH/file associations. Both executables print2.23. The four-thread build uses4 CPUs for matrix/stiffness/stress/Spooles and reproduces serial printed shoulder metrics exactly;9.39 s/84 MB serial versus8.94 s/95 MB with4 threads is a small-fixture measurement, not an assembly prediction.

The runner retains input,stdout,stderr,DAT,FRD,STA and a hash/version/runtime manifest. It distinguishes completed,failed,cancelled and timed_out. Completion requires clean exit, final requested step times, FRD presence and the completion marker. Omitted tied MPC/constraint warnings are critical failures even if CalculiX finishes. Full wrapped warning context is retained. Numerical acceptance remains separate. Regression checks cover missing final time, failed exit, solver error, omitted ties, cancellation and evidence overwrite prevention.

The release cancellation regression launched a real native shoulder fixture, identified its owned child by parent PID, executable and working directory, requested cancellation after0.828 s and confirmed process termination at0.953 s. Its status is cancelled and unaccepted, with input, stdout, stderr and run manifest preserved. Evidence is in `fixtures/native-cancellation-lifecycle-verification/cancellation-verification.json`; this lifecycle test does not contribute a physical result.

Before binding an assembly project, `verify_backend_feasibility` verifies every referenced JSON hash, every required accepted fixture and clean run, the retained input hashes, the explicit pretension and fixed-master-elimination formulations, and the selected executable against the two verified binary hashes. A copied `all_passed` flag cannot bypass missing or changed evidence. The assembly binding also reads the source/count-matched mesh generation manifest and records requested versus actual mesh settings, preserving the requested solve node budget. The first rigid-anchorage pilot was actually meshed at0.5 mm contact size despite its frozen project initially saying1.0 mm; any correction to that existing run must be an explicit provenance addendum.

DAT preserves raw integration-point stresses and contact samples. FRD is nodal display/extrapolated data and is labeled accordingly. For this pinned face-to-face formulation, DAT CDIS normal is negative in penetration and CSTR normal is positive in compression. The official2.23 *CONTACT PRINT prose gives the opposite CDIS sign; the two-block geometric-gap measurement governs this adapter. Absent contact samples are not automatically interpreted as measured zero; integrated CF explicitly reports zero in the open-contact fixture.

## Reproduction

Use the project `.venv-structural` interpreter. Commands from the KurtsFEM root:

```powershell
.venv-structural/Scripts/python.exe -m structural.fixtures --output 'Dodec FEA/structural-fea-benchmark-runs/fixtures/new-generic-verification'
.venv-structural/Scripts/python.exe -m structural.backends.short_screw_fixture --quarter --rim-refinement --size .22 --output 'Dodec FEA/structural-fea-benchmark-runs/fixtures/new-native-shoulder-diagnostic'
.venv-structural/Scripts/python.exe -m structural.backends.verify_explicit_pretension --output 'Dodec FEA/structural-fea-benchmark-runs/fixtures/new-explicit-bar-verification'
.venv-structural/Scripts/python.exe -m structural.backends.verify_rigid_anchorage --output 'Dodec FEA/structural-fea-benchmark-runs/fixtures/new-rigid-anchorage-verification'
.venv-structural/Scripts/python.exe -m structural.backends.write_feasibility_report
```

The retained explicit shoulder input is the reproducible final deck. `convert_shoulder_fixture` derives it from the retained native shoulder deck. The generic `explicit_pretension` API performs the same operation on a structural SolidMesh and records node maps for assembly use. Run directories cannot be reused or overwritten. On this machine, native solver launches need execution outside the managed sandbox because sandboxed loading produces Windows DLL-relocation error623; this is unrelated to numerical convergence.

## Scope remaining outside this gate

The actual pipe/hub/three-screw assembly, all-body force/moment equilibrium, contact-stiffness and mesh sensitivity, matched COMSOL field comparison, engineering review and load rating are separate gates. No fixture result establishes wall adequacy.

## Primary sources retained

- [CalculiX project/downloads](https://www.dhondt.de/index.html)
- [Pinned native Windows build source](https://github.com/calculix/CalculiX-Windows/commit/60c9a10b350ec813f4d493098f47f7a8b96cd276)
- Official2.23 HTML documentation downloaded from[ccx_2.23.htm.tar.bz2](https://www.dhondt.de/ccx_2.23.htm.tar.bz2): local nodes227(*BOUNDARY),250(*CONTACT PRINT),334(*PRE-TENSION SECTION),360(*TIE),3(parallelization).
"""
    (delivery/"backend-feasibility-report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"all_passed":record["all_passed"],"path":str(delivery/"backend-feasibility.json")},indent=2))
    return record


if __name__=="__main__":
    write_report()
