import copy
import json

import numpy as np
import pytest

from structural.project import StructuralProject
from structural.results import (REQUIRED_RESULT_GATE_NAMES, friction_configuration_evidence,
                                local_coulomb_diagnostics, report_is_accepted)
from structural.validation import assess_assembly


def qualified_contact():
    proof={"verified":True,"friction_coefficient":.2,
           "coulomb_absolute_tolerance_mpa":1e-5,"coulomb_relative_tolerance":1e-6}
    contact={"status":"available","paired_active_samples":True,"unmapped_active_samples":0,
             "pairs":[{"active_samples":1,"zero_area_inactive_pair":False}],
             "samples":[{"pressure_MPa":100.,"shear_local_1_MPa":12.,"shear_local_2_MPa":16.,"gap_mm":-.001}]}
    return contact,proof


def test_coulomb_checks_full_norm_printed_precision_and_no_custom_relaxation():
    contact,proof=qualified_contact()
    assert local_coulomb_diagnostics(contact,.2,proof)["passed"] is True
    contact["samples"][0]["shear_local_2_MPa"]+=1e-5
    assert local_coulomb_diagnostics(contact,.2,proof)["passed"] is True
    contact["samples"][0]["shear_local_2_MPa"]=17.
    # Each component is below mu*p, but their Euclidean norm exceeds it.
    assert local_coulomb_diagnostics(contact,.2,proof)["passed"] is False
    proof["coulomb_absolute_tolerance_mpa"]=10.
    assert local_coulomb_diagnostics(contact,.2,proof)["passed"] is None


@pytest.mark.parametrize("missing",["proof","shear","gap","pairing","coverage"])
def test_coulomb_requires_complete_qualified_paired_data(missing):
    contact,proof=qualified_contact()
    if missing=="proof":proof["verified"]=False
    elif missing=="shear":del contact["samples"][0]["shear_local_2_MPa"]
    elif missing=="gap":contact["samples"][0]["gap_mm"]=np.nan
    elif missing=="pairing":contact["paired_active_samples"]=False
    else:contact["pairs"][0]["active_samples"]=2
    assert local_coulomb_diagnostics(contact,.2,proof)["status"]=="missing"


def test_coulomb_inactive_exception_requires_independent_zero_area_proof():
    contact,proof=qualified_contact();contact["samples"]=[];contact["pairs"][0]["active_samples"]=0
    assert local_coulomb_diagnostics(contact,.2,proof)["passed"] is None
    contact["pairs"][0]["zero_area_inactive_pair"]=True
    assert local_coulomb_diagnostics(contact,.2,proof)["passed"] is True


def test_nonzero_report_cannot_inherit_sixteen_frictionless_gates():
    report={"accepted":True,"diagnostic_only":False,"solver_status":"completed",
        "assessment":{"status":"numerically_accepted","gates":[{"name":name,"status":"passed"} for name in REQUIRED_RESULT_GATE_NAMES]}}
    assert len(REQUIRED_RESULT_GATE_NAMES)==16 and report_is_accepted(report)
    report["friction_coefficient"]=.2
    assert not report_is_accepted(report)
    report["assessment"]["gates"].append({"name":"local_coulomb_bound","status":"passed"})
    assert not report_is_accepted(report)
    report["friction_qualification_evidence"]={"verified":True,"friction_coefficient":.2,"qualification":qualified_contact()[1]}
    assert report_is_accepted(report)
    for coefficient in (None,float("nan"),-.2,True):
        report["friction_coefficient"]=coefficient
        assert not report_is_accepted(report)
    assert len(assess_assembly({},"completed")["gates"])==11
    gate=assess_assembly({"friction_coefficient":.2},"completed")["gates"][-1]
    assert gate["name"]=="local_coulomb_bound" and gate["status"]=="missing"


def test_actual_deck_friction_requires_fresh_proof_and_bound_mu_k_slope_build(tmp_path,monkeypatch):
    from structural import results
    project=StructuralProject();project.contacts.friction_coefficient=.2;project.save(tmp_path/"project.json")
    path=tmp_path/"assembly.inp"
    path.write_text("*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR\n100000\n*FRICTION\n.2,10000\n")
    metadata={"friction_coefficient":.2,"normal_contact_stiffness_mpa_per_mm":100000.,"friction_stick_slope_mpa_per_mm":10000.}
    run={"executable_sha256":"fixture-build","threads":4,"solver_version":"2.23"}
    # The actual helper has no accepted production proof; absence fails closed.
    absent=friction_configuration_evidence(path,metadata,run)
    assert absent["required"] and not absent["verified"] and absent["missing_reasons"]
    proof={"verified":True,**metadata,"selected_executable_sha256":"fixture-build","threads":4,"version":"2.23",
           "coulomb_absolute_tolerance_mpa":1e-5,"coulomb_relative_tolerance":1e-6}
    metadata["friction_qualification_verification"]=copy.deepcopy(proof)
    monkeypatch.setattr(results.friction_module,"verify_friction_qualification",lambda *a,**kw:proof)
    assert friction_configuration_evidence(path,metadata,run)["verified"]
    for key in ("friction_coefficient","normal_contact_stiffness_mpa_per_mm","friction_stick_slope_mpa_per_mm"):
        altered=copy.deepcopy(metadata);altered[key]*=2
        assert not friction_configuration_evidence(path,altered,run)["verified"]
    assert not friction_configuration_evidence(path,metadata,{**run,"threads":1})["verified"]
    path.write_text("*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR\n200000\n*FRICTION\n.2,10000\n")
    assert not friction_configuration_evidence(path,metadata,run)["verified"]
    path.write_text("*FRICTION\n.2\n")
    assert friction_configuration_evidence(path,metadata,run)["required"]
    assert not friction_configuration_evidence(path,metadata,run)["verified"]


def test_legacy_frictionless_identity_is_unchanged(tmp_path):
    path=tmp_path/"legacy.inp";path.write_text("*HEADING\nlegacy frictionless\n")
    assert friction_configuration_evidence(path,{}, {}) is None
    StructuralProject().save(tmp_path/"project.json")
    assert friction_configuration_evidence(path,{}, {}) is None


def test_later_nonzero_friction_card_cannot_hide_behind_first_zero_card(tmp_path):
    StructuralProject().save(tmp_path/"project.json")
    path=tmp_path/"conflicting.inp";path.write_text("*FRICTION\n0,10000\n*FRICTION\n.2,10000\n")
    result=friction_configuration_evidence(path,{"friction_coefficient":0},{})
    assert result["required"] and not result["verified"]


def test_legacy_report_without_coefficient_reads_retained_frozen_project(tmp_path):
    project=StructuralProject();project.save(tmp_path/"project.json")
    report={"accepted":True,"diagnostic_only":False,"solver_status":"completed","source_run":str(tmp_path/"assembly.run.json"),
        "assessment":{"status":"numerically_accepted","gates":[{"name":name,"status":"passed"} for name in REQUIRED_RESULT_GATE_NAMES]}}
    assert report_is_accepted(report)
    project.contacts.friction_coefficient=.2;project.save(tmp_path/"project.json")
    assert not report_is_accepted(report)
