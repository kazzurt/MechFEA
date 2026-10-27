"""Numerical declarations cannot conceal different native convergence cards."""
import copy

import pytest

from structural.backends.contact_controls import contact_control_cards, verify_contact_control_binding
from structural.project import StructuralProject


def deck(cards=""):
    return "*HEADING\nControl binding fixture\n" + (
        "*STEP,NLGEOM\n" + cards + "*STATIC\n.1,1.,1e-6,.25\n*END STEP\n") * 2


def test_optional_contact_controls_preserve_legacy_project_and_deck(tmp_path):
    project = StructuralProject()
    legacy = copy.deepcopy(project.to_dict())
    fingerprint = project.fingerprint()
    project.solver["contact_count_relative_tolerance"] = None
    assert project.to_dict() == legacy
    assert project.fingerprint() == fingerprint
    assert contact_control_cards(project) == ""
    assert verify_contact_control_binding(project, deck(), {})["verified"]
    project.save(tmp_path / "legacy.json")
    assert "contact_count_relative_tolerance" not in StructuralProject.load(tmp_path / "legacy.json").solver


def test_positive_control_binds_both_steps_and_round_trips(tmp_path):
    project = StructuralProject()
    original = project.fingerprint()
    project.solver["contact_count_relative_tolerance"] = 1e-8
    cards = contact_control_cards(project)
    assert cards == "*CONTROLS,PARAMETERS=CONTACT\n1e-08\n"
    assert project.fingerprint() != original
    metadata = {"contact_count_relative_tolerance": 1e-8}
    result = verify_contact_control_binding(project, deck(cards), metadata)
    assert result["verified"] and [row["step"] for row in result["actual_controls"]] == [1, 2]
    project.save(tmp_path / "controlled.json")
    assert StructuralProject.load(tmp_path / "controlled.json").contact_count_relative_tolerance() == 1e-8


@pytest.mark.parametrize("value", [0, -1, .002, float("nan"), float("inf"), True, "1e-8", {}])
def test_invalid_or_looser_control_rejected(value):
    project = StructuralProject()
    project.solver["contact_count_relative_tolerance"] = value
    with pytest.raises(ValueError, match="Contact count"):
        project.validate()


@pytest.mark.parametrize("mutation", [
    lambda text: text.replace("1e-08", "0.001", 1),
    lambda text: text.replace("*CONTROLS,PARAMETERS=CONTACT\n1e-08\n", "", 1),
    lambda text: text.replace("*STATIC", "*CONTROLS,PARAMETERS=FIELD\n,1e-6\n*STATIC", 1),
    lambda text: text + "*CONTROLS,PARAMETERS=CONTACT\n1e-08\n",
    lambda text: text.replace("1e-08\n", "1e-08,.5\n", 1),
    lambda text: text.replace("PARAMETERS=CONTACT", "RESET", 1),
])
def test_changed_or_unregistered_native_controls_fail_binding(mutation):
    project = StructuralProject()
    project.solver["contact_count_relative_tolerance"] = 1e-8
    result = verify_contact_control_binding(project, mutation(deck(contact_control_cards(project))),
                                            {"contact_count_relative_tolerance": 1e-8})
    assert not result["verified"] and result["failures"]


def test_missing_metadata_and_undeclared_card_cannot_inherit_a_verified_flag():
    project = StructuralProject()
    project.solver["contact_count_relative_tolerance"] = 1e-8
    text = deck(contact_control_cards(project))
    assert not verify_contact_control_binding(project, text, {"contact_controls_binding": {"verified": True}})["verified"]
    project.solver.pop("contact_count_relative_tolerance")
    assert not verify_contact_control_binding(project, text, {})["verified"]
    assert not verify_contact_control_binding(project, deck(), {"contact_count_relative_tolerance": 1e-8})["verified"]
