"""Pinned CalculiX contact-count controls and retained-deck binding.

This numerical criterion does not replace physical contact acceptance. Equal
active counts can conceal changes of identity; tensile contact still fails.
"""
from __future__ import annotations

from ..provenance import capture_module_source

IMPLEMENTATION_PROVENANCE = capture_module_source(__file__, __name__)


def contact_control_cards(project) -> str:
    value = project.contact_count_relative_tolerance()
    if value is None:
        return ""
    return f"*CONTROLS,PARAMETERS=CONTACT\n{value:.14g}\n"


def verify_contact_control_binding(project, deck_text: str, metadata: dict) -> dict:
    """Require exactly the declared controls in each of the two assembly steps.

    Unregistered FIELD/RESET/other controls are rejected as well: a declaration
    cannot establish a controlled study when the actual deck changes them.
    """
    expected = project.contact_count_relative_tolerance()
    failures = []
    recorded = metadata.get("contact_count_relative_tolerance")
    if isinstance(recorded, bool) or recorded != expected:
        failures.append("Contact-count metadata differs from the frozen project")
    lines = [line.strip() for line in deck_text.splitlines()
             if line.strip() and not line.lstrip().startswith("**")]
    step = None
    steps = 0
    found = []
    for index, line in enumerate(lines):
        if not line.startswith("*"):
            continue
        card = line.upper().replace(" ", "")
        keyword = card.split(",", 1)[0]
        if keyword == "*STEP":
            steps += 1
            step = steps
        elif keyword == "*ENDSTEP":
            step = None
        elif keyword == "*CONTROLS":
            values = []
            for following in lines[index + 1:]:
                if following.startswith("*"):
                    break
                values.append(following)
            found.append({"step": step, "card": card, "data": values})
    if steps != 2:
        failures.append("Contact-control binding requires both assembly steps")
    if expected is None:
        if found:
            failures.append("Undeclared native convergence controls occur in the deck")
    else:
        if len(found) != 2 or [row["step"] for row in found] != [1, 2]:
            failures.append("Each assembly step requires exactly one declared contact-control card")
        expected_data = f"{expected:.14g}"
        for row in found:
            if row["card"] != "*CONTROLS,PARAMETERS=CONTACT" or row["data"] != [expected_data]:
                failures.append("Native control card differs from the exact generated contact-count setting")
                break
    return {"verified": not failures, "contact_count_relative_tolerance": expected,
            "native_default_relative_tolerance": .001, "actual_controls": found,
            "failures": failures,
            "limitation": "The native criterion compares active contact counts, not point identities or pressure signs; independent contact gates remain required."}
