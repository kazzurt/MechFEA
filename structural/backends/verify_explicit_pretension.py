"""Independent analytical check of explicit shoulder-MPC pretension formula."""
from pathlib import Path
import json
import argparse
from .explicit_pretension import convert_shoulder_fixture
from .calculix import CalculixRunner
from ..fixtures import preload_lock_deck, evaluate


def verify(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    native=output/"native-bar-definition"
    native.mkdir()
    text,target=preload_lock_deck()
    lines=[]
    node_mode=False
    for line in text.splitlines():
        if line.startswith("*"):
            node_mode=line=="*NODE" or line.startswith("*NODE,")
            lines.append(line.replace("EALL","BOLT"))
        elif node_mode:
            tokens=line.split(",")
            tokens[1]=str(float(tokens[1])-10.)
            lines.append(",".join(tokens))
        else:
            lines.append(line)
    inp=native/"bar.inp"
    inp.write_text("\n".join(lines)+"\n")
    (native/"mesh-manifest.json").write_text(json.dumps({"symmetry_fraction":1.,"purpose":"Uniform bar analytic explicit-jump fixture","expected_preload_N":100.,"expected_service_force_N":150.}))
    explicit=convert_shoulder_fixture(inp,output/"explicit-bar-verification")
    run=CalculixRunner().run(explicit,expected_steps=2,timeout_s=60)
    check=evaluate("preload_lock",run,target)
    (output/"verification.json").write_text(json.dumps(check,indent=2))
    print(json.dumps(check,indent=2))
    return check


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    raise SystemExit(0 if verify(args.output)["accepted"] else 1)
