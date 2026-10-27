import copy
import json

import pytest

from structural.project import sha256
from structural.results import COMPONENT_DOMAINS, retained_result_source_binding, retained_component_field_binding


def retained_report(tmp_path):
    deck=tmp_path/"assembly.inp";deck.write_text("retained exact deck")
    extensions=(".inp",".assembly.json",".run.json",".dat",".frd",".sta",".stdout.log",".stderr.log")
    for extension in extensions[1:]:deck.with_suffix(extension).write_text("native fixture")
    run=deck.with_suffix(".run.json")
    run.write_text(json.dumps({"status":"completed","returncode":0,"input_path":str(deck),"input_sha256":sha256(deck)}))
    report={"source_run":str(run),"input_sha256":sha256(deck),
            "source_file_hashes":{extension:sha256(deck.with_suffix(extension)) for extension in extensions}}
    return deck,report


@pytest.mark.parametrize("mutation",["dat","missing_hash","different_run_path","failed_run","report_input"])
def test_all_eight_native_sources_status_and_exact_source_run_binding(tmp_path,mutation):
    deck,report=retained_report(tmp_path)
    assert retained_result_source_binding(report)["verified"]
    if mutation=="dat":deck.with_suffix(".dat").write_text("changed results after reviewed report")
    elif mutation=="missing_hash":del report["source_file_hashes"][".stderr.log"]
    elif mutation=="different_run_path":
        other=tmp_path/"other.run.json";other.write_bytes(deck.with_suffix(".run.json").read_bytes());report["source_run"]=str(other)
    elif mutation=="failed_run":
        path=deck.with_suffix(".run.json");data=json.loads(path.read_text());data["status"]="failed";path.write_text(json.dumps(data))
        report["source_file_hashes"][".run.json"]=sha256(path)
    else:report["input_sha256"]="0"*64
    result=retained_result_source_binding(report)
    assert not result["verified"] and result["failures"]


def test_relative_report_source_resolves_from_report_directory(tmp_path):
    deck,report=retained_report(tmp_path)
    destination=tmp_path/"review";destination.mkdir()
    report["source_run"]="../assembly.run.json"
    assert retained_result_source_binding(report,destination/"results.json")["verified"]
    assert not retained_result_source_binding(report)["verified"]


def test_component_fields_require_report_pinned_hashes_in_both_states(tmp_path):
    field=tmp_path/"component.vtu";field.write_text("exported component field")
    entry={"status":"available","vtu":field.name,"vtu_sha256":sha256(field)}
    report={"stages":{state:{"components":{name:copy.deepcopy(entry) for name in COMPONENT_DOMAINS}}
                      for state in ("preload","loaded")}}
    path=tmp_path/"results.json"
    assert retained_component_field_binding(report,path)["verified"]
    old=copy.deepcopy(report);del old["stages"]["loaded"]["components"]["pipe"]["vtu_sha256"]
    assert not retained_component_field_binding(old,path)["verified"]
    field.write_text("changed derived stress with unchanged native DAT")
    result=retained_component_field_binding(report,path)
    assert not result["verified"] and len(result["failures"])==10
