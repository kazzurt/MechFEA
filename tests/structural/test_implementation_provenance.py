import hashlib
from structural.provenance import capture_module_source, function_code_sha256, module_code_sha256


def test_import_snapshot_and_loaded_code_do_not_follow_later_file_edits(tmp_path):
    source=tmp_path/"retained.py"
    original="def value():\n    return 1\n"
    source.write_text(original,encoding="utf-8")
    original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot=capture_module_source(source,"fixture")
    namespace={"__name__":"fixture"}
    exec(compile(original,str(source),"exec"),namespace)
    function_hash=function_code_sha256(namespace["value"])
    module_hash=module_code_sha256(namespace)
    source.write_text("def value():\n    return 2\n",encoding="utf-8")
    assert snapshot["source_sha256_at_import"]==original_hash
    assert snapshot["source_sha256_at_import"]!=hashlib.sha256(source.read_bytes()).hexdigest()
    assert function_code_sha256(namespace["value"])==function_hash
    assert module_code_sha256(namespace)==module_hash
    assert namespace["value"]()==1
