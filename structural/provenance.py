"""Evidence for the implementation loaded in a long-running Python process."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import types


def capture_module_source(path, name):
    source=Path(path).resolve()
    return {"module":name,"source_path":str(source),
            "source_sha256_at_import":hashlib.sha256(source.read_bytes()).hexdigest(),
            "captured_at_utc":datetime.now(timezone.utc).isoformat(),"python_version":sys.version,
            "interpretation":"Source bytes captured during module import; later file edits do not update this record"}


def function_code_sha256(function):
    """Hash the function's actual loaded code object, without rereading its file."""
    # marshal includes interpreter reference/interning flags that can change
    # after execution. Normalize immutable code attributes instead.
    def normalize(value):
        if isinstance(value,types.CodeType):
            return {key:normalize(getattr(value,key)) for key in (
                "co_code","co_consts","co_names","co_varnames","co_freevars","co_cellvars",
                "co_argcount","co_posonlyargcount","co_kwonlyargcount","co_flags","co_exceptiontable")}
        if isinstance(value,bytes):return {"bytes":value.hex()}
        if isinstance(value,(tuple,list)):return [normalize(item) for item in value]
        if isinstance(value,frozenset):return {"frozenset":sorted((normalize(item) for item in value),key=repr)}
        if value is None or isinstance(value,(bool,int,str)):return value
        return {"type":type(value).__name__,"value":repr(value)}
    return hashlib.sha256(json.dumps(normalize(function.__code__),sort_keys=True,separators=(",",":")).encode()).hexdigest()


def module_code_sha256(namespace):
    digest=hashlib.sha256()
    for name,value in sorted(namespace.items()):
        if isinstance(value,types.FunctionType) and value.__module__==namespace.get("__name__"):
            digest.update(name.encode());digest.update(function_code_sha256(value).encode())
    return digest.hexdigest()
