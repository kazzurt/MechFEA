"""Exercise cancellation of a real, exclusively owned native solver process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import threading

import psutil

from .calculix import CalculixRunner, DEFAULT_EXECUTABLE, ROOT, sha256


def verify(output: str | Path) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "Dodec FEA/structural-fea-benchmark-runs/fixtures/rigid-anchorage-elimination-equivalence/short_screw.inp"
    executable = DEFAULT_EXECUTABLE.with_name("ccx_MT.exe")
    inp = output / "cancel_verification.inp"
    shutil.copyfile(source, inp)
    cancel = threading.Event()
    observed = {}
    parent = psutil.Process()

    def progress(record):
        if not observed:
            for child in parent.children(recursive=False):
                try:
                    if Path(child.exe()).resolve() == executable.resolve() and Path(child.cwd()).resolve() == output:
                        observed.update(pid=child.pid, create_time=child.create_time(), cwd=child.cwd(),
                                        executable=child.exe(), parent_pid=parent.pid)
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        if observed and record["elapsed_s"] >= .75:
            observed["cancel_requested_elapsed_s"] = record["elapsed_s"]
            cancel.set()

    run = CalculixRunner(executable, threads=4).run(inp, expected_steps=2, timeout_s=30,
                                                  cancel_event=cancel, on_progress=progress)
    alive = False
    if observed:
        try:
            process = psutil.Process(observed["pid"])
            alive = process.create_time() == observed["create_time"] and process.is_running()
        except psutil.NoSuchProcess:
            pass
    retained = {suffix: inp.with_suffix(suffix).is_file()
                for suffix in (".inp", ".stdout.log", ".stderr.log", ".run.json")}
    checks = {"owned_process_observed": bool(observed), "cancel_requested": cancel.is_set(),
              "owned_process_terminated": bool(observed) and not alive,
              "cancelled_status": run.status == "cancelled",
              "not_completed_or_accepted": not run.completed and not run.converged and not run.accepted,
              "logs_and_run_record_preserved": all(retained.values()),
              "input_unchanged": sha256(inp) == sha256(source)}
    result = {"accepted_cancellation_regression": all(checks.values()), "checks": checks,
              "owned_process": observed, "process_alive_after_runner_return": alive,
              "source_input": {"path":str(source), "sha256":sha256(source)},
              "retained": retained, "run":run.to_dict(),
              "scope":"Cancellation lifecycle only; partial fields are unaccepted diagnostic evidence."}
    (output / "cancellation-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.output)["accepted_cancellation_regression"] else 1)
