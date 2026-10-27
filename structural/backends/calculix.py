"""Pinned native CalculiX runner and loss-conscious ASCII result readers.

Completion is a process/convergence fact, never numerical acceptance. Input and
all solver outputs remain in their run directory; existing outputs cannot be
overwritten accidentally. N-mm-MPa is the structural workflow's unit convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Callable

from ..provenance import capture_module_source

IMPLEMENTATION_PROVENANCE = capture_module_source(__file__, __name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTABLE = Path(os.environ.get("MECHFEA_CCX", str(ROOT / "structural-fea-environment/solver/CalculiX-2.23.0-win-x64/bin/ccx.exe")))
FLOAT = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?"
DAT_EXPANDED_EXPONENT = r"[+-]?(?:\d+\.\d*|\.\d+)[+-]\d{3}(?!\d)"
DAT_NUMBER = rf"(?:[+-]?(?:nan|inf(?:inity)?)\b|{DAT_EXPANDED_EXPONENT}|{FLOAT})"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RunResult:
    status: str
    input_path: str
    executable: str
    returncode: int | None = None
    elapsed_s: float = 0.0
    expected_steps: int = 0
    completed_steps: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    critical_warnings: list[str] = field(default_factory=list)
    solver_version: str | None = None
    input_sha256: str = ""
    executable_sha256: str = ""
    peak_rss_bytes: int = 0
    threads: int = 1
    accepted: bool = False

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def converged(self) -> bool:
        return self.completed

    def to_dict(self) -> dict:
        return asdict(self)


class CalculixRunner:
    def __init__(self, executable: str | Path | None = None, expected_version: str = "2.23", threads: int = 1):
        self.executable = Path(executable or DEFAULT_EXECUTABLE).resolve()
        self.expected_version = expected_version
        if isinstance(threads,bool) or not isinstance(threads,int) or threads<1:
            raise ValueError("Thread count must be a positive integer")
        self.threads = threads

    def run(self, input_path: str | Path, expected_steps: int | None = None,
            timeout_s: float = 1800, cancel_event: threading.Event | None = None,
            on_progress: Callable[[dict], None] | None = None,
            expected_step_times: list[float] | None = None) -> RunResult:
        """Run an isolated deck, preserving failure/timeout/cancel states.

        A directory should contain one immutable job. ``expected_step_times``
        defaults to 1.0 for each step and must be supplied for other durations.
        """
        input_path = Path(input_path).resolve()
        if isinstance(timeout_s,bool) or not isinstance(timeout_s,(int,float)) or not math.isfinite(timeout_s) or timeout_s<=0:
            raise ValueError("Solver timeout must be finite and positive")
        if input_path.suffix.lower() != ".inp" or not input_path.is_file():
            raise ValueError("An existing .inp file is required")
        if not self.executable.is_file():
            raise FileNotFoundError(f"CalculiX executable not found: {self.executable}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", input_path.stem):
            raise ValueError("Solver job name must contain only letters, numbers, _ or -")
        deck = input_path.read_text(encoding="utf-8")
        actual_steps = len(re.findall(r"(?im)^\*STEP(?:\s|,|$)", deck))
        expected_steps = actual_steps if expected_steps is None else expected_steps
        if expected_steps < 1 or actual_steps != expected_steps:
            raise ValueError("Expected step count must match the input deck")
        expected_step_times = expected_step_times or [1.0] * expected_steps
        if len(expected_step_times) != expected_steps or any(
                isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0
                for x in expected_step_times):
            raise ValueError("Provide one positive duration per step")
        for ext in (".dat", ".frd", ".sta", ".cvg", ".stdout.log", ".stderr.log", ".run.json"):
            if input_path.with_suffix(ext).exists():
                raise FileExistsError(f"Refusing to overwrite run evidence: {input_path.with_suffix(ext)}")
        result = RunResult("running", str(input_path), str(self.executable), expected_steps=expected_steps,
                           input_sha256=sha256(input_path), executable_sha256=sha256(self.executable), threads=self.threads)
        start = time.monotonic()
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS=str(self.threads), CCX_NPROC_STIFFNESS=str(self.threads),
                           CCX_NPROC_RESULTS=str(self.threads), CCX_NPROC_EQUATION_SOLVER=str(self.threads))
        environment["PATH"] = str(self.executable.parent) + os.pathsep + environment.get("PATH", "")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = None
        try:
            with input_path.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, \
                    input_path.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
                process = subprocess.Popen([str(self.executable), "-i", input_path.stem],
                                           cwd=input_path.parent, stdout=stdout, stderr=stderr,
                                           env=environment, creationflags=creationflags)
                monitor = None
                if getattr(process,"pid",None):
                    try:
                        import psutil
                    except ImportError:
                        pass
                    else:
                        try:
                            monitor = psutil.Process(process.pid)
                        except (OSError,psutil.Error):
                            # Sampling can race with an already-finished child;
                            # process completion remains determined by Popen.
                            pass
                while process.poll() is None:
                    elapsed = time.monotonic() - start
                    if monitor:
                        try:
                            result.peak_rss_bytes=max(result.peak_rss_bytes,monitor.memory_info().rss)
                        except (OSError,psutil.Error):
                            pass
                    if (cancel_event and cancel_event.is_set()) or elapsed > timeout_s:
                        result.status = "cancelled" if cancel_event and cancel_event.is_set() else "timed_out"
                        break
                    if on_progress:
                        on_progress({"elapsed_s": elapsed, "peak_rss_bytes":result.peak_rss_bytes,"increments": parse_sta(input_path.with_suffix(".sta"))})
                    time.sleep(0.1)
                result.returncode = process.returncode
        except (Exception, KeyboardInterrupt) as exc:
            result.status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            result.errors.append(f"Runner {type(exc).__name__}: {exc}")
        finally:
            # A progress callback or I/O failure must never orphan this solver.
            # Only the exact Popen child owned by this invocation is stopped.
            if process is not None:
                try:
                    if process.poll() is None:
                        try:
                            process.terminate()
                        except OSError as exc:
                            result.errors.append(f"Owned solver terminate failed; trying forced stop: {exc}")
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        if process.returncode is None:
                            raise RuntimeError("Owned solver did not report termination")
                    result.returncode = process.returncode
                except Exception as exc:
                    result.status = "failed"
                    result.errors.append(f"Owned solver stop unconfirmed: {type(exc).__name__}: {exc}")
        result.elapsed_s = time.monotonic() - start
        stdout_path, stderr_path = input_path.with_suffix(".stdout.log"), input_path.with_suffix(".stderr.log")
        stdout_text = stdout_path.read_text(errors="replace") if stdout_path.is_file() else ""
        stderr_text = stderr_path.read_text(errors="replace") if stderr_path.is_file() else ""
        logs = stdout_text + "\n" + stderr_text
        version = re.search(r"(?:CalculiX Version|This is Version)\s+([\d.]+)", logs)
        result.solver_version = version.group(1) if version else None
        if result.solver_version != self.expected_version:
            result.errors.append(f"Expected solver version {self.expected_version}, observed {result.solver_version}")
        result.errors.extend(line.strip() for line in logs.splitlines() if re.search(r"\*ERROR|fatal error", line, re.I))
        result.warnings = log_warning_blocks(logs)
        result.critical_warnings = critical_log_issues(logs)
        result.errors.extend("Critical constraint warning: "+warning for warning in result.critical_warnings)
        increments = parse_sta(input_path.with_suffix(".sta"))
        result.completed_steps = sorted({r["step"] for r in increments if r["step"] <= expected_steps and
                                         r["step_time"] >= expected_step_times[r["step"] - 1] * (1 - 1e-7)})
        if result.status == "running":
            finished = "Job finished" in stdout_text
            has_fields = input_path.with_suffix(".frd").is_file() and input_path.with_suffix(".frd").stat().st_size > 100
            result.status = "completed" if (result.returncode == 0 and finished and has_fields and not result.errors
                                           and result.completed_steps == list(range(1, expected_steps + 1))) else "failed"
            if not finished:
                result.errors.append("Solver completion marker missing")
            if not has_fields:
                result.errors.append("ASCII FRD result fields missing")
            if result.completed_steps != list(range(1, expected_steps + 1)):
                result.errors.append("Not all requested steps reached their final time")
        manifest = result.to_dict() | {"recorded_utc": datetime.now(timezone.utc).isoformat(),
                                       "units": "N-mm-MPa", "expected_step_times": expected_step_times,
                                       "expected_solver_version": self.expected_version,
                                       "command": [str(self.executable), "-i", input_path.stem]}
        input_path.with_suffix(".run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return result


def log_warning_blocks(log_text: str) -> list[str]:
    """Keep wrapped solver warning context, not merely its introductory line."""
    blocks=[]
    lines=log_text.splitlines()
    for index,line in enumerate(lines):
        if "*WARNING" not in line.upper():
            continue
        block=[line.strip()]
        for following in lines[index+1:index+14]:
            if not following.strip() or following.lstrip().startswith("*"):
                break
            block.append(following.strip())
        blocks.append("\n".join(block))
    return blocks


def critical_log_issues(log_text: str) -> list[str]:
    """Find evidence that a requested connection was omitted by the solver.

    CalculiX can exit successfully after rejecting tied nodes. These are model
    definition failures, even if the remaining system reaches all step times.
    This helper can also audit an existing run made with an older runner.
    """
    issues=[]
    for warning in log_warning_blocks(log_text):
        if (re.search(r"gentiedmpc|tiedcontact",warning,re.I) and
                re.search(r"no tied (?:MPC|constraint)|not generated|not active|too far|not found|miss",warning,re.I)):
            issues.append(warning)
    # Preserve protection if a truncated/log-filtered stream lost its header.
    if not issues and re.search(r"no tied (?:MPC|constraint)(?:\s|$)",log_text,re.I):
        issues.append("Solver reports omitted tied MPC/constraint; inspect full stdout and failed-node sets")
    return issues


def parse_sta(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < 7 or not all(t.isdigit() for t in tokens[:4]):
            continue
        try:
            rows.append(dict(zip(("step", "increment", "attempt", "iterations", "total_time", "step_time", "increment_time"),
                                 [*map(int, tokens[:4]), *[float(t.replace("D", "E")) for t in tokens[4:7]]])))
        except ValueError:
            continue
    return rows


def parse_dat(path: str | Path) -> list[dict]:
    """Return every numeric DAT table with its original header and time.

    IDs and integration point columns are retained in ``rows``. Native NaN/Inf
    values stay nonfinite, with their original tokens recorded separately;
    consumers must establish whether a quantity is undefined or invalid.
    No smoothing, stress averaging or contact sign change occurs here.
    """
    blocks: list[dict] = []
    header: list[str] = []
    current = None
    context_time = None
    for line in Path(path).read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = re.findall(DAT_NUMBER, stripped, flags=re.I)
        remainder = re.sub(DAT_NUMBER, "", stripped, flags=re.I).replace(" ", "").replace("\t", "")
        numeric = bool(tokens) and remainder == ""
        if numeric:
            if current is None:
                raw_title = " ".join(header)
                # The first table in a step is preceded by a spaced STEP banner.
                # Keep that original text, but expose the same canonical table
                # label for first and subsequent frames to every reader.
                title = re.sub(r"^\s*(?:S\s*T\s*E\s*P\s+\d+\s+)?INCREMENT\s+\d+\s+", "", raw_title, flags=re.I)
                tm = re.search(r"(?:time|TIME)\s*(?:=)?\s*(" + FLOAT + ")", title)
                if tm:
                    context_time = float(tm.group(1).replace("D", "E"))
                set_match = re.search(r"for set\s+(\S+)", title, re.I)
                current = {"header": title, "raw_header": raw_title, "time": context_time,
                           "set": set_match.group(1) if set_match else None, "rows": []}
                blocks.append(current)
                header = []
            values = [float(re.sub(r"([+-]\d{3})$", r"E\1", v)
                            if re.fullmatch(DAT_EXPANDED_EXPONENT, v)
                            else v.replace("D", "E").replace("d", "e")) for v in tokens]
            for column, (token, value) in enumerate(zip(tokens, values)):
                if not math.isfinite(value):
                    current.setdefault("nonfinite_entries", []).append(
                        {"row":len(current["rows"]),"column":column,"raw_token":token})
            current["rows"].append(values)
        else:
            if current is not None:
                current = None
                header = []
            header.append(stripped)
    return blocks


def parse_frd(path: str | Path) -> dict:
    """Read ASCII FRD coordinates/connectivity/fields using fixed-width floats.

    Field names/components and step metadata are preserved. The FRD STRESS field
    is extrapolated/averaged nodal display data, NOT integration-point raw stress.
    Request *EL PRINT,S separately when quantitative raw stresses are needed.
    """
    result = {"nodes": {}, "elements": {}, "fields": [], "stress_location": "extrapolated_nodal_display"}
    mode = None
    field = None
    element = None
    time_value = None
    pst = None
    for line in Path(path).read_text(errors="replace").splitlines():
        code = line[:3].strip()
        if line.startswith("    2C"):
            mode = "nodes"
        elif line.startswith("    3C"):
            mode = "elements"
        elif line.startswith("    1PSTEP"):
            pst = [int(x) for x in line.split()[1:]]
        elif line.startswith("  100C"):
            values = line.split()
            time_value = float(values[2])
            mode = "fields"
        elif code == "-4":
            fields = line.split()
            field = {"name": fields[1], "components": [], "time": time_value, "pstep": pst, "values": {}}
            result["fields"].append(field)
            mode = "fields"
        elif code == "-5" and field is not None:
            field["components"].append(line.split()[1])
        elif code == "-1":
            identifier = int(line[3:13])
            if mode == "nodes":
                result["nodes"][identifier] = _frd_floats(line[13:])
            elif mode == "elements":
                metadata = [int(x) for x in line[13:].split()]
                element = {"type": metadata[0], "metadata": metadata[1:], "nodes": []}
                result["elements"][identifier] = element
            elif mode == "fields" and field is not None:
                field["values"][identifier] = _frd_floats(line[13:])
        elif code == "-2" and mode == "elements" and element is not None:
            element["nodes"].extend(int(x) for x in line[3:].split())
        elif code == "-3":
            mode = None
    return result


def _frd_floats(text: str) -> list[float]:
    # The pinned MinGW build emits three exponent digits and may expand a
    # negative value beyond the nominal 12-character field. Fixed slicing fails.
    values = re.findall(r"[+-]?\d\.\d{5}[Ee][+-]\d{2,3}", text)
    if re.sub(r"[+-]?\d\.\d{5}[Ee][+-]\d{2,3}", "", text).strip():
        raise ValueError(f"Unrecognized ASCII FRD numeric record: {text!r}")
    return [float(value) for value in values]
