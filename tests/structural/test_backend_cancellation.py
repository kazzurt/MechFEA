"""Real cancellation evidence and the desktop's delayed-kill ownership guard."""
import json
from types import SimpleNamespace

from structural.backends.calculix import ROOT


def test_old_cancellation_timer_cannot_stop_a_new_task(tmp_path, monkeypatch):
    from structural import gui
    callbacks = []
    monkeypatch.setattr(gui.QTimer, "singleShot", lambda milliseconds, callback: callbacks.append(callback))
    monkeypatch.setattr(gui.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("A stale timer must not call taskkill")))
    old_process = SimpleNamespace(processId=lambda: 888888)
    new_process = SimpleNamespace(state=lambda: gui.QProcess.ProcessState.Running,
                                  processId=lambda: 999999,
                                  kill=lambda: (_ for _ in ()).throw(AssertionError("New process was killed")))
    window = SimpleNamespace(process=old_process, active_output=tmp_path, active_command="solve",
                             cancelled=False, status=SimpleNamespace(setText=lambda text: None))
    window.force_stop = lambda expected_process=None: gui.StructuralWindow.force_stop(window, expected_process)
    gui.StructuralWindow.cancel_run(window)
    assert callbacks
    assert json.loads((tmp_path / "cancelled.json").read_text())["status"] == "cancel_requested"
    window.process = new_process
    callbacks[0]()


def test_stop_failure_stays_unconfirmed(tmp_path, monkeypatch):
    from structural import gui
    messages = []
    process = SimpleNamespace(state=lambda: gui.QProcess.ProcessState.Running, processId=lambda: 888888)
    window = SimpleNamespace(process=process, active_output=tmp_path,
                             log=SimpleNamespace(appendPlainText=messages.append),
                             status=SimpleNamespace(setText=messages.append))
    monkeypatch.setattr(gui.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=5, stderr=b"Access denied"))
    assert gui.StructuralWindow.force_stop(window, process) is False
    record = json.loads((tmp_path / "cancelled.json").read_text())
    assert record["status"] == "stop_unconfirmed"
    assert not record["accepted"]


def test_confirmed_cancellation_invalidates_coordinator_result_but_preserves_solver_evidence(tmp_path):
    from structural import gui
    from structural.results import report_is_accepted
    result_path = tmp_path / "structural-results/results.json"
    result_path.parent.mkdir()
    result_path.write_text(json.dumps({"solver_status":"completed", "accepted":True,
                                     "assessment":{"status":"accepted", "gates":[]}, "raw_metric":123.45}))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"status":"completed", "solver_completion":"completed", "accepted":True}))
    solver_path = tmp_path / "dodec_assembly.run.json"
    solver_evidence = '{"status":"completed","returncode":0,"accepted":false}'
    solver_path.write_text(solver_evidence)
    window = SimpleNamespace(active_output=tmp_path, active_command="solve")
    gui.StructuralWindow.finalize_cancelled_task(window, 1)
    record = json.loads(result_path.read_text())
    assert not report_is_accepted(record)
    assert record["solver_status"] == "completed" and record["raw_metric"] == 123.45
    assert record["task_status"] == "cancelled" and record["diagnostic_only"]
    assert any(gate["name"] == "task_not_cancelled" and gate["status"] == "failed" for gate in record["assessment"]["gates"])
    assert json.loads(manifest_path.read_text())["status"] == "cancelled"
    assert solver_path.read_text() == solver_evidence
    cancellation = json.loads((tmp_path / "cancelled.json").read_text())
    assert cancellation["stop_confirmed"] and cancellation["status"] == "cancelled"


def test_native_cancellation_evidence_when_available():
    import pytest
    path = ROOT / "Dodec FEA/structural-fea-benchmark-runs/fixtures/native-cancellation-lifecycle-verification/cancellation-verification.json"
    if not path.exists():
        pytest.skip("Run the native cancellation fixture first")
    evidence = json.loads(path.read_text())
    assert evidence["accepted_cancellation_regression"]
    assert all(evidence["checks"].values())
    assert evidence["run"]["status"] == "cancelled"
    assert evidence["run"]["accepted"] is False
    assert evidence["process_alive_after_runner_return"] is False
