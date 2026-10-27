"""Result format and failure-state regressions derived from native 2.23 runs."""
import json
import math
from pathlib import Path
import threading
import sys

import pytest

from structural.backends.calculix import CalculixRunner, parse_dat, parse_frd, parse_sta, critical_log_issues


def test_dat_increment_only_banner_retains_later_history(tmp_path):
    path=tmp_path/'increment-history.dat'
    path.write_text('INCREMENT     10\n\n displacements (vx,vy,vz) for set NALL and time 0.1000000E+01\n\n 1 0.1 0.0 0.0\n')
    block=parse_dat(path)[0]
    assert block['header'].startswith('displacements')
    assert block['raw_header'].startswith('INCREMENT')
    assert block['time']==1. and block['rows']==[[1.,.1,0.,0.]]


def test_dat_native_omitted_e_three_digit_exponents_are_single_values(tmp_path):
    path=tmp_path/'opened-contact-tiny-reactions.dat'
    path.write_text('forces (fx,fy,fz) for set NALL and time 0.5700000E+01\n'
                    '1 2.083940-101 6.763533-102 -7.711638-103\n'
                    '2 -1.234567+101 0.000000E+00 4.500000D-03\n')
    block=parse_dat(path)[0]
    assert len(block['rows'][0])==4
    assert block['rows'][0]==[1.,2.083940e-101,6.763533e-102,-7.711638e-103]
    assert block['rows'][1]==[2.,-1.234567e101,0.,.0045]


def test_frd_windows_expanded_exponent_and_negative_adjacent_field(tmp_path):
    path=tmp_path/"actual-format.frd"
    path.write_text("    2C                           1                                     1\n"
                    " -1         12.50000E+001-5.00000E+0000.00000E+000\n -3\n"
                    "    1PSTEP                         1           7           2\n"
                    "  100CL  101 2.000000000           1                     0    1           1\n"
                    " -4  DISP        4    1\n -5  D1          1    2    1    0\n"
                    " -1         11.21951E-003-7.07317E-005-7.07317E-005\n -3\n")
    result=parse_frd(path)
    assert result["nodes"][1]==[25.,-5.,0.]
    assert result["fields"][0]["values"][1]==pytest.approx([.00121951,-.0000707317,-.0000707317])
    assert result["fields"][0]["time"]==2.
    assert result["fields"][0]["pstep"]==[1,7,2]


def test_frd_standard_two_digit_exponents(tmp_path):
    path=tmp_path/"standard.frd"
    path.write_text("    2C                           1                                     1\n"
                    " -1         1 2.50000E+01-5.00000E+00 0.00000E+00\n -3\n")
    assert parse_frd(path)["nodes"][1]==[25.,-5.,0.]


def test_contact_dat_keeps_signed_gap_and_statistic_time(tmp_path):
    path=tmp_path/"contact.dat"
    path.write_text("relative contact displacement (slave element+face,normal,tang1,tang2) for all contact elements and time 0.1000000E+01\n"
                    "9 1 -1.000000E-06 0.000000E+00 0.000000E+00\n"
                    "statistics for slave set SLAVE, master set MASTER and time 0.1000000E+01\n"
                    "total surface force (fx,fy,fz) and moment about the origin (mx,my,mz)\n"
                    "0 0 1000 5000 -5000 0\n"
                    "area, normal force (+ = tension) and shear force (size)\n100 -1000 0\n")
    blocks=parse_dat(path)
    assert blocks[0]["rows"][0][2]==-1e-6
    assert all(b["time"]==1 for b in blocks)
    assert blocks[1]["rows"][0][2]==1000
    assert blocks[2]["rows"][0][1]==-1000


def test_first_dat_table_step_banner_is_preserved_but_not_part_of_canonical_label(tmp_path):
    path=tmp_path/"first-table.dat"
    path.write_text(" S T E P       1\n INCREMENT     7\n\n"
                    " displacements (vx,vy,vz) for set ALL_NODES and time 0.1000000E+01\n"
                    "1 1.000000E-03 0.000000E+00 0.000000E+00\n")
    block=parse_dat(path)[0]
    assert block["header"].startswith("displacements (")
    assert block["raw_header"].startswith("S T E P")
    assert block["time"]==1. and block["set"]=="ALL_NODES"
    assert block["rows"]==[[1.,.001,0.,0.]]


def test_inactive_contact_zero_area_preserves_undefined_nan_scalars(tmp_path):
    path=tmp_path/"inactive-contact.dat"
    path.write_text("statistics for slave set PIPE_BORE, master set HUB_SNOUT and time 0.2000000E+01\n"
                    "total surface force (fx,fy,fz) and moment about the origin (mx,my,mz)\n"
                    "0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00\n"
                    "center of gravity and mean normal\n"
                    "NaN NaN NaN NaN NaN NaN\n"
                    "area, normal force (+ = tension) and shear force (size)\n"
                    "0.000000E+00 NaN NaN\n"
                    "contact stress for all contact elements and time 0.2000000E+01\n"
                    "10 1 1.000000E+02 0.000000E+00 0.000000E+00\n")
    blocks=parse_dat(path)
    assert len(blocks)==4 and all(block["time"]==2. for block in blocks)
    area=blocks[2]
    assert area["header"].startswith("area,") and area["rows"][0][0]==0.
    assert all(math.isnan(value) for value in area["rows"][0][1:])
    assert area["nonfinite_entries"]==[{"row":0,"column":1,"raw_token":"NaN"},
                                        {"row":0,"column":2,"raw_token":"NaN"}]
    assert all(math.isnan(value) for value in blocks[1]["rows"][0])
    assert blocks[3]["header"].startswith("contact stress")


def test_sta_ignores_unconverged_attempt_markers(tmp_path):
    path=tmp_path/"job.sta"
    path.write_text("STEP INC ATT ITRS TOT TIME STEP TIME INC TIME\n"
                    "1 1 1U 20 0.1 0.1 0.1\n1 1 2 5 0.05 0.05 0.05\n")
    assert len(parse_sta(path))==1
    assert parse_sta(path)[0]["step_time"]==.05


@pytest.mark.parametrize("finish_time,returncode,error,expected",[(.5,0,"","failed"),(1.,1,"","failed"),(1.,0,"*ERROR convergence failed","failed"),(1.,0,"*WARNING in gentiedmpc: no tied MPC\n generated for node 2244\n opposite master face found but\n too far away; distance: .00138","failed"),(1.,0,"","completed")])
def test_completion_requires_final_step_time_and_clean_exit(tmp_path,monkeypatch,finish_time,returncode,error,expected):
    inp=tmp_path/"job.inp"
    inp.write_text("*STEP\n*STATIC\n*END STEP\n")
    class FinishedProcess:
        def __init__(self,*args,**kwargs):
            self.returncode=returncode
            kwargs["stdout"].write("CalculiX Version 2.23\n"+error+"\nJob finished\n")
            inp.with_suffix(".sta").write_text(f"1 1 1 1 {finish_time} {finish_time} .1\n")
            inp.with_suffix(".frd").write_text("x"*101)
        def poll(self): return self.returncode
    monkeypatch.setattr("structural.backends.calculix.subprocess.Popen",FinishedProcess)
    result=CalculixRunner(sys.executable).run(inp)
    assert result.status==expected
    assert result.accepted is False
    assert inp.with_suffix(".run.json").exists()


def test_wrapped_omitted_tie_warning_is_critical_but_increment_warning_is_not():
    log="*WARNING reading *STATIC:\n minimum increment changed to1.e-6\n\n*WARNING in gentiedmpc:\n DOF3 of node651 is not active;\n no tied constraint is generated\n\nJob finished\n"
    issues=critical_log_issues(log)
    assert len(issues)==1
    assert "node651" in issues[0]
    assert "minimum increment" not in issues[0]


def test_cancellation_is_not_a_completed_run(tmp_path,monkeypatch):
    inp=tmp_path/"job.inp"
    inp.write_text("*STEP\n*STATIC\n*END STEP\n")
    class RunningProcess:
        returncode=None
        def __init__(self,*args,**kwargs): pass
        def poll(self): return None
        def terminate(self): self.returncode=-15
        def wait(self,timeout=None): return self.returncode
    monkeypatch.setattr("structural.backends.calculix.subprocess.Popen",RunningProcess)
    event=threading.Event()
    event.set()
    result=CalculixRunner(sys.executable).run(inp,cancel_event=event)
    assert result.status=="cancelled"
    assert not result.converged and not result.accepted


@pytest.mark.parametrize("callback_error",[RuntimeError("progress widget failed"),BrokenPipeError("progress consumer closed")])
def test_progress_failure_stops_owned_child_and_preserves_failure_manifest(tmp_path,monkeypatch,callback_error):
    inp=tmp_path/"job.inp"
    inp.write_text("*STEP\n*STATIC\n*END STEP\n")
    instances=[]
    class OwnedProcess:
        returncode=None
        terminated=False
        waited=False
        def __init__(self,*args,**kwargs):
            instances.append(self)
            kwargs["stdout"].write("CalculiX Version 2.23\n")
        def poll(self):return self.returncode
        def terminate(self):self.terminated=True;self.returncode=-15
        def wait(self,timeout=None):self.waited=True;return self.returncode
    def failing_callback(record):raise callback_error
    monkeypatch.setattr("structural.backends.calculix.subprocess.Popen",OwnedProcess)
    result=CalculixRunner(sys.executable).run(inp,on_progress=failing_callback)
    assert instances[0].terminated and instances[0].waited
    assert result.status=="failed" and result.returncode==-15 and not result.accepted
    assert any(type(callback_error).__name__ in error for error in result.errors)
    assert json.loads(inp.with_suffix(".run.json").read_text())["status"]=="failed"
    assert inp.with_suffix(".stdout.log").read_text().startswith("CalculiX Version")


@pytest.mark.parametrize("extension",[".dat",".cvg",".stderr.log"])
def test_prevents_overwriting_run_evidence(tmp_path,extension):
    inp=tmp_path/"job.inp"
    inp.write_text("*STEP\n*END STEP\n")
    inp.with_suffix(extension).write_text("earlier failed run")
    with pytest.raises(FileExistsError):
        CalculixRunner(sys.executable).run(inp)


@pytest.mark.parametrize("timeout",[0,-1,float("nan"),float("inf"),True])
def test_invalid_timeout_never_launches_solver(tmp_path,monkeypatch,timeout):
    inp=tmp_path/"job.inp";inp.write_text("*STEP\n*END STEP\n")
    monkeypatch.setattr("structural.backends.calculix.subprocess.Popen",lambda *a,**k:pytest.fail("Invalid timeout launched a process"))
    with pytest.raises(ValueError,match="finite and positive"):
        CalculixRunner(sys.executable).run(inp,timeout_s=timeout)


def test_optional_resource_monitor_race_does_not_fail_valid_completed_run(tmp_path,monkeypatch):
    import psutil
    inp=tmp_path/"job.inp";inp.write_text("*STEP\n*END STEP\n")
    class FinishedProcess:
        pid=123
        returncode=0
        def __init__(self,*args,**kwargs):
            kwargs["stdout"].write("CalculiX Version 2.23\nJob finished\n")
            inp.with_suffix(".sta").write_text("1 1 1 1 1 1 1\n")
            inp.with_suffix(".frd").write_text("x"*101)
        def poll(self):return self.returncode
    def gone(pid):raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr("structural.backends.calculix.subprocess.Popen",FinishedProcess)
    monkeypatch.setattr(psutil,"Process",gone)
    result=CalculixRunner(sys.executable).run(inp)
    assert result.completed and not result.errors


def test_native_numerical_evidence_when_available():
    root=Path(__file__).resolve().parents[2]
    record=root/"Dodec FEA/structural-fea-benchmark-runs/fixtures/release-backend-verification/fixture-summary.json"
    if not record.exists():
        pytest.skip("Native archived fixture runs not present; run python -m structural.fixtures")
    results=json.loads(record.read_text())
    assert all(row["accepted"] for row in results.values())
    assert results["preload_contact_open"]["metrics"]["service_contact_N"]==0
    assert results["contact"]["checks"]["gap_sign_verified"]
