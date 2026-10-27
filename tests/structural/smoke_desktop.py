"""Hidden native desktop smoke check with retained review screenshots."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
os.environ.setdefault("QT_API","pyside6")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from structural.gui import StructuralWindow
from structural.project import StructuralProject

app=QApplication([])
if not QFontDatabase.families():
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QFont('Segoe UI',10))
window=StructuralWindow()
window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen,True)
window.show()
app.processEvents()
window.show_cad()
app.processEvents()
output=ROOT/"Dodec FEA/structural-fea-validation-deliverables/desktop-smoke"
output.mkdir(parents=True,exist_ok=True)
window.project_path=output/"reopened-desktop-project.json"
window.fields["preload_2"].setValue(125.0)
window.fields["bore"].setValue(1.5)
window.fields["slot"].setValue(.75)
window.apply_fields()
window.project.save(window.project_path)
reopened=StructuralProject.load(window.project_path,verify_source=True)
assert reopened.loads.screw_preloads_n==[100.0,125.0,100.0]
assert reopened.mesh.bore_snout_size_mm==1.5
assert reopened.mesh.slot_size_mm==.75
assert reopened.fingerprint()==window.project.fingerprint()
window.direction.setText('1, 1, 0')
window.apply_fields()
assert abs(window.project.loads.direction_reference[0]-2**-.5)<1e-12
window.direction.setText('0, 1, 0')
window.fields["preload_2"].setValue(100.0)
window.fields["bore"].setValue(0.)
window.fields["slot"].setValue(0.)
window.apply_fields()
window.grab().save(str(output/"structural-desktop-setup.png"))
if window.viewer:
    window.viewer.render()
    window.viewer.screenshot(str(output/"registered-cad-view.png"))
assert window.tabs.count()==6
assert window.study_kind.findData("slot_mesh")>=0
window.message=lambda message: (_ for _ in ()).throw(AssertionError(str(message)))
prepared_mesh=ROOT/'Dodec FEA/structural-fea-model-preparation/assembly-study-35mm-2-radial-layers/structural-mesh.npz'
if prepared_mesh.exists():
    window.fields['preload_2'].setValue(125.0)
    window.load_mesh(prepared_mesh)
    app.processEvents()
    assert window.mesh_path==prepared_mesh
    assert window.project.loads.screw_preloads_n==[100.0,125.0,100.0]
    assert window.selection.findText('screw_1_anchor')>=0
    window.selection.setCurrentText('screw_1_anchor')
    app.processEvents()
    assert window.viewer is not None and 'selection' in window.viewer.actors
    assert window.readiness.toPlainText().startswith('Preparation checks passed.')
    window.tabs.setCurrentIndex(3)
    window.grab().save(str(output/'prepared-mesh-anchorage-selection.png'))
    window.fields['preload_2'].setValue(100.0)
    window.apply_fields()
failed_report=ROOT/'Dodec FEA/structural-fea-benchmark-runs/assembly-35mm-pilot-first/structural-results/results.json'
if failed_report.exists():
    window.load_results(failed_report)
    app.processEvents()
    assert not window.result_is_accepted and not window.result_files
    assert window.gate_table.rowCount()>0
    assert window.compare_button.isEnabled()
    window.grab().save(str(output/'failed-run-diagnostics.png'))
    window.clear_project_views()
    assert window.results_path is None and not window.result_files
    assert window.mesh_path is None and window.gate_table.rowCount()==0
    assert not window.compare_button.isEnabled()
print("PASS: six workflow tabs, CAD rendering, setup editing, exact save/reopen, prepared mesh and anchorage selection when available, failed-run diagnostics")
window.close()
app.processEvents()
