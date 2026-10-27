"""Desktop setup and diagnostic viewer for the separate structural workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from PySide6.QtCore import QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QTabWidget, QSplitter, QFileDialog, QDoubleSpinBox, QSpinBox,
    QPlainTextEdit, QLineEdit, QComboBox, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView)

from .project import ROOT, DODEC, StructuralProject, write_json
from .workflow import initialize_project_from_evidence, new_run_directory, verify_evidence


class StructuralWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KurtsFEM · Structural")
        self.resize(1440,940)
        self.project_path = DODEC / "structural-fea-projects/dodec-35mm-baseline.json"
        self.project = StructuralProject.load(self.project_path) if self.project_path.exists() else initialize_project_from_evidence()
        self.mesh_path = None
        self.active_output = None
        self.active_command = None
        self.process = None
        self.cancelled = False
        self.viewer = None
        self.result_files = []
        self.result_is_accepted = False
        self.results_path = None
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        toolbar = QHBoxLayout()
        for label,callback in (("Open project",self.open_project),("Save",self.save_project),("Save as…",self.save_as),("Validation record",self.open_record)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        toolbar.addStretch()
        self.title = QLabel()
        toolbar.addWidget(self.title)
        layout.addLayout(toolbar)
        self.status = QLabel("Setup — no accepted assembly result")
        self.status.setStyleSheet("padding: 10px; background: #fff3cd; color: #49380b; font-weight: 600;")
        layout.addWidget(self.status)
        splitter = QSplitter()
        layout.addWidget(splitter,1)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.West)
        splitter.addWidget(self.tabs)
        self.fields = {}
        self.geometry_panel()
        self.material_panel()
        self.connections_panel()
        self.mesh_panel()
        self.solve_panel()
        self.results_panel()
        right = QWidget()
        right_layout = QVBoxLayout(right)
        controls = QHBoxLayout()
        for label,callback in (("Show CAD",self.show_cad),("Show mesh",self.show_mesh),("Fit",self.fit_view)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            controls.addWidget(button)
        self.selection = QComboBox()
        self.selection.addItem("All components")
        self.selection.currentTextChanged.connect(self.show_selection)
        controls.addWidget(self.selection)
        right_layout.addLayout(controls)
        try:
            from pyvistaqt import QtInteractor
            self.viewer = QtInteractor(right,off_screen=os.environ.get("QT_QPA_PLATFORM")=="offscreen")
            self.viewer.set_background("#f4f6f8")
            self.viewer.add_axes()
            right_layout.addWidget(self.viewer.interactor,1)
        except Exception as exc:
            right_layout.addWidget(QLabel(f"3D viewer unavailable: {exc}"))
        self.frame_label=QLabel()
        right_layout.addWidget(self.frame_label)
        splitter.addWidget(right)
        splitter.setSizes([510,900])
        self.fill_fields()
        self.refresh_readiness()
        QTimer.singleShot(200,self.show_cad)

    def panel(self,title):
        widget=QWidget()
        layout=QVBoxLayout(widget)
        self.tabs.addTab(widget,title)
        return layout

    def number(self,form,label,key,minimum,maximum,decimals=3):
        box=QDoubleSpinBox()
        box.setRange(minimum,maximum)
        box.setDecimals(decimals)
        box.setKeyboardTracking(False)
        self.fields[key]=box
        form.addRow(label,box)
        return box

    def geometry_panel(self):
        layout=self.panel("Geometry")
        self.geometry_summary=QPlainTextEdit()
        self.geometry_summary.setReadOnly(True)
        layout.addWidget(self.geometry_summary)
        note=QLabel("Original CAD stays unchanged. Derived geometry excludes only the two verified duplicate screws, preserves clearances, and registers the frame.")
        note.setWordWrap(True)
        layout.addWidget(note)

    def material_panel(self):
        layout=self.panel("Materials / Supports")
        form=QFormLayout()
        for name,title in (("pipe","Pipe"),("screw","Screws · 17-4 PH H1150")):
            self.number(form,title+" E (MPa)",name+"_e",1,1e7,1)
            self.number(form,title+" Poisson ratio",name+"_nu",-.99,.499,4)
        form.addRow("Hub",QLabel("Fixed rigid baseline; stress is not interpreted"))
        form.addRow("Midpoint",QLabel("Normal displacement only: Uz=0 in reference frame"))
        form.addRow("Pipe dimensions",QLabel("OD 38 · ID 35 · wall 1.5 · half-length 233.68 mm"))
        form.addRow("Free end",QLabel("Gasket omitted; no extra support"))
        layout.addLayout(form)
        note=QLabel("Elastic material inputs are recorded assumptions. Plasticity and strength certification are outside this implementation.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()

    def connections_panel(self):
        layout=self.panel("Connections / Preload")
        form=QFormLayout()
        for i,label in enumerate(("Right screw","Lower-left screw","Upper-left screw"),1):
            self.number(form,label+" preload (N)",f"preload_{i}",0,100000,2)
        self.number(form,"Service force (N)","force",0,1e6,4)
        self.direction=QLineEdit()
        form.addRow("Direction x, y, z (normalized on save)",self.direction)
        self.patch=QComboBox()
        self.patch.addItems(["35 mm projected · baseline","30 mm projected · reference diagnostic"])
        form.addRow("Patch on −Y, 25 mm axial",self.patch)
        self.number(form,"Coulomb friction coefficient","friction",0,2,3)
        self.number(form,"Normal contact stiffness (MPa/mm)","stiffness",1,1e10,1)
        form.addRow("Anchorage",QLabel("Smooth cylinder tied to matching hub hole; no threads"))
        form.addRow("Opening interfaces",QLabel("Three cone / pipe seats and pipe bore / snout"))
        form.addRow("Stage A",QLabel("Establish each whole-screw preload; patch force off"))
        form.addRow("Stage B",QLabel("Lock tightening adjustment; ramp patch force"))
        layout.addLayout(form)
        note=QLabel("Screw forces remain free to redistribute in Stage B. Contact bodies retain separate nodes. The 35 mm baseline differs from the 30 mm reference partition.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()

    def mesh_panel(self):
        layout=self.panel("Mesh")
        form=QFormLayout()
        self.number(form,"Far-field target (mm)","far",.1,20,3)
        self.number(form,"Thickness target (mm)","thickness",.1,3,3)
        self.number(form,"Screw seat target (mm)","contact",.05,5,3)
        self.number(form,"Bore / snout target (mm)","bore",0,20,3).setSpecialValueText("Far-field sizing")
        self.number(form,"Slot perimeter target (mm)","slot",0,20,3).setSpecialValueText("Far-field sizing")
        self.node_limit=QSpinBox();self.node_limit.setRange(1000,5000000);self.node_limit.setSingleStep(10000)
        form.addRow("Pilot node budget",self.node_limit)
        form.addRow("Element family",QLabel("Quadratic C3D10 solids"))
        layout.addLayout(form)
        for label,callback in (("Generate derived mesh",self.generate_mesh),("Open saved mesh…",self.open_mesh)):
            button=QPushButton(label);button.clicked.connect(callback);layout.addWidget(button)
        self.mesh_summary=QPlainTextEdit();self.mesh_summary.setReadOnly(True)
        layout.addWidget(self.mesh_summary)
        self.mesh_summary.setPlainText("No mesh loaded. A pilot mesh is not a mesh-convergence result.")
        self.study_kind=QComboBox()
        for label,kind in (("Three wall meshes","mesh"),("Screw-seat refinement","contact_mesh"),("Bore / snout refinement","bore_contact_mesh"),("Slot refinement","slot_mesh"),("Contact stiffness","contact"),("35 / 30 mm patch","patch"),
                           ("Friction sensitivity","friction"),("Service load sensitivity","load"),("Preload sensitivity","preload")):
            self.study_kind.addItem(label,kind)
        layout.addWidget(self.study_kind)
        button=QPushButton("Run controlled study");button.clicked.connect(self.run_study);layout.addWidget(button)
        note=QLabel("Mesh and patch studies generate new meshes. Contact, friction, load and preload studies reuse the loaded mesh. Each case retains its own diagnostics.")
        note.setWordWrap(True);layout.addWidget(note)

    def solve_panel(self):
        layout=self.panel("Solve")
        self.readiness=QPlainTextEdit();self.readiness.setReadOnly(True)
        layout.addWidget(self.readiness,1)
        buttons=QHBoxLayout()
        self.solve_button=QPushButton("Run preload + service")
        self.solve_button.clicked.connect(self.solve)
        buttons.addWidget(self.solve_button)
        self.cancel_button=QPushButton("Cancel run")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_run)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log,2)

    def results_panel(self):
        layout=self.panel("Results / Compare")
        button=QPushButton("Open run results…");button.clicked.connect(self.open_results);layout.addWidget(button)
        self.compare_button=QPushButton("Compare loaded run with COMSOL")
        self.compare_button.clicked.connect(self.compare_results);self.compare_button.setEnabled(False);layout.addWidget(self.compare_button)
        button=QPushButton("Open results folder");button.clicked.connect(self.open_results_folder);layout.addWidget(button)
        button=QPushButton("Open comparison / study report…");button.clicked.connect(self.open_analysis_report);layout.addWidget(button)
        self.result_state=QComboBox();self.result_state.currentIndexChanged.connect(self.render_result);layout.addWidget(self.result_state)
        self.result_field=QComboBox()
        for label,kind in (("Displacement magnitude","displacement"),("von Mises display stress","mises"),
                           ("Maximum principal display stress","principal_max"),("Minimum principal display stress","principal_min"),
                           ("Ux","ux"),("Uy","uy"),("Uz","uz"),
                           ("Sxx","sxx"),("Syy","syy"),("Szz","szz"),("Sxy","sxy"),("Sxz","sxz"),("Syz","syz")):
            self.result_field.addItem(label,kind)
        self.result_field.currentIndexChanged.connect(self.render_result);layout.addWidget(self.result_field)
        self.gate_table=QTableWidget(0,3);self.gate_table.setHorizontalHeaderLabels(["Check","Status","Value / limit"])
        self.gate_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.gate_table)
        self.result_summary=QPlainTextEdit();self.result_summary.setReadOnly(True);layout.addWidget(self.result_summary,1)
        reference=DODEC/"structural-fea-validation-deliverables/comsol-reference-manifest.json"
        if reference.exists():
            self.result_summary.setPlainText("COMSOL reference available. No candidate assembly result loaded.\n\n"+reference.read_text(encoding="utf-8")[:6000])

    def fill_fields(self):
        p=self.project
        values={"pipe_e":p.pipe.youngs_modulus_mpa,"pipe_nu":p.pipe.poisson_ratio,
                "screw_e":p.screw.youngs_modulus_mpa,"screw_nu":p.screw.poisson_ratio,
                "force":p.loads.service_force_n,"friction":p.contacts.friction_coefficient,
                "stiffness":p.contacts.normal_stiffness_mpa_per_mm,"far":p.mesh.global_size_mm,
                "thickness":p.mesh.thickness_size_mm,"contact":p.mesh.contact_size_mm,"bore":p.mesh.bore_snout_size_mm or 0.,
                "slot":p.mesh.slot_size_mm or 0.}
        values.update({f"preload_{i}":v for i,v in enumerate(p.loads.screw_preloads_n,1)})
        for key,value in values.items():self.fields[key].setValue(value)
        self.patch.setCurrentIndex(1 if p.loads.patch_width_mm==30 else 0)
        self.direction.setText(", ".join(format(value,".17g") for value in p.loads.direction_reference))
        self.frame_label.setText(f"Reference frame: pipe toward hub −Z · load direction {p.loads.direction_reference} · midpoint Z=0. Dimensions in mm.")
        self.node_limit.setValue(p.mesh.max_nodes)
        self.title.setText(p.name)
        self.geometry_summary.setPlainText(f"Source CAD\n{p.source_path}\n\nOne pipe · one fixed hub · three complete screws\n\nPipe dimensions\n38 mm OD · 35 mm ID · 1.5 mm wall\n233.68 mm half length · 467.36 mm full length\n\nRegistered frame\nPipe extends toward hub along −Z\nMidpoint at Z=0, normal displacement constrained\nLoad acts +Y on the −Y side\n\nPhysical contact geometry\n0.45 mm nominal radial bore-to-snout clearance\nSmooth screw cylinders tied to hub holes\nCone seats and bore retain separating contact\n\nSource identity\nSHA-256: {p.source_sha256}\n\nRegistration status: {p.registration.get('status','unverified')}\nComparison: {p.registration.get('comparison_classification','unverified')}\n\nUnits: N / mm / MPa")
        self.geometry_summary.setPlainText(self.geometry_summary.toPlainText().replace("Load acts +Y on the −Y side",f"Load direction {p.loads.direction_reference} on the −Y side"))

    def apply_fields(self):
        p=self.project
        p.pipe.youngs_modulus_mpa=self.fields["pipe_e"].value();p.pipe.poisson_ratio=self.fields["pipe_nu"].value()
        p.screw.youngs_modulus_mpa=self.fields["screw_e"].value();p.screw.poisson_ratio=self.fields["screw_nu"].value()
        p.loads.screw_preloads_n=[self.fields[f"preload_{i}"].value() for i in range(1,4)]
        p.loads.service_force_n=self.fields["force"].value();p.loads.patch_width_mm=30. if self.patch.currentIndex() else 35.
        import math
        direction=[float(value.strip()) for value in self.direction.text().split(",")]
        if len(direction)!=3 or not all(math.isfinite(value) for value in direction):raise ValueError("Enter three finite direction components separated by commas")
        length=math.sqrt(sum(value*value for value in direction))
        if length==0:raise ValueError("Load direction cannot be zero")
        p.loads.direction_reference=[value/length for value in direction]
        self.direction.setText(", ".join(format(value,".17g") for value in p.loads.direction_reference))
        p.contacts.friction_coefficient=self.fields["friction"].value();p.contacts.normal_stiffness_mpa_per_mm=self.fields["stiffness"].value()
        p.mesh.global_size_mm=self.fields["far"].value();p.mesh.thickness_size_mm=self.fields["thickness"].value();p.mesh.contact_size_mm=self.fields["contact"].value()
        p.mesh.bore_snout_size_mm=self.fields["bore"].value() or None
        p.mesh.slot_size_mm=self.fields["slot"].value() or None
        p.mesh.max_nodes=self.node_limit.value()
        p.registration["comparison_classification"]=("derived_geometry_diagnostic" if p.geometry_revision else
            "matched_partition_diagnostic" if p.loads.patch_width_mm==30 else "revised_model")
        p.validate()

    def message(self,text):
        QMessageBox.warning(self,"Structural workflow",str(text))

    def open_project(self):
        if self.process and self.process.state()!=QProcess.ProcessState.NotRunning:
            self.message("Finish or cancel the current task before opening another project.");return
        filename,_=QFileDialog.getOpenFileName(self,"Open structural project",str(DODEC/"structural-fea-projects"),"Project JSON (*.json)")
        if not filename:return
        try:
            self.project=StructuralProject.load(filename,verify_source=True);self.project_path=Path(filename)
            self.clear_project_views();self.fill_fields();self.refresh_readiness();self.show_cad()
        except Exception as exc:self.message(exc)

    def clear_project_views(self):
        self.mesh_path=None
        if hasattr(self,"mesh"):del self.mesh
        self.selection.blockSignals(True);self.selection.clear();self.selection.addItem("All components");self.selection.blockSignals(False)
        self.mesh_summary.setPlainText("No mesh loaded for this project.")
        self.results_path=None;self.result_files=[];self.result_is_accepted=False
        self.compare_button.setEnabled(False)
        self.result_state.blockSignals(True);self.result_state.clear();self.result_state.blockSignals(False)
        self.gate_table.setRowCount(0);self.result_summary.clear()
        self.status.setText("Setup · no accepted result loaded for this project")
        self.status.setStyleSheet("padding: 10px; background: #fff3cd; color: #49380b; font-weight: 600;")
        if self.viewer:self.viewer.clear()

    def save_project(self):
        try:
            self.apply_fields();self.project.save(self.project_path);self.refresh_readiness()
            self.statusBar().showMessage(f"Saved {self.project_path}",5000)
        except Exception as exc:self.message(exc)

    def save_as(self):
        filename,_=QFileDialog.getSaveFileName(self,"Save structural project",str(self.project_path),"Project JSON (*.json)")
        if filename:self.project_path=Path(filename);self.save_project()

    def open_record(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(DODEC/"structural-fea-validation-deliverables/implementation-status.md")))

    def refresh_readiness(self):
        issues=verify_evidence(self.project)+self.project.readiness_issues()
        if not self.mesh_path:
            issues.append("No prepared mesh is loaded for this project")
        elif self.mesh.provenance.get("patch_width_mm")!=self.project.loads.patch_width_mm:
            issues.append("Loaded mesh patch width differs from the project; load or generate its matching mesh")
        text=("Physical preparation still required:\n\n"+"\n".join(issues) if issues else
              "Preparation checks passed. Solver completion and numerical acceptance will be evaluated separately.")
        if self.project.geometry_revision:
            text+="\n\nDerived geometry: seat trims follow the exact original cone/cylinder intersections. The supplied STEP and earlier meshes are retained."
        self.readiness.setPlainText(text)

    def show_cad(self):
        if not self.viewer:return
        manifest=DODEC/"structural-fea-model-preparation/registered-geometry/registered-geometry-map.json"
        if not manifest.exists():return
        try:
            from .display import cad_surface_meshes
            self.viewer.clear()
            colors={"New Pipe":"#c0cbd4","Hub":"#677c8a","Bolt 1":"#bb7e46","Bolt 2":"#bb7e46","Bolt 3":"#bb7e46"}
            for name,mesh in cad_surface_meshes(manifest).items():
                self.viewer.add_mesh(mesh,color=colors[name],name=name,smooth_shading=True)
            if self.project.geometry_revision:
                self.viewer.add_text("Source CAD view\nDerived seat-trim geometry is recorded with the prepared mesh",font_size=10)
            self.add_load_guides();self.viewer.camera_position=[(-300,-400,80),(0,0,-120),(0,0,1)];self.viewer.reset_camera()
        except Exception as exc:self.statusBar().showMessage(f"CAD view: {exc}")

    def add_load_guides(self):
        import numpy as np
        import pyvista as pv
        half=np.arcsin(self.project.loads.patch_width_mm/38)
        theta,z=np.meshgrid(np.linspace(-np.pi/2-half,-np.pi/2+half,25),np.linspace(-25,0,3),indexing="ij")
        patch=pv.StructuredGrid(19.025*np.cos(theta),19.025*np.sin(theta),z)
        self.viewer.add_mesh(patch,color="#ed9c38",opacity=.8,name="load footprint")
        direction=np.asarray(self.project.loads.direction_reference)
        self.viewer.add_arrows(np.array([[0.,-19.,-12.5]])-18*direction,np.array([18*direction]),color="#ca542c")
        self.viewer.add_arrows(np.array([[0.,0.,0.]]),np.array([[0.,0.,15.]]),color="#246bb3")
        self.viewer.add_axes()

    def fit_view(self):
        if self.viewer:self.viewer.reset_camera()

    def open_mesh(self):
        filename,_=QFileDialog.getOpenFileName(self,"Open structural mesh",str(DODEC/"structural-fea-model-preparation"),"Structural mesh (*.npz)")
        if filename:self.load_mesh(Path(filename))

    def load_mesh(self,path):
        from .mesh import SolidMesh
        try:
            self.apply_fields()
            candidate=SolidMesh.load(path)
            if candidate.provenance.get("patch_width_mm")!=self.project.loads.patch_width_mm:
                raise ValueError("Mesh patch width differs from the project. Select the matching project/patch first.")
            evidence=DODEC/"structural-fea-validation-deliverables/backend-feasibility.json"
            if not evidence.exists():raise ValueError("Verified backend evidence is required before loading an assembly mesh")
            from .assembly import bind_mesh_evidence
            bound_project=bind_mesh_evidence(self.project,candidate,backend_fixture_report=json.loads(evidence.read_text(encoding="utf-8")))
            self.mesh=candidate;self.mesh_path=path;self.project=bound_project
            self.mesh_summary.setPlainText(f"{path}\n\n{len(self.mesh.node_ids):,} nodes\n{len(self.mesh.element_ids):,} quadratic tetrahedra\n\n"+json.dumps(self.mesh.provenance,indent=2))
            self.selection.blockSignals(True);self.selection.clear();self.selection.addItem("All components");self.selection.addItems(sorted(self.mesh.surfaces));self.selection.blockSignals(False)
            self.fill_fields()
            self.refresh_readiness();self.show_mesh()
        except Exception as exc:self.refresh_readiness();self.message(exc)

    def show_mesh(self, *, opacity=1.):
        if not self.viewer or not self.mesh_path:return
        try:
            import pyvista as pv
            self.viewer.clear()
            grid=pv.UnstructuredGrid({24:self.mesh.connectivity_indices()},self.mesh.points_mm)
            self.viewer.add_mesh(grid,color="#bbcad4",show_edges=True,edge_color="#657886",opacity=opacity)
            self.add_load_guides();self.viewer.reset_camera()
        except Exception as exc:self.message(exc)

    def show_selection(self,name):
        if name=="All components":self.show_mesh();return
        if not self.viewer or not hasattr(self,"mesh") or name not in self.mesh.surfaces:return
        import numpy as np
        import pyvista as pv
        points=[];faces=[]
        for eid,face in self.mesh.surfaces[name]:
            q=self.mesh.face_points(eid,face);start=len(points);points.extend(q)
            for tri in ((0,3,5),(3,1,4),(5,4,2),(3,4,5)):faces.append([3,*[start+n for n in tri]])
        self.show_mesh(opacity=.16)
        selected=pv.PolyData(np.array(points),np.array(faces))
        self.viewer.add_mesh(selected,color="#f28b23",show_edges=False,name="selection")
        self.viewer.reset_camera(bounds=selected.bounds)
        self.viewer.camera.zoom(.7)

    def generate_mesh(self):
        self.start_task("mesh")

    def solve(self):
        if not self.mesh_path:self.message("Generate or open a prepared mesh first.");return
        self.start_task("solve")

    def run_study(self):
        kind=self.study_kind.currentData()
        if kind in ("contact","friction","load","preload") and not self.mesh_path:
            self.message("Open the mesh to reuse across this sensitivity study first.");return
        self.start_task("study")

    def compare_results(self):
        if self.results_path:self.start_task("compare")

    def open_analysis_report(self):
        filename,_=QFileDialog.getOpenFileName(self,"Open comparison or study report",str(DODEC),"Analysis JSON (*.json)")
        if filename:self.load_analysis_report(Path(filename))

    def load_analysis_report(self,path):
        try:
            report=json.loads(Path(path).read_text(encoding="utf-8"))
            if not ("screen_status" in report or "study" in report):
                raise ValueError("Select a comparison.json or study.json report.")
            self.result_summary.setPlainText(json.dumps(report,indent=2)[:100000])
            self.status.setText("Analysis report · "+str(report.get("screen_status",report.get("status","incomplete"))))
            self.status.setStyleSheet("padding: 10px; background: #e7edf3; color: #263b4c; font-weight: 600;")
            if self.viewer:
                self.viewer.clear();self.viewer.add_text("Analysis report loaded\nOpen a component result to view fields",font_size=11)
            self.tabs.setCurrentIndex(5)
        except Exception as exc:self.message(exc)

    def start_task(self,command):
        if self.process and self.process.state()!=QProcess.ProcessState.NotRunning:
            self.message("A structural task is already running.");return
        try:
            self.apply_fields()
            if self.project.geometry_revision and (command=="mesh" or
                    (command=="study" and self.study_kind.currentData() in ("mesh","contact_mesh","bore_contact_mesh","slot_mesh","patch"))):
                raise ValueError("This project uses verified derived seat-trim geometry. Open its prepared mesh to solve. For spatial or patch studies, use separately repaired meshes and the retained-run assessment recipe in the user guide.")
            output=new_run_directory("desktop-"+command,DODEC/("structural-fea-model-preparation" if command=="mesh" else "structural-fea-benchmark-runs"))
            request=output/"request.project.json";self.project.save(request)
            self.active_output=output;self.active_command=command;self.cancelled=False
            args=["-m","structural",command,str(self.results_path if command=="compare" else request)]
            if command=="solve":args += [str(self.mesh_path),"--cancel-file",str(output/"cancel.request")]
            task_output=output
            if command=="compare":task_output=output/"comsol-comparison"
            if command=="study":
                kind=self.study_kind.currentData();args += [kind]
                if kind in ("contact","friction","load","preload"):args += ["--mesh",str(self.mesh_path)]
                task_output=output/"controlled-study"
            self.active_task_output=task_output
            args += ["--output",str(task_output)]
            self.process=QProcess(self);self.process.setWorkingDirectory(str(ROOT))
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(lambda p=self.process:self.read_process(p))
            self.process.finished.connect(lambda code,exit_status,p=self.process:self.task_finished(code,exit_status,expected_process=p))
            self.process.errorOccurred.connect(lambda error,p=self.process:self.process_error(error,p))
            self.log.clear();self.log.appendPlainText("Output: "+str(output))
            self.status.setText("Running "+command+" · results are not yet accepted")
            self.cancel_button.setEnabled(True);self.solve_button.setEnabled(False)
            self.tabs.setCurrentIndex(4);self.process.start(sys.executable,args)
        except Exception as exc:self.message(exc)

    def read_process(self,expected_process=None):
        if expected_process is not None and expected_process is not self.process:return
        self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode("utf-8",errors="replace"))

    def process_error(self,error,process):
        if process is not self.process:return
        self.log.appendPlainText(process.errorString())
        if error==QProcess.ProcessError.FailedToStart:
            self.cancel_button.setEnabled(False);self.solve_button.setEnabled(True)
            self.status.setText("Task failed to start · no accepted result")
            if self.active_output:
                write_json(self.active_output/"task-failed-to-start.json",{"status":"failed_to_start","accepted":False,"error":process.errorString()})

    def cancel_run(self):
        if not self.process:return
        self.cancelled=True
        if self.active_output:
            (self.active_output/"cancel.request").write_text("User cancelled\n",encoding="utf-8")
            write_json(self.active_output/"cancelled.json",{"status":"cancel_requested","accepted":False,"process_id":int(self.process.processId())})
        if self.active_command!="solve":self.force_stop()
        else:QTimer.singleShot(10000,lambda process=self.process:self.force_stop(process))
        self.status.setText("Cancelling · any partial fields remain diagnostic only")

    def force_stop(self,expected_process=None):
        # A cancellation timer from a finished task must never stop a later task.
        if expected_process is not None and expected_process is not self.process:return False
        if self.process and self.process.state()!=QProcess.ProcessState.NotRunning:
            if os.name=="nt":
                stopped=subprocess.run(["taskkill","/PID",str(self.process.processId()),"/T","/F"],creationflags=subprocess.CREATE_NO_WINDOW,capture_output=True)
                if stopped.returncode and self.process.state()!=QProcess.ProcessState.NotRunning:
                    self.log.appendPlainText("Stop was not confirmed: "+stopped.stderr.decode(errors="replace"))
                    if self.active_output:write_json(self.active_output/"cancelled.json",{"status":"stop_unconfirmed","accepted":False,"taskkill_return_code":stopped.returncode})
                    self.status.setText("Cancellation requested · process stop is not yet confirmed")
                    return False
            else:self.process.kill()
            return self.process.waitForFinished(2000)
        return True

    def finalize_cancelled_task(self,code):
        """Only called after this owned QProcess has finished; keep raw solver evidence."""
        if not self.active_output:return
        write_json(self.active_output/"cancelled.json",{"status":"cancelled","accepted":False,"process_exit_code":code,"stop_confirmed":True})
        paths=[self.active_output/"manifest.json",self.active_output/"structural-results/results.json"]
        if self.active_command=="study":paths.append(self.active_task_output/"study.json")
        for path in paths:
            if not path.exists():continue
            record=json.loads(path.read_text(encoding="utf-8"))
            record["task_status"]="cancelled";record["accepted"]=False
            if path.name=="results.json":
                record["diagnostic_only"]=True
                record.setdefault("assessment",{})["status"]="unaccepted"
                record["assessment"].setdefault("gates",[]).append({"name":"task_not_cancelled","status":"failed","measured":False,"limit":True,"detail":"Desktop task cancelled; raw solver status and fields retained separately"})
            else:
                record["status"]="cancelled";record["numerical_acceptance"]="unaccepted";record["numerical_convergence_accepted"]=False
            write_json(path,record)

    def task_finished(self,code,*_,expected_process=None):
        if expected_process is not None and expected_process is not self.process:return
        self.read_process();self.cancel_button.setEnabled(False);self.solve_button.setEnabled(True)
        if self.cancelled:self.finalize_cancelled_task(code)
        state="Cancelled" if self.cancelled else "Completed" if code==0 else "Failed / incomplete"
        self.status.setText(state+" · numerical acceptance is shown separately")
        if self.active_command=="mesh" and code==0 and not self.cancelled:
            self.load_mesh(self.active_output/"structural-mesh.npz")
        if self.active_command=="solve" and not self.cancelled and (self.active_output/"structural-results/results.json").exists():
            self.load_results(self.active_output/"structural-results/results.json")
        report_name={"compare":"comparison.json","study":"study.json"}.get(self.active_command)
        if report_name and not self.cancelled and (self.active_task_output/report_name).exists():
            self.load_analysis_report(self.active_task_output/report_name)

    def open_results(self):
        filename,_=QFileDialog.getOpenFileName(self,"Open result diagnostics",str(DODEC/"structural-fea-benchmark-runs"),"Result JSON (*.json)")
        if filename:self.load_results(Path(filename))

    def open_results_folder(self):
        if self.results_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.results_path.parent)))

    def load_results(self,path):
        try:
            from .results import report_is_accepted, retained_result_source_binding, retained_component_field_binding
            path=Path(path)
            report=json.loads(path.read_text(encoding="utf-8"))
            self.results_path=path
            self.compare_button.setEnabled(True)
            assessment=report.get("assessment",report.get("acceptance",{}))
            gates=assessment.get("gates",[])
            self.gate_table.setRowCount(len(gates))
            for row,gate in enumerate(gates):
                for column,value in enumerate((gate["name"],gate["status"],f"{gate.get('measured')} / {gate.get('limit')}")):
                    self.gate_table.setItem(row,column,QTableWidgetItem(str(value)))
            source_binding=retained_result_source_binding(report,path)
            field_binding=retained_component_field_binding(report,path)
            self.result_summary.setPlainText(json.dumps({"current_source_binding":source_binding,
                "current_field_binding":field_binding,"retained_report":report},indent=2)[:60000])
            self.result_is_accepted=report_is_accepted(report) and source_binding["verified"] and field_binding["verified"]
            status="numerically accepted · mesh/contact study remains separate" if self.result_is_accepted else "unaccepted · diagnostic fields only"
            solver_status=report.get("solver_status","unknown solver status")
            source_run=Path(report["source_run"]) if report.get("source_run") else None
            if source_run is not None and not source_run.is_absolute():source_run=path.parent/source_run
            run_name=source_run.parent.name if source_run is not None else path.parent.parent.name
            self.status.setText(f"Run {run_name} · {solver_status} · {status} · design adequacy not assessed")
            self.status.setStyleSheet("padding: 10px; font-weight: 600; background: "+("#dff0e1; color: #21452a;" if self.result_is_accepted else "#fff3cd; color: #49380b;"))
            # Only display files explicitly linked to this report, not unrelated nearby VTUs.
            self.result_files=[]
            for stage in report.get("stages",{}).values():
                for component in stage.get("components",{}).values():
                    if component.get("status")=="available" and component.get("vtu"):
                        candidate=Path(component["vtu"])
                        if not candidate.is_absolute():candidate=path.parent/candidate
                        if candidate.is_file():self.result_files.append(candidate)
            self.result_state.blockSignals(True);self.result_state.clear()
            self.result_state.addItems([f.stem for f in self.result_files]);self.result_state.blockSignals(False)
            if self.viewer:
                self.viewer.clear()
                if not self.result_files:self.viewer.add_text("No final component fields available · diagnostic only",font_size=11)
            self.render_result()
            self.tabs.setCurrentIndex(5)
        except Exception as exc:self.message(exc)

    def render_result(self,*_):
        index=self.result_state.currentIndex()
        if not self.viewer or index<0 or index>=len(self.result_files):return
        try:
            import pyvista as pv
            from .results import result_display_field
            grid=pv.read(self.result_files[index]);self.viewer.clear()
            requested=self.result_field.currentData()
            field=result_display_field(grid.point_data,requested)
            if field["name"]:
                grid.point_data[field["name"]]=field["values"]
                self.viewer.add_mesh(grid,scalars=field["name"],cmap="viridis",show_edges=False,scalar_bar_args={"title":field["label"]})
            else:self.viewer.add_mesh(grid,color="#9aa9b4",show_edges=False)
            annotation="Numerical checks passed; design adequacy not assessed" if self.result_is_accepted else "UNACCEPTED · DIAGNOSTIC ONLY"
            if field["note"]:annotation+="\n"+field["note"]
            self.viewer.add_text(annotation,font_size=10)
            self.viewer.add_axes();self.viewer.reset_camera()
        except Exception as exc:self.message(exc)

    def closeEvent(self,event):
        if self.process and self.process.state()!=QProcess.ProcessState.NotRunning:
            answer=QMessageBox.question(self,"Task running","Cancel the active structural task and close?")
            if answer!=QMessageBox.StandardButton.Yes:event.ignore();return
            self.cancel_run()
            if not self.force_stop():event.ignore();return
        if self.viewer:self.viewer.close()
        super().closeEvent(event)
