"""Step 6 — solve for the extrinsic, and show what the answer rests on.

Runs detection on every captured scene, fits one transform to all of them, and
then says how much each scene mattered. The original tool takes whatever the
last three entries in a log file happen to be; here scenes are ticked on and off
and the effect is visible immediately.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gui.core.bag_reader import BagSource, accumulate_cloud
from gui.core.decode import image_to_bgr
from gui.core.detect_camera import detect as detect_camera
from gui.core.detect_lidar import (
    DetectParams, apply_box, detect as detect_lidar, point_spacing,
)
from gui.core.evaluation.calibration_correspondence import (
    SceneMetadata,
    export_correspondence_diagnostics,
)
from gui.core.evaluation.correspondence_symmetry import (
    analyze_calibration_symmetry,
    decision_to_dict,
    export_candidate_selection_evidence,
)
from gui.core.project import Project, RESULT_DIR
from gui.core.solve import Solution, coverage, leave_one_out, solve
from gui.ui.steps import StepPage
from gui.ui.steps.step_filter import derive_params


class _Worker(QtCore.QObject):
    progress = QtCore.Signal(str)
    scene_done = QtCore.Signal(str, object, object, object, str)  # id, lidar pts, cam pts, metadata, note
    finished = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._src: BagSource | None = None

    def run(self, jobs, lidar_topic: str, camera_topic: str, camera, target,
            method: str = DetectParams.method):
        """`jobs` is (scene, bag path) -- scenes may come from different bags."""
        for scene, bag in jobs:
            self.progress.emit(f"{scene.id} 처리 중…")
            try:
                if self._src is None or str(self._src.path) != bag:
                    if self._src is not None:
                        self._src.close()
                    self._src = BagSource(bag)
                    self._src.open()
                camera_stamp_ns, img_msg = self._src.first_after(camera_topic, scene.t_ns)
                cam_det = detect_camera(image_to_bgr(img_msg), camera, target) if img_msg else None
                if cam_det is None or not cam_det.ok:
                    self.scene_done.emit(scene.id, None, None, None, f"Camera: {cam_det.reason if cam_det else 'no image'}")
                    continue

                cloud, _, single, _, ring = accumulate_cloud(
                    self._src, lidar_topic, scene.t_ns, scene.frames
                )
                inside = apply_box(single, scene.filter)
                params = derive_params(point_spacing(inside) if len(inside) > 20 else 0.0)
                params.method = method  # keep step 5 and step 6 on the same detector
                lid_det = detect_lidar(cloud, scene.filter, target, params, ring=ring)
                if not lid_det.ok:
                    self.scene_done.emit(scene.id, None, None, None, f"LiDAR: {lid_det.reason}")
                    continue

                metadata = SceneMetadata(
                    scene_index=scene.id,
                    camera_timestamp_ns=int(camera_stamp_ns) if camera_stamp_ns is not None else None,
                    lidar_timestamp_ns=int(scene.t_ns),
                    camera_header_stamp_ns=_header_stamp_ns(img_msg),
                    lidar_header_stamp_ns=int(scene.t_ns),
                    camera_physical_ids=list(getattr(cam_det, "center_ids", []) or []),
                    camera_marker_ids=list(getattr(cam_det, "center_marker_ids", []) or []),
                    camera_board_xy=[list(xy) for xy in (getattr(cam_det, "center_board_xy", []) or [])],
                )
                self.scene_done.emit(scene.id, lid_det.centers, cam_det.centers, metadata, "")
            except Exception as exc:  # noqa: BLE001 - reported per scene
                self.scene_done.emit(scene.id, None, None, None, f"{type(exc).__name__}: {exc}")
        self.finished.emit()


def _header_stamp_ns(msg) -> int | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", getattr(stamp, "secs", 0)) or 0)
    nsec = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)) or 0)
    return sec * 1_000_000_000 + nsec


class CalibrateStep(StepPage):
    title = "6. 캘리브레이션"
    subtitle = "외부 파라미터 계산"

    request_run = QtCore.Signal(object, str, str, object, object, str)

    def __init__(self, project: Project, parent=None):
        super().__init__(project, parent)
        self.detections: dict[str, tuple] = {}  # id -> (lidar centres, camera centres)
        self.scene_metadata: dict[str, SceneMetadata] = {}
        self.notes: dict[str, str] = {}
        self.solution: Solution | None = None
        self.correspondence_diagnostics: dict[str, object] = {}
        self.symmetry_gate: dict[str, object] = {}
        # leave_one_out() output for `solution`, kept so step 7's diagnostics
        # can read it without re-running the solver.
        self.loo: dict = {}

        self.run_btn = QtWidgets.QPushButton("모든 scene 검출 후 계산")
        self.run_btn.clicked.connect(self._run)
        self.status = QtWidgets.QLabel("—")
        self.status.setStyleSheet("color: palette(mid);")

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["사용", "scene", "RMSE", "제외 시 변화", "비고"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_toggle)

        self.result = QtWidgets.QLabel("—")
        self.result.setTextFormat(QtCore.Qt.RichText)
        self.result.setAlignment(QtCore.Qt.AlignTop)
        self.result.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.coverage_label = QtWidgets.QLabel()
        self.coverage_label.setTextFormat(QtCore.Qt.RichText)
        self.coverage_label.setWordWrap(True)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.run_btn)
        top.addWidget(self.status, 1)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("결과"))
        right.addWidget(self.result)
        right.addSpacing(8)
        right.addWidget(QtWidgets.QLabel("커버리지"))
        right.addWidget(self.coverage_label)
        right.addStretch(1)
        right_box = QtWidgets.QWidget()
        right_box.setLayout(right)
        right_box.setFixedWidth(360)

        middle = QtWidgets.QHBoxLayout()
        middle.addWidget(self.table, 1)
        middle.addWidget(right_box)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(middle, 1)

        self._thread = QtCore.QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._worker.progress.connect(self.status.setText)
        self._worker.scene_done.connect(self._on_scene)
        self._worker.finished.connect(self._on_finished)
        self.request_run.connect(self._worker.run)
        self._thread.start()

    # -------------------------------------------------------------------- run

    def on_enter(self):
        p = self.project
        # Drop results for scenes that no longer exist, so a deleted scene cannot
        # keep contributing to the fit.
        live = {s.id for s in p.scenes}
        if not live.issuperset(self.detections):
            for sid in set(self.detections) - live:
                self.detections.pop(sid, None)
                self.notes.pop(sid, None)
            self._solve()
        # Naming what is missing rather than "finish steps 1-5": five separate
        # things gate this button, and a greyed-out button with no reason is a
        # dead end.
        missing = []
        if not p.bag_paths:
            missing.append("1단계 bag")
        if not p.lidar_topic:
            missing.append("1단계 LiDAR 토픽")
        if not p.camera_topic:
            missing.append("1단계 카메라 토픽")
        if not p.camera.is_set:
            missing.append("2단계 카메라 내부 파라미터")
        if not p.scenes:
            missing.append("4단계 scene 캡처")
        ready = not missing
        self.run_btn.setEnabled(ready)
        if missing:
            self.status.setText("아직 없습니다 — " + ", ".join(missing))
        elif not self.detections:
            self.status.setText(f"scene {len(p.scenes)}개 준비됨.")

    def _run(self):
        self.detections.clear()
        self.scene_metadata.clear()
        self.notes.clear()
        self.solution = None
        self.correspondence_diagnostics = {}
        self.symmetry_gate = {}
        self.table.setRowCount(0)
        self.run_btn.setEnabled(False)
        p = self.project
        # Step 5 owns the choice; fall back to the declared default, not a copy
        # of it, so the two steps cannot drift apart.
        method = DetectParams.method
        window = self.window()
        for page in getattr(window, "pages", []):
            combo = getattr(page, "method_combo", None)
            if combo is not None:
                method = combo.currentData()
                break
        self.request_run.emit(
            [(s, p.bag_for(s)) for s in p.scenes], p.lidar_topic, p.camera_topic,
            p.camera, p.target, method,
        )

    def _on_scene(self, sid: str, lidar_pts, cam_pts, metadata, note: str):
        if lidar_pts is not None:
            self.detections[sid] = (lidar_pts, cam_pts)
        if metadata is not None:
            self.scene_metadata[sid] = metadata
        self.notes[sid] = note

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.status.setText(f"검출 성공 {len(self.detections)} / {len(self.project.scenes)} scene")
        self._solve()

    # ------------------------------------------------------------------ solve

    def _enabled_scenes(self):
        return [
            (s.id, *self.detections[s.id])
            for s in self.project.scenes
            if s.enabled and s.id in self.detections
        ]

    def _solve(self):
        scenes = self._enabled_scenes()
        self.solution = solve(scenes) if scenes else None
        loo = leave_one_out(scenes) if len(scenes) >= 3 else {}
        self.loo = loo
        self._export_correspondences(scenes)
        self._evaluate_symmetry_gate(scenes)
        self._fill_table(loo)
        self._show_result(scenes)
        self.changed.emit()

    def _export_correspondences(self, scenes):
        self.correspondence_diagnostics = {}
        sol = self.solution
        if sol is None or not sol.ok or sol.R is None:
            return
        try:
            scene_diags, summary, out_dir = export_correspondence_diagnostics(
                scenes,
                sol.R,
                sol.t,
                metadata=self.scene_metadata,
            )
            self.correspondence_diagnostics = {
                "scenes": {d.scene_index: d for d in scene_diags},
                "summary": summary,
                "out_dir": out_dir,
            }
        except Exception as exc:  # noqa: BLE001 - diagnostic must not block calibration
            self.correspondence_diagnostics = {"error": str(exc)}

    def _evaluate_symmetry_gate(self, scenes):
        self.symmetry_gate = {}
        sol = self.solution
        if sol is None or not sol.ok or sol.R is None or not scenes:
            return
        try:
            candidates, decision = analyze_calibration_symmetry(
                scenes,
                seed_R=None,
                seed_t=None,
                include_all_permutations=False,
            )
            out_dir = export_candidate_selection_evidence(
                candidates,
                decision,
                RESULT_DIR / "candidate_selection_evidence",
                legacy_solution={
                    "R_legacy": sol.R.tolist(),
                    "t_legacy": [float(x) for x in sol.t],
                    "rmse_mm": float(sol.rmse * 1000.0),
                    "n_pairs": int(sol.n_pairs),
                },
            )
            self.symmetry_gate = {
                "candidates": candidates,
                "decision": decision,
                "decision_dict": decision_to_dict(decision),
                "out_dir": out_dir,
            }
        except Exception as exc:  # noqa: BLE001 - gate evidence must not block diagnostics
            self.symmetry_gate = {"error": str(exc)}

    def _fill_table(self, loo: dict):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.project.scenes))
        for r, scene in enumerate(self.project.scenes):
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            usable = scene.id in self.detections
            check.setCheckState(
                QtCore.Qt.Checked if (scene.enabled and usable) else QtCore.Qt.Unchecked
            )
            if not usable:
                check.setFlags(QtCore.Qt.NoItemFlags)
            self.table.setItem(r, 0, check)
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(scene.id))

            diag = (self.correspondence_diagnostics.get("scenes", {}) or {}).get(scene.id)
            rmse = (diag.fit_stats_mm["rmse"] / 1000.0) if diag else (
                self.solution.per_scene_rmse.get(scene.id) if self.solution else None
            )
            self.table.setItem(
                r, 2, QtWidgets.QTableWidgetItem(f"{rmse * 1000:.1f} mm" if rmse else "—")
            )

            m = loo.get(scene.id)
            if diag:
                text = (
                    f"LOO max {diag.loo_stats_mm['max']:.1f} mm / "
                    f"dR {diag.delta_rotation_vs_global_deg:.2f} deg / "
                    f"dt {diag.delta_translation_vs_global_mm:.0f} mm"
                )
                item = QtWidgets.QTableWidgetItem(text)
                if diag.status != "OK":
                    item.setForeground(QtGui.QColor("#d9534f"))
                self.table.setItem(r, 3, item)
                note = f"{diag.status}; cond={diag.geometry['condition_indicator']:.2f}; {self.notes.get(scene.id, '')}"
                self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(note))
                continue
            if m:
                text = f"{m['shift_mm']:.0f} mm / {m['rotation_deg']:.2f}°"
                item = QtWidgets.QTableWidgetItem(text)
                if m["shift_mm"] > 30 or m["rotation_deg"] > 0.5:
                    item.setForeground(QtGui.QColor("#d9534f"))
            else:
                item = QtWidgets.QTableWidgetItem("—")
            self.table.setItem(r, 3, item)
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(self.notes.get(scene.id, "")))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.table.blockSignals(False)

    def _on_toggle(self, item: QtWidgets.QTableWidgetItem):
        if item.column() != 0:
            return
        row = item.row()
        if 0 <= row < len(self.project.scenes):
            self.project.scenes[row].enabled = item.checkState() == QtCore.Qt.Checked
            self._solve()

    def _show_result(self, scenes):
        sol = self.solution
        if sol is None or not sol.ok:
            self.result.setText("사용 가능한 scene이 없습니다.")
            self.coverage_label.clear()
            return

        R, t = sol.R, sol.t
        rows = "".join(
            "<tr>" + "".join(f'<td align="right" style="padding:0 6px">{v:9.6f}</td>' for v in row) + "</tr>"
            for row in R
        )
        warn = ""
        if len(scenes) < 3:
            warn = ('<div style="color:#d9534f">scene이 3개 미만입니다 — 회전이 잘 결정되지 않고, '
                    "보드의 180° 대칭 때문에 방향이 뒤집힐 수 있습니다.</div>")
        symmetry = self._symmetry_html()
        self.result.setText(
            f"<b>RMSE {sol.rmse * 1000:.2f} mm</b>  ({sol.n_pairs}쌍, scene {len(scenes)}개)<br><br>"
            f"<b>Rcl</b><table>{rows}</table>"
            f"<b>Pcl</b> [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]<br>"
            f"|t| = {np.linalg.norm(t):.4f} m{warn}{symmetry}"
        )

        c = coverage(scenes)
        if not c:
            self.coverage_label.clear()
            return

        def bar(value, good):
            n = int(min(value / good, 1.0) * 10)
            colour = "#2e9e4f" if n >= 6 else "#d9534f"
            return f'<span style="color:{colour}">{"█" * n}{"░" * (10 - n)}</span>'

        self.coverage_label.setText(
            "보드가 움직인 범위 — 좁으면 회전이 노이즈에 휘둘립니다.<table>"
            f"<tr><td>가로</td><td>{bar(c['x_range'], 2.0)}</td><td>{c['x_range']:.2f} m</td></tr>"
            f"<tr><td>세로</td><td>{bar(c['y_range'], 2.0)}</td><td>{c['y_range']:.2f} m</td></tr>"
            f"<tr><td>높이</td><td>{bar(c['z_range'], 1.0)}</td><td>{c['z_range']:.2f} m</td></tr>"
            f"<tr><td>기울기</td><td>{bar(c['tilt_range'], 30)}</td>"
            f"<td>{c['tilt_min']:.0f}~{c['tilt_max']:.0f}°</td></tr></table>"
        )

    def _symmetry_html(self) -> str:
        if not self.symmetry_gate:
            return ""
        if "error" in self.symmetry_gate:
            return (
                "<br><br><b>Correspondence symmetry</b><br>"
                f"<span style='color:#d9534f'>diagnostic failed: {self.symmetry_gate['error']}</span>"
            )
        decision = self.symmetry_gate.get("decision")
        if decision is None:
            return ""
        colour = "#2e9e4f" if decision.status == "CALIBRATION_ACCEPTABLE" else "#d9534f"
        selected = decision.preferred_candidate_name or "None"
        out_dir = self.symmetry_gate.get("out_dir")
        return (
            "<br><br><b>Correspondence symmetry</b><br>"
            f"<span style='color:{colour}'>{decision.status}</span><br>"
            f"Selection source: {decision.selection_source}<br>"
            f"Selected candidate: {selected}<br>"
            f"Confidence: {decision.confidence}<br>"
            f"<span style='color:palette(mid)'>{decision.reason}</span><br>"
            f"<span style='color:palette(mid)'>Evidence: {out_dir}</span>"
        )

    # ------------------------------------------------------------------ state

    def shutdown(self):
        self._thread.quit()
        self._thread.wait(5000)

    def is_complete(self) -> bool:
        return self.solution is not None and self.solution.ok

    def status_text(self) -> str:
        if self.solution and self.solution.ok:
            return f"RMSE {self.solution.rmse * 1000:.1f} mm"
        return "미계산"
