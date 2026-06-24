"""AI 안면인식 출퇴근 시스템 — 통합 GUI 애플리케이션 (PyQt5)."""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
)
from app import config as appcfg
from app import tts
from app.core import AttendanceDB
from app.face_engine import (
    FaceEngine,
    average_embeddings,
    cosine_best_match,
    open_camera_safe,
)

ACTION_KEYS = {
    "출근": "1",
    "외출": "2",
    "복귀": "3",
    "퇴근": "4",
    "퇴근갱신": "5",
}
ACTION_LABELS = {"퇴근갱신": "퇴근 시각 갱신"}

LIGHT_QSS = """
* { font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; font-size: 10pt; }
QMainWindow, QDialog { background: #f4f6fa; }
QTabWidget::pane { border: none; background: #f4f6fa; }
QTabBar::tab {
    background: transparent; color: #6b7280; padding: 9px 20px;
    border: none; border-bottom: 2px solid transparent; margin-right: 2px;
}
QTabBar::tab:selected { color: #2563eb; border-bottom: 2px solid #2563eb; }
QTabBar::tab:hover:!selected { color: #374151; }
QLabel { color: #1f2937; }
QLabel[muted="true"] { color: #9ca3af; font-size: 9pt; }
QLabel[hint="true"] {
    color: #1e40af; background: #eef4ff; padding: 10px 12px; border-radius: 8px;
}
QLabel[viewport="true"] {
    background: #0f1115; color: #6b7280; border-radius: 12px;
}
QPushButton {
    background: #ffffff; color: #374151; border: 1px solid #d8dde6;
    border-radius: 8px; padding: 8px 16px;
}
QPushButton:hover { background: #f0f4ff; border-color: #2563eb; color: #2563eb; }
QPushButton:pressed { background: #e2e9fb; }
QPushButton:disabled { color: #b4b9c2; background: #f3f4f6; border-color: #e5e7eb; }
QPushButton[accent="true"] {
    background: #2563eb; color: #ffffff; border: none; font-weight: 600;
}
QPushButton[accent="true"]:hover { background: #1d4ed8; }
QPushButton[accent="true"]:pressed { background: #1e40af; }
QLineEdit {
    background: #ffffff; border: 1px solid #d8dde6; border-radius: 8px;
    padding: 6px 10px; color: #1f2937; selection-background-color: #bfdbfe;
}
QLineEdit:focus { border: 1px solid #2563eb; }
QDoubleSpinBox, QSpinBox, QComboBox, QDateEdit {
    background: #ffffff; border: 1px solid #d8dde6; border-radius: 6px;
    padding: 4px 6px; color: #1f2937;
}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QDateEdit:focus {
    border: 1px solid #2563eb;
}
QComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #d8dde6;
    selection-background-color: #dbeafe; selection-color: #1e3a8a;
    outline: none;
}
QTableWidget {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
    gridline-color: #eef0f4; selection-background-color: #dbeafe;
    selection-color: #1e3a8a; alternate-background-color: #f8fafc;
}
QHeaderView::section {
    background: #f8fafc; color: #6b7280; padding: 8px; border: none;
    border-bottom: 1px solid #e5e7eb; font-weight: 600;
}
QHeaderView::section:hover { background: #eef4ff; color: #2563eb; }
QTableWidget::item { padding: 8px 6px; border-bottom: 1px solid #f1f3f6; }
QTableWidget::item:selected { background: #dbeafe; color: #1e3a8a; }
QStatusBar { background: #ffffff; color: #6b7280; border-top: 1px solid #e5e7eb; }
QCheckBox { color: #374151; }
QMessageBox { background: #ffffff; }
"""


def bgr_to_qpixmap(frame_bgr) -> QtGui.QPixmap:
    """OpenCV BGR 프레임 → QPixmap (미러링은 호출부에서 처리)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QtGui.QImage(
        rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888
    )
    return QtGui.QPixmap.fromImage(img.copy())


class CameraScanThread(QtCore.QThread):
    """앱 시작 시 백그라운드로 사용 가능한 카메라를 검색 (UI를 막지 않음)."""

    done = QtCore.pyqtSignal(list)

    def __init__(self, max_index: int = 5):
        super().__init__()
        self.max_index = max_index

    def run(self):
        try:
            from app.face_engine import list_cameras

            found = list_cameras(self.max_index)
        except Exception:
            found = []
        self.done.emit(found)


class ModelCheckThread(QtCore.QThread):
    """설정 화면의 '모델 확인/다운로드' 버튼용 — 백그라운드에서 모델 로드를 시도."""

    progress = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(
        bool, str
    )  # (성공 여부, 메시지)

    def __init__(self, engine: FaceEngine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            ok = self.engine.load(
                progress_cb=self.progress.emit
            )
        except Exception as e:
            ok = False
            self.engine._load_error = str(e)
        if ok:
            dev = (
                "GPU"
                if self.engine.device
                and self.engine.device.type == "cuda"
                else "CPU"
            )
            self.done.emit(
                True,
                f"모델이 정상적으로 준비되었습니다. ({dev})",
            )
        else:
            err = (
                self.engine.get_load_error()
                or "알 수 없는 오류"
            )
            self.done.emit(
                False,
                f"모델을 불러오지 못했습니다.\n\n{err}",
            )


class LoadingDialog(QtWidgets.QDialog):
    """검색 등 짧은 작업 동안 띄우는 모달 로딩창 (진행 표시줄 포함)."""

    def __init__(self, text="처리 중...", parent=None):
        super().__init__(parent)
        self.setWindowTitle("")
        self.setModal(True)
        # 닫기 버튼·테두리 없는 단순한 창
        self.setWindowFlags(
            QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint
        )
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 24)
        self.label = QtWidgets.QLabel(text)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.label)
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 0)  # 불확정(빙글빙글) 모드
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        lay.addWidget(bar)
        self.setFixedWidth(260)

    def showEvent(self, e):
        super().showEvent(e)
        if self.parent():
            pg = self.parent().frameGeometry()
            self.adjustSize()
            self.move(pg.center() - self.rect().center())


class RecognizeThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    recognized = QtCore.pyqtSignal(str, str, float)
    auto_recognized = QtCore.pyqtSignal(
        str, str, float
    )  # 자동 모드 처리용
    status = QtCore.pyqtSignal(str)

    def __init__(
        self,
        engine: FaceEngine,
        db: AttendanceDB,
        sim_thr: float,
        vote_n: int,
        camera_index: int = 0,
        cooldown_sec: float = 1.0,
        auto_mode: bool = False,
        dwell_sec: float = 2.0,
        auto_preview: bool = True,
    ):
        super().__init__()
        self.engine = engine
        self.db = db
        self.sim_thr = sim_thr
        self.vote_n = vote_n
        self.camera_index = camera_index
        self.cooldown_sec = cooldown_sec
        self.auto_mode = auto_mode
        self.dwell_sec = dwell_sec
        self.auto_preview = auto_preview
        self._running = False
        self._paused = False
        self.db_embeddings = {}

    def reload_embeddings(self):
        self.db_embeddings = self.db.load_all_embeddings()

    def run(self):
        import time as _t

        self._running = True
        self.status.emit("카메라 여는 중...")
        cap = open_camera_safe(
            self.camera_index, timeout=3.0
        )
        if cap is None:
            self.status.emit(
                "카메라를 열 수 없습니다. 설정에서 다른 카메라를 선택하세요."
            )
            return

        model_ready = self.engine.load(
            progress_cb=self.status.emit
        )
        if not model_ready:
            err = self.engine.get_load_error()
            self.status.emit(
                f"AI 모델 로딩 실패 — 얼굴 인식 없이 카메라만 표시됩니다. ({err})"
            )
        self.reload_embeddings()
        if model_ready:
            self.status.emit(
                f"인식 대기 중 — 등록 직원 {len(self.db_embeddings)}명"
            )
        vote = deque(maxlen=self.vote_n)
        frame_no = 0
        read_fail = 0
        cooldown_until = (
            0.0  # 이 시각 전까지는 재인식 안 함
        )
        rearm = True  # 빈 화면(얼굴 없음)을 본 뒤에야 다시 인식 준비
        dwell_id = (
            None  # 자동 모드: 현재 연속 체류 중인 사람
        )
        dwell_start = 0.0  # 자동 모드: 체류 시작 시각
        dwell_action = None  # 자동 모드: 예정 동작(체류 시작 시 1회 조회)
        while self._running:
            if cap is None:
                # 재연결 대기 중 — 다시 열어본다
                self.msleep(1000)
                cap = open_camera_safe(
                    self.camera_index, timeout=3.0
                )
                continue
            ret, frame = cap.read()
            if not ret:
                # 카메라 일시적 끊김 → 재연결 시도 (장기 운영 대비)
                read_fail += 1
                self.status.emit(
                    f"카메라 신호 없음… 재연결 시도 ({read_fail})"
                )
                cap.release()
                self.msleep(1000)
                if not self._running:
                    break
                cap = open_camera_safe(
                    self.camera_index, timeout=3.0
                )
                if cap is None:
                    # 재연결 실패 — 다음 루프에서 다시 시도
                    continue
                if (
                    read_fail > 30
                ):  # 30초 넘게 실패하면 잠깐 길게 대기
                    self.msleep(3000)
                continue
            read_fail = 0
            frame_no += 1
            now = _t.time()
            if (
                model_ready
                and not self._paused
                and self.db_embeddings
                and now >= cooldown_until
            ):
                try:
                    emb = self.engine.embed_face(frame)
                except Exception as e:
                    emb = None
                    self.status.emit(f"인식 오류: {e}")
                if emb is None:
                    # 얼굴이 없는 프레임 → 다음 사람 인식 준비(re-arm)
                    rearm = True
                    vote.clear()
                    dwell_id = None
                elif rearm:
                    emp_id, name, sim = cosine_best_match(
                        emb,
                        self.db_embeddings,
                        self.sim_thr,
                    )
                    if emp_id is None:
                        vote.clear()
                        dwell_id = None
                    elif self.auto_mode:
                        # 자동 모드: 같은 사람이 dwell_sec 이상 연속 체류하면 자동 처리
                        if emp_id != dwell_id:
                            dwell_id, dwell_start = (
                                emp_id,
                                now,
                            )
                            # 체류 시작 시 예정 동작을 한 번만 조회
                            try:
                                acts = self.db.get_available_actions(
                                    emp_id
                                )
                                dwell_action = (
                                    acts[0]
                                    if acts
                                    else None
                                )
                            except Exception:
                                # DB가 교체/종료 중이면 이번 프레임은 건너뜀
                                dwell_id = None
                                continue
                        held = now - dwell_start
                        remain = self.dwell_sec - held
                        if remain > 0:
                            if (
                                self.auto_preview
                                and dwell_action
                            ):
                                lab = ACTION_LABELS.get(
                                    dwell_action,
                                    dwell_action,
                                )
                                self.status.emit(
                                    f"{name} — 곧 [{lab}] 처리 · {remain:.1f}초"
                                )
                            elif (
                                self.auto_preview
                                and dwell_action is None
                            ):
                                self.status.emit(
                                    f"{name} — 당일 처리 완료 (추가 동작 없음)"
                                )
                            else:
                                self.status.emit(
                                    f"{name} 인식 중… {remain:.1f}초"
                                )
                        else:
                            self.auto_recognized.emit(
                                emp_id, name, sim
                            )
                            self._paused = True
                            cooldown_until = (
                                now + self.cooldown_sec
                            )
                            rearm = False
                            dwell_id = None
                    else:
                        # 수동 모드: vote_n 프레임 연속 일치 시 동작 선택창
                        if vote and vote[-1] != emp_id:
                            vote.clear()
                        vote.append(emp_id)
                        if (
                            len(vote) == self.vote_n
                            and len(set(vote)) == 1
                        ):
                            self.recognized.emit(
                                emp_id, name, sim
                            )
                            vote.clear()
                            self._paused = True
                            cooldown_until = (
                                now + self.cooldown_sec
                            )
                            rearm = False
            self.frame_ready.emit(frame)
            # 주기적 GPU 캐시 정리 (장시간 운영 시 VRAM 누적 방지)
            if (
                model_ready and frame_no % 1800 == 0
            ):  # 대략 수십 초~1분마다
                self.engine.cleanup()
            self.msleep(15)
        if cap is not None:
            cap.release()

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False
        self.wait(2000)


class VideoRegisterThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)

    def __init__(
        self,
        engine: FaceEngine,
        video_path: str,
        frame_skip: int = 5,
        max_frames: int = 30,
    ):
        super().__init__()
        self.engine = engine
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.max_frames = max_frames

    def run(self):
        if not self.engine.load(
            progress_cb=self.status.emit
        ):
            err = self.engine.get_load_error()
            self.status.emit(
                f"AI 모델을 불러올 수 없어 처리할 수 없습니다. ({err})"
            )
            self.done.emit(None)
            return

        # 빌드된 exe에서는 기본(FFmpeg) 백엔드의 DLL이 제대로 안 잡혀서
        # 동영상 "파일"을 못 여는 경우가 있음. 카메라가 쓰는 MSMF를 먼저
        # 시도하고, 안 되면 기본 방식으로 재시도 (open_camera_safe와 동일 패턴).
        cap = None
        if hasattr(cv2, "CAP_MSMF"):
            cap = cv2.VideoCapture(
                self.video_path, cv2.CAP_MSMF
            )
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status.emit("동영상을 열 수 없습니다.")
            self.done.emit(None)
            return
        embs = []
        idx = 0
        while (
            cap.isOpened() and len(embs) < self.max_frames
        ):
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.frame_skip == 0:
                try:
                    emb = self.engine.embed_face(frame)
                except Exception as e:
                    emb = None
                    self.status.emit(
                        f"프레임 처리 오류: {e}"
                    )
                if emb is not None:
                    embs.append(emb)
                    self.progress.emit(
                        len(embs), self.max_frames
                    )
            idx += 1
        cap.release()
        if not embs:
            self.status.emit(
                "얼굴을 찾지 못했습니다. 다른 동영상을 시도하세요."
            )
            self.done.emit(None)
        else:
            self.status.emit(
                f"{len(embs)}개 프레임에서 임베딩 추출 완료"
            )
            self.done.emit(average_embeddings(embs))


class RegisterDialog(QtWidgets.QDialog):
    def __init__(
        self,
        engine: FaceEngine,
        db: AttendanceDB,
        parent=None,
        camera_index: int = 0,
    ):
        super().__init__(parent)
        self.engine = engine
        self.db = db
        self.camera_index = camera_index
        self.setWindowTitle("신규 직원 등록")
        self.resize(640, 620)
        self._cam_embs = []
        self._cap = None
        self._timer = None
        self._recording = False
        self._build()

    def _build(self):
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QHBoxLayout()
        form.addWidget(QtWidgets.QLabel("이름:"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText(
            "등록할 직원 이름 (입력해야 등록 시작)"
        )
        self.name_edit.textChanged.connect(
            self._on_name_changed
        )
        form.addWidget(self.name_edit)
        nid = self.db.next_emp_id()
        self.id_label = QtWidgets.QLabel(
            f"자동 ID: <b>{nid}</b>"
        )
        form.addWidget(self.id_label)
        v.addLayout(form)

        self.preview = QtWidgets.QLabel(
            "이름을 입력한 뒤 등록 방법을 선택하세요"
        )
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(360)
        self.preview.setProperty("viewport", True)
        v.addWidget(self.preview)

        self.guide = QtWidgets.QLabel(self._guide_text())
        self.guide.setWordWrap(True)
        self.guide.setProperty("hint", True)
        v.addWidget(self.guide)

        # 등록 방법 3종 — 이름 입력 전엔 모두 비활성
        btns = QtWidgets.QHBoxLayout()
        self.capture_mode_btn = QtWidgets.QPushButton(
            "캡처로 등록"
        )
        self.capture_mode_btn.clicked.connect(
            self.start_capture_mode
        )
        self.record_mode_btn = QtWidgets.QPushButton(
            "영상으로 등록"
        )
        self.record_mode_btn.clicked.connect(
            self.start_record_mode
        )
        self.video_btn = QtWidgets.QPushButton(
            "영상 업로드"
        )
        self.video_btn.clicked.connect(self.pick_video)
        for b in (
            self.capture_mode_btn,
            self.record_mode_btn,
            self.video_btn,
        ):
            b.setEnabled(False)
            btns.addWidget(b)
        v.addLayout(btns)

        # 캡처 모드 전용 버튼
        cam_actions = QtWidgets.QHBoxLayout()
        self.capture_btn = QtWidgets.QPushButton(
            "현재 얼굴 캡처 (0/10)"
        )
        self.capture_btn.clicked.connect(self.capture_one)
        self.capture_btn.setEnabled(False)
        cam_actions.addWidget(self.capture_btn)
        v.addLayout(cam_actions)

        self.status = QtWidgets.QLabel("")
        self.status.setProperty("muted", True)
        v.addWidget(self.status)
        self.register_btn = QtWidgets.QPushButton(
            "등록 완료"
        )
        self.register_btn.setProperty("accent", True)
        self.register_btn.setMinimumHeight(40)
        self.register_btn.clicked.connect(self.do_register)
        self.register_btn.setEnabled(False)
        v.addWidget(self.register_btn)

        self._pending_emb = None
        self._recording = False
        self.RECORD_SECONDS = 5
        self.RECORD_TARGET = (
            12  # 5초 동안 모을 목표 프레임 수
        )

    def _on_name_changed(self, text):
        """이름이 있어야 등록 방법 버튼이 활성화된다."""
        has = bool(text.strip())
        # 카메라/녹화 세션이 진행 중이면 방법 버튼 상태를 건드리지 않는다
        session_active = (
            self._cap is not None or self._recording
        )
        if not session_active:
            for b in (
                self.capture_mode_btn,
                self.record_mode_btn,
                self.video_btn,
            ):
                b.setEnabled(has)
        if not has:
            self.capture_btn.setEnabled(False)

    def _guide_text(self):
        return (
            "<b>등록 방법</b> (이름 입력 후 선택):<br>"
            "• <b>캡처로 등록</b>: 미리보기를 보며 ‘현재 얼굴 캡처’로 여러 장(권장 10장) 수집.<br>"
            "• <b>영상으로 등록</b>: 버튼을 누르면 약 5초간 자동으로 여러 각도를 수집합니다. "
            "그동안 고개를 아주 조금씩 좌우/상하로 움직이세요.<br>"
            "• <b>영상 업로드</b>: 미리 찍어둔 5~10초 얼굴 영상 파일을 선택.<br>"
            "• 밝은 곳에서 정면을 보고, 안경 착용자는 평소처럼 쓰고 등록하세요."
        )

    def _open_cam(self):
        """카메라를 열고 미리보기 타이머를 시작 (캡처/녹화 공통)."""
        if self._cap is not None:
            return True
        self.status.setText("카메라 여는 중...")
        QtWidgets.QApplication.processEvents()
        # 가상캠 등이 멈추게 하는 것을 막기 위해 timeout 적용해 안전하게 연다
        from app.face_engine import open_camera_safe

        self._cap = open_camera_safe(
            self.camera_index, timeout=3.0
        )
        if self._cap is None:
            self.status.setText(
                "카메라를 열 수 없습니다. 설정 탭에서 다른 카메라를 선택하세요."
            )
            return False
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._cam_tick)
        self._timer.start(30)
        return True

    def start_capture_mode(self):
        """캡처로 등록: 한 장씩 수집."""
        if not self._open_cam():
            return
        self._cam_embs = []
        self._recording = False
        self.capture_btn.setEnabled(True)
        self.capture_btn.setText("현재 얼굴 캡처 (0/10)")
        self.record_mode_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.status.setText(
            "거울 미리보기 — 얼굴을 맞추고 ‘현재 얼굴 캡처’를 누르세요."
        )

    def start_record_mode(self):
        """영상으로 등록: 약 5초간 자동 수집."""
        if not self._open_cam():
            return
        self._cam_embs = []
        self.capture_btn.setEnabled(False)
        self.capture_mode_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self._recording = True
        self._record_start = None  # 첫 프레임에서 설정
        self.status.setText(
            f"{self.RECORD_SECONDS}초 자동 수집 시작 — 고개를 천천히 움직이세요."
        )

    def _cam_tick(self):
        try:
            ret, frame = self._cap.read()
        except Exception:
            ret, frame = False, None
        if not ret or frame is None:
            return
        self._last_frame = frame
        mirror = cv2.flip(frame, 1)
        pix = bgr_to_qpixmap(mirror).scaled(
            self.preview.width(),
            self.preview.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

        if not self._recording:
            return
        # 녹화 모드: 일정 간격으로 임베딩 수집
        import time as _t

        now = _t.time()
        if self._record_start is None:
            self._record_start = now
            self._last_grab = 0
        elapsed = now - self._record_start
        # 목표 프레임을 5초에 고르게 분배
        interval = self.RECORD_SECONDS / max(
            1, self.RECORD_TARGET
        )
        if (
            now - self._last_grab >= interval
            and len(self._cam_embs) < self.RECORD_TARGET
        ):
            try:
                emb = self.engine.embed_face(frame)
            except Exception as e:
                emb = None
                self.status.setText(
                    f"분석 오류 (계속 수집 중): {e}"
                )
            self._last_grab = now
            if emb is not None:
                self._cam_embs.append(emb)
            self.status.setText(
                f"수집 중… {len(self._cam_embs)}장 ({min(elapsed, self.RECORD_SECONDS):.1f}/{self.RECORD_SECONDS}초)"
            )
        if elapsed >= self.RECORD_SECONDS:
            self._recording = False
            self._finish_record()

    def _finish_record(self):
        n = len(self._cam_embs)
        if n < 3:
            self.status.setText(
                f"수집된 얼굴이 부족합니다 ({n}장). 다시 시도하세요."
            )
            # 버튼 복구
            self._on_name_changed(self.name_edit.text())
            self.capture_mode_btn.setEnabled(True)
            self.record_mode_btn.setEnabled(True)
            return
        self._pending_emb = average_embeddings(
            self._cam_embs
        )
        self.register_btn.setEnabled(True)
        self.status.setText(
            f"{n}장 수집 완료 — ‘등록 완료’를 누르세요."
        )

    def capture_one(self):
        if not hasattr(self, "_last_frame"):
            return
        self.status.setText("얼굴 분석 중...")
        QtWidgets.QApplication.processEvents()
        try:
            emb = self.engine.embed_face(self._last_frame)
        except Exception as e:
            self.status.setText(
                f"얼굴 분석 중 오류가 발생했습니다: {e}"
            )
            return
        if emb is None:
            if not self.engine.is_loaded():
                err = self.engine.get_load_error()
                self.status.setText(
                    f"AI 모델을 사용할 수 없습니다. ({err})"
                )
            else:
                self.status.setText(
                    "얼굴이 감지되지 않았습니다. 다시 시도하세요."
                )
            return
        self._cam_embs.append(emb)
        n = len(self._cam_embs)
        self.capture_btn.setText(f"현재 얼굴 캡처 ({n}/10)")
        self.status.setText(
            f"{n}장 캡처됨"
            + (" — 등록 가능" if n >= 3 else "")
        )
        if n >= 3:
            self._pending_emb = average_embeddings(
                self._cam_embs
            )
            self.register_btn.setEnabled(True)

    def pick_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "동영상 선택",
            "",
            "동영상 (*.mp4 *.avi *.mov *.mkv)",
        )
        if not path:
            return
        self.capture_mode_btn.setEnabled(False)
        self.record_mode_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.preview.setText("동영상 처리 중...")
        self.vt = VideoRegisterThread(self.engine, path)
        self.vt.status.connect(self.status.setText)
        self.vt.progress.connect(
            lambda c, t: self.status.setText(
                f"임베딩 추출 {c}/{t}"
            )
        )
        self.vt.done.connect(self._video_done)
        self.vt.start()

    def _video_done(self, emb_bytes):
        if emb_bytes is None:
            # 실패 → 방법 버튼 복구 (이름이 있다면)
            self._on_name_changed(self.name_edit.text())
            return
        self._pending_emb = emb_bytes
        self.register_btn.setEnabled(True)
        self.preview.setText(
            "처리 완료 — '등록 완료'를 누르세요."
        )

    def do_register(self):
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(
                self, "확인", "이름을 입력하세요."
            )
            return
        if self._pending_emb is None:
            QtWidgets.QMessageBox.warning(
                self, "확인", "먼저 얼굴을 캡처/처리하세요."
            )
            return
        emp_id = self.db.add_employee(
            name, self._pending_emb
        )
        QtWidgets.QMessageBox.information(
            self,
            "등록 완료",
            f"{name} ({emp_id}) 등록되었습니다.",
        )
        self.accept()

    def closeEvent(self, e):
        if self._timer:
            self._timer.stop()
        if self._cap:
            self._cap.release()
        super().closeEvent(e)


class ActionDialog(QtWidgets.QDialog):
    def __init__(
        self, name: str, actions: list, parent=None
    ):
        super().__init__(parent)
        self.chosen = None
        self.setWindowTitle("동작 선택")
        self.resize(360, 220)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(10)
        title = QtWidgets.QLabel(
            f"<h2>{name}</h2>님, 동작을 선택하세요"
        )
        title.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(title)

        for act in actions:
            label = ACTION_LABELS.get(act, act)
            b = QtWidgets.QPushButton(
                f"[{ACTION_KEYS[act]}] {label}"
            )
            b.setProperty("accent", True)
            b.setMinimumHeight(46)
            b.clicked.connect(
                lambda _, a=act: self._choose(a)
            )
            v.addWidget(b)
        cancel = QtWidgets.QPushButton("[ESC] 취소")
        cancel.setMinimumHeight(38)
        cancel.clicked.connect(self.reject)
        v.addWidget(cancel)
        self._actions = actions

        QtCore.QTimer.singleShot(5000, self.reject)

    def _choose(self, act):
        self.chosen = act
        self.accept()

    def keyPressEvent(self, e):
        key = e.text()
        for act in self._actions:
            if key == ACTION_KEYS[act]:
                self._choose(act)
                return
        if e.key() == QtCore.Qt.Key_Escape:
            self.reject()


class SortKeyItem(QtWidgets.QTableWidgetItem):
    """정렬 시 표시 텍스트가 아니라 UserRole 키로 비교하는 셀.

    날짜 컬럼이 'YYYY-MM-DD'만 보여주더라도 'YYYY-MM-DD HH:MM:SS' 키로
    정렬되게 해, 같은 날짜 안에서도 시각순이 유지된다.
    """

    def __lt__(self, other):
        a = self.data(QtCore.Qt.UserRole)
        b = other.data(QtCore.Qt.UserRole)
        if a is not None and b is not None:
            return a < b
        return super().__lt__(other)


class ProportionalTable(QtWidgets.QTableWidget):
    """컬럼 폭을 글자수 비율로 유지하는 표.

    set_ratios([4,3,10,8,2]) 처럼 비율을 주면 창 크기가 바뀌어도
    가용 폭을 그 비율대로 나눠 각 컬럼에 배분한다.
    """

    def __init__(self, rows, cols, parent=None):
        super().__init__(rows, cols, parent)
        self._ratios = []
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Fixed
        )

    def set_ratios(self, ratios):
        self._ratios = list(ratios)
        self._apply_ratios()

    def _apply_ratios(self):
        if not self._ratios:
            return
        total = sum(self._ratios)
        if total <= 0:
            return
        vw = self.viewport().width()
        used = 0
        for i, r in enumerate(self._ratios[:-1]):
            wpx = int(vw * r / total)
            self.setColumnWidth(i, wpx)
            used += wpx
        self.setColumnWidth(
            len(self._ratios) - 1, max(40, vw - used)
        )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_ratios()


class AutoToast(QtWidgets.QDialog):
    """자동 출퇴근 완료를 큰 글씨로 띄우고 일정 시간 후 자동으로 닫히는 팝업."""

    def __init__(
        self,
        name: str,
        action_label: str,
        seconds: float,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            QtCore.Qt.Dialog
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setModal(False)
        # 창 자체는 투명 — 둥근 모서리가 밖으로 비치지 않게
        self.setAttribute(
            QtCore.Qt.WA_TranslucentBackground, True
        )

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # 실제 배경/테두리는 안쪽 컨테이너에만 그린다
        self.card = QtWidgets.QFrame()
        self.card.setObjectName("toastCard")
        self.card.setStyleSheet(
            "#toastCard { background: #ffffff; border: 3px solid #16a34a;"
            " border-radius: 16px; }"
        )
        inner = QtWidgets.QVBoxLayout(self.card)
        inner.setContentsMargins(48, 34, 48, 34)
        text = QtWidgets.QLabel(
            f"{name}\n{action_label} 완료"
        )
        text.setAlignment(QtCore.Qt.AlignCenter)
        text.setStyleSheet(
            "font-size: 34px; font-weight: 800; color: #16a34a; border: none;"
        )
        inner.addWidget(text)
        outer.addWidget(self.card)

        # seconds 후 자동 닫힘
        QtCore.QTimer.singleShot(
            int(seconds * 1000), self.close
        )

    def showEvent(self, e):
        super().showEvent(e)
        # 부모(메인창) 중앙에 배치
        if self.parent():
            pg = self.parent().frameGeometry()
            self.adjustSize()
            self.move(pg.center() - self.rect().center())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = appcfg.load_config()
        self.engine = FaceEngine(
            use_fp16=self.cfg.get("use_fp16", True)
        )
        self.db = None
        self.rec_thread = None
        self.cam_scan_thread = None
        self.setWindowTitle("AI 안면인식 출퇴근 시스템")
        icon_path = appcfg.resource_path("favicon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
        self.resize(960, 700)
        self._open_db()
        self._build()
        self._start_camera_scan()

    def _start_camera_scan(self):
        """앱 시작 직후 로딩창을 띄우고 카메라를 검색해 콤보를 채운다."""
        # 창이 먼저 그려진 뒤 로딩창이 뜨도록 약간 지연
        QtCore.QTimer.singleShot(
            150,
            lambda: self._run_camera_scan(
                show_warning_if_none=False
            ),
        )

    def _on_cameras_found(self, found):
        """카메라 검색 결과로 콤보를 갱신 (현재 설정값은 유지). 로딩창을 닫는다."""
        # 로딩창이 떠 있으면 닫기
        dlg = getattr(self, "_scan_dialog", None)
        if dlg is not None:
            dlg.accept()
            self._scan_dialog = None
        if not hasattr(self, "cam_combo"):
            return
        cur = self.cfg.get("camera_index", 0)
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        if found:
            for i in found:
                self.cam_combo.addItem(f"카메라 {i}", i)
            idx = found.index(cur) if cur in found else 0
            self.cam_combo.setCurrentIndex(idx)
            self.statusBar().showMessage(
                f"카메라 {len(found)}개 발견: {found}"
            )
        else:
            self.cam_combo.addItem(f"카메라 {cur}", cur)
            self.statusBar().showMessage(
                "카메라 검색: 발견된 카메라 없음"
            )
        self.cam_combo.blockSignals(False)

        manual = getattr(self, "_scan_warn", False)
        if found:
            # 선택된 카메라를 한 번 더 테스트해 해상도까지 확인 (수동/자동 공통)
            from app.face_engine import test_camera

            sel = self.cam_combo.currentData()
            ok, res = test_camera(sel, timeout=3.0)
            if ok:
                self.statusBar().showMessage(
                    f"카메라 {found} 발견 · 카메라 {sel} 정상 ({res[0]}x{res[1]})"
                )
                if manual:
                    QtWidgets.QMessageBox.information(
                        self,
                        "카메라 검색 완료",
                        f"사용 가능한 카메라 {len(found)}개: {found}\n\n"
                        f"선택된 카메라 {sel} 정상 동작 (해상도 {res[0]} x {res[1]}).",
                    )
            else:
                self.statusBar().showMessage(
                    f"카메라 {found} 발견 · 카메라 {sel} 테스트 실패 — 다른 카메라를 선택하세요"
                )
                if manual:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "카메라 검색 완료",
                        f"카메라 {found} 를 찾았으나, 선택된 카메라 {sel} 테스트에 실패했습니다.\n"
                        "목록에서 다른 카메라를 선택해 보세요.",
                    )
        else:
            self.statusBar().showMessage(
                "카메라 검색: 발견된 카메라 없음"
            )
            if manual:
                QtWidgets.QMessageBox.warning(
                    self,
                    "카메라 검색",
                    "사용 가능한 카메라를 찾지 못했습니다.\n"
                    "카메라 연결 상태를 확인하거나, 다른 앱이 사용 중인지 확인하세요.",
                )
        self._scan_warn = False

    def _open_db(self):
        try:
            self.db = AttendanceDB(
                self.cfg["db_path"],
                self.cfg.get("attendance_mode", "full"),
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "DB 오류",
                f"DB를 열 수 없습니다:\n{e}\n설정에서 경로를 변경하세요.",
            )

    def _build(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(
            self._tab_recognize(), "실시간 인식"
        )
        self.tabs.addTab(self._tab_employees(), "직원 관리")
        self.tabs.addTab(self._tab_logs(), "근태 기록")
        self.tabs.addTab(self._tab_settings(), "설정")
        self.statusBar().showMessage(
            f"DB: {self.cfg['db_path']}"
        )

    def _tab_recognize(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(12)
        self.cam_view = QtWidgets.QLabel(
            "‘인식 시작’을 누르세요"
        )
        self.cam_view.setAlignment(QtCore.Qt.AlignCenter)
        self.cam_view.setMinimumHeight(360)
        self.cam_view.setProperty("viewport", True)
        # 큰 화면에서 카메라 영역이 과도하게 늘어나지 않도록 최대 폭 제한 + 중앙 정렬
        self.cam_view.setMaximumWidth(1100)
        self.cam_view.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        cam_row = QtWidgets.QHBoxLayout()
        cam_row.addStretch(1)
        cam_row.addWidget(self.cam_view, 8)
        cam_row.addStretch(1)
        v.addLayout(cam_row, 1)

        self.rec_status = QtWidgets.QLabel("대기 중")
        self.rec_status.setProperty("muted", True)
        self.rec_status.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.rec_status)

        h = QtWidgets.QHBoxLayout()
        h.addStretch(1)
        self.start_btn = QtWidgets.QPushButton("인식 시작")
        self.start_btn.setProperty("accent", True)
        self.start_btn.setMinimumHeight(44)
        self.start_btn.setMinimumWidth(220)
        self.start_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.start_btn.clicked.connect(
            self.toggle_recognize
        )
        h.addWidget(self.start_btn)
        h.addStretch(1)
        v.addLayout(h)
        return w

    def toggle_recognize(self):
        if self.rec_thread and self.rec_thread.isRunning():
            self.rec_thread.stop()
            self.rec_thread = None
            self.start_btn.setText("인식 시작")
            self.rec_status.setText("정지됨")
            return
        if self.db is None:
            return
        # 자동 모드는 2단계(출/퇴)에서만 — 4단계면 강제로 끔
        auto = (
            self.cfg.get("auto_mode", False)
            and self.cfg.get("attendance_mode", "full")
            == "simple"
        )
        self.rec_thread = RecognizeThread(
            self.engine,
            self.db,
            self.cfg.get("sim_threshold", 0.85),
            self.cfg.get("vote_n", 5),
            self.cfg.get("camera_index", 0),
            self.cfg.get("cooldown_sec", 1.0),
            auto,
            self.cfg.get("dwell_sec", 2.0),
            self.cfg.get("auto_preview", True),
        )
        self.rec_thread.frame_ready.connect(
            self._show_frame
        )
        self.rec_thread.recognized.connect(
            self._on_recognized
        )
        self.rec_thread.auto_recognized.connect(
            self._on_auto_recognized
        )
        self.rec_thread.status.connect(
            self.rec_status.setText
        )
        self.rec_thread.start()
        self.start_btn.setText("인식 정지")

    def _show_frame(self, frame):
        mirror = cv2.flip(frame, 1)
        pix = bgr_to_qpixmap(mirror).scaled(
            self.cam_view.width(),
            self.cam_view.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.cam_view.setPixmap(pix)

    def _on_recognized(self, emp_id, name, sim):
        actions = self.db.get_available_actions(emp_id)
        if not actions:
            self.rec_status.setText(
                f"{name}: 당일 퇴근 완료 — 추가 동작 없음"
            )
            if self.rec_thread:
                self.rec_thread.resume()
            return
        dlg = ActionDialog(name, actions, self)
        if (
            dlg.exec_() == QtWidgets.QDialog.Accepted
            and dlg.chosen
        ):
            self.db.log_action(emp_id, dlg.chosen)
            label = ACTION_LABELS.get(
                dlg.chosen, dlg.chosen
            )
            self.rec_status.setText(
                f"{name}: {label} 완료 (유사도 {sim:.2f})"
            )
            self.refresh_logs()
            if self.cfg.get("tts_enabled", False):
                tts.speak_async(f"{name}님 {label}")
        else:
            self.rec_status.setText("취소됨")
        if self.rec_thread:
            self.rec_thread.resume()

    def _on_auto_recognized(self, emp_id, name, sim):
        """자동 모드: 동작 선택창 없이 가능한 동작을 바로 기록."""
        actions = self.db.get_available_actions(emp_id)
        if not actions:
            self.rec_status.setText(
                f"{name}: 당일 퇴근 완료 — 추가 동작 없음"
            )
            if self.rec_thread:
                self.rec_thread.resume()
            return
        # 2단계에서 가능한 동작은 항상 하나(출근 또는 퇴근 또는 퇴근갱신)
        action = actions[0]
        try:
            self.db.log_action(emp_id, action)
            label = ACTION_LABELS.get(action, action)
            self.rec_status.setText(
                f"{name}: {label} 완료 (자동)"
            )
            self.refresh_logs()
            # 완료 팝업 (옵션)
            if self.cfg.get("auto_popup", True):
                toast = AutoToast(
                    name,
                    label,
                    self.cfg.get("popup_sec", 2.0),
                    self,
                )
                toast.show()
            # 음성 안내 (옵션)
            if self.cfg.get("tts_enabled", False):
                tts.speak_async(f"{name}님 {label}")
        except Exception as e:
            self.rec_status.setText(f"처리 오류: {e}")
        if self.rec_thread:
            self.rec_thread.resume()

    @staticmethod
    def _style_table(table):
        """표 공통 스타일: 줄무늬, 정렬, 행 선택, 행 높이, hover."""
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.setShowGrid(False)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(38)
        table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel
        )
        table.horizontalHeader().setHighlightSections(False)

    def _tab_employees(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(12)
        top = QtWidgets.QHBoxLayout()
        self.emp_search = QtWidgets.QLineEdit()
        self.emp_search.setPlaceholderText(
            "이름 또는 ID 검색"
        )
        self.emp_search.setMinimumHeight(34)
        self.emp_search.textChanged.connect(
            self.refresh_employees
        )
        top.addWidget(self.emp_search, 1)
        add_btn = QtWidgets.QPushButton("신규 등록")
        add_btn.setProperty("accent", True)
        add_btn.setFixedWidth(110)
        add_btn.setMinimumHeight(34)
        add_btn.clicked.connect(self.open_register)
        top.addWidget(add_btn)
        del_btn = QtWidgets.QPushButton("선택 삭제")
        del_btn.setFixedWidth(110)
        del_btn.setMinimumHeight(34)
        del_btn.clicked.connect(self.delete_employee)
        top.addWidget(del_btn)
        v.addLayout(top)
        self.emp_table = ProportionalTable(0, 3)
        self.emp_table.setHorizontalHeaderLabels(
            ["ID", "이름", "등록일시"]
        )
        self._style_table(self.emp_table)
        # 글자수 비율 — ID:2, 이름:2, 등록일시:6
        self.emp_table.set_ratios([2, 2, 6])
        v.addWidget(self.emp_table)
        self.refresh_employees()
        return w

    def refresh_employees(self):
        if self.db is None:
            return
        kw = self.emp_search.text().strip()
        rows = self.db.list_employees(kw)
        self.emp_table.setSortingEnabled(False)
        self.emp_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self._set_cell(
                self.emp_table,
                i,
                0,
                r["emp_id"],
                center=True,
            )
            self._set_cell(
                self.emp_table, i, 1, r["name"], center=True
            )
            self._set_cell(
                self.emp_table,
                i,
                2,
                r["created_at"],
                center=True,
            )
        self.emp_table.setSortingEnabled(True)

    @staticmethod
    def _set_cell(
        table, row, col, text, center=False, color=None
    ):
        """셀 생성 + 정렬/색상 옵션."""
        item = QtWidgets.QTableWidgetItem(
            str(text) if text is not None else ""
        )
        if center:
            item.setTextAlignment(QtCore.Qt.AlignCenter)
        if color:
            item.setForeground(QtGui.QColor(color))
        table.setItem(row, col, item)

    def open_register(self):
        if self.db is None:
            return
        dlg = RegisterDialog(
            self.engine,
            self.db,
            self,
            self.cfg.get("camera_index", 0),
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.refresh_employees()
            if (
                self.rec_thread
                and self.rec_thread.isRunning()
            ):
                self.rec_thread.reload_embeddings()

    def delete_employee(self):
        row = self.emp_table.currentRow()
        if row < 0:
            return
        emp_id = self.emp_table.item(row, 0).text()
        name = self.emp_table.item(row, 1).text()
        ok = QtWidgets.QMessageBox.question(
            self,
            "삭제 확인",
            f"{name} ({emp_id}) 직원과 모든 근태 기록을 삭제할까요?",
        )
        if ok == QtWidgets.QMessageBox.Yes:
            self.db.delete_employee(emp_id)
            self.refresh_employees()
            if (
                self.rec_thread
                and self.rec_thread.isRunning()
            ):
                self.rec_thread.reload_embeddings()

    def _tab_logs(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(12)
        top = QtWidgets.QHBoxLayout()
        self.log_search = QtWidgets.QLineEdit()
        self.log_search.setPlaceholderText("이름/ID")
        self.log_search.setMinimumHeight(34)
        self.log_search.returnPressed.connect(
            self.refresh_logs
        )
        top.addWidget(self.log_search, 2)

        today = QtCore.QDate.currentDate()
        first_day = QtCore.QDate(
            today.year(), today.month(), 1
        )

        self.date_from = QtWidgets.QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_from.setDisplayFormat("yyyy-MM-dd")
        self.date_from.setDate(
            first_day
        )  # 기본: 이번 달 1일
        self.date_from.setMinimumHeight(34)
        top.addWidget(QtWidgets.QLabel("시작"))
        top.addWidget(self.date_from, 1)

        self.date_to = QtWidgets.QDateEdit()
        self.date_to.setCalendarPopup(True)
        self.date_to.setDisplayFormat("yyyy-MM-dd")
        self.date_to.setDate(today)  # 기본: 오늘
        self.date_to.setMinimumHeight(34)
        top.addWidget(QtWidgets.QLabel("종료"))
        top.addWidget(self.date_to, 1)

        search_btn = QtWidgets.QPushButton("검색")
        search_btn.setFixedWidth(90)
        search_btn.setMinimumHeight(34)
        search_btn.clicked.connect(self.refresh_logs)
        top.addWidget(search_btn)
        export_btn = QtWidgets.QPushButton("엑셀 추출")
        export_btn.setProperty("accent", True)
        export_btn.setFixedWidth(110)
        export_btn.setMinimumHeight(34)
        export_btn.clicked.connect(self.export_logs)
        top.addWidget(export_btn)
        v.addLayout(top)
        self.log_table = ProportionalTable(0, 5)
        self.log_table.setHorizontalHeaderLabels(
            ["직원ID", "이름", "날짜", "시각", "상태"]
        )
        self._style_table(self.log_table)
        # 글자수 비율 — ID(E001):4, 이름:3, 날짜(yyyy-mm-dd):10, 시각(hh:mm:ss):8, 상태:2
        self.log_table.set_ratios([4, 3, 10, 8, 2])
        v.addWidget(self.log_table)
        self.refresh_logs()
        return w

    def _current_log_rows(self):
        return self.db.query_logs(
            self.log_search.text().strip(),
            self.date_from.date().toString("yyyy-MM-dd"),
            self.date_to.date().toString("yyyy-MM-dd"),
        )

    def refresh_logs(self):
        if self.db is None:
            return
        rows = self._current_log_rows()
        status_color = {
            "출근": "#16a34a",
            "외출": "#d97706",
            "복귀": "#2563eb",
            "퇴근": "#6b7280",
        }
        self.log_table.setSortingEnabled(False)
        self.log_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            emp_id, name, date_, time_, state = (
                list(r) + [""] * 5
            )[:5]
            self._set_cell(
                self.log_table, i, 0, emp_id, center=True
            )
            self._set_cell(
                self.log_table, i, 1, name, center=True
            )
            # 날짜 셀에 "날짜 시각" 정렬키를 심어, 날짜 기준 정렬 시 시각까지 반영
            date_item = SortKeyItem(str(date_))
            date_item.setTextAlignment(
                QtCore.Qt.AlignCenter
            )
            date_item.setData(
                QtCore.Qt.UserRole, f"{date_} {time_}"
            )
            self.log_table.setItem(i, 2, date_item)
            self._set_cell(
                self.log_table, i, 3, time_, center=True
            )
            self._set_cell(
                self.log_table,
                i,
                4,
                state,
                center=True,
                color=status_color.get(state),
            )
        self.log_table.setSortingEnabled(True)
        # 기본 정렬: 날짜(+시각) 최신순. 헤더 화살표도 날짜 컬럼에 표시.
        self.log_table.sortByColumn(
            2, QtCore.Qt.DescendingOrder
        )
        if not rows:
            self.statusBar().showMessage("조회 결과 없음")
        else:
            self.statusBar().showMessage(
                f"근태 기록 {len(rows)}건"
            )

    def export_logs(self):
        if self.db is None:
            return
        rows = self._current_log_rows()
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "추출", "추출할 기록이 없습니다."
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "엑셀로 저장",
            "근태기록.xlsx",
            "Excel (*.xlsx)",
        )
        if not path:
            return
        if not path.endswith(".xlsx"):
            path += ".xlsx"
        self.db.export_logs_xlsx(path, rows)
        QtWidgets.QMessageBox.information(
            self, "추출 완료", f"저장됨:\n{path}"
        )

    def _tab_settings(self):
        outer = QtWidgets.QWidget()
        outer_h = QtWidgets.QHBoxLayout(outer)
        outer_h.setContentsMargins(24, 24, 24, 24)
        outer_h.addStretch(1)

        col = QtWidgets.QVBoxLayout()
        col.setSpacing(14)
        card = QtWidgets.QWidget()
        card.setFixedWidth(520)
        card.setLayout(col)

        def section_label(text):
            lb = QtWidgets.QLabel(text)
            lb.setStyleSheet(
                "color:#374151; font-weight:600;"
            )
            return lb

        # AI 얼굴 인식 모델 상태 확인 / 수동 다운로드
        col.addWidget(section_label("AI 얼굴 인식 모델"))
        model_row = QtWidgets.QHBoxLayout()
        self.model_status_label = QtWidgets.QLabel(
            self._model_status_text()
        )
        self.model_status_label.setWordWrap(True)
        model_row.addWidget(self.model_status_label, 1)
        model_check_btn = QtWidgets.QPushButton(
            "모델 확인 / 다운로드"
        )
        model_check_btn.setFixedWidth(150)
        model_check_btn.clicked.connect(self._check_model)
        model_row.addWidget(model_check_btn)
        col.addLayout(model_row)

        # DB 경로 (라벨 위 / 입력칸 아래)
        col.addWidget(section_label("DB 경로"))
        path_row = QtWidgets.QHBoxLayout()
        self.db_path_edit = QtWidgets.QLineEdit(
            self.cfg["db_path"]
        )
        self.db_path_edit.setMinimumHeight(34)
        path_row.addWidget(self.db_path_edit, 1)
        browse = QtWidgets.QPushButton("찾아보기")
        browse.setFixedWidth(90)
        browse.clicked.connect(self._browse_db)
        path_row.addWidget(browse)
        col.addLayout(path_row)

        # 임계값 / 투표 프레임 수 — 한 줄에 나란히
        twin = QtWidgets.QHBoxLayout()
        twin.setSpacing(24)
        thr_box = QtWidgets.QVBoxLayout()
        thr_box.addWidget(section_label("유사도 임계값"))
        self.thr_spin = QtWidgets.QDoubleSpinBox()
        self.thr_spin.setRange(0.30, 0.95)
        self.thr_spin.setSingleStep(0.01)
        self.thr_spin.setValue(
            self.cfg.get("sim_threshold", 0.85)
        )
        self.thr_spin.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.thr_spin.setMinimumHeight(34)
        self.thr_spin.setAlignment(QtCore.Qt.AlignCenter)
        thr_box.addWidget(self.thr_spin)
        twin.addLayout(thr_box, 1)

        vote_box = QtWidgets.QVBoxLayout()
        vote_box.addWidget(section_label("투표 프레임 수"))
        self.vote_spin = QtWidgets.QSpinBox()
        self.vote_spin.setRange(1, 15)
        self.vote_spin.setValue(self.cfg.get("vote_n", 5))
        self.vote_spin.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.vote_spin.setMinimumHeight(34)
        self.vote_spin.setAlignment(QtCore.Qt.AlignCenter)
        vote_box.addWidget(self.vote_spin)
        twin.addLayout(vote_box, 1)
        col.addLayout(twin)

        # 카메라
        col.addWidget(section_label("카메라"))
        cam_row = QtWidgets.QHBoxLayout()
        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.addItem(
            f"카메라 {self.cfg.get('camera_index', 0)}",
            self.cfg.get("camera_index", 0),
        )
        self.cam_combo.setMinimumHeight(34)
        cam_row.addWidget(self.cam_combo, 1)
        scan_btn = QtWidgets.QPushButton("카메라 검색")
        scan_btn.setFixedWidth(120)
        scan_btn.clicked.connect(self._scan_cameras)
        cam_row.addWidget(scan_btn)
        test_btn = QtWidgets.QPushButton("카메라 테스트")
        test_btn.setFixedWidth(120)
        test_btn.clicked.connect(self._test_camera)
        cam_row.addWidget(test_btn)
        col.addLayout(cam_row)

        col.addWidget(section_label("근태 모드"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setMinimumHeight(34)
        # data = (attendance_mode, auto_mode)
        self.mode_combo.addItem(
            "출근 / 퇴근 (수동)", ("simple", False)
        )
        self.mode_combo.addItem(
            "출근 / 외출 / 복귀 / 퇴근 (수동)",
            ("full", False),
        )
        self.mode_combo.addItem(
            "출근 / 퇴근 (자동 — 카메라 앞에 서면 자동 처리)",
            ("simple", True),
        )
        cur_mode = self.cfg.get("attendance_mode", "full")
        cur_auto = self.cfg.get("auto_mode", False)
        if cur_auto and cur_mode == "simple":
            self.mode_combo.setCurrentIndex(2)
        else:
            self.mode_combo.setCurrentIndex(
                0 if cur_mode == "simple" else 1
            )
        self.mode_combo.currentIndexChanged.connect(
            self._on_mode_changed
        )
        col.addWidget(self.mode_combo)

        # 자동 모드 전용 옵션(체류 시간 / 예정 동작 미리보기) — 자동 선택 시에만 표시
        dwell_row = QtWidgets.QHBoxLayout()
        dwell_row.addWidget(
            QtWidgets.QLabel("자동 처리까지 체류 시간(초):")
        )
        self.dwell_spin = QtWidgets.QDoubleSpinBox()
        self.dwell_spin.setRange(1.0, 10.0)
        self.dwell_spin.setSingleStep(0.5)
        self.dwell_spin.setValue(
            self.cfg.get("dwell_sec", 2.0)
        )
        self.dwell_spin.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.dwell_spin.setFixedWidth(90)
        self.dwell_spin.setMinimumHeight(30)
        self.dwell_spin.setAlignment(QtCore.Qt.AlignCenter)
        dwell_row.addWidget(self.dwell_spin)
        dwell_row.addStretch(1)
        self.dwell_widget = QtWidgets.QWidget()
        self.dwell_widget.setLayout(dwell_row)
        col.addWidget(self.dwell_widget)

        self.preview_chk = QtWidgets.QCheckBox(
            '자동 처리 전 예정 동작 미리 표시 (예: "홍길동 — 곧 [퇴근] 처리")'
        )
        self.preview_chk.setChecked(
            self.cfg.get("auto_preview", True)
        )
        col.addWidget(self.preview_chk)

        # 완료 팝업
        self.popup_chk = QtWidgets.QCheckBox(
            "처리 완료 시 큰 팝업 표시"
        )
        self.popup_chk.setChecked(
            self.cfg.get("auto_popup", True)
        )
        col.addWidget(self.popup_chk)

        popup_row = QtWidgets.QHBoxLayout()
        popup_row.addWidget(
            QtWidgets.QLabel("팝업 표시 시간(초):")
        )
        self.popup_spin = QtWidgets.QDoubleSpinBox()
        self.popup_spin.setRange(0.5, 10.0)
        self.popup_spin.setSingleStep(0.5)
        self.popup_spin.setValue(
            self.cfg.get("popup_sec", 2.0)
        )
        self.popup_spin.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.popup_spin.setFixedWidth(90)
        self.popup_spin.setMinimumHeight(30)
        self.popup_spin.setAlignment(QtCore.Qt.AlignCenter)
        popup_row.addWidget(self.popup_spin)
        popup_row.addStretch(1)
        self.popup_widget = QtWidgets.QWidget()
        self.popup_widget.setLayout(popup_row)
        col.addWidget(self.popup_widget)

        self._on_mode_changed()  # 자동 전용 옵션 표시 상태 반영

        # 음성 안내(TTS) — 수동/자동 모드 모두에서 동작 (항상 표시)
        self.tts_chk = QtWidgets.QCheckBox(
            "처리 완료 시 음성 안내 (이름+상태 읽기)"
        )
        self.tts_chk.setChecked(
            self.cfg.get("tts_enabled", False)
        )
        if not tts.is_available():
            self.tts_chk.setEnabled(False)
            self.tts_chk.setText(
                "음성 안내 — pyttsx3 미설치 (pip install pyttsx3)"
            )
        col.addWidget(self.tts_chk)

        self.fp16_chk = QtWidgets.QCheckBox(
            "GPU FP16 가속 사용"
        )
        self.fp16_chk.setChecked(
            self.cfg.get("use_fp16", True)
        )
        # GPU(CUDA)가 없는 환경(CPU 빌드)에서는 의미가 없으므로 숨긴다
        try:
            import torch

            _has_cuda = torch.cuda.is_available()
        except Exception:
            _has_cuda = False
        if _has_cuda:
            col.addWidget(self.fp16_chk)
        else:
            self.fp16_chk.setVisible(False)

        # 저장
        save = QtWidgets.QPushButton(
            "설정 저장 (DB 없으면 자동 생성)"
        )
        save.setProperty("accent", True)
        save.setMinimumHeight(42)
        save.clicked.connect(self._save_settings)
        col.addWidget(save)

        note = QtWidgets.QLabel(
            "DB 경로를 바꾸면 해당 위치에 DB가 없을 경우 자동으로 생성됩니다.\n"
            "설정은 실행 파일 옆 config.json 에 저장되어 다음 실행 시 자동 적용됩니다."
        )
        note.setProperty("muted", True)
        note.setWordWrap(True)
        col.addWidget(note)

        wrap = QtWidgets.QVBoxLayout()
        wrap.addWidget(card)
        wrap.addStretch(1)
        ww = QtWidgets.QWidget()
        ww.setLayout(wrap)
        outer_h.addWidget(ww)
        outer_h.addStretch(1)
        return outer

    def _model_status_text(self) -> str:
        if self.engine.is_loaded():
            dev = (
                "GPU"
                if self.engine.device
                and self.engine.device.type == "cuda"
                else "CPU"
            )
            return f"✅ 모델 준비됨 ({dev})"
        err = self.engine.get_load_error()
        if err:
            return f"⚠ 모델 로드 실패: {err}"
        return "❔ 아직 확인되지 않음 — 버튼을 눌러 확인하세요."

    def _check_model(self):
        """수동으로 AI 모델 로드를 시도(다운로드 포함). 스레드로 돌려 UI가 멈추지 않게 함."""
        if (
            getattr(self, "model_check_thread", None)
            and self.model_check_thread.isRunning()
        ):
            return
        self._model_dialog = LoadingDialog(
            "AI 모델 확인 중...\n(최초 1회는 가중치 다운로드로 시간이 걸립니다)",
            self,
        )
        self.model_check_thread = ModelCheckThread(
            self.engine
        )
        self.model_check_thread.progress.connect(
            self._on_model_progress
        )
        self.model_check_thread.done.connect(
            self._on_model_checked
        )
        self.model_check_thread.start()
        self._model_dialog.exec_()  # 완료되면 _on_model_checked에서 닫음

    def _on_model_progress(self, msg: str):
        if getattr(self, "_model_dialog", None):
            self._model_dialog.label.setText(msg)
        self.statusBar().showMessage(msg)

    def _on_model_checked(self, ok: bool, msg: str):
        if getattr(self, "_model_dialog", None):
            self._model_dialog.accept()
            self._model_dialog = None
        self.model_status_label.setText(
            self._model_status_text()
        )
        if ok:
            QtWidgets.QMessageBox.information(
                self, "모델 확인", msg
            )
        else:
            QtWidgets.QMessageBox.warning(
                self, "모델 확인", msg
            )

    def _scan_cameras(self):
        """수동 카메라 검색: 로딩창을 띄우고 스레드에서 검색 (UI 멈춤/튕김 방지)."""
        self._run_camera_scan(show_warning_if_none=True)

    def _run_camera_scan(self, show_warning_if_none=False):
        """카메라 검색을 스레드로 수행하고, 그동안 모달 로딩창을 표시."""
        # 이미 검색 중이면 무시
        if (
            getattr(self, "cam_scan_thread", None)
            and self.cam_scan_thread.isRunning()
        ):
            return
        self._scan_warn = show_warning_if_none
        self._scan_dialog = LoadingDialog(
            "카메라를 검색하는 중입니다...", self
        )
        self.cam_scan_thread = CameraScanThread(5)
        self.cam_scan_thread.done.connect(
            self._on_cameras_found
        )
        self.cam_scan_thread.start()
        # 안전장치: 검색이 비정상적으로 오래 걸리면 로딩창을 강제로 닫음
        QtCore.QTimer.singleShot(
            15000, self._force_close_scan_dialog
        )
        self._scan_dialog.exec_()  # 검색이 끝나면 _on_cameras_found에서 닫음

    def _force_close_scan_dialog(self):
        dlg = getattr(self, "_scan_dialog", None)
        if dlg is not None:
            dlg.accept()
            self._scan_dialog = None
            self.statusBar().showMessage(
                "카메라 검색 시간 초과 — 설정에서 다시 시도하세요."
            )

    def _test_camera(self):
        from app.face_engine import test_camera

        idx = self.cam_combo.currentData()
        if idx is None or idx < 0:
            QtWidgets.QMessageBox.warning(
                self, "테스트", "먼저 카메라를 선택하세요."
            )
            return
        self.statusBar().showMessage(
            f"카메라 {idx} 테스트 중..."
        )
        QtWidgets.QApplication.processEvents()
        ok, res = test_camera(idx)
        if ok:
            QtWidgets.QMessageBox.information(
                self,
                "카메라 테스트",
                f"카메라 {idx} 정상 동작.\n해상도: {res[0]} x {res[1]}",
            )
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "카메라 테스트",
                f"카메라 {idx} 를 열 수 없습니다.\n"
                "다른 앱이 사용 중이거나 권한이 없는지 확인하세요.",
            )

    def _browse_db(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "DB 파일 선택/생성",
            self.cfg["db_path"],
            "SQLite DB (*.db)",
            options=QtWidgets.QFileDialog.DontConfirmOverwrite,
        )
        if path:
            if not path.endswith(".db"):
                path += ".db"
            self.db_path_edit.setText(path)

    def _on_mode_changed(self):
        """자동 모드 선택 시에만 체류 시간·미리보기·팝업 옵션을 보이게."""
        data = self.mode_combo.currentData()
        is_auto = bool(data and data[1])
        self.dwell_widget.setVisible(is_auto)
        self.preview_chk.setVisible(is_auto)
        self.popup_chk.setVisible(is_auto)
        self.popup_widget.setVisible(is_auto)

    def _save_settings(self):
        new_path = self.db_path_edit.text().strip()
        self.cfg["db_path"] = new_path
        self.cfg["sim_threshold"] = self.thr_spin.value()
        self.cfg["vote_n"] = self.vote_spin.value()
        # FP16은 GPU 환경에서만 의미가 있으므로, 체크박스가 보일 때만 저장
        if self.fp16_chk.isVisible():
            self.cfg["use_fp16"] = self.fp16_chk.isChecked()
        cam_data = self.cam_combo.currentData()
        if cam_data is not None and cam_data >= 0:
            self.cfg["camera_index"] = cam_data
        mode, auto = self.mode_combo.currentData() or (
            "full",
            False,
        )
        self.cfg["attendance_mode"] = mode
        self.cfg["auto_mode"] = auto
        self.cfg["dwell_sec"] = self.dwell_spin.value()
        self.cfg["auto_preview"] = (
            self.preview_chk.isChecked()
        )
        self.cfg["auto_popup"] = self.popup_chk.isChecked()
        self.cfg["popup_sec"] = self.popup_spin.value()
        self.cfg["tts_enabled"] = self.tts_chk.isChecked()
        appcfg.save_config(self.cfg)

        # 인식 스레드가 옛 DB를 쓰는 중이면 먼저 멈춘다 (닫힌 DB 접근 방지)
        was_running = bool(
            self.rec_thread and self.rec_thread.isRunning()
        )
        if was_running:
            self.rec_thread.stop()
            self.rec_thread = None
            self.start_btn.setText("인식 시작")

        try:
            if self.db:
                self.db.close()
            self.db = AttendanceDB(new_path, mode)
            self.statusBar().showMessage(f"DB: {new_path}")
            self.refresh_employees()
            self.refresh_logs()
            # 인식 중이었다면 새 설정·새 DB로 다시 시작
            if was_running:
                self.toggle_recognize()
            QtWidgets.QMessageBox.information(
                self,
                "저장 완료",
                f"설정이 저장되었습니다.\nDB: {new_path}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "DB 오류", str(e)
            )

    def closeEvent(self, e):
        if self.rec_thread:
            self.rec_thread.stop()
        if (
            self.cam_scan_thread
            and self.cam_scan_thread.isRunning()
        ):
            self.cam_scan_thread.wait(3000)
        if self.db:
            self.db.close()
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(LIGHT_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
