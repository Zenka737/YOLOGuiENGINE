import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSlider, QGroupBox, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
from core.detector import DetectorThread, _list_cameras

BUILTIN_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt"]


class DetectionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = DetectorThread()
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.status_update.connect(self._on_status)
        self.thread.error.connect(self._on_error)
        self._build_ui()
        self._rebuild_camera_list()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(10)
        left = QVBoxLayout()
        left.setSpacing(8)

        grp_model = QGroupBox("Модель")
        g1 = QVBoxLayout(grp_model)
        self.model_combo = QComboBox()
        self.model_combo.addItems(BUILTIN_MODELS)
        g1.addWidget(self.model_combo)
        btn_model_file = QPushButton("📂 Выбрать файл модели")
        btn_model_file.clicked.connect(self._browse_model)
        g1.addWidget(btn_model_file)
        left.addWidget(grp_model)

        grp_src = QGroupBox("Источник")
        g2 = QVBoxLayout(grp_src)
        self.src_combo = QComboBox()
        self._camera_indices = []
        self.src_combo.currentIndexChanged.connect(self._on_source_change)
        g2.addWidget(self.src_combo)
        btn_refresh_cam = QPushButton("🔄 Найти камеры")
        btn_refresh_cam.clicked.connect(self._rebuild_camera_list)
        g2.addWidget(btn_refresh_cam)
        self.btn_browse_src = QPushButton("📂 Выбрать файл")
        self.btn_browse_src.clicked.connect(self._browse_source)
        self.btn_browse_src.setVisible(False)
        g2.addWidget(self.btn_browse_src)
        self.lbl_src = QLabel("")
        self.lbl_src.setWordWrap(True)
        g2.addWidget(self.lbl_src)
        left.addWidget(grp_src)

        grp_params = QGroupBox("Параметры")
        g3 = QVBoxLayout(grp_params)
        g3.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_lbl = QLabel("0.25")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_lbl.setText(f"{v/100:.2f}"))
        row = QHBoxLayout()
        row.addWidget(self.conf_slider)
        row.addWidget(self.conf_lbl)
        g3.addLayout(row)
        g3.addWidget(QLabel("IoU:"))
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(45)
        self.iou_lbl = QLabel("0.45")
        self.iou_slider.valueChanged.connect(
            lambda v: self.iou_lbl.setText(f"{v/100:.2f}"))
        row2 = QHBoxLayout()
        row2.addWidget(self.iou_slider)
        row2.addWidget(self.iou_lbl)
        g3.addLayout(row2)
        left.addWidget(grp_params)

        grp_ctrl = QGroupBox("Управление")
        g4 = QVBoxLayout(grp_ctrl)
        self.btn_start = QPushButton("▶  Старт")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._start)
        self.btn_pause = QPushButton("⏸  Пауза")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_stop = QPushButton("⏹  Стоп")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        g4.addWidget(self.btn_start)
        g4.addWidget(self.btn_pause)
        g4.addWidget(self.btn_stop)
        left.addWidget(grp_ctrl)

        left.addStretch()
        self.status_lbl = QLabel("Статус: ожидание")
        self.status_lbl.setWordWrap(True)
        left.addWidget(self.status_lbl)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(240)
        main.addWidget(left_widget)

        self.video_label = QLabel("Нажмите Старт для запуска детекции")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "border: 2px solid #2d4a7a; border-radius: 8px; background: #0d1b2e;")
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
        main.addWidget(self.video_label)

    def _rebuild_camera_list(self):
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        self._camera_indices = []
        cameras = _list_cameras()
        if cameras:
            for i in cameras:
                self.src_combo.addItem(f"Камера {i}")
                self._camera_indices.append(i)
        else:
            self.src_combo.addItem("Камеры не найдены")
            self._camera_indices.append(None)
        self.src_combo.addItem("Видеофайл")
        self._camera_indices.append("video")
        self.src_combo.addItem("Изображение")
        self._camera_indices.append("image")
        self.src_combo.blockSignals(False)
        self.src_combo.setCurrentIndex(0)
        self._on_source_change(0)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать модель", "", "YOLO Models (*.pt *.onnx)")
        if path:
            self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)

    def _on_source_change(self, idx):
        if idx < 0 or idx >= len(self._camera_indices):
            return
        val = self._camera_indices[idx]
        is_file = val in ("video", "image")
        self.btn_browse_src.setVisible(is_file)
        if not is_file:
            self.lbl_src.setText(
                "" if val is None else f"Источник: Камера {val}")

    def _browse_source(self):
        idx = self.src_combo.currentIndex()
        val = self._camera_indices[idx] if idx < len(self._camera_indices) else None
        if val == "video":
            path, _ = QFileDialog.getOpenFileName(
                self, "Видео", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Изображение", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.lbl_src.setText(f"Файл: {os.path.basename(path)}")
            self._src_path = path

    def _get_source(self):
        idx = self.src_combo.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            return 0
        val = self._camera_indices[idx]
        if val in ("video", "image"):
            return getattr(self, "_src_path", 0)
        return val if val is not None else 0

    def _start(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        self.thread.load_model(self.model_combo.currentText())
        self.thread.set_source(self._get_source())
        self.thread.conf = self.conf_slider.value() / 100
        self.thread.iou = self.iou_slider.value() / 100
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def _pause(self):
        self.thread.pause()
        self.btn_pause.setText(
            "▶  Продолжить" if self.thread.paused else "⏸  Пауза")

    def _stop(self):
        self.thread.stop()
        self.thread.wait()
        self.video_label.setText("Детекция остановлена")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Пауза")
        self.btn_stop.setEnabled(False)

    @pyqtSlot(QImage)
    def _on_frame(self, qimg):
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    @pyqtSlot(str)
    def _on_status(self, msg):
        self.status_lbl.setText(f"Статус: {msg}")

    @pyqtSlot(str)
    def _on_error(self, msg):
        self.status_lbl.setText(f"⚠ {msg}")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
