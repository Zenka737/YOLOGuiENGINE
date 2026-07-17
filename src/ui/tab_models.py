import os
import shutil
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QGroupBox, QFileDialog,
    QProgressBar, QTextEdit, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

YOLO_MODELS = [
    ("yolov8n.pt", "YOLOv8 Nano — 6 MB, самая быстрая"),
    ("yolov8s.pt", "YOLOv8 Small — 22 MB"),
    ("yolov8m.pt", "YOLOv8 Medium — 52 MB, баланс"),
    ("yolov8l.pt", "YOLOv8 Large — 87 MB, точная"),
    ("yolov8x.pt", "YOLOv8 XLarge — 137 MB, максимальная"),
    ("yolov8n-seg.pt", "YOLOv8 Nano Segmentation"),
    ("yolov8s-seg.pt", "YOLOv8 Small Segmentation"),
    ("yolov8n-pose.pt", "YOLOv8 Nano Pose"),
    ("yolov8s-pose.pt", "YOLOv8 Small Pose"),
]
MODELS_DIR = os.path.join(os.path.expanduser("~"), ".yologuiengine", "models")


class DownloadThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            from ultralytics import YOLO
            self.log.emit(f"Загрузка {self.model_name}...")
            os.makedirs(MODELS_DIR, exist_ok=True)
            dest = os.path.join(MODELS_DIR, self.model_name)
            YOLO(self.model_name)
            src = self.model_name
            if os.path.exists(src) and not os.path.exists(dest):
                shutil.move(src, dest)
            elif not os.path.exists(dest):
                dest = src
            self.log.emit(f"Модель сохранена: {dest}")
            self.progress.emit(100)
            self.finished.emit(dest)
        except Exception as e:
            self.error.emit(str(e))


class ModelsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._download_thread = None
        os.makedirs(MODELS_DIR, exist_ok=True)
        self._build_ui()
        self._refresh_local()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(10)
        left = QVBoxLayout()

        grp_available = QGroupBox("Доступные модели")
        g1 = QVBoxLayout(grp_available)
        self.available_list = QListWidget()
        for name, desc in YOLO_MODELS:
            item = QListWidgetItem(f"{name}\n  {desc}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.available_list.addItem(item)
        g1.addWidget(self.available_list)
        self.btn_download = QPushButton("⬇  Скачать")
        self.btn_download.setObjectName("btn_start")
        self.btn_download.clicked.connect(self._download_selected)
        g1.addWidget(self.btn_download)
        left.addWidget(grp_available)

        grp_import = QGroupBox("Импорт")
        g2 = QVBoxLayout(grp_import)
        btn_import = QPushButton("📂 Импортировать .pt / .onnx")
        btn_import.clicked.connect(self._import_model)
        g2.addWidget(btn_import)
        left.addWidget(grp_import)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left.addWidget(self.progress_bar)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(120)
        self.log_view.setPlaceholderText("Лог...")
        left.addWidget(self.log_view)
        left.addStretch()
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(300)
        main.addWidget(left_w)

        right = QVBoxLayout()
        grp_local = QGroupBox("Локальные модели")
        g3 = QVBoxLayout(grp_local)
        self.local_list = QListWidget()
        g3.addWidget(self.local_list)
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Обновить")
        btn_refresh.clicked.connect(self._refresh_local)
        btn_delete = QPushButton("🗑 Удалить")
        btn_delete.setObjectName("btn_stop")
        btn_delete.clicked.connect(self._delete_selected)
        btn_open_dir = QPushButton("📁 Папка")
        btn_open_dir.clicked.connect(self._open_models_dir)
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(btn_delete)
        btn_row.addWidget(btn_open_dir)
        g3.addLayout(btn_row)
        self.info_lbl = QLabel("")
        self.info_lbl.setWordWrap(True)
        g3.addWidget(self.info_lbl)
        right.addWidget(grp_local)
        right.addStretch()
        right_w = QWidget()
        right_w.setLayout(right)
        main.addWidget(right_w)

    def _refresh_local(self):
        self.local_list.clear()
        if not os.path.isdir(MODELS_DIR):
            return
        for f in sorted(os.listdir(MODELS_DIR)):
            if not f.endswith((".pt", ".onnx")):
                continue
            path = os.path.join(MODELS_DIR, f)
            size_mb = os.path.getsize(path) / 1024 / 1024
            item = QListWidgetItem(f"{f}  ({size_mb:.1f} MB)")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.local_list.addItem(item)
        self.info_lbl.setText(f"Папка: {MODELS_DIR}")

    def _download_selected(self):
        item = self.available_list.currentItem()
        if not item:
            self.log_view.append("⚠ Выберите модель")
            return
        model_name = item.data(Qt.ItemDataRole.UserRole)
        self.progress_bar.setValue(0)
        self._download_thread = DownloadThread(model_name)
        self._download_thread.log.connect(lambda m: self.log_view.append(m))
        self._download_thread.progress.connect(self.progress_bar.setValue)
        self._download_thread.error.connect(
            lambda e: self.log_view.append(f"❌ {e}"))
        self._download_thread.finished.connect(self._on_download_done)
        self._download_thread.start()
        self.btn_download.setEnabled(False)

    def _on_download_done(self, path):
        self.btn_download.setEnabled(True)
        self._refresh_local()
        self.log_view.append(f"✅ {path}")

    def _import_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Импорт", "", "YOLO Models (*.pt *.onnx)")
        if not path:
            return
        dest = os.path.join(MODELS_DIR, os.path.basename(path))
        if os.path.exists(dest):
            self.log_view.append(f"⚠ Уже есть: {os.path.basename(path)}")
            return
        shutil.copy2(path, dest)
        self.log_view.append(f"✅ Импортирован: {os.path.basename(path)}")
        self._refresh_local()

    def _delete_selected(self):
        item = self.local_list.currentItem()
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        name = os.path.basename(path)
        reply = QMessageBox.question(
            self,
            "Удаление",
            f"Удалить {name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            os.remove(path)
            self._refresh_local()
            self.log_view.append(f"🗑 {name}")

    def _open_models_dir(self):
        import subprocess
        try:
            subprocess.Popen(["xdg-open", MODELS_DIR])
        except Exception:
            self.info_lbl.setText(f"Папка: {MODELS_DIR}")
