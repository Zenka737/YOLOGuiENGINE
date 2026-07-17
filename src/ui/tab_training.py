import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QFileDialog,
    QTextEdit, QProgressBar, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt


def _scan_cpu():
    """Always available. Returns list of one (device_id, label)."""
    import platform
    name = platform.processor() or "CPU"
    return [("cpu", name)]


def _scan_gpu():
    """Returns list of (device_id, label) for all CUDA/MPS GPUs."""
    found = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                found.append((str(i), torch.cuda.get_device_name(i)))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            found.append(("mps", "Apple MPS"))
    except Exception:
        pass
    return found


def _scan_npu():
    """Returns list of (device_id, label) for Intel NPU / OpenVINO."""
    found = []
    try:
        import openvino.runtime as ov
        core = ov.Core()
        for dev in core.available_devices:
            if "NPU" in dev or "GNA" in dev:
                found.append((dev, dev))
    except Exception:
        pass
    return found


class TrainThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.params["model"])
            self.log.emit(f"Начало обучения: {self.params}")
            model.train(
                data=self.params["data"],
                epochs=self.params["epochs"],
                imgsz=self.params["imgsz"],
                batch=self.params["batch"],
                lr0=self.params["lr"],
                project=self.params["project"],
                name=self.params["name"],
                device=self.params["device"],
            )
            self.log.emit("Обучение завершено!")
            self.log.emit(
                f"Результаты: {self.params['project']}/{self.params['name']}")
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.train_thread = None
        self._build_ui()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(10)
        left = QVBoxLayout()

        grp_data = QGroupBox("Данные")
        g1 = QVBoxLayout(grp_data)
        self.data_lbl = QLabel("data.yaml: не выбран")
        self.data_lbl.setWordWrap(True)
        btn_data = QPushButton("📂 Выбрать data.yaml")
        btn_data.clicked.connect(self._browse_data)
        g1.addWidget(self.data_lbl)
        g1.addWidget(btn_data)
        left.addWidget(grp_data)

        grp_model = QGroupBox("Базовая модель")
        g2 = QVBoxLayout(grp_model)
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        g2.addWidget(self.model_combo)
        left.addWidget(grp_model)

        grp_params = QGroupBox("Параметры обучения")
        g3 = QVBoxLayout(grp_params)

        def row(label, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget)
            return h

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        g3.addLayout(row("Эпохи:", self.epochs_spin))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        g3.addLayout(row("Batch:", self.batch_spin))
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(
            ["320", "416", "512", "640", "800", "1024", "1280"])
        self.imgsz_combo.setCurrentText("640")
        g3.addLayout(row("imgsz:", self.imgsz_combo))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.01)
        g3.addLayout(row("LR:", self.lr_spin))
        left.addWidget(grp_params)

        # --- Device selector ---
        grp_device = QGroupBox("Устройство")
        gd = QVBoxLayout(grp_device)

        btn_row = QHBoxLayout()
        self._dev_btn_group = QButtonGroup(self)
        for i, (label, dtype) in enumerate([("CPU", "cpu"), ("GPU", "gpu"), ("NPU", "npu")]):
            rb = QRadioButton(label)
            rb.setProperty("devtype", dtype)
            self._dev_btn_group.addButton(rb, i)
            btn_row.addWidget(rb)
        self._dev_btn_group.button(0).setChecked(True)
        gd.addLayout(btn_row)

        self.device_combo = QComboBox()
        self.device_combo.setVisible(False)
        gd.addWidget(self.device_combo)

        self.device_status_lbl = QLabel("")
        self.device_status_lbl.setWordWrap(True)
        self.device_status_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        gd.addWidget(self.device_status_lbl)

        self._dev_btn_group.idClicked.connect(self._on_device_type_changed)
        left.addWidget(grp_device)

        # Trigger initial scan for CPU
        self._on_device_type_changed(0)

        grp_out = QGroupBox("Папка")
        g4 = QVBoxLayout(grp_out)
        self.out_lbl = QLabel("runs/train")
        btn_out = QPushButton("📂 Изменить")
        btn_out.clicked.connect(self._browse_output)
        g4.addWidget(self.out_lbl)
        g4.addWidget(btn_out)
        left.addWidget(grp_out)

        self.btn_train = QPushButton("🚀  Начать обучение")
        self.btn_train.setObjectName("btn_start")
        self.btn_train.clicked.connect(self._start_train)
        left.addWidget(self.btn_train)
        self.btn_stop_train = QPushButton("⏹  Остановить")
        self.btn_stop_train.setObjectName("btn_stop")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self._stop_train)
        left.addWidget(self.btn_stop_train)
        left.addStretch()
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(260)
        main.addWidget(left_w)

        right = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        right.addWidget(QLabel("Прогресс:"))
        right.addWidget(self.progress_bar)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Лог обучения...")
        right.addWidget(self.log_view)
        right_w = QWidget()
        right_w.setLayout(right)
        main.addWidget(right_w)

    def _on_device_type_changed(self, btn_id):
        dtype = self._dev_btn_group.button(btn_id).property("devtype")
        self.device_combo.clear()
        self._devices_raw = []

        if dtype == "cpu":
            results = _scan_cpu()
        elif dtype == "gpu":
            results = _scan_gpu()
        else:
            results = _scan_npu()

        if results:
            for dev_id, label in results:
                self.device_combo.addItem(label)
                self._devices_raw.append(dev_id)
            self.device_combo.setVisible(len(results) > 1)
            self.device_status_lbl.setText("")
            self.device_status_lbl.setStyleSheet("")
        else:
            self.device_combo.setVisible(False)
            name = dtype.upper()
            self.device_status_lbl.setText(f"❌ {name} не найден")
            self.device_status_lbl.setStyleSheet("color: #ff4444; font-weight: bold;")

    def _selected_device(self):
        i = self.device_combo.currentIndex()
        if not self._devices_raw:
            return "cpu"
        if i < 0 or i >= len(self._devices_raw):
            return self._devices_raw[0]
        return self._devices_raw[i]

    def _browse_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "data.yaml", "", "YAML (*.yaml *.yml)")
        if path:
            self._data_path = path
            self.data_lbl.setText(f"data.yaml: {os.path.basename(path)}")

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Папка")
        if path:
            self._out_path = path
            self.out_lbl.setText(path)

    def _validate_yaml(self, path):
        import yaml
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            return f"Не удалось прочитать yaml: {e}"
        missing = [k for k in ("train", "val") if k not in cfg]
        if missing:
            return (
                f"В файле data.yaml отсутствуют обязательные поля: {', '.join(missing)}.\n"
                "Добавьте в yaml:\n"
                "  train: images/train\n"
                "  val:   images/val\n"
                "  nc:    1\n"
                "  names: ['class1']")
        return None

    def _start_train(self):
        data = getattr(self, "_data_path", None)
        if not data:
            self.log_view.append("⚠ Выберите data.yaml")
            return
        err = self._validate_yaml(data)
        if err:
            self.log_view.append(f"❌ {err}")
            return
        if not self._devices_raw:
            self.log_view.append("❌ Выбранное устройство недоступно")
            return
        device = self._selected_device()
        params = {
            "model": self.model_combo.currentText(),
            "data": data,
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": int(self.imgsz_combo.currentText()),
            "lr": self.lr_spin.value(),
            "project": getattr(self, "_out_path", "runs/train"),
            "name": "yologuiengine_train",
            "device": device,
        }
        self.train_thread = TrainThread(params)
        self.train_thread.log.connect(lambda m: self.log_view.append(m))
        self.train_thread.progress.connect(self.progress_bar.setValue)
        self.train_thread.error.connect(
            lambda e: self.log_view.append(f"❌ {e}"))
        self.train_thread.finished.connect(self._on_train_done)
        self.train_thread.start()
        self.btn_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.log_view.append(f"▶ Обучение запущено на {device.upper()}...")

    def _stop_train(self):
        if self.train_thread:
            self.train_thread.terminate()
        self.log_view.append("⏹ Обучение остановлено")
        self._on_train_done()

    def _on_train_done(self):
        self.btn_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
