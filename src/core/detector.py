import os
import platform
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


def _camera_backend():
    """Pick the OpenCV capture backend appropriate for the current OS."""
    system = platform.system()
    default = getattr(cv2, "CAP_ANY", 0)
    if system == "Windows":
        return getattr(cv2, "CAP_DSHOW", default)
    if system == "Darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", default)
    return default


def _is_cuda_device(device) -> bool:
    return isinstance(device, str) and (device.isdigit() or device.startswith("cuda"))


def _is_rtx_device(device) -> bool:
    """Check whether the given CUDA device index is an RTX card (has tensor cores)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        idx = int(device) if str(device).isdigit() else torch.cuda.current_device()
        return "rtx" in torch.cuda.get_device_name(idx).lower()
    except Exception:
        return False


def _enable_tensor_cores():
    """Enable TF32/tensor-core math for RTX GPUs (Turing/Ampere/Ada, RTX 20xx+)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _list_cameras(max_test=4):
    """Return list of working camera indices, suppressing OpenCV stderr."""
    found = []
    backend = _camera_backend()
    devnull = open(os.devnull, "w")
    old_stderr = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    try:
        for i in range(max_test):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                found.append(i)
                cap.release()
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        devnull.close()
    return found


class DetectorThread(QThread):
    frame_ready = pyqtSignal(QImage)
    status_update = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.source = 0
        self.running = False
        self.paused = False
        self.conf = 0.25
        self.iou = 0.45
        self.device = "cpu"

    def load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.status_update.emit(f"Модель загружена: {model_path}")
        except Exception as e:
            self.error.emit(f"Ошибка загрузки модели: {e}")

    def set_source(self, source):
        try:
            self.source = int(source)
        except (ValueError, TypeError):
            self.source = source

    def run(self):
        if self.model is None:
            self.error.emit("Модель не загружена")
            return
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error.emit(f"Не удалось открыть источник: {self.source}")
            return
        self.running = True
        self.status_update.emit("Детекция запущена")
        use_half = _is_cuda_device(self.device) and _is_rtx_device(self.device)
        if use_half:
            _enable_tensor_cores()
        while self.running:
            if self.paused:
                self.msleep(50)
                continue
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            results = self.model(frame, conf=self.conf, iou=self.iou,
                                 device=self.device, half=use_half, verbose=False)
            annotated = results[0].plot()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qimg.copy())
        cap.release()
        self.status_update.emit("Детекция остановлена")

    def stop(self):
        self.running = False
        self.paused = False

    def pause(self):
        self.paused = not self.paused
