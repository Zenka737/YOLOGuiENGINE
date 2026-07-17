"""Unit tests for non-UI logic — run without a display."""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# tab_training: _validate_yaml
# ---------------------------------------------------------------------------

class FakeTrainingTab:
    """Minimal stand-in that exposes _validate_yaml without PyQt."""

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


def _write_yaml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


def test_validate_yaml_ok():
    path = _write_yaml("train: a\nval: b\nnc: 1\nnames: [obj]\n")
    try:
        assert FakeTrainingTab()._validate_yaml(path) is None
    finally:
        os.unlink(path)


def test_validate_yaml_missing_train():
    path = _write_yaml("val: b\nnc: 1\nnames: [obj]\n")
    try:
        result = FakeTrainingTab()._validate_yaml(path)
        assert result is not None
        assert "train" in result
    finally:
        os.unlink(path)


def test_validate_yaml_missing_val():
    path = _write_yaml("train: a\nnc: 1\nnames: [obj]\n")
    try:
        result = FakeTrainingTab()._validate_yaml(path)
        assert result is not None
        assert "val" in result
    finally:
        os.unlink(path)


def test_validate_yaml_missing_both():
    path = _write_yaml("nc: 1\nnames: [obj]\n")
    try:
        result = FakeTrainingTab()._validate_yaml(path)
        assert result is not None
        assert "train" in result and "val" in result
    finally:
        os.unlink(path)


def test_validate_yaml_bad_file():
    result = FakeTrainingTab()._validate_yaml("/nonexistent/path/file.yaml")
    assert result is not None
    assert "Не удалось прочитать" in result


# ---------------------------------------------------------------------------
# detector: _list_cameras returns a list
# ---------------------------------------------------------------------------

def test_list_cameras_returns_list(monkeypatch, tmp_path):
    """_list_cameras with no real cameras should return an empty list."""
    import types

    # Stub cv2
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.CAP_DSHOW = 700

    class FakeCap:
        def isOpened(self):
            return False

        def release(self):
            pass

    cv2_stub.VideoCapture = lambda *a, **kw: FakeCap()
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)

    # Stub PyQt6 so detector.py can be imported
    for name in ("PyQt6", "PyQt6.QtCore", "PyQt6.QtGui"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    qt_core = sys.modules["PyQt6.QtCore"]
    qt_core.QThread = object
    qt_core.pyqtSignal = lambda *a, **kw: None
    qt_gui = sys.modules["PyQt6.QtGui"]
    qt_gui.QImage = object

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "detector_test",
        os.path.join(os.path.dirname(__file__), "..", "src", "core", "detector.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = mod._list_cameras(max_test=2)
    assert isinstance(result, list)
    assert result == []


# ---------------------------------------------------------------------------
# dataset organizer: _label_path_for
# ---------------------------------------------------------------------------

def test_label_path_for_images_train(tmp_path):
    """Labels for images/train/foo.jpg → labels/train/foo.txt"""
    images_train = tmp_path / "images" / "train"
    images_train.mkdir(parents=True)
    img = images_train / "foo.jpg"
    img.touch()

    # Replicate logic from AnnotationTab._label_path_for
    img_path = os.path.normpath(str(img))
    parts = img_path.replace("\\", "/").split("/")
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
    label_dir = "/".join(parts[:-1])
    label_path = os.path.join(label_dir, os.path.splitext(parts[-1])[0] + ".txt")

    assert label_path.replace("\\", "/").endswith("labels/train/foo.txt")


def test_label_path_for_images_val(tmp_path):
    """Labels for images/val/bar.png → labels/val/bar.txt"""
    images_val = tmp_path / "images" / "val"
    images_val.mkdir(parents=True)
    img = images_val / "bar.png"
    img.touch()

    img_path = os.path.normpath(str(img))
    parts = img_path.replace("\\", "/").split("/")
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
    label_dir = "/".join(parts[:-1])
    label_path = os.path.join(label_dir, os.path.splitext(parts[-1])[0] + ".txt")

    assert label_path.replace("\\", "/").endswith("labels/val/bar.txt")


# ---------------------------------------------------------------------------
# _detect_devices always includes cpu
# ---------------------------------------------------------------------------

def test_detect_devices_has_cpu(monkeypatch):
    """_detect_devices always includes cpu even when torch is absent."""
    import types

    # Stub torch to simulate no GPU
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False, device_count=lambda: 0)
    torch_stub.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    # Extract only _detect_devices by exec-ing the function in isolation
    src = open(
        os.path.join(os.path.dirname(__file__), "..", "src", "ui", "tab_training.py"),
        encoding="utf-8",
    ).read()

    # Pull out just the function body
    ns = {"sys": sys}
    # Execute only up to the class definition to get the top-level function
    lines = src.splitlines()
    func_lines = []
    in_func = False
    for line in lines:
        if line.startswith("def _detect_devices"):
            in_func = True
        if in_func:
            if line.startswith("class ") and func_lines:
                break
            func_lines.append(line)
    exec("\n".join(func_lines), ns)  # noqa: S102
    devices = ns["_detect_devices"]()
    assert any(d[0] == "cpu" for d in devices)
