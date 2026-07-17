import os
import shutil
import random
import yaml
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QListWidget, QComboBox, QGroupBox, QFileDialog, QInputDialog,
    QSizePolicy, QMessageBox, QDialog, QSpinBox, QLineEdit, QFormLayout,
    QDialogButtonBox
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont


class AnnotationCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(
            Qt.AlignmentFlag.AlignTop
            | Qt.AlignmentFlag.AlignLeft)
        self.setStyleSheet("border: 2px solid #2d4a7a; background: #0d1b2e;")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)
        self._pixmap_orig = None
        self._boxes = []
        self._classes = ["object"]
        self._current_class = 0
        self._drawing = False
        self._start = QPoint()
        self._end = QPoint()
        self._colors = [
            QColor(255, 80, 80), QColor(80, 255, 80), QColor(80, 180, 255),
            QColor(255, 200, 0), QColor(200, 80, 255), QColor(0, 255, 200)]

    def load_image(self, path):
        self._pixmap_orig = QPixmap(path)
        self._boxes = []
        self._redraw()

    def set_classes(self, classes):
        self._classes = classes

    def set_current_class(self, idx):
        self._current_class = idx

    def undo_last(self):
        if self._boxes:
            self._boxes.pop()
            self._redraw()

    def clear_boxes(self):
        self._boxes.clear()
        self._redraw()

    def get_yolo_labels(self):
        if not self._pixmap_orig:
            return []
        w, h = self._pixmap_orig.width(), self._pixmap_orig.height()
        lines = []
        for b in self._boxes:
            cx = ((b["x1"] + b["x2"]) / 2) / w
            cy = ((b["y1"] + b["y2"]) / 2) / h
            bw = abs(b["x2"] - b["x1"]) / w
            bh = abs(b["y2"] - b["y1"]) / h
            lines.append(
                f"{b['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    def _scaled_size(self):
        if not self._pixmap_orig:
            return self.width(), self.height()
        pix = self._pixmap_orig.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation)
        return pix.width(), pix.height()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._pixmap_orig:
            self._drawing = True
            self._start = e.position().toPoint()
            self._end = self._start

    def mouseMoveEvent(self, e):
        if self._drawing:
            self._end = e.position().toPoint()
            self._redraw()

    def mouseReleaseEvent(self, e):
        if self._drawing and self._pixmap_orig:
            self._drawing = False
            x1, y1 = self._start.x(), self._start.y()
            x2, y2 = self._end.x(), self._end.y()
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                pw, ph = self._pixmap_orig.width(), self._pixmap_orig.height()
                disp_w, disp_h = self._scaled_size()
                sx, sy = pw / disp_w, ph / disp_h
                self._boxes.append({
                    "class_id": self._current_class,
                    "x1": int(min(x1, x2) * sx), "y1": int(min(y1, y2) * sy),
                    "x2": int(max(x1, x2) * sx), "y2": int(max(y1, y2) * sy),
                })
            self._redraw()

    def _redraw(self):
        if not self._pixmap_orig:
            return
        pix = self._pixmap_orig.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        pw, ph = self._pixmap_orig.width(), self._pixmap_orig.height()
        sx, sy = pix.width() / pw, pix.height() / ph
        painter = QPainter(pix)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        for b in self._boxes:
            color = self._colors[b["class_id"] % len(self._colors)]
            painter.setPen(QPen(color, 2))
            rx, ry = int(b["x1"] * sx), int(b["y1"] * sy)
            rw = int((b["x2"] - b["x1"]) * sx)
            rh = int((b["y2"] - b["y1"]) * sy)
            painter.drawRect(rx, ry, rw, rh)
            cls_name = (
                self._classes[b["class_id"]]
                if b["class_id"] < len(self._classes)
                else str(b["class_id"]))
            painter.fillRect(rx, ry - 16, len(cls_name) * 8 + 4, 16, color)
            painter.setPen(QPen(Qt.GlobalColor.black))
            painter.drawText(rx + 2, ry - 3, cls_name)
        if self._drawing:
            painter.setPen(QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine))
            x1 = min(self._start.x(), self._end.x())
            y1 = min(self._start.y(), self._end.y())
            painter.drawRect(x1, y1,
                             abs(self._end.x() - self._start.x()),
                             abs(self._end.y() - self._start.y()))
        painter.end()
        self.setPixmap(pix)


class DatasetOrganizerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Организовать датасет")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("Папка с исходными фото")
        btn_src = QPushButton("📂")
        btn_src.setFixedWidth(32)
        btn_src.clicked.connect(self._pick_src)
        src_row = QHBoxLayout()
        src_row.addWidget(self.src_edit)
        src_row.addWidget(btn_src)
        form.addRow("Исходные фото:", src_row)

        self.dst_edit = QLineEdit()
        self.dst_edit.setPlaceholderText("Папка датасета (создастся структура)")
        btn_dst = QPushButton("📂")
        btn_dst.setFixedWidth(32)
        btn_dst.clicked.connect(self._pick_dst)
        dst_row = QHBoxLayout()
        dst_row.addWidget(self.dst_edit)
        dst_row.addWidget(btn_dst)
        form.addRow("Датасет (корень):", dst_row)

        self.class_edit = QLineEdit("pringles")
        form.addRow("Название класса:", self.class_edit)

        self.val_spin = QSpinBox()
        self.val_spin.setRange(5, 40)
        self.val_spin.setValue(20)
        self.val_spin.setSuffix(" %")
        form.addRow("Доля val:", self.val_spin)

        layout.addLayout(form)

        self.info_lbl = QLabel("")
        self.info_lbl.setWordWrap(True)
        layout.addWidget(self.info_lbl)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._run)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _pick_src(self):
        p = QFileDialog.getExistingDirectory(self, "Папка с фото")
        if p:
            self.src_edit.setText(p)

    def _pick_dst(self):
        p = QFileDialog.getExistingDirectory(self, "Корень датасета")
        if p:
            self.dst_edit.setText(p)

    def _run(self):
        src = self.src_edit.text().strip()
        dst = self.dst_edit.text().strip()
        cls_name = self.class_edit.text().strip() or "object"
        val_pct = self.val_spin.value() / 100

        if not src or not os.path.isdir(src):
            self.info_lbl.setText("❌ Укажите папку с фото")
            return
        if not dst:
            self.info_lbl.setText("❌ Укажите папку датасета")
            return

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [
            f for f in os.listdir(src)
            if os.path.splitext(f)[1].lower() in exts]

        if not images:
            self.info_lbl.setText("❌ В папке нет изображений")
            return

        random.shuffle(images)
        n_val = max(1, int(len(images) * val_pct))
        val_files = images[:n_val]
        train_files = images[n_val:]

        for split in ("train", "val"):
            os.makedirs(os.path.join(dst, "images", split), exist_ok=True)
            os.makedirs(os.path.join(dst, "labels", split), exist_ok=True)

        for fname in train_files:
            shutil.copy2(
                os.path.join(src, fname),
                os.path.join(dst, "images", "train", fname))
        for fname in val_files:
            shutil.copy2(
                os.path.join(src, fname),
                os.path.join(dst, "images", "val", fname))

        yaml_path = os.path.join(dst, "data.yaml")
        cfg = {
            "train": os.path.join(dst, "images", "train").replace("\\", "/"),
            "val": os.path.join(dst, "images", "val").replace("\\", "/"),
            "nc": 1,
            "names": [cls_name],
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        self.info_lbl.setText(
            f"✅ Готово!\n"
            f"train: {len(train_files)} фото\n"
            f"val: {len(val_files)} фото\n"
            f"data.yaml: {yaml_path}")
        self._yaml_path = yaml_path
        self._dst = dst

    def result_paths(self):
        return getattr(self, "_dst", ""), getattr(self, "_yaml_path", "")


class AnnotationTab(QWidget):
    def __init__(self):
        super().__init__()
        self._images = []
        self._current_idx = -1
        self._folder = ""
        self._build_ui()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(8)
        left = QVBoxLayout()

        grp_dataset = QGroupBox("Датасет")
        g0 = QVBoxLayout(grp_dataset)
        btn_organize = QPushButton("⚙ Организовать датасет")
        btn_organize.clicked.connect(self._organize_dataset)
        g0.addWidget(btn_organize)
        left.addWidget(grp_dataset)

        grp_folder = QGroupBox("Папка для разметки")
        g1 = QVBoxLayout(grp_folder)
        btn_open = QPushButton("📂 Открыть папку")
        btn_open.clicked.connect(self._open_folder)
        g1.addWidget(btn_open)
        left.addWidget(grp_folder)

        grp_cls = QGroupBox("Классы")
        g2 = QVBoxLayout(grp_cls)
        self.class_combo = QComboBox()
        self.class_combo.addItems(["object"])
        self.class_combo.currentIndexChanged.connect(
            lambda i: self.canvas.set_current_class(i))
        g2.addWidget(self.class_combo)
        btn_add_cls = QPushButton("➕ Добавить класс")
        btn_add_cls.clicked.connect(self._add_class)
        g2.addWidget(btn_add_cls)
        left.addWidget(grp_cls)

        grp_nav = QGroupBox("Навигация")
        g3 = QVBoxLayout(grp_nav)
        self.img_list = QListWidget()
        self.img_list.setMaximumHeight(180)
        self.img_list.currentRowChanged.connect(self._load_image)
        g3.addWidget(self.img_list)
        nav_row = QHBoxLayout()
        btn_prev = QPushButton("◀")
        btn_prev.clicked.connect(lambda: self._navigate(-1))
        btn_next = QPushButton("▶")
        btn_next.clicked.connect(lambda: self._navigate(1))
        nav_row.addWidget(btn_prev)
        nav_row.addWidget(btn_next)
        g3.addLayout(nav_row)
        left.addWidget(grp_nav)

        grp_actions = QGroupBox("Действия")
        g4 = QVBoxLayout(grp_actions)
        btn_undo = QPushButton("↩ Отменить")
        btn_undo.clicked.connect(lambda: self.canvas.undo_last())
        btn_clear = QPushButton("🗑 Очистить")
        btn_clear.clicked.connect(lambda: self.canvas.clear_boxes())
        btn_save = QPushButton("💾 Сохранить")
        btn_save.setObjectName("btn_start")
        btn_save.clicked.connect(self._save_label)
        btn_save_all = QPushButton("💾 Сохранить всё")
        btn_save_all.clicked.connect(self._save_all)
        g4.addWidget(btn_undo)
        g4.addWidget(btn_clear)
        g4.addWidget(btn_save)
        g4.addWidget(btn_save_all)
        left.addWidget(grp_actions)

        self.status_lbl = QLabel("Откройте папку")
        self.status_lbl.setWordWrap(True)
        left.addWidget(self.status_lbl)
        left.addStretch()
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(220)
        main.addWidget(left_w)
        self.canvas = AnnotationCanvas()
        main.addWidget(self.canvas)

    def _organize_dataset(self):
        dlg = DatasetOrganizerDialog(self)
        dlg.exec()
        dst, yaml_path = dlg.result_paths()
        if dst:
            reply = QMessageBox.question(
                self, "Открыть для разметки?",
                f"Открыть images/train для разметки?\n{os.path.join(dst, 'images', 'train')}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self._load_folder(os.path.join(dst, "images", "train"))

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка")
        if folder:
            self._load_folder(folder)

    def _load_folder(self, folder):
        self._folder = folder
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self._images = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in exts]
        self.img_list.clear()
        self.img_list.addItems([os.path.basename(p) for p in self._images])
        if self._images:
            self.img_list.setCurrentRow(0)
        self.status_lbl.setText(f"Загружено: {len(self._images)} изображений")

    def _load_image(self, idx):
        if 0 <= idx < len(self._images):
            self._current_idx = idx
            self.canvas.load_image(self._images[idx])

    def _navigate(self, delta):
        new_idx = self._current_idx + delta
        if 0 <= new_idx < len(self._images):
            self.img_list.setCurrentRow(new_idx)

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "Класс", "Название:")
        if ok and name.strip():
            self.class_combo.addItem(name.strip())
            self.canvas.set_classes([
                self.class_combo.itemText(i)
                for i in range(self.class_combo.count())])

    def _label_path_for(self, img_path):
        """Return labels/... path mirroring the images/... structure."""
        img_path = os.path.normpath(img_path)
        parts = img_path.replace("\\", "/").split("/")
        # replace 'images' segment with 'labels'
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
        label_dir = "/".join(parts[:-1])
        os.makedirs(label_dir, exist_ok=True)
        return os.path.join(label_dir, os.path.splitext(parts[-1])[0] + ".txt")

    def _save_label(self):
        if self._current_idx < 0:
            return
        img_path = self._images[self._current_idx]
        label_path = self._label_path_for(img_path)
        with open(label_path, "w") as f:
            f.write("\n".join(self.canvas.get_yolo_labels()))
        self.status_lbl.setText(f"Сохранено: {os.path.basename(label_path)}")

    def _save_all(self):
        self._save_label()
        if self._folder:
            classes = [
                self.class_combo.itemText(i)
                for i in range(self.class_combo.count())]
            with open(os.path.join(self._folder, "classes.txt"), "w") as f:
                f.write("\n".join(classes))
        self.status_lbl.setText("Все метки сохранены")
