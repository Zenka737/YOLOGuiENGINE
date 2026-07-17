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
from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont

HANDLE_SIZE = 8  # pixels for corner resize handles


class AnnotationCanvas(QLabel):
    """Interactive canvas: draw, select, resize and delete bounding boxes."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setStyleSheet("border: 2px solid #2d4a7a; background: #0d1b2e;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._pixmap_orig = None
        self._boxes = []        # list of {"class": int, "x", "y", "w", "h"} — YOLO normalised
        self._suggestions = []  # same format, rendered dashed
        self._selected = -1
        self._classes = ["object"]
        self._current_class = 0
        self._drag_mode = None  # "draw" | "move" | "resize_NW/NE/SW/SE"
        self._drag_start = QPoint()
        self._drag_cur = QPoint()
        self._drag_box_orig = None
        self._template_box = None
        self._colors = [
            QColor(255, 80, 80), QColor(80, 255, 80), QColor(80, 180, 255),
            QColor(255, 200, 0), QColor(200, 80, 255), QColor(0, 255, 200)]

    # ------------------------------------------------------------------ public

    def load_image(self, path):
        self._pixmap_orig = QPixmap(path)
        self._boxes = []
        self._suggestions = []
        self._selected = -1
        self._drag_mode = None
        self._redraw()

    def load_boxes_from_labels(self, label_path):
        """Populate boxes from an existing YOLO .txt label file."""
        self._boxes = []
        self._selected = -1
        if not os.path.exists(label_path):
            self._redraw()
            return
        with open(label_path, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    self._boxes.append({"class": cls, "x": x, "y": y, "w": w, "h": h})
        self._redraw()

    def set_classes(self, classes):
        self._classes = classes

    def set_current_class(self, idx):
        self._current_class = idx

    def set_suggestions(self, boxes):
        """Display suggestion boxes (dashed orange)."""
        self._suggestions = list(boxes)
        self._redraw()

    def accept_suggestions(self):
        """Move all suggestion boxes into confirmed boxes."""
        self._boxes.extend(self._suggestions)
        self._suggestions = []
        self._redraw()

    def clear_suggestions(self):
        self._suggestions = []
        self._redraw()

    def set_template_highlight(self, box):
        self._template_box = dict(box) if box else None
        self._redraw()

    def undo_last(self):
        if self._boxes:
            self._boxes.pop()
            if self._selected >= len(self._boxes):
                self._selected = -1
            self._redraw()

    def clear_boxes(self):
        self._boxes.clear()
        self._suggestions.clear()
        self._selected = -1
        self._redraw()

    def selected_box(self):
        """Return the currently selected box dict or None."""
        if 0 <= self._selected < len(self._boxes):
            return self._boxes[self._selected]
        return None

    def get_yolo_labels(self):
        lines = []
        for b in self._boxes:
            lines.append(f"{b['class']} {b['x']:.6f} {b['y']:.6f} {b['w']:.6f} {b['h']:.6f}")
        return lines

    # ---------------------------------------------------------------- geometry

    def _disp_size(self):
        """Return (dw, dh) of the scaled image rendered on the canvas."""
        if not self._pixmap_orig:
            return self.width(), self.height()
        pix = self._pixmap_orig.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation)
        return pix.width(), pix.height()

    def _norm_to_disp(self, x, y, w, h):
        """Normalised YOLO box → display pixel rect (rx, ry, rw, rh)."""
        dw, dh = self._disp_size()
        rx = int((x - w / 2) * dw)
        ry = int((y - h / 2) * dh)
        rw = int(w * dw)
        rh = int(h * dh)
        return rx, ry, rw, rh

    def _disp_to_norm(self, rx, ry, rw, rh):
        """Display pixel rect → normalised YOLO box (x, y, w, h)."""
        dw, dh = self._disp_size()
        if dw == 0 or dh == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (rx + rw / 2) / dw, (ry + rh / 2) / dh, rw / dw, rh / dh

    def _hit_corner(self, box, mx, my):
        """Return corner label ('NW','NE','SW','SE') if mouse is near a handle."""
        rx, ry, rw, rh = self._norm_to_disp(box["x"], box["y"], box["w"], box["h"])
        corners = [("NW", (rx, ry)), ("NE", (rx + rw, ry)),
                   ("SW", (rx, ry + rh)), ("SE", (rx + rw, ry + rh))]
        for name, (cx, cy) in corners:
            if abs(mx - cx) <= HANDLE_SIZE and abs(my - cy) <= HANDLE_SIZE:
                return name
        return None

    def _hit_box(self, mx, my):
        """Return index of the topmost box under (mx, my), or -1."""
        for i in range(len(self._boxes) - 1, -1, -1):
            b = self._boxes[i]
            rx, ry, rw, rh = self._norm_to_disp(b["x"], b["y"], b["w"], b["h"])
            if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                return i
        return -1

    def _hit_suggestion(self, mx, my):
        """Return index of suggestion box under (mx, my), or -1."""
        for i in range(len(self._suggestions) - 1, -1, -1):
            b = self._suggestions[i]
            rx, ry, rw, rh = self._norm_to_disp(b["x"], b["y"], b["w"], b["h"])
            if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                return i
        return -1

    # ------------------------------------------------------------ mouse events

    def mousePressEvent(self, e):
        if not self._pixmap_orig:
            return
        mx, my = int(e.position().x()), int(e.position().y())

        # Right-click on suggestion → accept it
        if e.button() == Qt.MouseButton.RightButton:
            si = self._hit_suggestion(mx, my)
            if si >= 0:
                self._boxes.append(self._suggestions.pop(si))
                self._selected = len(self._boxes) - 1
                self._redraw()
            return

        # Left-click on suggestion → reject it
        if e.button() == Qt.MouseButton.LeftButton and self._suggestions:
            si = self._hit_suggestion(mx, my)
            if si >= 0:
                self._suggestions.pop(si)
                self._redraw()
                return

        if e.button() != Qt.MouseButton.LeftButton:
            return

        # Corner handle on selected box?
        if self._selected >= 0:
            corner = self._hit_corner(self._boxes[self._selected], mx, my)
            if corner:
                self._drag_mode = f"resize_{corner}"
                self._drag_start = QPoint(mx, my)
                self._drag_box_orig = dict(self._boxes[self._selected])
                return

        # Click on existing box?
        hit = self._hit_box(mx, my)
        if hit >= 0:
            self._selected = hit
            self._drag_mode = "move"
            self._drag_start = QPoint(mx, my)
            self._drag_box_orig = dict(self._boxes[hit])
            self._redraw()
            return

        # Start drawing a new box
        self._selected = -1
        self._drag_mode = "draw"
        self._drag_start = QPoint(mx, my)
        self._drag_cur = QPoint(mx, my)
        self._redraw()

    def mouseMoveEvent(self, e):
        if self._drag_mode is None:
            return
        mx, my = int(e.position().x()), int(e.position().y())
        dw, dh = self._disp_size()

        if self._drag_mode == "draw":
            self._drag_cur = QPoint(mx, my)

        elif self._drag_mode == "move":
            dx = (mx - self._drag_start.x()) / dw
            dy = (my - self._drag_start.y()) / dh
            orig = self._drag_box_orig
            b = self._boxes[self._selected]
            b["x"] = max(0.0, min(1.0, orig["x"] + dx))
            b["y"] = max(0.0, min(1.0, orig["y"] + dy))

        elif self._drag_mode.startswith("resize_"):
            corner = self._drag_mode[7:]
            orig = self._drag_box_orig
            b = self._boxes[self._selected]
            x1 = orig["x"] - orig["w"] / 2
            y1 = orig["y"] - orig["h"] / 2
            x2 = orig["x"] + orig["w"] / 2
            y2 = orig["y"] + orig["h"] / 2
            nx, ny = mx / dw, my / dh
            if "W" in corner:
                x1 = max(0.0, min(nx, x2 - 0.01))
            if "E" in corner:
                x2 = min(1.0, max(nx, x1 + 0.01))
            if "N" in corner:
                y1 = max(0.0, min(ny, y2 - 0.01))
            if "S" in corner:
                y2 = min(1.0, max(ny, y1 + 0.01))
            b["x"] = (x1 + x2) / 2
            b["y"] = (y1 + y2) / 2
            b["w"] = x2 - x1
            b["h"] = y2 - y1

        self._redraw()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        mx, my = int(e.position().x()), int(e.position().y())

        if self._drag_mode == "draw":
            x1, y1 = self._drag_start.x(), self._drag_start.y()
            x2, y2 = mx, my
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                rx, ry = min(x1, x2), min(y1, y2)
                rw, rh = abs(x2 - x1), abs(y2 - y1)
                nx, ny, nw, nh = self._disp_to_norm(rx, ry, rw, rh)
                self._boxes.append({"class": self._current_class, "x": nx, "y": ny, "w": nw, "h": nh})
                self._selected = len(self._boxes) - 1

        self._drag_mode = None
        self._drag_box_orig = None
        self._redraw()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Delete and self._selected >= 0:
            self._boxes.pop(self._selected)
            self._selected = min(self._selected, len(self._boxes) - 1)
            self._redraw()

    # ----------------------------------------------------------------- drawing

    def _redraw(self):
        if not self._pixmap_orig:
            return
        pix = self._pixmap_orig.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(pix)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))

        for i, b in enumerate(self._boxes):
            color = self._colors[b["class"] % len(self._colors)]
            rx, ry, rw, rh = self._norm_to_disp(b["x"], b["y"], b["w"], b["h"])
            pw = 3 if i == self._selected else 2
            painter.setPen(QPen(color, pw))
            painter.drawRect(rx, ry, rw, rh)
            cls_name = self._classes[b["class"]] if b["class"] < len(self._classes) else str(b["class"])
            painter.fillRect(rx, ry - 16, len(cls_name) * 8 + 4, 16, color)
            painter.setPen(QPen(Qt.GlobalColor.black))
            painter.drawText(rx + 2, ry - 3, cls_name)
            if i == self._selected:
                for hx, hy in [(rx, ry), (rx + rw, ry), (rx, ry + rh), (rx + rw, ry + rh)]:
                    painter.fillRect(hx - HANDLE_SIZE // 2, hy - HANDLE_SIZE // 2,
                                     HANDLE_SIZE, HANDLE_SIZE, color)

        if self._template_box:
            b = self._template_box
            rx, ry, rw, rh = self._norm_to_disp(b["x"], b["y"], b["w"], b["h"])
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine))
            painter.drawRect(rx, ry, rw, rh)
            painter.setPen(QPen(QColor(255, 255, 0), 1))
            painter.drawText(rx + 2, ry + 14, "шаблон")

        for b in self._suggestions:
            color = QColor(255, 165, 0)
            rx, ry, rw, rh = self._norm_to_disp(b["x"], b["y"], b["w"], b["h"])
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            painter.drawRect(rx, ry, rw, rh)
            cls_name = self._classes[b["class"]] if b["class"] < len(self._classes) else str(b["class"])
            painter.setPen(QPen(color, 1))
            painter.drawText(rx + 2, ry + 14, f"? {cls_name}")

        if self._drag_mode == "draw":
            painter.setPen(QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine))
            x1 = min(self._drag_start.x(), self._drag_cur.x())
            y1 = min(self._drag_start.y(), self._drag_cur.y())
            painter.drawRect(x1, y1,
                             abs(self._drag_cur.x() - self._drag_start.x()),
                             abs(self._drag_cur.y() - self._drag_start.y()))

        painter.end()
        self.setPixmap(pix)


# ---------------------------------------------------------------------------
# Background YOLO suggestion thread
# ---------------------------------------------------------------------------

class TemplateSuggestThread(QThread):
    """Find regions similar to a template box using cv2.matchTemplate."""

    results = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, image_path, box, cls, threshold=0.55):
        super().__init__()
        self._image_path = image_path
        self._box = box          # normalised YOLO dict
        self._cls = cls
        self._threshold = threshold

    def run(self):
        try:
            import cv2
            import numpy as np
            img = cv2.imread(self._image_path)
            if img is None:
                self.error.emit("Не удалось открыть изображение")
                return
            ih, iw = img.shape[:2]
            b = self._box
            tx = int((b["x"] - b["w"] / 2) * iw)
            ty = int((b["y"] - b["h"] / 2) * ih)
            tw = int(b["w"] * iw)
            th = int(b["h"] * ih)
            tx, ty = max(0, tx), max(0, ty)
            tw, th = min(tw, iw - tx), min(th, ih - ty)
            if tw < 8 or th < 8:
                self.error.emit("Бокс слишком мал для поиска")
                return
            tmpl = img[ty:ty + th, tx:tx + tw]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
            boxes = []
            used = []
            for scale in [0.8, 1.0, 1.2]:
                sw = max(8, int(tw * scale))
                sh = max(8, int(th * scale))
                if sw > iw or sh > ih:
                    continue
                t_scaled = cv2.resize(gray_tmpl, (sw, sh))
                res = cv2.matchTemplate(gray, t_scaled, cv2.TM_CCOEFF_NORMED)
                locs = np.where(res >= self._threshold)
                for pt_x, pt_y in zip(locs[1], locs[0]):
                    # suppress duplicates within half template size
                    dup = any(abs(pt_x - ux) < sw // 2 and abs(pt_y - uy) < sh // 2
                              for ux, uy in used)
                    if dup:
                        continue
                    used.append((pt_x, pt_y))
                    cx = (pt_x + sw / 2) / iw
                    cy = (pt_y + sh / 2) / ih
                    nw = sw / iw
                    nh = sh / ih
                    # skip if nearly identical to the original template box
                    if abs(cx - b["x"]) < 0.02 and abs(cy - b["y"]) < 0.02:
                        continue
                    boxes.append({"class": self._cls, "x": cx, "y": cy, "w": nw, "h": nh})
            self.results.emit(boxes)
        except Exception as ex:
            self.error.emit(str(ex))


# ---------------------------------------------------------------------------
# Dataset organizer dialog (unchanged)
# ---------------------------------------------------------------------------

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
        images = [f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in exts]

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
            shutil.copy2(os.path.join(src, fname), os.path.join(dst, "images", "train", fname))
        for fname in val_files:
            shutil.copy2(os.path.join(src, fname), os.path.join(dst, "images", "val", fname))

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
            f"✅ Готово!\ntrain: {len(train_files)} фото\nval: {len(val_files)} фото\ndata.yaml: {yaml_path}")
        self._yaml_path = yaml_path
        self._dst = dst

    def result_paths(self):
        return getattr(self, "_dst", ""), getattr(self, "_yaml_path", "")


# ---------------------------------------------------------------------------
# Main annotation tab
# ---------------------------------------------------------------------------

class AnnotationTab(QWidget):
    def __init__(self):
        super().__init__()
        self._images = []
        self._current_idx = -1
        self._folder = ""
        self._suggest_thread = None
        self._template_box = None   # persists across image navigation
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
        self.class_combo.currentIndexChanged.connect(lambda i: self.canvas.set_current_class(i))
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

        grp_suggest = QGroupBox("Авто-выделение")
        g5 = QVBoxLayout(grp_suggest)
        hint = QLabel("Выдели бокс → «Найти похожие»\nПКМ — принять, ЛКМ — отклонить")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 10px;")
        g5.addWidget(hint)
        self.btn_find = QPushButton("🔍 Найти похожие")
        self.btn_find.clicked.connect(self._find_similar)
        g5.addWidget(self.btn_find)
        self.tmpl_lbl = QLabel("Шаблон: не задан")
        self.tmpl_lbl.setWordWrap(True)
        self.tmpl_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        g5.addWidget(self.tmpl_lbl)
        btn_set_tmpl = QPushButton("📌 Запомнить выделенный")
        btn_set_tmpl.clicked.connect(self._set_template)
        g5.addWidget(btn_set_tmpl)
        left.addWidget(grp_suggest)

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

    # -------------------------------------------------------- dataset / folder

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

    # ---------------------------------------------------------- image loading

    def _load_image(self, idx):
        if 0 <= idx < len(self._images):
            self._current_idx = idx
            path = self._images[idx]
            self.canvas.load_image(path)
            self.canvas.load_boxes_from_labels(self._label_path_for(path))
            self.canvas.clear_suggestions()
            self.canvas.set_template_highlight(self._template_box)

    def _navigate(self, delta):
        new_idx = self._current_idx + delta
        if 0 <= new_idx < len(self._images):
            self.img_list.setCurrentRow(new_idx)

    # --------------------------------------------------------------- classes

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "Класс", "Название:")
        if ok and name.strip():
            self.class_combo.addItem(name.strip())
            self.canvas.set_classes([
                self.class_combo.itemText(i) for i in range(self.class_combo.count())])

    def _current_class_names(self):
        return [self.class_combo.itemText(i) for i in range(self.class_combo.count())]

    # ---------------------------------------------------------- label I/O

    def _label_path_for(self, img_path):
        """Return labels/... path mirroring the images/... structure."""
        img_path = os.path.normpath(img_path)
        parts = img_path.replace("\\", "/").split("/")
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
            with open(os.path.join(self._folder, "classes.txt"), "w") as f:
                f.write("\n".join(self._current_class_names()))
        self.status_lbl.setText("Все метки сохранены")

    # --------------------------------------------------------- YOLO suggest

    def _set_template(self):
        sel = self.canvas.selected_box()
        if sel is None:
            self.status_lbl.setText("⚠ Сначала выделите бокс на холсте")
            return
        self._template_box = dict(sel)
        self.canvas.set_template_highlight(self._template_box)
        cls_name = self._current_class_names()[sel["class"]] if sel["class"] < self.class_combo.count() else "?"
        self.tmpl_lbl.setText(f"Шаблон: {cls_name} ✓")
        self.tmpl_lbl.setStyleSheet("color: #80ff80; font-size: 10px;")
        self.status_lbl.setText("✅ Шаблон сохранён — переходи к следующему кадру")

    def _find_similar(self):
        if self._current_idx < 0:
            self.status_lbl.setText("Откройте изображение")
            return
        # prefer currently selected box, fall back to saved template
        sel = self.canvas.selected_box() or self._template_box
        if sel is None:
            self.status_lbl.setText("⚠ Выделите бокс или нажмите «Запомнить выделенный»")
            return
        if self._suggest_thread and self._suggest_thread.isRunning():
            return
        self.btn_find.setEnabled(False)
        self.status_lbl.setText("Поиск похожих…")
        self.canvas.clear_suggestions()
        self._suggest_thread = TemplateSuggestThread(
            self._images[self._current_idx], sel, sel["class"])
        self._suggest_thread.results.connect(self._on_suggestions)
        self._suggest_thread.error.connect(self._on_suggest_error)
        self._suggest_thread.start()

    def _on_suggestions(self, boxes):
        self.btn_find.setEnabled(True)
        if not boxes:
            self.status_lbl.setText("Похожих не найдено")
            return
        self.canvas.set_suggestions(boxes)
        self.status_lbl.setText(f"Найдено: {len(boxes)} (пунктир). ПКМ — принять, ЛКМ — отклонить")

    def _on_suggest_error(self, msg):
        self.btn_find.setEnabled(True)
        self.status_lbl.setText(f"Ошибка: {msg}")
