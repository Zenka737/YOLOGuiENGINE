from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QStatusBar
from PyQt6.QtCore import Qt
from ui.tab_detection import DetectionTab
from ui.tab_training import TrainingTab
from ui.tab_annotation import AnnotationTab
from ui.tab_models import ModelsTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOGuiENGINE v1.0")
        self.setMinimumSize(1100, 720)
        self.resize(1200, 780)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 6)
        layout.setSpacing(6)
        title = QLabel("🎯  YOLOGuiENGINE")
        title.setObjectName("label_title")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)
        tabs = QTabWidget()
        tabs.addTab(DetectionTab(), "🔍  Детекция")
        tabs.addTab(TrainingTab(), "🏋️  Обучение")
        tabs.addTab(AnnotationTab(), "✏️  Аннотация")
        tabs.addTab(ModelsTab(), "📦  Модели")
        layout.addWidget(tabs)
        self.status = QStatusBar()
        self.status.showMessage("Готов к работе")
        self.setStatusBar(self.status)
