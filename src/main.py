import sys
import os
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("YOLOGuiENGINE")
    app.setApplicationVersion("1.0.0")
    app.setStyle("Fusion")
    qss_path = os.path.join(os.path.dirname(__file__), "ui", "style.qss")
    if os.path.exists(qss_path):
        app.setStyleSheet(open(qss_path).read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
