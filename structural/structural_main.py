"""Separate Windows desktop entry point: python -m structural gui."""
import os
import sys


def main():
    os.environ["QT_API"] = "pyside6"
    from PySide6.QtWidgets import QApplication
    from .gui import StructuralWindow
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("KurtsFEM Structural")
    app.setOrganizationName("KurtsFEM")
    app.setStyle("Fusion")
    window = StructuralWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
