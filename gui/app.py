"""Entry point.

    python3.10 gui/app.py [프로젝트.calib.yaml]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6 import QtWidgets  # noqa: E402

from gui.core.project import Project  # noqa: E402
from gui.ui.main_window import MainWindow  # noqa: E402


def main():
    project = Project.load(sys.argv[1]) if len(sys.argv) > 1 else Project()
    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow(project)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
