"""Wizard steps.

Each step is a page that reads and writes the shared Project. A step reports
whether it is satisfied; the navigator uses that to decide what the user is
allowed to move on to, and to show the tick marks down the left side.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from gui.core.project import Project


class StepPage(QtWidgets.QWidget):
    """Base class for a wizard page."""

    # Emitted whenever this page's completeness may have changed.
    changed = QtCore.Signal()

    title = "Step"
    subtitle = ""

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project

    def is_complete(self) -> bool:
        """True when the user may move past this step."""
        return True

    def status_text(self) -> str:
        """One line shown next to the step name in the navigator."""
        return ""

    def on_enter(self):
        """Called each time the page becomes visible."""

    def shutdown(self):
        """Stop any background work. Called once when the window closes."""


class PlaceholderPage(StepPage):
    """A step that is not built yet, so the shape of the app is visible."""

    def __init__(self, title: str, subtitle: str, note: str, project: Project, parent=None):
        super().__init__(project, parent)
        self.title = title
        self.subtitle = subtitle

        label = QtWidgets.QLabel(note)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("color: palette(mid); font-size: 14px;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)

    def is_complete(self) -> bool:
        return False

    def status_text(self) -> str:
        return "미구현"
