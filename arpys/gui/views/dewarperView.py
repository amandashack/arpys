import logging

from gui.widgets.dewarperControlsWidget import DewarperControlsWidget
from gui.widgets.dewarperImageWidget import DewarperImageWidget
from PyQt5.QtWidgets import QVBoxLayout, QWidget

log = logging.getLogger('pydm')
log.setLevel('CRITICAL')


class DewarperView(QWidget):
    def __init__(self, context, signals):
        super(DewarperView, self).__init__()
        self.signals = signals
        self.context = context
        self.camera = ""
        self.mainLayout = QVBoxLayout()
        self.imageWidget = None
        self.editorWidget = None
        self.create_image_widget()
        self.create_editor_widget()
        self.mainLayout.addWidget(self.editorWidget, 25)
        self.mainLayout.addWidget(self.imageWidget, 75)
        self.setLayout(self.mainLayout)
        self.make_connections()

    def make_connections(self):
        pass

    def create_image_widget(self):
        self.imageWidget = DewarperImageWidget(self.context, self.signals)

    def create_editor_widget(self):
        self.editorWidget = DewarperControlsWidget(self.context, self.signals)