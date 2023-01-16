import logging

from gui.widgets.baseControlsWidget import BaseControlsWidget
from gui.widgets.ssImageWidget import SSImageWidget
from PyQt5.QtWidgets import QHBoxLayout, QWidget

log = logging.getLogger('pydm')
log.setLevel('CRITICAL')


class SingleScanView(QWidget):
    def __init__(self, context, signals):
        super(SingleScanView, self).__init__()
        self.signals = signals
        self.context = context
        self.camera = ""
        self.mainLayout = QHBoxLayout()
        self.imageWidget = None
        self.editorWidget = None
        self.create_image_widget()
        self.create_editor_widget()
        self.mainLayout.addWidget(self.imageWidget, 75)
        self.mainLayout.addWidget(self.editorWidget, 25)
        self.setLayout(self.mainLayout)
        self.make_connections()

    def make_connections(self):
        pass

    def create_image_widget(self):
        self.imageWidget = SSImageWidget(self.context, self.signals, "single")

    def create_editor_widget(self):
        self.editorWidget = BaseControlsWidget(self.context, self.signals, "single")
