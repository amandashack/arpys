import logging

from gui.widgets.fmControlsWidget import FermiMapControlsWidget
from gui.widgets.fmImageWidget import FMImageWidget
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QFileDialog

log = logging.getLogger('pydm')
log.setLevel('CRITICAL')


class FermiMapView(QWidget):
    def __init__(self, context, signals):
        super(FermiMapView, self).__init__()
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
        self.imageWidget = FMImageWidget(self.context, self.signals, "fermi_map")

    def create_editor_widget(self):
        self.editorWidget = FermiMapControlsWidget(self.context, self.signals)

    def save_figure(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        if filePath == "":
            return
        self.imageWidget.canvas.fig.savefig(filePath)
