import logging

from gui.widgets.fileLoaderWidget import FileLoaderWidget
from PyQt5.QtWidgets import QHBoxLayout, QWidget

log = logging.getLogger('pydm')
log.setLevel('CRITICAL')


class FileLoaderView(QWidget):
    def __init__(self, context, signals):
        super(FileLoaderView, self).__init__()
        self.signals = signals
        self.context = context
        self.fm_filename = ''
        self.cut_filename = ''
        self.hv_scan_filename = ''
        self.mainLayout = QHBoxLayout()
        self.file_loader_widget = None
        self.create_file_loader_widget()
        self.mainLayout.addWidget(self.file_loader_widget)
        self.setLayout(self.mainLayout)
        self.make_connections()

    def make_connections(self):
        pass

    def create_file_loader_widget(self):
        self.file_loader_widget = FileLoaderWidget(self.context, self.signals)
