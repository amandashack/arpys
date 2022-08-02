from gui.views.fileLoaderView import FileLoaderView
from gui.windows.fileLoaderWindowUi import FileLoaderWindow_Ui
from PyQt5.QtWidgets import QMainWindow


class FileLoaderWindow(QMainWindow, FileLoaderWindow_Ui):
    def __init__(self, context, signals):
        super(FileLoaderWindow, self).__init__()
        self.signals = signals
        self.context = context
        self.setupUi(self)
        self.file_loader_view = None
        self.create_views_and_dialogs()
        self.setCentralWidget(self.file_loader_view)

    def create_views_and_dialogs(self):
        self.file_loader_view = FileLoaderView(self.context, self.signals)

    def closeEvent(self, e):
        pass
