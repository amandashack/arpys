from gui.views.dewarperView import DewarperView
from gui.windows.dewarpToolWindowUi import DewarperWindow_Ui
from PyQt5.QtWidgets import QMainWindow


class DewarperWindow(QMainWindow, DewarperWindow_Ui):
    def __init__(self, context, signals, scan_type):
        super(DewarperWindow, self).__init__()
        self.signals = signals
        self.context = context
        self.setupUi(self)
        self.dewarper_view = None
        self.scan_type = scan_type
        self.create_views_and_dialogs()
        self.make_connections()
        self.setCentralWidget(self.dewarper_view)

    def make_connections(self):
        pass

    def create_views_and_dialogs(self):
        self.dewarper_view = DewarperView(self.context, self.signals)

    def set_scan_type(self, st):
        self.dewarper_view.update_scan_type(st)

    def closeEvent(self, e):
        self.close()