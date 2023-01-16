from gui.views.edcView import EDCView
from gui.windows.edcToolWindowUi import EDCWindow_Ui
from PyQt5.QtWidgets import QMainWindow


class EDCWindow(QMainWindow, EDCWindow_Ui):
    def __init__(self, context, signals, scan_type):
        super(EDCWindow, self).__init__()
        self.signals = signals
        self.context = context
        self.setupUi(self)
        self.EDC_view = None
        self.scan_type = scan_type
        self.create_views_and_dialogs()
        self.make_connections()
        self.setCentralWidget(self.EDC_view)

    def make_connections(self):
        pass

    def create_views_and_dialogs(self):
        self.EDC_view = EDCView(self.context, self.signals)

    def set_scan_type(self, st):
        self.EDC_view.update_scan_type(st)

    def closeEvent(self, e):
        self.close()