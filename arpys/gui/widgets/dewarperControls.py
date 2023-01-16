import logging

from gui.widgets.dewarperControlsUi import DewarperControls_Ui
# from users.ajshack.hv_scan_dev_ssrl import fix_array
from PyQt5.QtWidgets import QFrame, QMessageBox, QFileDialog
import sys, types, importlib.util
from collections import defaultdict
from pathlib import Path
from pathlib import Path

log = logging.getLogger(__name__)

BASEDIR = Path(__file__).resolve().parents[2].joinpath('loaders')
DATADIR = Path(__file__).resolve().parents[4].joinpath('data/ssrl_071522')


# TODO: if the graph loads, close the window, if not give another popup
#  that says the issue and let them try again


class DewarperControls(QFrame, DewarperControls_Ui):
    def __init__(self, context, signals):
        super(DewarperControls, self).__init__()
        self.signals = signals
        self.context = context
        self.scan_type = "fermi_map"
        self.data = self.context.master_dict["data"]["fermi_map"]
        self.coord1 = self.data.coords[self.data.dims[0]].values
        self.coord2 = self.data.coords[self.data.dims[1]].values
        self.coord3 = self.data.coords[self.data.dims[0]].values
        self.setupUi(self)
        self.make_connections()
        #self.update_allowed_positions()

    def make_connections(self):
        # TODO: make step size selector and connect to these instead
        pass

    def initialize_data(self, st):
        self.data = self.context.master_dict['data'][st]
        self.coord1 = self.data.coords[self.data.dims[0]].values
        self.coord2 = self.data.coords[self.data.dims[1]].values
        self.coord3 = self.data.coords[self.data.dims[0]].values
        self.update_allowed_positions()

    def update_allowed_positions(self):
        pass