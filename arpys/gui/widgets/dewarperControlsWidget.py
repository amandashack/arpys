import logging

from gui.widgets.dewarperControlsWidgetUi import DewarperControlsWidget_Ui
# from users.ajshack.hv_scan_dev_ssrl import fix_array
from PyQt5.QtWidgets import QFrame, QMessageBox, QFileDialog
import sys, types, importlib.util
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

BASEDIR = Path(__file__).resolve().parents[2].joinpath('loaders')
DATADIR = Path(__file__).resolve().parents[4].joinpath('data/ssrl_071522')


# TODO: if the graph loads, close the window, if not give another popup
#  that says the issue and let them try again


class DewarperControlsWidget(QFrame, DewarperControlsWidget_Ui):
    def __init__(self, context, signals):
        super(DewarperControlsWidget, self).__init__()
        self.signals = signals
        self.context = context
        self.scan_type = "fermi_map"
        self.data = self.context.master_dict["data"]["fermi_map"]
        self.coord1 = self.data.coords[self.data.dims[0]].values
        self.coord2 = self.data.coords[self.data.dims[1]].values
        self.coord3 = self.data.coords[self.data.dims[0]].values
        self.setupUi(self)
        self.make_connections()
        self.update_allowed_positions()

    def make_connections(self):
        self.lw_z_bin.valueChanged.connect(self.capture_change_z)
        self.lw_y_bin.valueChanged.connect(self.capture_change_y)

    def update_data(self, st):
        self.data = self.context.master_dict['data'][st]
        self.coord1 = self.data.coords[self.data.dims[0]].values
        self.coord2 = self.data.coords[self.data.dims[1]].values
        self.coord3 = self.data.coords[self.data.dims[2]].values
        self.update_allowed_positions()

    def capture_change_z(self, v):
        self.lw_z_position.setSingleStep(v)

    def capture_change_y(self, v):
        self.lw_y_position.setSingleStep(v)

    def update_allowed_positions(self):
        self.lw_z_position.set_values(self.coord2)
        self.lw_z_position.setMinimum(self.lw_z_position.valueFromText(min(self.coord2)))
        self.lw_z_position.setMaximum(self.lw_z_position.valueFromText(max(self.coord2)))
        self.lw_z_position.valueFromText(str(self.coord2.tolist()[0]))

        self.lw_y_position.set_values(self.coord1)
        self.lw_y_position.setMinimum(self.lw_y_position.valueFromText(min(self.coord1)))
        self.lw_y_position.setMaximum(self.lw_y_position.valueFromText(max(self.coord1)))
        self.lw_y_position.valueFromText(str(self.coord1.tolist()[0]))

