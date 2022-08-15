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
        self.data = self.context.fm_xar_data
        self.setupUi(self)
        self.make_connections()

    def make_connections(self):
        self.lw_x_bin.valueChanged.connect(self.capture_change_x)
        self.lw_y_bin.valueChanged.connect(self.capture_change_y)

    def capture_change_x(self, v):
        self.lw_x_position.setSingleStep(v)

    def capture_change_y(self, v):
        self.lw_y_position.setSingleStep(v)

    def update_scan_type(self, st):
        self.scan_type = st
        if st == "fermi_map":
            self.data = self.context.fm_xar_data
            self.coord1 = self.data.slit.values.tolist()
            self.coord2 = self.data.perp.values.tolist()
            self.coord3 = self.data.energy.values.tolist()
            self.lw_x_position.set_values(self.coord2)
            self.lw_x_position.setMinimum(self.lw_x_position.valueFromText(min(self.coord2)))
            self.lw_x_position.setMaximum(self.lw_x_position.valueFromText(max(self.coord2)))

            self.lw_y_position.set_values(self.coord1)
            self.lw_y_position.setMinimum(self.lw_y_position.valueFromText(min(self.coord1)))
            self.lw_y_position.setMaximum(self.lw_y_position.valueFromText(max(self.coord1)))

