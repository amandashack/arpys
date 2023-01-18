import logging

from gui.widgets.dewarperControlsUi import DewarperControls_Ui
from PyQt5.QtWidgets import QFrame
from pathlib import Path
import numpy as np
from scipy.optimize import fmin
import math

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
        self.coord0 = self.data.coords[self.data.dims[0]].values
        self.coord1 = self.data.coords[self.data.dims[1]].values
        self.coord2 = self.data.coords[self.data.dims[2]].values
        self.a = 25
        self.b = .1
        self.c = 4
        self.y = 1.9
        self.min_eloss = 0
        self.min_eloss_y = 0
        self.setupUi(self)
        self.set_min_eloss()
        self.make_connections()

    def make_connections(self):
        self.signals.changeEminText.connect(self.change_emin)
        self.signals.changeEmaxText.connect(self.change_emax)
        self.cb_conv_binding.stateChanged.connect(self.set_widgets_enabled)

    def set_min_eloss(self):
        a = fmin(self.f, 0.005, full_output=True)
        self.min_eloss = np.round(a[0][0], 5)
        self.min_eloss_y = np.round(a[1], 5)
        self.le_min_eloss.setText(str(self.min_eloss))
        self.le_min_eloss_y.setText(str(self.min_eloss_y))

    def initialize_data(self, st):
        self.data = self.context.master_dict['data'][st]
        self.coord0 = self.data.coords[self.data.dims[0]].values
        self.coord1 = self.data.coords[self.data.dims[1]].values
        self.coord2 = self.data.coords[self.data.dims[2]].values

    def update_allowed_positions(self):
        # this is not completely right because if the lines are in a certain
        # location then these should be set based on those.
        self.le_cut.valRange(self.coord1[0], self.coord1[-1])
        self.le_slice_range_min.valRange(self.coord2[0], self.coord2[-2])
        self.le_slice_range_max.valRange(self.coord2[1], self.coord2[-1])
        self.le_cut.setText(str(self.coord1[0]))
        self.le_slice_range_min.setText(str(self.coord2[0]))
        self.le_slice_range_max.setText(str(self.coord2[-1]))

    def change_emin(self, t):
        self.le_slice_range_min.setText(t)

    def change_emax(self, t):
        self.le_slice_range_max.setText(t)

    def set_widgets_enabled(self, s):
        if not s:
            self.le_hv.setEnabled(True)
            self.bttn_convert.setEnabled(True)
            self.bttn_fit_cut.setEnabled(False)
            self.bttn_fit_3d.setEnabled(False)
        else:
            self.le_hv.setEnabled(False)
            self.bttn_convert.setEnabled(False)
            self.bttn_fit_cut.setEnabled(True)
            self.bttn_fit_3d.setEnabled(True)

    def f(self, x):
        return np.power((np.power(self.a * x, self.c) / 2) -
                        np.log(self.b * x) - self.y, 2)
