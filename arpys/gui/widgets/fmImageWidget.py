import logging
from users.ajshack import convert_2d_normal_emission
from gui.widgets.fmImageWidgetUi import FermiMapImageWidget_Ui
from PyQt5.QtWidgets import QFrame
from pyimagetool import RegularDataArray, PGImageTool
from fmPlotting import fmPlotWidget, PlotCanvas, graph_setup
import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# TODO: for converting to k-space for a single cut, you must know the photon energy -
#  Try to get it from the metadata and if not, have a pop up which asks for the
#  photon energy.


class FermiMapImageWidget(QFrame, FermiMapImageWidget_Ui):

    def __init__(self, context, signals):
        super(FermiMapImageWidget, self).__init__()
        self.signals = signals
        self.context = context
        self.setupUi(self)
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        z = np.linspace(-1, 1, 51)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        d = np.sin(np.pi * np.exp(-1 * (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))) * np.cos(np.pi / 2 * xyz[1])
        self.xar = xr.DataArray(d, coords={"slit": x, 'perp': y, "energy": z}, dims=["slit", "perp", "energy"])
        self.data = RegularDataArray(d, delta=[x[1] - x[0], y[1] - y[0], z[1] - z[0]], coord_min=[x[0], y[0], z[0]])
        self.cut = self.xar.sel({"perp": 0}, method='nearest')
        self.k_data = None
        self.k = False
        self.which_cut = 1
        self.dims = []
        self.x_min = -10
        self.x_max = 10
        self.y_min = -10
        self.y_max = 10
        self.x_label = "$k_(x)$ ($\AA^(-1)$)"
        self.y_label = "Binding Energy"
        self.title = "Fermi Map Cut"
        self.initialize_vals()
        self.connect_signals()

    def initialize_vals(self):
        self.context.fm_xar_data = self.xar
        self.context.fm_reg_data = self.data
        self.context.upload_fm_data(self.xar)

    def connect_signals(self):
        self.signals.fmData.connect(self.handle_plotting)
        self.signals.updateRealSpace.connect(self.convert_k)
        self.signals.axslitOffset.connect(self.update_axslit)
        self.signals.alslitOffset.connect(self.update_alslit)
        self.signals.azimuthOffset.connect(self.update_azimuth)
        self.signals.axesChanged.connect(self.change_axes)
        self.signals.updateXYTLabel.connect(self.update_xyt)

    def update_xyt(self, scan_type, xyt):
        if scan_type == "fermi_map":
            self.clearLayout(self.layout)
            self.fm_pyplot = PlotCanvas()
            self.fm_pyqtplot = PGImageTool(self.data, layout=1)
            self.x_label = xyt[0]
            self.y_label = xyt[1]
            self.title = xyt[2]
            self.fm_pyplot.plot(self.cut)
            self.fm_pyplot.set_xyt(self.x_label, self.y_label, self.title)
            self.layout.addWidget(self.fm_pyqtplot)
            self.layout.addWidget(self.fm_pyplot)

    def change_axes(self, scan_type, a):
        if scan_type == "fermi_map":
            self.clearLayout(self.layout)
            self.fm_pyplot = PlotCanvas()
            self.fm_pyqtplot = PGImageTool(self.data, layout=1)
            self.x_min = a[0]
            self.x_max = a[1]
            self.y_min = a[2]
            self.y_max = a[3]
            self.fm_pyplot.plot(self.cut)
            self.fm_pyplot.set_xlim(self.x_min, self.x_max)
            self.fm_pyplot.set_ylim(self.y_min, self.y_max)
            self.layout.addWidget(self.fm_pyqtplot)
            self.layout.addWidget(self.fm_pyplot)

    def update_axslit(self, axs):
        """function for shifting across slit"""
        pass

    def update_alslit(self, als):
        """function for shifting along slit"""
        pass

    def update_azimuth(self, az):
        """function for shifting in azimuth"""
        pass

    def convert_k(self, scan_type, k_space):
        if scan_type == "fermi_map":
            if k_space:

                pass
            else:
                # go back to real - not convert, just use original data
                pass

    def ranges(self):
        if self.which_cut == 1:
            x_min = self.xar.energy.values[0]
            x_max = self.xar.energy.values[-1]
            y_min = self.xar.slit.values[0]
            y_max = self.xar.slit.values[-1]
            self.context.update_all_axes("fermi_map", [["x_min", x_min],
                                                       ["x_max", x_max],
                                                       ["y_min", y_min],
                                                       ["y_max", y_max]])

    def handle_plotting(self, xar):
        self.xar = xar
        self.data = RegularDataArray(self.xar)
        self.cut = self.xar.sel({"perp": 0}, method='nearest')
        self.ranges()
        self.refresh_plots()

    def refresh_plots(self):
        self.clearLayout(self.layout)
        self.fm_pyplot = PlotCanvas()
        self.fm_pyplot.plot(self.cut)
        self.fm_pyqtplot = PGImageTool(self.data, layout=1)
        self.layout.addWidget(self.fm_pyqtplot)
        self.layout.addWidget(self.fm_pyplot)

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def plot_data(self, buf):
        pass

    def set_x_axis(self):
        pass
