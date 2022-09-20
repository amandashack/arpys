import logging
from gui.widgets.hvImageWidgetUi import HVImageWidget_Ui
from PyQt5.QtWidgets import QFrame
from pyimagetool import RegularDataArray
from plottingTools import PlotCanvas, PlotWidget
from Arpes import Arpes
import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# TODO: for converting to k-space for a single cut, you must know the photon energy -
#  Try to get it from the metadata and if not, have a pop up which asks for the
#  photon energy.


class HVImageWidget(QFrame, HVImageWidget_Ui):

    def __init__(self, context, signals, scan_type):
        super(HVImageWidget, self).__init__()
        self.signals = signals
        self.context = context
        self.setupUi(self)
        self.scan_type = scan_type
        self.xar = xr.DataArray()
        self.data = None
        self.cut = xr.DataArray()
        self.imagetool = None
        self.canvas = None
        self.k_cut = None
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
        self.work_function = 4.2
        self.inner_potential = 14
        self.photon_energy = 150
        self.initialize_canvases()
        self.connect_signals()

    def initialize_canvases(self):
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        z = np.linspace(-1, 1, 51)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        d = np.sin(np.pi * np.exp(-1 * (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))) * np.cos(np.pi / 2 * xyz[1])
        self.xar = xr.DataArray(d, coords={"photon_energy": x, 'slit': y, "energy": z},
                                dims=["photon_energy", "slit", "energy"])
        self.data = RegularDataArray(d, delta=[x[1] - x[0], y[1] - y[0], z[1] - z[0]], coord_min=[x[0], y[0], z[0]])
        self.cut = self.xar.sel({"slit": 0}, method='nearest')
        self.imagetool = PlotWidget(self.data, layout=1)
        self.canvas = PlotCanvas()
        self.canvas.plot(self.cut.transpose())
        self.layout.addWidget(self.imagetool)
        self.layout.addWidget(self.canvas)

    def initialize_vals(self):
        self.context.hv_xar_data = self.xar
        self.context.hv_reg_data = self.data

    def connect_signals(self):
        self.signals.updateData.connect(self.change_data)
        self.signals.updateRealSpace.connect(self.convert_k)
        self.signals.axslitOffset.connect(self.update_axslit)
        self.signals.alslitOffset.connect(self.update_alslit)
        self.signals.azimuthOffset.connect(self.update_azimuth)
        self.signals.workFunctionChanged.connect(self.update_wf)
        self.signals.innerPotentialChanged.connect(self.update_ip)
        self.signals.hvChanged.connect(self.update_hv)
        self.signals.axesChanged.connect(self.change_axes)
        self.signals.updateXYTLabel.connect(self.update_xyt)

    def update_xyt(self, xyt, scan_type):
        if scan_type == self.scan_type:
            self.clear_canvases()
            self.x_label = xyt[0]
            self.y_label = xyt[1]
            self.title = xyt[2]
            self.canvas.set_xyt(self.x_label, self.y_label, self.title)
            self.add_canvases()

    def change_axes(self, a, scan_type):
        if scan_type == self.scan_type:
            self.clear_canvases()
            self.x_min = a[0]
            self.x_max = a[1]
            self.y_min = a[2]
            self.y_max = a[3]
            self.canvas.set_xlim(self.x_min, self.x_max)
            self.canvas.set_ylim(self.y_min, self.y_max)
            self.add_canvases()

    def update_axslit(self, axs):
        """function for shifting across slit"""
        pass

    def update_alslit(self, als):
        """function for shifting along slit"""
        pass

    def update_azimuth(self, az):
        """function for shifting in azimuth"""
        pass
    
    def update_wf(self, wf, scan_type):
        if scan_type == self.scan_type:
            self.work_function = wf

    def update_ip(self, ip, scan_type):
        if scan_type == self.scan_type:
            self.inner_potential = ip

    def update_hv(self):
        self.photon_energy = self.context.master_dict['hv'][self.scan_type]

    def convert_k(self):
        self.k = self.context.master_dict['real_space'][self.scan_type]
        if self.k:
            spectra_ek = self.convert_to_ke()
            self.k_cut = spectra_ek.arpes.spectra_k_irreg(phi0=0)
            self.clear_canvases()
            self.add_canvases()
        else:
            self.clear_canvases()
            self.add_canvases()
            pass

    def ranges(self):
        if self.which_cut == 1:
            x_min = self.xar.slit.values[0]
            x_max = self.xar.slit.values[-1]
            y_min = self.xar.energy.values[0]
            y_max = self.xar.energy.values[-1]
            self.context.update_all_axes(self.scan_type, [["x_min", x_min],
                                                       ["x_max", x_max],
                                                       ["y_min", y_min],
                                                       ["y_max", y_max]])

    def change_data(self, st):
        if st == "fhv_scan":
            self.xar = self.context.master_dict['data']['hv_scan']
            self.data = RegularDataArray(self.xar)
            self.handle_plotting()

    def handle_plotting(self, xar):
        self.cut = self.xar.sel({"slit": 0}, method='nearest')
        self.ranges()
        self.refresh_plots()

    def refresh_plots(self):
        self.clear_canvases()
        self.add_canvases()

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def clear_canvases(self):
        self.clearLayout(self.layout)
        self.canvas = PlotCanvas()
        self.imagetool = PlotWidget(self.data, layout=1)
        if self.k:
            self.canvas.plot(self.k_cut.transpose())
        else:
            self.canvas.plot(self.cut.transpose())

    def add_canvases(self):
        self.layout.addWidget(self.imagetool)
        self.layout.addWidget(self.canvas)

    def convert_to_ke(self):
        binding_energies = self.cut.energy
        kinetic_energies = binding_energies + (self.photon_energy - self.work_function)
        v = self.cut.assign_coords({'energy': kinetic_energies})
        ef = self.photon_energy - self.work_function
        v.arpes.ef = ef
        return v
