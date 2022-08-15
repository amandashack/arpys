import logging

from gui.widgets.fileLoaderWidgetUi import FileLoaderWidget_Ui
# from users.ajshack.hv_scan_dev_ssrl import fix_array
from PyQt5.QtWidgets import QFrame, QMessageBox, QFileDialog
import sys, types, importlib.util
from collections import defaultdict
from pathlib import Path
import xarray as xr
import numpy as np
log = logging.getLogger(__name__)

BASEDIR = Path(__file__).resolve().parents[2].joinpath('loaders')
DATADIR = Path(__file__).resolve().parents[4].joinpath('data/ssrl_071522')

# TODO: if the graph loads, close the window, if not give another popup
#  that says the issue and let them try again


class FileLoaderWidget(QFrame, FileLoaderWidget_Ui):
    def __init__(self, context, signals):
        super(FileLoaderWidget, self).__init__()
        self.signals = signals
        self.context = context
        self.setupUi(self)
        self.cur_file = ""
        self.plot_type = ""
        self.beamline_options = {}
        self.beamline_loaders = defaultdict(dict)
        self.make_connections()
        self.scan_options()

    def make_connections(self):
        self.cbox_beamline.currentTextChanged.connect(self.handle_options)
        self.bttn_loader.clicked.connect(self.handle_load)
        self.bttn_select_file.clicked.connect(self.handle_select)

    def scan_options(self):
        for entry in BASEDIR.iterdir():
            if entry.is_file() and entry.match('[!_.]*.py'):
                self.beamline_options[entry.stem] = entry
        self.cbox_beamline.addItems(self.beamline_options)

    def get_loaders(self, option):
        if option not in self.beamline_loaders:
            entry = self.beamline_options[option]
            modname = f'{BASEDIR.name}.{option}'
            spec = importlib.util.spec_from_file_location(modname, entry)
            module = importlib.util.module_from_spec(spec)
            sys.modules[modname] = module
            spec.loader.exec_module(module)
            for name in dir(module):
                if not name.startswith('_'):
                    obj = getattr(module, name)
                    if isinstance(obj, types.FunctionType):
                        self.beamline_loaders[option][name] = obj
        return self.beamline_loaders[option]

    def get_loader(self, option, loader):
        if option and loader:
            return self.get_loaders(option)[loader]

    def handle_options(self, option):
        self.cbox_loader_type.clear()
        self.cbox_loader_type.addItems(self.get_loaders(option))

    def handle_select(self):
        #qd = QFileDialog()
        #qd.setDirectory(DATADIR)
        file = QFileDialog.getOpenFileName(self, 'Open File') # , directory=DATADIR)
        self.le_select_file.setText(list(file)[0])
        self.cur_file = list(file)[0]

    @staticmethod
    def fix_array(ar, scan_type):
        """
        make your array uniform based on what type of scan it is
        :param ar: input xarray
        :param scan_type: the scan type can be "hv_scan"
        :return: the fixed array
        """

        def np_transpose(xar, tr):
            """Transpose the RegularSpacedData
            :param xar: starting xarray
            :param tr: list of the new transposed order
            """
            coords = {}
            dims = []
            for i in tr:
                name = list(xar.dims)[i]
                coords[name] = xar[name].values
                dims.append(list(xar.dims)[i])
            return xr.DataArray(np.transpose(xar.data, tr), dims=dims, coords=coords)
        if scan_type == "hv_scan":
            photon_energy = ar.photon_energy.values
            slit = ar.slit.values
            energy = ar.energy.values
            size_new = [len(photon_energy), len(slit), len(energy)]
            size_ar = list(ar.values.shape)
            tr = [size_ar.index(i) for i in size_new]
            print(size_new, size_ar, tr)
            return np_transpose(ar, tr)
        if scan_type == "fermi_map":
            slit = ar.slit.values
            perp = ar.perp.values
            energy = ar.energy.values
            size_new = [len(slit), len(perp), len(energy)]
            size_ar = list(ar.values.shape)
            tr = [size_ar.index(i) for i in size_new]
            print(size_new, size_ar, tr)
            return np_transpose(ar, tr)

    def determine_information(self, x):
        """
        Determine what tab the plot should go to.
        Parameters
        ----------
        x

        Returns
        -------

        """
        dims = sorted(list(x.dims))
        hv_scan = sorted(['photon_energy', 'slit', 'energy'])
        fermi_map = sorted(['slit', 'perp', 'energy'])
        single = sorted(['slit', 'energy'])
        if dims == hv_scan:
            x = self.fix_array(x, 'hv_scan')
            self.plot_type = "hv_scan"
        elif dims == fermi_map:
            x = self.fix_array(x, 'fermi_map')
            self.plot_type = "fermi_map"
        elif dims == single:
            x = self.fix_array(x, 'single')
            self.plot_type = "single"
        return x

    def handle_load(self):
        option = self.cbox_beamline.currentText()
        loader = self.cbox_loader_type.currentText()
        func = self.get_loader(option, loader)
        #QMessageBox.information(self, 'Load', repr(func))
        try:
            xar = func(self.cur_file)
            xar = self.determine_information(xar)
            if self.plot_type == "hv_scan":
                self.context.upload_hv_data(xar)
            elif self.plot_type == "fermi_map":
                self.context.upload_fm_data(xar)
            elif self.plot_type == "single":
                self.context.upload_ss_data(xar)
            self.signals.closeLoader.emit()
        except Exception as e:
            print("There was an exception while trying to load in file: " + self.cur_file, " \nException: ", Exception)
            print("Try again... make sure you have the correct loader selected. "
                  "If you do, it may not be working properly")




