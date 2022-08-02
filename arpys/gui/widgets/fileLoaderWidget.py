import logging

from gui.widgets.fileLoaderWidgetUi import FileLoaderWidget_Ui
# from users.ajshack.hv_scan_dev_ssrl import fix_array
from PyQt5.QtWidgets import QFrame, QMessageBox, QFileDialog
import sys, types, importlib.util
from collections import defaultdict
from pathlib import Path
log = logging.getLogger(__name__)

BASEDIR = Path(__file__).resolve().parents[2].joinpath('loaders')
DATADIR = Path(__file__).resolve().parents[4].joinpath('data/ssrl_071522')

class FileLoaderWidget(QFrame, FileLoaderWidget_Ui):
    def __init__(self, context, signals):
        super(FileLoaderWidget, self).__init__()
        self.signals = signals
        self.context = context
        self.setupUi(self)
        self.cur_file = ""
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
        print(DATADIR)
        file = QFileDialog.getOpenFileName(self, 'Open File') # , directory=DATADIR)
        self.le_select_file.setText(list(file)[0])
        self.cur_file = list(file)[0]

    def determine_information(self, x):
        dims = list(x.dims)
        if all(dims) in ['photon_energy', 'slit', 'energy']:
            print("here is where you would fix the array")
            #fix_array(x, 'hv_scan')

    def handle_load(self):
        option = self.cbox_beamline.currentText()
        loader = self.cbox_loader_type.currentText()
        func = self.get_loader(option, loader)
        #QMessageBox.information(self, 'Load', repr(func))
        print(self.cur_file)
        xar = func(self.cur_file)
        self.determine_information(xar)
        # need to determine here what tab it should be loaded into
        # for now I am just going to try hard coding to work with FMs
        self.context.upload_fm_data(xar)


