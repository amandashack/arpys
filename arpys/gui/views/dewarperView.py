import logging

from gui.widgets.dewarperControls import DewarperControls
from gui.widgets.dewarperImageWidget import DewarperImageWidget
from gui.windows.paramsTableWindow import ParamsTableWindow
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import numpy as np
from users.ajshack import normalize_scan
import arpys
from pyimagetool import RegularDataArray

log = logging.getLogger('pydm')
log.setLevel('CRITICAL')


class DewarperView(QWidget):
    def __init__(self, context, signals):
        super(DewarperView, self).__init__()
        self.signals = signals
        self.context = context
        self.camera = ""
        self.mainLayout = QVBoxLayout()
        self.scan_type = "fermi_map"
        self.imageWidget = None
        self.controlWidget = None
        self.table_window = None
        self.create_image_widget()
        self.create_control_widget()
        self.mainLayout.addWidget(self.controlWidget, 20)
        self.mainLayout.addWidget(self.imageWidget, 80)
        self.setLayout(self.mainLayout)
        self.make_connections()

    def make_connections(self):
        self.controlWidget.sb_x_downsample.valueChanged.connect(self.capture_x_downsample)
        self.controlWidget.sb_y_downsample.valueChanged.connect(self.capture_y_downsample)

    def capture_x_downsample(self, v):
        y_ds = self.controlWidget.sb_y_downsample.value()
        try:
            self.imageWidget.xar = self.context.master_dict['data'][self.imageWidget.st].arpes.downsample(
                {self.imageWidget.dim0: v})
            self.controlWidget.data = self.context.master_dict['data'][self.imageWidget.st].arpes.downsample(
                {self.imageWidget.dim0: v})
            self.imageWidget.xar = self.imageWidget.xar.arpes.downsample({self.imageWidget.dim1: y_ds})
            self.controlWidget.data = self.imageWidget.xar.arpes.downsample({self.imageWidget.dim1: y_ds})
        except ZeroDivisionError:
            print("got a division by zero error, did you try to downsample to 0?")

        self.imageWidget.data = RegularDataArray(self.imageWidget.xar)
        self.imageWidget.coord0 = self.imageWidget.xar[self.imageWidget.dim0].values
        self.imageWidget.coord1 = self.imageWidget.xar[self.imageWidget.dim1].values
        self.imageWidget.cut_val = min(self.imageWidget.coord1,
                                       key=lambda f: abs(f - self.imageWidget.cut_val))
        self.imageWidget.y_edc = min(self.imageWidget.coord0,
                                     key=lambda f: abs(f - self.imageWidget.y_edc))
        self.imageWidget.cut = self.imageWidget.xar.sel({self.imageWidget.dim1: self.imageWidget.cut_val},
                                                        method='nearest')
        self.imageWidget.refresh_plots()
        self.controlWidget.update_allowed_positions()

    def capture_y_downsample(self, v):
        x_ds = self.controlWidget.sb_x_downsample.value()
        try:
            self.imageWidget.xar = self.context.master_dict['data'][self.imageWidget.st].arpes.downsample(
                {self.imageWidget.dim1: v})
            self.controlWidget.data = self.context.master_dict['data'][self.imageWidget.st].arpes.downsample(
                {self.imageWidget.dim1: v})
            self.imageWidget.xar = self.imageWidget.xar.arpes.downsample({self.imageWidget.dim0: x_ds})
            self.controlWidget.data = self.imageWidget.xar.arpes.downsample({self.imageWidget.dim0: x_ds})
        except ZeroDivisionError:
            print("got a division by zero error, did you try to downsample to 0?")

        self.imageWidget.data = RegularDataArray(self.imageWidget.xar)
        self.imageWidget.coord0 = self.imageWidget.xar[self.imageWidget.dim0].values
        self.imageWidget.coord1 = self.imageWidget.xar[self.imageWidget.dim1].values
        self.imageWidget.cut_val = min(self.imageWidget.coord1,
                                       key=lambda f: abs(f - self.imageWidget.cut_val))
        self.imageWidget.y_edc = min(self.imageWidget.coord0,
                                       key=lambda f: abs(f - self.imageWidget.y_edc))
        self.imageWidget.cut = self.imageWidget.xar.sel({self.imageWidget.dim1: self.imageWidget.cut_val},
                                                        method='nearest')
        self.imageWidget.refresh_plots()
        self.controlWidget.update_allowed_positions()

    def create_image_widget(self):
        self.imageWidget = DewarperImageWidget(self.context, self.signals)
        self.set_image_values()

    def create_control_widget(self):
        self.controlWidget = DewarperControls(self.context, self.signals)
        self.set_control_values()

    def update_scan_type(self, st):
        """
        Used to set the scan type before the dewarper is opened so that it
        pulls the correct data.
        Parameters
        ----------
        st: scan type

        Returns
        -------

        """
        self.scan_type = st
        self.set_image_values()
        self.set_control_values()

    def set_image_values(self):
        self.imageWidget.initialize_data(self.scan_type)
        self.imageWidget.handle_plotting()

    def set_control_values(self):
        self.controlWidget.initialize_data(self.scan_type)
