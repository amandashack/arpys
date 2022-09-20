import logging

from gui.widgets.dewarperControlsWidget import DewarperControlsWidget
from gui.widgets.dewarperImageWidget import DewarperImageWidget
from gui.windows.paramsTableWindow import ParamsTableWindow
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import numpy as np
from users.ajshack import normalize_scan
import arpys

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
        self.editorWidget = None
        self.table_window = None
        self.create_image_widget()
        self.create_editor_widget()
        self.mainLayout.addWidget(self.editorWidget, 25)
        self.mainLayout.addWidget(self.imageWidget, 75)
        self.setLayout(self.mainLayout)
        self.make_connections()

    def make_connections(self):
        self.editorWidget.lw_z_bin.valueChanged.connect(self.capture_change_binz)
        self.editorWidget.lw_y_bin.valueChanged.connect(self.capture_change_biny)
        self.editorWidget.lw_z_position.valueChanged.connect(self.capture_change_posz)
        self.editorWidget.lw_y_position.valueChanged.connect(self.capture_change_posy)
        self.editorWidget.bttn_fit_edc.pressed.connect(self.imageWidget.fit_edc)
        self.editorWidget.bttn_change_fit_params.pressed.connect(self.open_params)

    def capture_change_binz(self, v):
        try:
            self.imageWidget.xar = self.imageWidget.xar.arpes.downsample({self.imageWidget.dim1: v})
        except ZeroDivisionError:
            print("got a division by zero error, did you try to downsample to 0?")
        self.imageWidget.coord1 = self.imageWidget.xar[self.imageWidget.dim1].values
        self.imageWidget.cut_val = min(self.imageWidget.coord1,
                                       key=lambda f: abs(f - self.imageWidget.cut_val))
        self.imageWidget.handle_plotting()

    def capture_change_biny(self, v):
        try:
            self.imageWidget.xar = self.context.master_dict['data'][self.imageWidget.st].arpes.downsample(
                {self.imageWidget.dim0: v})
            self.imageWidget.xar = normalize_scan(self.imageWidget.xar,
                                                  {self.imageWidget.dim0: [-12, 21],
                                                   self.imageWidget.dim2: [self.imageWidget.x_left[0],
                                                                           self.imageWidget.x_right[0]]},
                                                  self.imageWidget.dim1)
        except ZeroDivisionError:
            print("got a division by zero error, did you try to downsample to 0?")
        self.imageWidget.coord0 = self.imageWidget.xar[self.imageWidget.dim0].values
        fits = {}
        params = {}
        for y_edc in self.imageWidget.edc_fit_params:
            fit = self.imageWidget.edc_fit_params[y_edc]
            new_edc_val = min(self.imageWidget.coord0,
                              key=lambda f: abs(f - y_edc))
            fits[new_edc_val] = fit
        self.imageWidget.edc_fit_params = fits
        for cut in self.imageWidget.params.keys():
            for y_edc in self.imageWidget.params[cut].keys():
                if self.imageWidget.params[cut][y_edc] != self.imageWidget.default_params:
                    p = self.imageWidget.params[cut][y_edc]
                    new_edc_val = min(self.imageWidget.coord0,
                                      key=lambda f: abs(f - y_edc))
                    params[new_edc_val] = p
        self.imageWidget.params = {}
        for cut in self.imageWidget.coord1:
            self.imageWidget.params[cut] = {}
            for y_edc in self.imageWidget.coord0:
                if y_edc in params.keys():
                    self.imageWidget.params[cut][y_edc] = params[y_edc]
                else:
                    self.imageWidget.params[cut][y_edc] = self.imageWidget.default_params
        self.imageWidget.y_edc = min(self.imageWidget.coord0,
                                     key=lambda f: abs(f - self.imageWidget.y_edc))
        self.imageWidget.cut = self.imageWidget.xar.sel({self.imageWidget.dim1: self.imageWidget.cut_val},
                                                        method='nearest')
        self.imageWidget.edc = self.imageWidget.cut.sel({self.imageWidget.dim0: self.imageWidget.y_edc})
        self.imageWidget.edc = self.imageWidget.edc[self.imageWidget.x_left[1]: self.imageWidget.x_right[1]]
        self.imageWidget.handle_plotting()

    def capture_change_posz(self, v):
        self.imageWidget.cut_val = float(v)
        self.imageWidget.handle_plotting()

    def capture_change_posy(self, v):
        v = float(self.editorWidget.lw_y_position.textFromValue(v))
        rel_pos = lambda x: abs(self.imageWidget.scene.sceneRect().height() - x)
        bbox = self.imageWidget.canvas_cut.axes.spines['left'].get_window_extent()
        plot_bbox = [bbox.y0, bbox.y1]
        size_range = len(self.imageWidget.coord2)
        r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
        corr = list(zip(r, self.imageWidget.coord2))
        what_index = self.imageWidget.coord2.tolist().index(v)
        y = corr[what_index][0]
        y = rel_pos(y)
        self.imageWidget.y_edc = v
        self.imageWidget.line_item_hor.setPos(0, y)
        self.imageWidget.handle_plotting_edc()

    def create_image_widget(self):
        self.imageWidget = DewarperImageWidget(self.context, self.signals)
        self.set_image_values()

    def create_editor_widget(self):
        self.editorWidget = DewarperControlsWidget(self.context, self.signals)
        self.set_editor_values()

    def update_scan_type(self, st):
        self.scan_type = st
        self.set_image_values()
        self.set_editor_values()

    def set_image_values(self):
        self.imageWidget.update_data(self.scan_type)
        self.imageWidget.handle_plotting()

    def set_editor_values(self):
        self.editorWidget.update_data(self.scan_type)
        self.editorWidget.update_allowed_positions()

    def open_params(self):
        self.table_window = ParamsTableWindow(self.context, self.signals,
                                              self.imageWidget.params[self.imageWidget.cut_val][self.imageWidget.y_edc])
        d = self.imageWidget.params[self.imageWidget.cut_val][self.imageWidget.y_edc]
        #self.table_window.set_data(d)
        self.table_window.show()


