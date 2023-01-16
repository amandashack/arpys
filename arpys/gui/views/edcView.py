import logging

from gui.widgets.edcControlsWidget import EDCControlsWidget
from gui.widgets.edcImageWidget import EDCImageWidget
from gui.windows.paramsTableWindow import ParamsTableWindow
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import numpy as np
from users.ajshack import normalize_scan
import arpys

log = logging.getLogger('pydm')
log.setLevel('CRITICAL')


class EDCView(QWidget):
    def __init__(self, context, signals):
        super(EDCView, self).__init__()
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

    def rebuild_dicts(self):
        fits = {}
        params = {}
        for cut in self.imageWidget.edc_fit_results.keys():
            new_cut_val = min(self.imageWidget.coord1,
                              key=lambda f: abs(f - cut))
            fits[new_cut_val] = {}
            for edc in self.imageWidget.edc_fit_results[cut].keys():
                fit = self.imageWidget.edc_fit_results[cut][edc]
                new_edc_val = min(self.imageWidget.coord0,
                              key=lambda f: abs(f - edc))
                fits[cut][new_edc_val] = fit
        self.imageWidget.edc_fit_results = fits
        for cut in self.imageWidget.fit_params.keys():
            new_cut_val = min(self.imageWidget.coord1,
                              key=lambda f: abs(f - cut))
            fits[new_cut_val] = {}
            for edc in self.imageWidget.fit_params[cut].keys():
                fit = self.imageWidget.fit_params[cut][edc]
                new_edc_val = min(self.imageWidget.coord0,
                                  key=lambda f: abs(f - edc))
                params[cut][new_edc_val] = fit
        self.imageWidget.fit_params = fits

    def capture_change_binz(self, v):
        try:
            self.imageWidget.xar = self.context.master_dict['data'][self.imageWidget.st].arpes.downsample(
                {self.imageWidget.dim1: v})
            self.editorWidget.data = self.context.master_dict['data'][self.imageWidget.st].arpes.downsample(
                {self.imageWidget.dim1: v})
        except ZeroDivisionError:
            print("got a division by zero error, did you try to downsample to 0?")
        fits = {}
        params = {}

        self.imageWidget.coord1 = self.imageWidget.xar[self.imageWidget.dim1].values
        self.imageWidget.cut_val = min(self.imageWidget.coord1,
                                       key=lambda f: abs(f - self.imageWidget.cut_val))
        self.imageWidget.handle_plotting()
        self.editorWidget.update_allowed_positions()

    def capture_change_biny(self, v):
        # TODO: think about the range on dim0 and how to make that into values that are not
        #  hard coded. Either convince yourself that starting from the beginning and going
        #  to the end is ok, otherwise add more plot lines to handle it.
        #  lastly, check if the rest of these capture functions need normalize scan (probably)
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

        self.imageWidget.y_edc = min(self.imageWidget.coord0,
                                     key=lambda f: abs(f - self.imageWidget.y_edc))
        self.imageWidget.cut = self.imageWidget.xar.sel({self.imageWidget.dim1: self.imageWidget.cut_val},
                                                        method='nearest')
        self.imageWidget.edc = self.imageWidget.cut.sel({self.imageWidget.dim0: self.imageWidget.y_edc})
        self.imageWidget.edc = self.imageWidget.edc[self.imageWidget.x_left[1]: self.imageWidget.x_right[1]]
        self.imageWidget.handle_plotting()
        self.editorWidget.data = self.context.master_dict['data'][self.imageWidget.st].arpes.downsample(
                {self.imageWidget.dim0: v})
        self.editorWidget.update_allowed_positions()

    def capture_change_posz(self, v):
        # TODO: IndexError: index 12 is out of bounds for axis 0 with size 10
        self.imageWidget.cut_val = self.imageWidget.coord1[int(v)]
        self.imageWidget.update_cut()
        self.imageWidget.handle_plotting()

    def capture_change_posy(self, v):
        v = float(self.editorWidget.lw_y_position.textFromValue(v))
        rel_pos = lambda x: abs(self.imageWidget.scene.sceneRect().height() - x)
        bbox = self.imageWidget.canvas_cut.axes.spines['left'].get_window_extent()
        plot_bbox = [bbox.y0, bbox.y1]
        size_range = len(self.imageWidget.coord0)
        r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
        corr = list(zip(r, self.imageWidget.coord0))
        what_index = self.imageWidget.coord0.tolist().index(v)
        y = corr[what_index][0]
        y = rel_pos(y)
        self.imageWidget.y_edc = v
        self.imageWidget.line_item_hor.setPos(0, y)
        self.imageWidget.handle_plotting()

    def create_image_widget(self):
        self.imageWidget = EDCImageWidget(self.context, self.signals)
        self.set_image_values()

    def create_editor_widget(self):
        self.editorWidget = EDCControlsWidget(self.context, self.signals)
        self.set_editor_values()

    def update_scan_type(self, st):
        """
        Used to set the scan type before the EDC is opened so that it
        pulls the correct data.
        Parameters
        ----------
        st: scan type

        Returns
        -------

        """
        self.scan_type = st
        self.set_image_values()
        self.set_editor_values()

    def set_image_values(self):
        self.imageWidget.initialize_data(self.scan_type)
        self.imageWidget.handle_plotting()

    def set_editor_values(self):
        self.editorWidget.initialize_data(self.scan_type)

    def open_params(self):
        if self.imageWidget.cut_val not in self.imageWidget.fit_params.keys():
            self.imageWidget.fit_params[self.cut_val] = {}
        self.table_window = ParamsTableWindow(self.context, self.signals,
                                              self.imageWidget.fit_params[self.imageWidget.cut_val][self.imageWidget.y_edc])
        #d = self.imageWidget.fit_params[self.imageWidget.cut_val][self.imageWidget.y_edc]
        #self.table_window.set_data(d)
        self.table_window.show()


