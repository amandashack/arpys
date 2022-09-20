import logging
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QFrame, QHBoxLayout
from gui.widgets.basicWidgets import HLineItem, VLineItem
from users.ajshack import standardize, normalize_scan, generate_fit
from PyQt5.QtCore import Qt
import numpy as np
import xarray as xr
from plottingTools import PlotCanvas

log = logging.getLogger(__name__)


class DewarperImageWidget(QFrame):

    def __init__(self, context, signals):
        super(DewarperImageWidget, self).__init__()
        self.context = context
        self.signals = signals
        self.st = "fermi_map"
        self.horizontal_labels = ['Temperature', 'fermi_kt', 'fermi_center',
                             'fermi_amplitude', 'linear_slope', 'linear_intercept',
                             'constant_c']
        self.default_params = [[13.6, None, None], [True, 0.03, 0.001],
                          [0.1, 0.15, True], [1, False, None], [-0.1, 50, False],
                          [1, True, None], [0.002, 100, None]]
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        z = np.linspace(-1, 1, 51)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        d = np.sin(np.pi * np.exp(-1 * (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))) * np.cos(np.pi / 2 * xyz[1])
        self.xar = xr.DataArray(d, coords={"slit": x, 'perp': y, "energy": z}, dims=["slit", "perp", "energy"])
        self.cut_val = 0
        self.y_edc = 0
        self.x_left = [-1, 0]
        self.x_right = [1, -1]
        self.cut = self.xar.sel({"perp": self.cut_val}, method='nearest')
        self.edc = self.cut.sel({'slit': self.y_edc}, method='nearest')
        self.edc_fit_params = {}
        self.params = {}
        self.canvas_cut = PlotCanvas()
        self.canvas_edc = PlotCanvas()
        self.canvas_cut.plot(self.cut)
        self.canvas_edc.plot(self.edc)

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.coord0 = self.xar.slit.values
        self.coord1 = self.xar.perp.values
        self.coord2 = self.xar.energy.values
        self.dim0 = self.xar.dims[0]
        self.dim1 = self.xar.dims[1]
        self.dim2 = self.xar.dims[2]
        self.line_item_hor = HLineItem(self.signals)
        self.line_item_vert_left = VLineItem(self.signals, 0)
        self.line_item_vert_right = VLineItem(self.signals, 1)
        self.layout_cut = QHBoxLayout()
        self.layout_edc = QHBoxLayout()
        self.connect_scene()

        self.layout_cut.addWidget(self.view)
        self.layout_edc.addWidget(self.canvas_edc)
        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.layout_cut)
        self.main_layout.addLayout(self.layout_edc)
        self.setLayout(self.main_layout)

        self.make_connections()

    def connect_scene(self):
        s = self.canvas_cut.figure.get_size_inches() * self.canvas_cut.figure.dpi
        self.view.setScene(self.scene)
        self.scene.addWidget(self.canvas_cut)
        self.scene.setSceneRect(0, 0, s[0], s[1])
        self.scene.addItem(self.line_item_vert_right)
        self.scene.addItem(self.line_item_vert_left)
        self.scene.addItem(self.line_item_hor)
        self.capture_scene_change()

    def make_connections(self):
        self.scene.sceneRectChanged.connect(self.capture_scene_change)
        self.signals.hbttnReleased.connect(self.plot_position)
        self.signals.vbttnReleased.connect(self.adjust_range)
        self.signals.tableData.connect(self.update_params)

    def update_data(self, st):
        self.st = st
        self.xar = self.context.master_dict['data'][st]
        self.xar = normalize_scan(self.xar, {self.dim0: [-12, 21],
                                             self.dim2: [self.x_left[0], self.x_right[0]]},
                                  self.dim1)
        self.dim0 = self.xar.dims[0]
        self.dim1 = self.xar.dims[1]
        self.dim2 = self.xar.dims[2]
        self.coord0 = self.xar[self.dim0].values
        self.coord1 = self.xar[self.dim1].values
        self.coord2 = self.xar[self.dim2].values
        self.cut_val = self.coord1[0]
        self.y_edc = self.coord0[0]
        self.x_left = [self.coord2[0], 0]
        self.x_right = [self.coord2[-1], -1]
        self.cut = self.xar.sel({self.dim1: self.cut_val}, method='nearest')
        self.edc = self.cut.sel({self.dim0: self.y_edc}, method='nearest')
        self.params = {i: {j: self.default_params for j in self.coord0} for i in self.coord1}

    def update_params(self, p):
        self.params[self.cut_val][self.y_edc] = p

    def handle_plotting(self):
        self.clearLayout(self.layout_cut)
        self.clearLayout(self.layout_edc)
        self.refresh_plots()

    def refresh_plots(self):
        self.canvas_cut = PlotCanvas()
        self.canvas_edc = PlotCanvas()
        self.canvas_cut.plot(self.cut)
        self.canvas_edc.plot(self.edc)
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.line_item_hor = HLineItem(self.signals)
        self.line_item_vert_left = VLineItem(self.signals, 0)
        self.line_item_vert_right = VLineItem(self.signals, 1)
        # move the lines to the correct positions
        self.connect_scene()
        self.layout_cut.addWidget(self.view)
        self.layout_edc.addWidget(self.canvas_edc)

    def handle_plotting_edc(self):
        self.edc = self.cut.sel({self.dim0: self.y_edc})#.sel({self.dim2: slice(self.x_left, self.x_right)})
        self.edc = self.edc[self.x_left[1]: self.x_right[1]]
        self.clearLayout(self.layout_edc)
        self.refresh_edc()

    def refresh_edc(self):
        self.canvas_edc = PlotCanvas()
        self.canvas_edc.plot(self.edc)
        if self.edc_fit_params:
            if self.y_edc in self.edc_fit_params:
                self.canvas_edc.axes.plot(self.edc_fit_params[self.y_edc][3],
                                          self.edc_fit_params[self.y_edc][1].best_fit)
        self.layout_edc.addWidget(self.canvas_edc)

    def fit_edc(self):
        edc_sum_1 = self.cut.sel({self.dim2: slice(self.x_left[0], self.x_right[0])}).sum(self.dim0)
        edc_sum = edc_sum_1.values / (len(edc_sum_1.values))
        edc = xr.DataArray(edc_sum, coords={self.dim2: edc_sum_1[self.dim2].values}, dims=[self.dim2])
        model, out, params = generate_fit(edc, self.x_left[0], self.x_right[0],
                                          self.params[self.cut_val][self.y_edc])
        dos = edc.values
        ene = edc['energy'].values

        out = model.fit(dos, params, x=ene)
        self.edc_fit_params[self.y_edc] = [model, out, params, ene]
        self.handle_plotting_edc()

    def adjust_range(self, x, id):
        bbox = self.canvas_cut.axes.spines['top'].get_window_extent()
        plot_bbox = [bbox.x0, bbox.x1]
        if x < plot_bbox[0]:
            self.line_item_vert_left.setPos(plot_bbox[0], 0)
            x = plot_bbox[0]
        elif x > plot_bbox[1]:
            self.line_item_vert_right.setPos(plot_bbox[1], 0)
            x = plot_bbox[1]
        size_range = len(self.coord2)
        r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
        sel_val = min(r, key=lambda f: abs(f - x))
        what_index = r.index(sel_val)
        if id == 0:
            self.x_left = [self.coord2[what_index], what_index]
        elif id == 1:
            self.x_right = [self.coord2[what_index], what_index]
        self.handle_plotting_edc()

    def plot_position(self, y):
        rel_pos = lambda x: abs(self.scene.sceneRect().height() - x)
        bbox = self.canvas_cut.axes.spines['left'].get_window_extent()
        plot_bbox = [bbox.y0, bbox.y1]
        if rel_pos(y) < plot_bbox[0]:
            self.line_item_hor.setPos(0, rel_pos(plot_bbox[0]))
        elif rel_pos(y) > plot_bbox[1]:
            self.line_item_hor.setPos(0, rel_pos(plot_bbox[1]))
        self.line_pos = self.line_item_hor.pos().y()
        size_range = len(self.coord0)
        r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
        corr = list(zip(r, self.coord0))
        sel_val = min(r, key=lambda f: abs(f - rel_pos(self.line_pos)))
        what_index = r.index(sel_val)
        self.y_edc = corr[what_index][1]
        self.handle_plotting_edc()

    def clear_canvases(self):
        self.clearLayout(self.layout)
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.line_item_hor = HLineItem(self.signals)
        self.line_item_vert_left = VLineItem(self.signals)
        self.line_item_vert_right = VLineItem(self.signals)
        self.canvas_cut = PlotCanvas()
        self.canvas_edc = PlotCanvas()
        self.canvas_cut.plot(self.cut)
        self.connect_scene()

    def add_canvases(self):
        self.layout_cut = QHBoxLayout()
        self.layout_edc = QHBoxLayout()
        self.layout_cut.addWidget(self.view)
        self.layout_edc.addWidget(self.canvas_edc)
        self.layout.addLayout(self.layout_cut)
        self.layout.addLayout(self.layout_edc)
        self.resize_event()

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def capture_scene_change(self):
        self.line_item_hor.setLine(0, 0, self.scene.sceneRect().width(), 0)
        self.line_item_hor.setPos(0, self.scene.sceneRect().height())
        self.line_item_vert_left.setLine(0, 0, 0, self.scene.sceneRect().height())
        self.line_item_vert_left.setPos(0, 0)
        self.line_item_vert_right.setLine(0, 0, 0, self.scene.sceneRect().height())
        self.line_item_vert_right.setPos(self.scene.sceneRect().width(), 0)

    def resize_event(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.view.fitInView(self.line_item_vert_left, Qt.KeepAspectRatio)
        self.view.fitInView(self.line_item_hor, Qt.KeepAspectRatio)
        self.view.fitInView(self.line_item_vert_right, Qt.KeepAspectRatio)

