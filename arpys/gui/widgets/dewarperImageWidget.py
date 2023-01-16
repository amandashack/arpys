import logging
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QFrame, QHBoxLayout, QVBoxLayout
from gui.widgets.basicWidgets import HLineItem, VLineItem
from users.ajshack import standardize, normalize_scan, generate_fit
from PyQt5.QtCore import Qt
import numpy as np
import xarray as xr
from plottingTools import PlotCanvas, PlotWidget
from pyimagetool import RegularDataArray

log = logging.getLogger(__name__)


class DewarperImageWidget(QFrame):

    def __init__(self, context, signals):
        super(DewarperImageWidget, self).__init__()
        self.context = context
        self.signals = signals
        self.st = "fermi_map"
        self.emin = -1
        self.emax = 1
        self.a = 5
        self.b = .1
        self.c = 4
        self.d = 2
        self.y = 1.9

        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        z = np.linspace(-1, 1, 51)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        d = np.sin(np.pi * np.exp(-1 * (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))) * np.cos(np.pi / 2 * xyz[1])
        self.xar = xr.DataArray(d, coords={"slit": x, 'perp': y, "energy": z}, dims=["slit", "perp", "energy"])
        self.data = RegularDataArray(d, delta=[x[1] - x[0], y[1] - y[0], z[1] - z[0]], coord_min=[x[0], y[0], z[0]])
        self.cut_val = 0
        self.y_edc = 0
        self.x_left = [-1, 0]
        self.x_right = [1, -1]
        self.cut = self.xar.sel({"perp": self.cut_val}, method='nearest')
        self.cut_fit = self.cut
        self.imagetool = PlotWidget(self.data, layout=1)
        self.canvas_cut = PlotCanvas()
        self.canvas_cut_fit = PlotCanvas()
        self.canvas_cut.plot(self.cut)
        self.canvas_cut_fit.plot(self.cut_fit)

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
        self.layout_cuts = QHBoxLayout()
        self.connect_scene()

        self.layout_cuts.addWidget(self.view)
        self.layout_cuts.addWidget(self.canvas_cut_fit)
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.layout_cuts)
        self.main_layout.addWidget(self.imagetool)
        self.setLayout(self.main_layout)

        self.make_connections()

    def connect_scene(self):
        s = self.canvas_cut.figure.get_size_inches() * self.canvas_cut.figure.dpi
        self.view.setScene(self.scene)
        self.scene.addWidget(self.canvas_cut)
        self.scene.setSceneRect(0, 0, s[0], s[1])
        self.scene.addItem(self.line_item_vert_right)
        self.scene.addItem(self.line_item_vert_left)
        self.scene.addItem(self.line_item_hor) # this could maybe be used for viewing where the minimum is for each edc
        self.capture_scene_change()

    def make_connections(self):
        pass

    def initialize_data(self, st):
        """
        Used to initialize the data. This should only be called once when the tool window is opened.
        Subsequently, data is changed by the view and retrieving data from the master dict would
        override changes made by the controller.
        Parameters
        ----------
        st: scan type
        """
        self.st = st
        self.xar = self.context.master_dict['data'][st]
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
        self.update_cut()
        self.plot_pos_to_line_pos()
        self.plot_pos_to_line_pos(horizontal=False)

    def update_cut(self):
        self.cut = self.xar.sel({self.dim1: self.cut_val}, method='nearest')
        self.cut_fit = self.cut

    def handle_plotting(self):
        self.handle_plotting_cut()
        self.main_layout

    def handle_plotting_cut(self):
        self.canvas_cut.axes.cla()
        self.canvas_cut.plot(self.cut)
        self.canvas_cut_fit.axes.cla()
        self.canvas_cut_fit.plot(self.cut)
        self.canvas_cut.draw()
        self.canvas_cut_fit.draw()

    def refresh_plots(self):
        self.clear_canvases()
        self.add_canvases()

    def clear_canvases(self):
        # I might be able to just remove the last layout (imagetool) and only refresh that one
        self.clearLayout(self.main_layout)
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.line_item_hor = HLineItem(self.signals)
        self.line_item_vert_left = VLineItem(self.signals, 0)
        self.line_item_vert_right = VLineItem(self.signals, 1)
        self.imagetool = PlotWidget(self.data, layout=1)
        self.canvas_cut = PlotCanvas()
        self.canvas_cut_fit = PlotCanvas()
        self.canvas_cut.plot(self.cut)
        self.canvas_cut_fit.plot(self.cut)
        self.connect_scene()

    def add_canvases(self):
        self.layout_cuts = QHBoxLayout()
        self.layout_cuts.addWidget(self.view)
        self.layout_cuts.addWidget(self.canvas_cut_fit)
        self.main_layout.addLayout(self.layout_cuts)
        self.main_layout.addWidget(self.imagetool)
        self.resize_event()

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def plot_pos_to_line_pos(self, horizontal=True):
        if horizontal:
            bbox = self.canvas_cut.axes.spines['left'].get_window_extent()
            plot_bbox = [bbox.y0, bbox.y1]
            size_range = len(self.coord0)
            r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
            corr = list(zip(r, self.coord0))
            what_index = self.coord0.tolist().index(self.y_edc)
            bbox_val = corr[what_index][0]
            rel_pos = lambda x: abs(self.scene.sceneRect().height() - x)
            self.line_item_hor.setPos(0, rel_pos(bbox_val))
        else:
            bbox = self.canvas_cut.axes.spines['top'].get_window_extent()
            plot_bbox = [bbox.x0, bbox.x1]
            self.line_item_vert_left.setPos(plot_bbox[0], 0)
            self.line_item_vert_right.setPos(plot_bbox[1], 0)

    def capture_scene_change(self):
        self.line_item_hor.setLine(0, 0, self.scene.sceneRect().width(), 0)
        self.plot_pos_to_line_pos()
        self.line_item_vert_left.setLine(0, 0, 0, self.scene.sceneRect().height())
        self.line_item_vert_right.setLine(0, 0, 0, self.scene.sceneRect().height())
        self.plot_pos_to_line_pos(horizontal=False)

    def resize_event(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.view.fitInView(self.line_item_vert_left, Qt.KeepAspectRatio)
        self.view.fitInView(self.line_item_hor, Qt.KeepAspectRatio)
        self.view.fitInView(self.line_item_vert_right, Qt.KeepAspectRatio)

