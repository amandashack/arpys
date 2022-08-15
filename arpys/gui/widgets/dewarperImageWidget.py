import logging
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView
from gui.widgets.basicWidgets import HLineItem, VLineItem
from PyQt5.QtCore import Qt
import numpy as np
import xarray as xr
from fmPlotting import PlotCanvas

log = logging.getLogger(__name__)


class DewarperImageWidget(QGraphicsView):

    def __init__(self, context, signals):
        super(DewarperImageWidget, self).__init__()
        self.signals = signals
        self.context = context
        self.scene = QGraphicsScene()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_image = PlotCanvas()
        self.data = xr.DataArray(np.zeros([500, 500, 3], dtype=np.uint8))
        self.plot_image.plot(self.data)
        self.line_item_hor_top = HLineItem()
        self.line_item_hor_bot = HLineItem()
        self.line_item_vert_left = VLineItem()
        self.line_item_vert_right = VLineItem()
        self.make_connections()
        self.connect_scene()

    def connect_scene(self):
        self.setScene(self.scene)
        self.scene.addWidget(self.plot_image)
        self.scene.addItem(self.line_item_vert_right)
        self.scene.addItem(self.line_item_vert_left)
        self.scene.addItem(self.line_item_hor_bot)
        self.scene.addItem(self.line_item_hor_top)

    def make_connections(self):
        pass
