import pyqtgraph as pg
import xarray
from PyQt5 import QtCore
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyimagetool import PGImageTool, RegularDataArray
import xarray as xr


def graph_setup(graph, title, y_axis, pen):
    graph.setTitle(title=title)
    graph.setLabels(left=y_axis, bottom="time (s)")
    graph.plotItem.showGrid(x=True, y=True)
    plot = pg.ScatterPlotItem(pen=pen, size=1)
    plot_average = pg.PlotCurveItem(pen=pg.mkPen(width=1, color='w'), size=1)
    graph.addPlot(plot)
    graph.addAvePlot(plot_average)


def add_calibration_graph(graph):
    plot_mean = pg.PlotCurveItem(
        pen=pg.mkPen(width=1, color=(255, 165, 0)), size=1)
    plot_sigma_low = pg.PlotCurveItem(
        pen=pg.mkPen(width=1, color=(255, 255, 0)),
        size=1, style=QtCore.Qt.DashLine)
    plot_sigma_high = pg.PlotCurveItem(
        pen=pg.mkPen(width=1, color=(255, 255, 0)),
        size=1, style=QtCore.Qt.DashLine)
    graph.addMeanPlot(plot_mean)
    graph.addSigmaPlots(plot_sigma_low, plot_sigma_high)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(PlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.data = xr.DataArray()
        self.axes = self.fig.add_subplot(111)
        self.axes.yaxis.set_ticks_position('both')
        self.axes.xaxis.set_ticks_position('both')

    def set_xyt(self, x, y, t):
        self.axes.set_xlabel(x)
        self.axes.set_ylabel(y)
        self.axes.set_title(t)

    def set_xlim(self, xmin, xmax):
        self.axes.set_xlim(xmin, xmax)

    def set_ylim(self, ymin, ymax):
        self.axes.set_ylim(ymin, ymax)

    def plot(self, data):
        self.data = data
        self.data.plot(ax=self.axes)
        self.draw()

"""
class PlotCanvas(FigureCanvas):

    def __init__(self, context, signals, parent=None, width=5, height=4, dpi=100):
        self.context = context
        self.signals = signals
        self.title = self.context.title
        self.x_label = self.context.x_label
        self.y_label = self.context.y_label
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(PlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.make_connections()

    def make_connections(self):
        self.signals.updateXYTLabel.connect(self.update_xyt)

    def update_xyt(self, x, y, t):
        self.axes.clear()
        self.x_label = x
        self.y_label = y
        self.title = t
        self.axes.set_title(self.title)
        self.axes.set_xlabel(self.x_label)
        self.axes.set_ylabel(self.y_label)
        self.plot(self.data)

    def plot(self, data):
        self.data = data
        ax = self.fig.add_subplot(111)
        self.data.plot(ax=ax)
        self.draw()
"""

class fmPlotWidget(PGImageTool):
    def __init__(self, parent=None):
        super(fmPlotWidget, self).__init__(parent)

