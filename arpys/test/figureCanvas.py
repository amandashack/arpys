import ctypes
import logging
import os
import sys

from PyQt5.Qt import Qt
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import xarray as xr
import numpy as np

log = logging.getLogger(__name__)


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
        self.axes = None

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
        self.axes = self.fig.add_subplot(111)
        self.data.plot(ax=self.axes)
        self.axes.set_xlim(-.5, .5)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        z = np.linspace(-1, 1, 51)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        d = np.sin(np.pi * np.exp(-1 * (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))) * np.cos(np.pi / 2 * xyz[1])
        self.xar = xr.DataArray(d, coords={"slit": x, 'perp': y, "energy": z}, dims=["slit", "perp", "energy"])
        self.cut = self.xar.sel({"perp": 0}, method='nearest')
        self.fm_pyplot = PlotCanvas()
        self.fm_pyplot.plot(self.cut)
        self.layout().addWidget(self.fm_pyplot)


class App(QApplication):
    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        log.debug("This is the mainThread")
        self.setAttribute(Qt.AA_EnableHighDpiScaling)
        self.mainWindow = MainWindow()
        self.mainWindow.setWindowTitle("arpys")
        self.mainWindow.show()


def main():

    app = App(sys.argv)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
