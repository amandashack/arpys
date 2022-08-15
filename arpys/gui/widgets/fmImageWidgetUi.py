import pyqtgraph as pg
from fmPlotting import fmPlotWidget, PlotCanvas, graph_setup
from PyQt5.QtWidgets import QVBoxLayout
from pyimagetool import RegularDataArray, PGImageTool
import xarray as xr
import numpy as np

class FermiMapImageWidget_Ui(object):

    def setupUi(self, obj):
        """
        used to setup the layout and initialize graphs
        """

        obj.layout = QVBoxLayout()
        obj.setLayout(obj.layout)
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        z = np.linspace(-1, 1, 51)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        d = np.sin(np.pi * np.exp(-1 * (xyz[0]**2 + xyz[1]**2 + xyz[2]**2))) * np.cos(np.pi / 2 * xyz[1])
        obj.xar = xr.DataArray(d, coords={"slit": x, 'perp': y, "energy": z}, dims=["slit", "perp", "energy"])
        obj.data = RegularDataArray(d, delta=[x[1] - x[0], y[1] - y[0], z[1] - z[0]], coord_min=[x[0], y[0], z[0]])
        obj.cut = obj.xar.sel({"perp": 0}, method='nearest')
        obj.fm_pyqtplot = PGImageTool(obj.data, layout=1)
        obj.fm_pyplot = PlotCanvas()
        obj.fm_pyplot.plot(obj.cut)
        obj.layout.addWidget(obj.fm_pyqtplot)
        obj.layout.addWidget(obj.fm_pyplot)
