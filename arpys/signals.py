import numpy as np
from PyQt5 import QtCore
import xarray


class Signals(QtCore.QObject):
    # emit in context
    # connect in fmImageWidget
    updateRealSpace = QtCore.pyqtSignal(bool)
    # emit in context
    # connect in fmImageWidget
    axslitOffset = QtCore.pyqtSignal(float)
    alslitOffset = QtCore.pyqtSignal(float)
    azimuthOffset = QtCore.pyqtSignal(float)
    # emit in context
    # connect in fmImageWidget
    fmData = QtCore.pyqtSignal(xarray.DataArray)
