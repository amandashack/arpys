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
    updateXYTLabel = QtCore.pyqtSignal(str, list)
    # emit in context
    # connect in fmImageWidget
    hvData = QtCore.pyqtSignal(xarray.DataArray)
    fmData = QtCore.pyqtSignal(xarray.DataArray)
    ssData = QtCore.pyqtSignal(xarray.DataArray)
    # emit in fileLoaderWidget
    # connect in fileLoaderWindow
    closeLoader = QtCore.pyqtSignal()
    # emit in context
    # connect in mainWindow
    startDewarper = QtCore.pyqtSignal(str)
    # emit in context
    # connect in all image widgets
    axesChanged = QtCore.pyqtSignal(str, list)
