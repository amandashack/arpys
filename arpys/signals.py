import numpy as np
from PyQt5 import QtCore
import xarray


class Signals(QtCore.QObject):
    # emit in context
    # connect in fmImageWidget
    updateRealSpace = QtCore.pyqtSignal()
    # emit in context
    # connect in ImageWidgets
    axslitOffset = QtCore.pyqtSignal(float)
    alslitOffset = QtCore.pyqtSignal(float)
    azimuthOffset = QtCore.pyqtSignal(float)
    updateXYTLabel = QtCore.pyqtSignal(list, str)
    workFunctionChanged = QtCore.pyqtSignal(float, str)
    innerPotentialChanged = QtCore.pyqtSignal(float, str)
    hvChanged = QtCore.pyqtSignal()
    # emit in context
    # connect in fmImageWidget
    updateData = QtCore.pyqtSignal(str)
    # emit in fileLoaderWidget
    # connect in fileLoaderWindow
    closeLoader = QtCore.pyqtSignal()
    # emit in context
    # connect in mainWindow
    startEDC = QtCore.pyqtSignal(str)
    # emit in context
    # connect in all image widgets
    axesChanged = QtCore.pyqtSignal(list, str)
    # emit in Context
    # connect in image widgets
    updateLines = QtCore.pyqtSignal(dict, str)
    # emit in basicWidgets
    # connect in EDC widget
    hbttnReleased = QtCore.pyqtSignal(float)
    # emit in basicWidgets
    # connect in EDC widget
    vbttnReleased = QtCore.pyqtSignal(float, int)
    # emit in paramsTableWindow
    # connect in EDCImageWidget
    tableData = QtCore.pyqtSignal(list)
    # ------------------------------------------------------
    # signals for dewarper
    # ------------------------------------------------------
    # emit in context
    # connect in mainWindow
    startDewarper = QtCore.pyqtSignal(str)
