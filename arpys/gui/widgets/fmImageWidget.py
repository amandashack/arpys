import logging
from users.ajshack import convert_2d_normal_emission
from gui.widgets.fmImageWidgetUi import FermiMapImageWidget_Ui
from PyQt5.QtWidgets import QFrame
from pyimagetool import imagetool
from pyimagetool import RegularDataArray

log = logging.getLogger(__name__)

# TODO: for converting to k-space for a single cut, you must know the photon energy -
#  Try to get it from the metadata and if not, have a pop up which asks for the
#  photon energy.


class FermiMapImageWidget(QFrame, FermiMapImageWidget_Ui):

    def __init__(self, context, signals):
        super(FermiMapImageWidget, self).__init__()
        self.signals = signals
        self.context = context
        self.setupUi(self)
        self.xar = None
        self.data = None
        self.k_data = None
        self.k = False
        self.dims = []
        self.initialize_vals()
        self.connect_signals()

    def initialize_vals(self):
        pass

    def connect_signals(self):
        self.signals.fmData.connect(self.handle_plotting)
        self.signals.updateRealSpace.connect(self.convert_k)
        self.signals.axslitOffset.connect(self.update_axslit)
        self.signals.alslitOffset.connect(self.update_alslit)
        self.signals.azimuthOffset.connect(self.update_azimuth)

    def update_axslit(self, axs):
        """function for shifting across slit"""
        pass

    def update_alslit(self, als):
        """function for shifting along slit"""
        pass

    def update_azimuth(self, az):
        """function for shifting in azimuth"""
        pass

    def convert_k(self, k_space):
        if k_space:
            # convert to k
            # if single cut:
            pass
        else:
            # go back to real - not convert, just use original data
            pass

    def handle_plotting(self, xar):
        self.xar = xar
        self.data = RegularDataArray(self.xar)
        print(self.data)
        self.refresh_plots()

    def refresh_plots(self):
        # self.fm_pyqtplot.plt.setData(self.data[0], self.data[1])
        self.fm_pyplot.plot(self.xar)

    def plot_data(self, buf):
        pass

    def set_x_axis(self):
        pass
