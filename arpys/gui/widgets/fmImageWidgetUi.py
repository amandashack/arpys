import pyqtgraph as pg
from fmPlotting import fmPlotWidget, PlotCanvas, graph_setup
from PyQt5.QtWidgets import QVBoxLayout


class FermiMapImageWidget_Ui(object):

    def setupUi(self, obj):
        """
        used to setup the layout and initialize graphs
        """

        obj.layout = QVBoxLayout()
        obj.setLayout(obj.layout)
        obj.fm_pyqtplot = fmPlotWidget(obj.context, obj.signals)
        obj.fm_pyplot = PlotCanvas(obj.context, obj.signals)
        graph_setup(obj.fm_pyqtplot, "Single Cut",
                    "I/I\N{SUBSCRIPT ZERO}", pg.mkPen(width=5, color='r'))
        #graph_setup(obj.i0_graph, "Initial Intensity", "I\N{SUBSCRIPT ZERO}",
        #            pg.mkPen(width=5, color='b'))
        #graph_setup(obj.diff_graph, "Intensity at the Detector",
        #            "Diffraction Intensity", pg.mkPen(width=5, color='g'))
        #add_calibration_graph(obj.ratio_graph)
        #add_calibration_graph(obj.i0_graph)
        #add_calibration_graph(obj.diff_graph)
        #obj.i0_graph.setXLink(obj.ratio_graph)
        #obj.diff_graph.setXLink(obj.ratio_graph)
        obj.layout.addWidget(obj.fm_pyqtplot)
        obj.layout.addWidget(obj.fm_pyplot)
