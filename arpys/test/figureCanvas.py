import sys
from PyQt5.Qt import Qt, QObject, QPen, QPointF
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsLineItem, QGraphicsView, \
    QGraphicsScene, QWidget, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import xarray as xr
import numpy as np


class Signals(QObject):
    bttnReleased = pyqtSignal(float)


class HLineItem(QGraphicsLineItem):

    def __init__(self, signals):
        super(HLineItem, self).__init__()
        self.signals = signals
        self.setPen(QPen(Qt.red, 3))
        self.setFlag(QGraphicsLineItem.ItemIsMovable)
        self.setCursor(Qt.OpenHandCursor)
        self.setAcceptHoverEvents(True)

    def mouseMoveEvent(self, event):
        orig_cursor_position = event.lastScenePos()
        updated_cursor_position = event.scenePos()

        orig_position = self.scenePos()
        updated_cursor_y = updated_cursor_position.y() - \
                           orig_cursor_position.y() + orig_position.y()
        self.setPos(QPointF(orig_position.x(), updated_cursor_y))

    def mouseReleaseEvent(self, event):
        y_pos = event.scenePos().y()
        self.signals.bttnReleased.emit(y_pos)


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

    def plot(self, data):
        self.data = data
        self.axes = self.fig.add_subplot(111)
        self.data.plot(ax=self.axes)
        self.fig.subplots_adjust(left=0.2)
        self.fig.subplots_adjust(bottom=0.2)
        self.axes.set_xlim(-.5, .5)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.signals = Signals()
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        z = np.linspace(-1, 1, 51)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        d = np.sin(np.pi * np.exp(-1 * (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))) * np.cos(np.pi / 2 * xyz[1])
        self.xar = xr.DataArray(d, coords={"slit": x, 'perp': y, "energy": z}, dims=["slit", "perp", "energy"])
        self.cut = self.xar.sel({"perp": 0}, method='nearest')
        self.edc = self.cut.sel({'slit': 0}, method='nearest')
        self.canvas = PlotCanvas()
        self.canvas_edc = PlotCanvas()
        self.canvas.plot(self.cut)
        self.canvas_edc.plot(self.edc)

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.line = HLineItem(self.signals)
        self.line_pos = [0, 0]
        self.layout1 = QHBoxLayout()
        self.layout2 = QHBoxLayout()
        self.connect_scene()

        self.layout1.addWidget(self.view)
        self.layout2.addWidget(self.canvas_edc)
        self.central = QWidget()
        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.layout1)
        self.main_layout.addLayout(self.layout2)
        self.central.setLayout(self.main_layout)
        self.setCentralWidget(self.central)

        self.signals.bttnReleased.connect(self.plot_position)

    def connect_scene(self):
        s = self.canvas.figure.get_size_inches() * self.canvas.figure.dpi
        self.view.setScene(self.scene)
        self.scene.addWidget(self.canvas)
        self.scene.setSceneRect(0, 0, s[0], s[1])
        # self.capture_scene_change()
        self.line.setLine(0, 0, self.scene.sceneRect().width(), 0)
        self.line.setPos(self.line_pos[0], self.line_pos[1])
        self.scene.addItem(self.line)

    def handle_plotting(self):
        self.clearLayout(self.layout2)
        self.refresh_edc()

    def refresh_edc(self):
        self.canvas_edc = PlotCanvas()
        self.canvas_edc.plot(self.edc)
        self.layout2.addWidget(self.canvas_edc)

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def plot_position(self, y):
        rel_pos = lambda x: abs(self.scene.sceneRect().height() - x)
        bbox = self.canvas.axes.spines['left'].get_window_extent()
        plot_bbox = [bbox.y0, bbox.y1]
        if rel_pos(y) < plot_bbox[0]:
            self.line.setPos(0, rel_pos(plot_bbox[0]))
        elif rel_pos(y) > plot_bbox[1]:
            self.line.setPos(0, rel_pos(plot_bbox[1]))
        self.line_pos = self.line.pos().y()
        size_range = len(self.cut.slit)
        r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
        corr = list(zip(r, self.cut.slit.values))
        sel_val = min(r, key=lambda f: abs(f - rel_pos(self.line_pos)))
        what_index = r.index(sel_val)
        self.edc = self.cut.sel({"slit": corr[what_index][1]}, method='nearest')
        self.handle_plotting()


class App(QApplication):
    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        self.setAttribute(Qt.AA_EnableHighDpiScaling)
        self.mainWindow = MainWindow()
        self.mainWindow.setWindowTitle("arpys")
        self.mainWindow.show()


def main():

    app = App(sys.argv)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
