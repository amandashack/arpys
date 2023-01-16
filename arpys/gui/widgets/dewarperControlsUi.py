from gui.widgets.basicWidgets import ComboBox, Label, LineEdit, QHLine, SpinBox
from PyQt5.QtWidgets import (QVBoxLayout, QSizePolicy, QLineEdit, QPushButton,
                             QHBoxLayout, QListWidget, QSpinBox, QGridLayout,
                             QSpacerItem)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class DewarperControls_Ui(object):

    def setupUi(self, obj):
        """
        used to set up the layout for controling parameters for dewarping
        dewarping parameters that are editable:
        - downsample along x and y directions
        - slice range along z axis, also controlable with bars
        - x-scaling and power in power part of loss function
        - x-scaling in log part of loss function
        - overall y subtraction
        - overall power scaling
        - polyfit order
        - buttons for run sinobj.layoute cut dewarping and for 3d dewarping
        - selector for which cut you would like to look at
        ------------------------------------
        possibly add later:
        - radiobutton to select whether the spectra is normalized
        - radiobutton to select whether to simply shift energy or to
          do a radial interpolation
        """
        
        
        # code for making the math equation at the top of the window
        """
        https://stackoverflow.com/questions/14097463/displaying-nicely-an-algebraic-expression-in-pyqt
        """

        obj.layout = QGridLayout()
        obj.setLayout(obj.layout)

        obj.eloss_def = Label("E = placeholder")
        
        obj.lbl_x_downsample = Label("Downsample x \n(slit or photon_energy): ")
        obj.sb_x_downsample = QSpinBox()
        obj.sb_x_downsample.setValue(1)
        obj.sb_x_downsample.setMinimum(1)
        obj.sb_x_downsample.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        obj.lbl_y_downsample = Label("Downsample y \n(perp or slit): ")
        obj.sb_y_downsample = QSpinBox()
        obj.sb_y_downsample.setValue(1)
        obj.sb_y_downsample.setMinimum(1)
        obj.sb_y_downsample.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # not completely sure how I want to do this yet, so this a placeholder
        obj.lbl_slice_range_min = Label("Select slice\n range minimum: ")
        obj.le_slice_range_min = LineEdit("-1")
        obj.le_slice_range_min.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_slice_range_min.valRange(-1, 0)
        obj.le_slice_range_min.setToolTip("Min energy range")

        obj.lbl_slice_range_max = Label("Select slice\n range maximum: ")
        obj.le_slice_range_max = LineEdit("1")
        obj.le_slice_range_max.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_slice_range_max.valRange(0, 1)
        obj.le_slice_range_max.setToolTip("Max energy range")

        obj.lbl_a = Label("Variable a: ")
        obj.le_a = LineEdit("5")
        obj.le_a.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_a.valRange(0, 100)
        obj.le_a.setToolTip("Set variable a")

        obj.lbl_b = Label("Variable b: ")
        obj.le_b = LineEdit("0.1")
        obj.le_b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_b.valRange(0, 100)
        obj.le_b.setToolTip("Set variable b")

        obj.lbl_power_c = Label("Power c: ")
        obj.le_power_c = LineEdit("4")
        obj.le_power_c.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        obj.le_power_c.valRange(0, 10)
        obj.le_power_c.setToolTip("Set variable c")

        obj.lbl_power_d = Label("Power d: ")
        obj.le_power_d = LineEdit("2")
        obj.le_power_d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_power_d.valRange(0, 10)
        obj.le_power_d.setToolTip("Set variable d")

        obj.lbl_subtraction_y = Label("Subtraction y: ")
        obj.le_subtraction_y = LineEdit("1.9")
        obj.le_subtraction_y.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_a.valRange(0, 100)
        obj.le_a.setToolTip("Set variable y")

        obj.lbl_polyfit_order = Label("Polyfit order: ")
        obj.le_polyfit_order = LineEdit("6")
        obj.le_polyfit_order.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_a.valRange(0, 10)
        obj.le_a.setToolTip("Set polynomial order for fitting")

        obj.bttn_fit_cut = QPushButton("Fit Current Cut")
        obj.bttn_fit_3d = QPushButton("Fit 3D")

        obj.layout_title = QHBoxLayout()
        obj.layout_x_downsample = QHBoxLayout()
        obj.layout_y_downsample = QHBoxLayout()
        obj.layout_slice_min = QHBoxLayout()
        obj.layout_slice_max = QHBoxLayout()
        obj.layout_variables = QHBoxLayout()
        obj.layout.addLayout(obj.layout_title, 0, 0, 1, 1)
        obj.layout.addLayout(obj.layout_x_downsample, 1, 0, 1, 1)
        obj.layout.addLayout(obj.layout_y_downsample, 1, 1, 1, 1)
        obj.layout.addLayout(obj.layout_slice_min, 2, 0, 1, 1)
        obj.layout.addLayout(obj.layout_slice_max, 2, 1, 1, 1)
        obj.layout.addLayout(obj.layout_variables, 3, 0, 1, 2)
        obj.layout.addWidget(obj.bttn_fit_cut, 4, 0, 1, 1)
        obj.layout.addWidget(obj.bttn_fit_3d, 4, 1, 1, 1)

        obj.layout_title.addWidget(obj.eloss_def)
        obj.layout_x_downsample.addWidget(obj.lbl_x_downsample)
        obj.layout_x_downsample.addWidget(obj.sb_x_downsample)
        obj.layout_y_downsample.addWidget(obj.lbl_y_downsample)
        obj.layout_y_downsample.addWidget(obj.sb_y_downsample)
        obj.layout_slice_min.addWidget(obj.lbl_slice_range_min)
        obj.layout_slice_min.addWidget(obj.le_slice_range_min)
        obj.layout_slice_max.addWidget(obj.lbl_slice_range_max)
        obj.layout_slice_max.addWidget(obj.le_slice_range_max)
        obj.layout_variables.addWidget(obj.lbl_a)
        obj.layout_variables.addWidget(obj.le_a)
        obj.layout_variables.addWidget(obj.lbl_b)
        obj.layout_variables.addWidget(obj.le_b)
        obj.layout_variables.addWidget(obj.lbl_power_c)
        obj.layout_variables.addWidget(obj.le_power_c)
        obj.layout_variables.addWidget(obj.lbl_power_d)
        obj.layout_variables.addWidget(obj.le_power_d)
        obj.layout_variables.addWidget(obj.lbl_subtraction_y)
        obj.layout_variables.addWidget(obj.le_subtraction_y)
        obj.layout_variables.addWidget(obj.lbl_polyfit_order)
        obj.layout_variables.addWidget(obj.le_polyfit_order)
        obj.layout.setRowStretch(0, 1)
        obj.layout.setRowStretch(1, 1)
        obj.layout.setRowStretch(2, 1)
        obj.layout.setRowStretch(3, 1)
        obj.layout.setColumnStretch(0, 1)
        obj.layout.setColumnStretch(1, 1)
