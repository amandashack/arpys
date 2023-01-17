from gui.widgets.basicWidgets import ComboBox, Label, LineEdit, QHLine, SpinBox
from PyQt5.QtWidgets import (QVBoxLayout, QSizePolicy, QPushButton,
                             QHBoxLayout, QSpinBox, QGridLayout,
                             QCheckBox, QLineEdit, QRadioButton,
                             QButtonGroup)
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

        obj.eloss_def = Label("E = (((A*x)^C)/2 - Ln(B*x) - Y)^2")

        obj.cb_conv_binding = QCheckBox("Currently in Binding Energy")
        obj.cb_conv_binding.setChecked(True)
        obj.lbl_hv = Label("Photon Energy: ")
        obj.le_hv = LineEdit("0")
        obj.le_hv.setEnabled(False)
        obj.bttn_convert = QPushButton("Convert")
        obj.bttn_convert.setEnabled(False)
        
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

        obj.lbl_cut = Label("Select Cut Value: ")
        obj.le_cut = LineEdit("-1")
        obj.le_cut.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_cut.valRange(-1, 1)
        obj.le_cut.setToolTip("Cut value")

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

        obj.lbl_min_eloss = Label("Minimum X of\n Loss function is:")
        obj.le_min_eloss = QLineEdit("0")
        obj.le_min_eloss.setReadOnly(True)

        obj.lbl_min_eloss_y = Label("y value at minimum:")
        obj.le_min_eloss_y = QLineEdit("0")
        obj.le_min_eloss_y.setReadOnly(True)

        obj.lbl_a = Label("Variable a: ")
        obj.le_a = LineEdit("25")
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

        obj.lbl_subtraction_y = Label("Subtraction y: ")
        obj.le_subtraction_y = LineEdit("1.9")
        obj.le_subtraction_y.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_subtraction_y.valRange(0, 100)
        obj.le_subtraction_y.setToolTip("Set variable y")

        obj.lbl_polyfit_order = Label("Polyfit order: ")
        obj.le_polyfit_order = LineEdit("6")
        obj.le_polyfit_order.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_polyfit_order.valRange(0, 10)
        obj.le_polyfit_order.setToolTip("Set polynomial order for fitting")

        obj.lbl_threshold = Label("Threshold: ")
        obj.le_threshold = LineEdit("0.001")
        obj.le_threshold.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        obj.le_threshold.valRange(0, 10)
        obj.le_threshold.setToolTip("Set the threshold for identifying points\n found where there's no fermi edge")

        obj.lbl_info_bar = Label("Read-Only Info Bar: ")
        obj.lbl_info_bar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        obj.le_info_bar = QLineEdit("This is for displaying messages to the user.")
        obj.le_info_bar.setReadOnly(True)

        obj.bttn_fit_cut = QPushButton("Fit Current Cut")
        obj.bttn_preview = QPushButton("Preview Cumulative Integral")
        obj.bttn_fit_3d = QPushButton("Fit 3D")

        obj.rdbttn_group = QButtonGroup()
        obj.rdbttn_og = QRadioButton("Plot Original/downsampled Data")
        obj.rdbttn_og.setChecked(True)
        obj.rdbttn_dewarped = QRadioButton("Plot Dewarped Data")
        obj.rdbttn_dewarped.setChecked(False)
        obj.rdbttn_group.addButton(obj.rdbttn_og, id=1)
        obj.rdbttn_group.addButton(obj.rdbttn_dewarped, id=0)

        obj.bttn_pop_out = QPushButton("Pop out current spectra")

        obj.layout_title = QHBoxLayout()
        obj.layout_convert = QHBoxLayout()
        obj.layout_downsample = QHBoxLayout()
        obj.layout_cut_min_max = QHBoxLayout()
        obj.layout_info = QHBoxLayout()
        obj.layout_variables = QHBoxLayout()
        obj.layout_bttns = QHBoxLayout()
        obj.layout_rdbttns = QHBoxLayout()
        obj.hline1 = QHLine()
        obj.hline2 = QHLine()
        obj.hline3 = QHLine()
        obj.hline4 = QHLine()
        obj.layout.addLayout(obj.layout_title, 0, 0, 1, 1)
        obj.layout.addWidget(obj.hline1, 1, 0, 1, 2)
        obj.layout.addLayout(obj.layout_convert, 2, 0, 1, 2)
        obj.layout.addWidget(obj.hline2, 3, 0, 1, 2)
        obj.layout.addWidget(obj.hline4, 4, 0, 1, 2)
        obj.layout.addLayout(obj.layout_downsample, 5, 0, 2, 1)
        obj.layout.addLayout(obj.layout_cut_min_max, 7, 0, 1, 2)
        obj.layout.addLayout(obj.layout_variables, 8, 0, 1, 2)
        obj.layout.addWidget(obj.hline3, 9, 0, 1, 2)
        obj.layout.addLayout(obj.layout_info, 10, 0, 1, 2)
        obj.layout.addLayout(obj.layout_bttns, 11, 0, 1, 2)
        obj.layout.addLayout(obj.layout_rdbttns, 12, 0, 1, 1)
        obj.layout.addWidget(obj.bttn_pop_out, 12, 1, 1, 1)

        obj.layout_title.addWidget(obj.eloss_def)
        obj.layout_convert.addWidget(obj.cb_conv_binding, stretch=2)
        obj.layout_convert.addWidget(obj.lbl_hv, stretch=1)
        obj.layout_convert.addWidget(obj.le_hv, stretch=2)
        obj.layout_convert.addWidget(obj.bttn_convert, stretch=4)
        obj.layout_downsample.addWidget(obj.lbl_x_downsample)
        obj.layout_downsample.addWidget(obj.sb_x_downsample)
        obj.layout_downsample.addWidget(obj.lbl_y_downsample)
        obj.layout_downsample.addWidget(obj.sb_y_downsample)
        obj.layout_cut_min_max.addWidget(obj.lbl_cut)
        obj.layout_cut_min_max.addWidget(obj.le_cut)
        obj.layout_cut_min_max.addWidget(obj.lbl_slice_range_min)
        obj.layout_cut_min_max.addWidget(obj.le_slice_range_min)
        obj.layout_cut_min_max.addWidget(obj.lbl_slice_range_max)
        obj.layout_cut_min_max.addWidget(obj.le_slice_range_max)
        obj.layout_variables.addWidget(obj.lbl_min_eloss)
        obj.layout_variables.addWidget(obj.le_min_eloss)
        obj.layout_variables.addWidget(obj.lbl_min_eloss_y)
        obj.layout_variables.addWidget(obj.le_min_eloss_y)
        obj.layout_variables.addWidget(obj.lbl_a)
        obj.layout_variables.addWidget(obj.le_a)
        obj.layout_variables.addWidget(obj.lbl_b)
        obj.layout_variables.addWidget(obj.le_b)
        obj.layout_variables.addWidget(obj.lbl_power_c)
        obj.layout_variables.addWidget(obj.le_power_c)
        obj.layout_variables.addWidget(obj.lbl_subtraction_y)
        obj.layout_variables.addWidget(obj.le_subtraction_y)
        obj.layout_variables.addWidget(obj.lbl_polyfit_order)
        obj.layout_variables.addWidget(obj.le_polyfit_order)
        obj.layout_variables.addWidget(obj.lbl_threshold)
        obj.layout_variables.addWidget(obj.le_threshold)
        obj.layout_info.addWidget(obj.lbl_info_bar, stretch=1)
        obj.layout_info.addWidget(obj.le_info_bar, stretch=9)
        obj.layout_bttns.addWidget(obj.bttn_fit_cut)
        obj.layout_bttns.addWidget(obj.bttn_preview)
        obj.layout_bttns.addWidget(obj.bttn_fit_3d)
        obj.layout_rdbttns.addWidget(obj.rdbttn_og)
        obj.layout_rdbttns.addWidget(obj.rdbttn_dewarped)
        obj.layout.setRowStretch(0, 1)
        obj.layout.setRowStretch(1, 1)
        obj.layout.setRowStretch(2, 1)
        obj.layout.setRowStretch(3, 1)
        obj.layout.setColumnStretch(0, 1)
        obj.layout.setColumnStretch(1, 1)
