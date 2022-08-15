from gui.widgets.basicWidgets import (CollapsibleBox, ComboBox, Label,
                                            LineEdit, QHLine)
from PyQt5.QtWidgets import (QButtonGroup, QFrame, QGridLayout, QHBoxLayout,
                             QLCDNumber, QPushButton, QRadioButton,
                             QSizePolicy, QTextEdit, QVBoxLayout, QLineEdit)
from PyQt5.QtCore import Qt


class SingleScanControls_Ui(object):
    def setupUi(self, obj):
        #####################################################################
        # set up user panel layout and give it a title
        #####################################################################
        obj.layout__single = QVBoxLayout(obj)
        obj.box_k_conv = CollapsibleBox("k conversion")
        obj.layout__single.addWidget(obj.box_k_conv)

        #####################################################################
        # set selections for converting to k-space
        #####################################################################

        obj.bttngrp1 = QButtonGroup()

        obj.rdbttn_real = QRadioButton("real space")
        obj.rdbttn_k = QRadioButton("k space")
        obj.rdbttn_real.setChecked(True)
        obj.bttngrp1.addButton(obj.rdbttn_real, id=1)
        obj.bttngrp1.addButton(obj.rdbttn_k, id=0)
        obj.bttngrp1.setExclusive(True)

        obj.bttngrp2 = QButtonGroup()
        obj.rdbttn_normal = QRadioButton("across slit\nnormal")
        obj.rdbttn_normal.setMinimumSize(10, 10)
        obj.rdbttn_off_normal = QRadioButton("across slit\noff normal")
        obj.rdbttn_off_normal.setMinimumSize(10, 10)
        obj.rdbttn_off_normal.setChecked(True)
        obj.bttngrp2.addButton(obj.rdbttn_off_normal, id=0)
        obj.bttngrp2.addButton(obj.rdbttn_normal, id=1)
        obj.bttngrp2.setExclusive(True)

        # setup layout
        ##############
        obj.layout_k_conv_params = QVBoxLayout()
        obj.layout_allrdbttns = QGridLayout()
        obj.layout_allrdbttns.setColumnStretch(0, 6)
        obj.layout_allrdbttns.setColumnStretch(1, 1)
        obj.layout_allrdbttns.setRowStretch(1, 5)
        obj.layout_allrdbttns.addWidget(obj.rdbttn_real, 0, 0)
        obj.layout_allrdbttns.addWidget(obj.rdbttn_k, 0, 1)
        obj.layout_allrdbttns.addWidget(obj.rdbttn_off_normal, 1, 0, 1, 2)
        obj.layout_allrdbttns.addWidget(obj.rdbttn_normal, 1, 1, 1, 2)
        obj.layout_allrdbttns.setRowMinimumHeight(0, 20)
        obj.layout_allrdbttns.setRowMinimumHeight(1, 30)
        obj.layout_k_conv_params.addLayout(obj.layout_allrdbttns)
        obj.layout_k_conv_params.addSpacing(5)

        #####################################################################
        # Allow user input for the offsets along the slit, across the slit,
        # and in the azimuthal direction. User can also indicate what the
        # slit orientation is for the k-conversion
        #####################################################################

        obj.lbl_along_slit_offset = Label("Along Slit Offset")
        obj.le_along_slit_offset = LineEdit("0")
        obj.le_along_slit_offset.valRange(-90, 90)
        obj.le_along_slit_offset.setToolTip('Sets the offset along the slit'
                                            ' in degrees. Values are allowed '
                                            'from -90 to 90 ')
        obj.lbl_across_slit_offset = Label("Across Slit Offset")
        obj.le_across_slit_offset = LineEdit('0')
        obj.le_across_slit_offset.valRange(-90, 90)
        obj.le_across_slit_offset.setToolTip('Sets the offset across the slit'
                                             ' in degrees. Values are allowed '
                                             'from -90 to 90 ')
        obj.lbl_azimuth_offset = Label('Azimuth Offset')
        obj.le_azimuth_offset = LineEdit("0")
        obj.le_azimuth_offset.valRange(-360, 360)
        obj.le_azimuth_offset.setToolTip('Sets the Azimuth offset in '
                                         'degrees. Values are allowed'
                                         'from -360 to 360')
        obj.lbl_slit_orientation = Label('Slit Orientation')
        obj.le_slit_orientation = LineEdit("0")
        obj.le_slit_orientation.valRange(0, 2)
        obj.le_slit_orientation.setToolTip('Sets the slit orientation. '
                                           '0=vertical slit, 1=horizontal slit,'
                                           ' 2=deflector+vertical slit')

        # setup layout
        ##############
        obj.layout_along_slit = QHBoxLayout()
        obj.layout_across_slit = QHBoxLayout()
        obj.layout_azimuth = QHBoxLayout()
        obj.layout_slit_orientation = QHBoxLayout()
        obj.layout_along_slit.addWidget(obj.lbl_along_slit_offset, 75, alignment=Qt.AlignLeft)
        obj.layout_along_slit.addWidget(obj.le_along_slit_offset, 25, alignment=Qt.AlignRight)
        obj.layout_across_slit.addWidget(obj.lbl_across_slit_offset, 75, alignment=Qt.AlignLeft)
        obj.layout_across_slit.addWidget(obj.le_across_slit_offset, 25, alignment=Qt.AlignRight)
        obj.layout_azimuth.addWidget(obj.lbl_azimuth_offset, 75, alignment=Qt.AlignLeft)
        obj.layout_azimuth.addWidget(obj.le_azimuth_offset, 25, alignment=Qt.AlignRight)
        obj.layout_slit_orientation.addWidget(obj.lbl_slit_orientation, 75, alignment=Qt.AlignLeft)
        obj.layout_slit_orientation.addWidget(obj.le_slit_orientation, 25, alignment=Qt.AlignRight)
        obj.hline1 = QHLine()
        obj.layout_k_conv_params.addLayout(obj.layout_along_slit)
        obj.layout_k_conv_params.addLayout(obj.layout_across_slit)
        obj.layout_k_conv_params.addLayout(obj.layout_azimuth)
        obj.layout_k_conv_params.addLayout(obj.layout_slit_orientation)
        obj.layout_k_conv_params.addWidget(obj.hline1)
        obj.box_k_conv.setContentLayout(obj.layout_k_conv_params)

        ###################################################################
        #  make section for editing the plot
        ###################################################################

        obj.box_plot_editor = CollapsibleBox("Plot Editor")
        obj.layout__single.addWidget(obj.box_plot_editor)

        obj.bttngrp3 = QButtonGroup()
        obj.rdbttn_pyplot_stylesheet = QRadioButton("Personal\nStylesheet")
        obj.rdbttn_custom_stylesheet = QRadioButton("Custom\nStylesheet")
        obj.rdbttn_pyplot_stylesheet.setChecked(True)
        obj.bttngrp3.addButton(obj.rdbttn_pyplot_stylesheet, id=1)
        obj.bttngrp3.addButton(obj.rdbttn_custom_stylesheet, id=0)
        obj.bttngrp3.setExclusive(True)

        obj.cbox_stylesheet = ComboBox()
        obj.cbox_stylesheet.setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Preferred)
        obj.cbox_stylesheet.setEnabled(True)
        obj.cbox_stylesheet.addItem("Dushyant")

        obj.lbl_xmin_max = Label("x-axis min-max")
        obj.lbl_ymin_max = Label("y-axis min-max")

        obj.le_xmin = LineEdit("-10")
        obj.le_xmin.valRange(-100, 100)
        obj.le_xmin.setToolTip('Min x-axis range.')

        obj.le_xmax = LineEdit("10")
        obj.le_xmax.valRange(-100, 100)
        obj.le_xmax.setToolTip('Max x-axis range.')

        obj.le_ymin = LineEdit("-10")
        obj.le_ymin.valRange(-100, 100)
        obj.le_ymin.setToolTip('Min y-axis')

        obj.le_ymax = LineEdit("10")
        obj.le_ymax.valRange(-100, 100)
        obj.le_ymax.setToolTip('Max y-axis')

        obj.lbl_aspect_ratio = Label("Plot Aspect Ratio")
        obj.le_aspect_ratio = LineEdit("1.5")
        obj.le_aspect_ratio.setToolTip('aspect ratio of the graph. '
                                       'Can take values between 0-5')
        obj.le_aspect_ratio.valRange(0, 5)

        obj.cbox_color_map = ComboBox()
        obj.cbox_color_map.setSizePolicy(QSizePolicy.Expanding,
                                         QSizePolicy.Preferred)
        obj.cbox_color_map.addItem("viridis")

        obj.lbl_colormap_normalization = Label("Colormap Normalization")
        obj.bttngrp4 = QButtonGroup()
        obj.rdbttn_auto_normalization = QRadioButton("auto")
        obj.rdbttn_custom_normalization = QRadioButton("custom")
        obj.bttngrp4.addButton(obj.rdbttn_auto_normalization, id=0)
        obj.bttngrp4.addButton(obj.rdbttn_custom_normalization, id=1)
        obj.rdbttn_auto_normalization.setChecked(True)
        obj.bttngrp4.setExclusive(True)

        obj.cbox_normalization = ComboBox()
        obj.cbox_normalization.setSizePolicy(QSizePolicy.Expanding,
                                             QSizePolicy.Preferred)
        obj.cbox_normalization.setEnabled(False)
        obj.cbox_normalization.addItem("linear")

        obj.lbl_title = Label("Title ")
        obj.le_title = QLineEdit("Title")

        obj.lbl_x_label = Label("X-axis Label")
        obj.le_x_label = QLineEdit("$k_(x)$ ($\AA^(-1)$)")

        obj.lbl_y_label = Label("Y-axis Label")
        obj.le_y_label = QLineEdit("Binding Energy")

        obj.bttn_apply = QPushButton("Apply")

        obj.layout_plot_tools = QVBoxLayout()
        obj.layout_stylesheet_rdbttns = QHBoxLayout()
        obj.layout_stylesheet_cbox = QHBoxLayout()
        obj.layout_x_y_axes = QGridLayout()
        obj.layout_title = QHBoxLayout()
        obj.layout_x_label = QHBoxLayout()
        obj.layout_y_label = QHBoxLayout()
        obj.layout_aspect_ratio = QHBoxLayout()
        obj.layout_color_map = QHBoxLayout()
        obj.layout_colormap_normalization = QHBoxLayout()
        obj.hline3 = QHLine()
        obj.hline4 = QHLine()
        obj.layout_plot_tools.addLayout(obj.layout_stylesheet_rdbttns)
        obj.layout_plot_tools.addLayout(obj.layout_stylesheet_cbox)
        obj.layout_plot_tools.addWidget(obj.hline3)
        obj.layout_plot_tools.addLayout(obj.layout_x_y_axes)
        obj.layout_plot_tools.addLayout(obj.layout_title)
        obj.layout_plot_tools.addLayout(obj.layout_x_label)
        obj.layout_plot_tools.addLayout(obj.layout_y_label)
        obj.layout_plot_tools.addLayout(obj.layout_aspect_ratio)
        obj.layout_plot_tools.addWidget(obj.hline4)
        obj.layout_plot_tools.addLayout(obj.layout_color_map)
        obj.layout_plot_tools.addWidget(obj.lbl_colormap_normalization)
        obj.layout_plot_tools.addLayout(obj.layout_colormap_normalization)
        obj.layout_plot_tools.addWidget(obj.bttn_apply)
        obj.layout_stylesheet_rdbttns.addWidget(obj.rdbttn_pyplot_stylesheet)
        obj.layout_stylesheet_rdbttns.addWidget(obj.rdbttn_custom_stylesheet)
        obj.layout_stylesheet_cbox.addWidget(obj.cbox_stylesheet)
        obj.layout_title.addWidget(obj.lbl_title)
        obj.layout_title.addWidget(obj.le_title)
        obj.layout_x_label.addWidget(obj.lbl_x_label)
        obj.layout_x_label.addWidget(obj.le_x_label)
        obj.layout_y_label.addWidget(obj.lbl_y_label)
        obj.layout_y_label.addWidget(obj.le_y_label)
        obj.layout_x_y_axes.addWidget(obj.lbl_xmin_max, 0, 0, 2, 1)
        obj.layout_x_y_axes.addWidget(obj.le_xmin, 0, 1, 2, 1)
        obj.layout_x_y_axes.addWidget(obj.le_xmax, 0, 2, 2, 1)
        obj.layout_x_y_axes.addWidget(obj.lbl_ymin_max, 3, 0, 2, 1)
        obj.layout_x_y_axes.addWidget(obj.le_ymin, 3, 1, 2, 1)
        obj.layout_x_y_axes.addWidget(obj.le_ymax, 3, 2, 2, 1)
        obj.layout_aspect_ratio.addWidget(obj.lbl_aspect_ratio)
        obj.layout_aspect_ratio.addWidget(obj.le_aspect_ratio)
        obj.layout_color_map.addWidget(obj.cbox_color_map)
        obj.layout_colormap_normalization.addWidget(obj.cbox_normalization)
        obj.box_plot_editor.setContentLayout(obj.layout_plot_tools)

        #######################################################################
        # text area for giving updates the user can see
        #######################################################################

        obj.text_area = QTextEdit("~~~Read Only information for user~~~")
        obj.text_area.setReadOnly(True)
        obj.layout__single.addWidget(obj.text_area)
