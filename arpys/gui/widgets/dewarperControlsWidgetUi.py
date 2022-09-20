from gui.widgets.basicWidgets import ComboBox, Label, LineEdit, QHLine, SpinBox
from PyQt5.QtWidgets import (QVBoxLayout, QSizePolicy, QLineEdit, QPushButton,
                             QHBoxLayout, QListWidget, QSpinBox, QGridLayout)


class DewarperControlsWidget_Ui(object):

    def setupUi(self, obj):
        """
        used to setup the layout and initialize graphs
        """

        obj.layout = QGridLayout()
        obj.setLayout(obj.layout)

        obj.lbl_z_position = Label("Select z position: ")
        obj.lw_z_position = SpinBox()
        obj.lw_z_position.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Maximum)

        obj.lbl_y_position = Label("Select y position: ")
        obj.lw_y_position = SpinBox()
        obj.lw_y_position.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Maximum)

        obj.lbl_z_bin = Label("Select z\nposition binning: ")
        obj.lw_z_bin = QSpinBox()
        obj.lw_z_bin.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Maximum)
        obj.lw_z_bin.setKeyboardTracking(False)

        obj.lbl_y_bin = Label("Select y\nposition binning: ")
        obj.lw_y_bin = QSpinBox()
        obj.lw_y_bin.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Maximum)
        obj.lw_y_bin.setKeyboardTracking(False)

        obj.bttn_fit_edc = QPushButton("Fit EDC")
        obj.bttn_change_fit_params = QPushButton("Change Fit Params")

        obj.layout_z_position = QHBoxLayout()
        obj.layout_y_position = QHBoxLayout()
        obj.layout_z_bin = QHBoxLayout()
        obj.layout_y_bin = QHBoxLayout()
        obj.layout.addLayout(obj.layout_z_position, 0, 0, 1, 1)
        obj.layout.addLayout(obj.layout_y_position, 0, 1, 1, 1)
        obj.layout.addLayout(obj.layout_z_bin, 0, 2, 1, 1)
        obj.layout.addLayout(obj.layout_y_bin, 0, 3, 1, 1)
        obj.layout.addWidget(obj.bttn_fit_edc, 1, 0, 1, 1)
        obj.layout.addWidget(obj.bttn_change_fit_params, 1, 1, 1, 1)

        obj.layout_z_position.addWidget(obj.lbl_z_position)
        obj.layout_z_position.addWidget(obj.lw_z_position)
        obj.layout_y_position.addWidget(obj.lbl_y_position)
        obj.layout_y_position.addWidget(obj.lw_y_position)
        obj.layout_z_bin.addWidget(obj.lbl_z_bin)
        obj.layout_z_bin.addWidget(obj.lw_z_bin)
        obj.layout_y_bin.addWidget(obj.lbl_y_bin)
        obj.layout_y_bin.addWidget(obj.lw_y_bin)