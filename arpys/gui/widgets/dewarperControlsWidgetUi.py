from gui.widgets.basicWidgets import ComboBox, Label, LineEdit, QHLine, SpinBox
from PyQt5.QtWidgets import (QVBoxLayout, QSizePolicy, QLineEdit, QPushButton,
                             QHBoxLayout, QListWidget, QSpinBox)


class DewarperControlsWidget_Ui(object):

    def setupUi(self, obj):
        """
        used to setup the layout and initialize graphs
        """

        obj.layout = QHBoxLayout()
        obj.setLayout(obj.layout)

        obj.lbl_x_position = Label("Select x position: ")
        obj.lw_x_position = SpinBox()
        obj.lw_x_position.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Maximum)

        obj.lbl_y_position = Label("Select y position: ")
        obj.lw_y_position = SpinBox()
        obj.lw_y_position.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Maximum)

        obj.lbl_x_bin = Label("Select x\nposition binning: ")
        obj.lw_x_bin = QSpinBox()
        obj.lw_x_bin.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Maximum)

        obj.lbl_y_bin = Label("Select y\nposition binning: ")
        obj.lw_y_bin = QSpinBox()
        obj.lw_y_bin.setSizePolicy(QSizePolicy.Expanding,
                                   QSizePolicy.Maximum)

        obj.layout_x_position = QHBoxLayout()
        obj.layout_y_position = QHBoxLayout()
        obj.layout_x_bin = QHBoxLayout()
        obj.layout_y_bin = QHBoxLayout()
        obj.layout.addLayout(obj.layout_x_position)
        obj.layout.addLayout(obj.layout_y_position)
        obj.layout.addLayout(obj.layout_x_bin)
        obj.layout.addLayout(obj.layout_y_bin)

        obj.layout_x_position.addWidget(obj.lbl_x_position)
        obj.layout_x_position.addWidget(obj.lw_x_position)
        obj.layout_y_position.addWidget(obj.lbl_y_position)
        obj.layout_y_position.addWidget(obj.lw_y_position)
        obj.layout_x_bin.addWidget(obj.lbl_x_bin)
        obj.layout_x_bin.addWidget(obj.lw_x_bin)
        obj.layout_y_bin.addWidget(obj.lbl_y_bin)
        obj.layout_y_bin.addWidget(obj.lw_y_bin)