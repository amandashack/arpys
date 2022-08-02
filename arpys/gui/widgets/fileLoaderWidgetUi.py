from gui.widgets.basicWidgets import ComboBox, Label, LineEdit, QHLine
from PyQt5.QtWidgets import (QVBoxLayout, QSizePolicy, QLineEdit, QPushButton,
                             QHBoxLayout)


class FileLoaderWidget_Ui(object):

    def setupUi(self, obj):
        """
        used to setup the layout and initialize graphs
        """

        obj.layout = QVBoxLayout()
        obj.setLayout(obj.layout)

        obj.lbl_beamline = Label("Select beamline: ")
        obj.cbox_beamline = ComboBox()
        obj.cbox_beamline.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Preferred)

        obj.lbl_loader_type = Label("Select type of Loader: ")
        obj.cbox_loader_type = ComboBox()
        obj.cbox_loader_type.setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Preferred)

        obj.lbl_select_file = Label("Select file: ")
        obj.le_select_file = QLineEdit()
        obj.bttn_select_file = QPushButton("Select")

        obj.bttn_loader = QPushButton("Load")

        obj.layout_beamline = QHBoxLayout()
        obj.layout_loader = QHBoxLayout()
        obj.layout_select = QHBoxLayout()
        obj.layout.addLayout(obj.layout_beamline)
        obj.layout.addLayout(obj.layout_loader)
        obj.layout.addLayout(obj.layout_select)

        obj.layout_beamline.addWidget(obj.lbl_beamline)
        obj.layout_beamline.addWidget(obj.cbox_beamline)
        obj.layout_loader.addWidget(obj.lbl_loader_type)
        obj.layout_loader.addWidget(obj.cbox_loader_type)
        obj.layout_select.addWidget(obj.lbl_select_file, 25)
        obj.layout_select.addWidget(obj.le_select_file, 50)
        obj.layout_select.addWidget(obj.bttn_select_file, 25)
        obj.layout.addWidget(obj.bttn_loader)
