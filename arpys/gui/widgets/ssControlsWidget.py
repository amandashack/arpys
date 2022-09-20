import logging

from gui.widgets.ssControlsWidgetUi import SingleScanControls_Ui
from PyQt5.QtWidgets import QFrame

log = logging.getLogger(__name__)


class SingleScanControlsWidget(QFrame, SingleScanControls_Ui):
    def __init__(self, context, signals):
        super(SingleScanControlsWidget, self).__init__()
        self.signals = signals
        self.context = context
        self.setupUi(self)
        self.set_k_space_options()
        self.set_plot_options()
        self.make_connections()

    def set_k_space_options(self):
        self.rdbttn_real.setChecked(self.context.real_space)
        self.rdbttn_normal.setChecked(self.context.normal_across_slit)
        self.le_across_slit_offset.setText(str(self.context.across_slit))
        self.le_along_slit_offset.setText(str(self.context.along_slit))
        self.le_azimuth_offset.setText(str(self.context.azimuth))

    def set_plot_options(self):
        self.rdbttn_custom_stylesheet.setChecked(self.context.custom_stylesheet)
        self.le_xmin.setText(str(self.context.master_dict["x_min"]["single"]))
        self.le_xmax.setText(str(self.context.master_dict['x_max']["single"]))
        self.le_ymin.setText(str(self.context.master_dict['y_min']['single']))
        self.le_ymax.setText(str(self.context.master_dict['y_max']['single']))
        self.le_aspect_ratio.setText(str(self.context.aspect_ratio))
        self.le_title.setText(str(self.context.master_dict['title']['single']))
        self.le_x_label.setText(str(self.context.master_dict['x_label']['single']))
        self.le_y_label.setText(str(self.context.master_dict['y_label']['single']))
        self.cbox_color_map.setCurrentText(self.context.master_dict['color_map']['single'])
        self.rdbttn_auto_normalization.setChecked(self.context.auto_normalize)
        self.cbox_normalization.setCurrentText(self.context.normalization_type)

    def make_connections(self):
        # QUESTION: does checkVal only emit when enter is pressed? need to
        # decide if I want to have the Apply button or not. Seems unnecessary
        self.le_across_slit_offset.checkVal.connect(self.context.update_axslit_offset)
        self.le_along_slit_offset.checkVal.connect(self.context.update_alslit_offset)
        self.le_azimuth_offset.checkVal.connect(self.context.update_azimuth_offset)
        self.le_xmin.checkVal.connect(lambda x: self.context.update_xmin(x, "single"))
        self.le_xmax.checkVal.connect(lambda x: self.context.update_xmax(x, "single"))
        self.le_ymin.checkVal.connect(lambda x: self.context.update_ymin(x, "single"))
        self.le_ymax.checkVal.connect(lambda x: self.context.update_ymax(x, "single"))
        #self.le_aspect_ratio.checkVal.connect(self.context.update_aspect_ratio)
        self.le_x_label.textChanged.connect(lambda x: self.context.update_x_label(x, "single"))
        self.le_y_label.textChanged.connect(lambda x: self.context.update_y_label(x, "single"))
        self.le_title.textChanged.connect(lambda x: self.context.update_title(x, "single"))
        self.le_x_label.returnPressed.connect(self.update_xyt)
        self.le_y_label.returnPressed.connect(self.update_xyt)
        self.le_title.returnPressed.connect(self.update_xyt)
        self.bttngrp1.buttonClicked.connect(self.checkBttn)
        self.bttngrp2.buttonClicked.connect(self.checkBttn)
        self.bttngrp3.buttonClicked.connect(self.checkBttn)
        self.bttngrp4.buttonClicked.connect(self.checkBttn)

    def start_dewarping(self):
        self.context.start_dewarper("single")

    def update_xyt(self):
        self.context.update_xyt("single")

    def checkBttn(self, button):
        bttn = button.text()
        if bttn == "real space":
            self.context.update_real_space(True, "single")
        elif bttn == "k space":
            self.context.update_real_space(False, "single")
        elif bttn == "across slit normal":
            self.context.update_normal_across_slit(True, "single")
        elif bttn == "across slit off normal":
            self.context.update_normal_across_slit(False, "single")
        elif bttn == "Personal\nStylesheet":
            self.context.update_custom_stylesheet(False, "single")
            self.cbox_stylesheet.setEnabled(True)
        elif bttn == "Custom\nStylesheet":
            self.context.update_custom_stylesheet(True, "single")
            self.cbox_stylesheet.setEnabled(True)
            # TODO: connect pop up toolbar for stylesheet editing here
        elif bttn == "auto":
            self.context.update_auto_normalize(True, "single")
        elif bttn == "custom":
            self.context.update_auto_normalize(False, "single")
            # TODO: connect pop up toolbar for normalization editing here

    def setDefaultStyleSheet(self):
        # This should be done with a json file

        self.setStyleSheet("\
            Label {\
                qproperty-alignment: AlignCenter;\
                border: 1px solid #FF17365D;\
                border-top-left-radius: 15px;\
                border-top-right-radius: 15px;\
                background-color: #FF17365D;\
                padding: 5px 0px;\
                color: rgb(255, 255, 255);\
                max-height: 35px;\
                font-size: 14px;\
            }")
