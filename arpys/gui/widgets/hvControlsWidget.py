import logging

from gui.widgets.hvControlsWidgetUi import HVScanControls_Ui
from PyQt5.QtWidgets import QFrame

log = logging.getLogger(__name__)


class HVScanControlsWidget(QFrame, HVScanControls_Ui):
    def __init__(self, context, signals):
        super(HVScanControlsWidget, self).__init__()
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
        self.le_xmin.setText(str(self.context.master_dict["x_min"]["hv_scan"]))
        self.le_xmax.setText(str(self.context.master_dict['x_max']["hv_scan"]))
        self.le_ymin.setText(str(self.context.master_dict['y_min']['hv_scan']))
        self.le_ymax.setText(str(self.context.master_dict['y_max']['hv_scan']))
        self.le_aspect_ratio.setText(str(self.context.aspect_ratio))
        self.le_title.setText(str(self.context.master_dict['title']['hv_scan']))
        self.le_x_label.setText(str(self.context.master_dict['x_label']['hv_scan']))
        self.le_y_label.setText(str(self.context.master_dict['y_label']['hv_scan']))
        self.cbox_color_map.setCurrentText(self.context.master_dict['color_map']['hv_scan'])
        self.rdbttn_auto_normalization.setChecked(self.context.auto_normalize)
        self.cbox_normalization.setCurrentText(self.context.normalization_type)

    def make_connections(self):
        # QUESTION: does checkVal only emit when enter is pressed? need to
        # decide if I want to have the Apply button or not. Seems unnecessary
        self.le_across_slit_offset.checkVal.connect(self.context.update_axslit_offset)
        self.le_along_slit_offset.checkVal.connect(self.context.update_alslit_offset)
        self.le_azimuth_offset.checkVal.connect(self.context.update_azimuth_offset)
        self.le_xmin.checkVal.connect(lambda x: self.context.update_xmin(x, "hv_scan"))
        self.le_xmax.checkVal.connect(lambda x: self.context.update_xmax(x, "hv_scan"))
        self.le_ymin.checkVal.connect(lambda x: self.context.update_ymin(x, "hv_scan"))
        self.le_ymax.checkVal.connect(lambda x: self.context.update_ymax(x, "hv_scan"))
        #self.le_aspect_ratio.checkVal.connect(self.context.update_aspect_ratio)
        self.le_x_label.textChanged.connect(lambda x: self.context.update_x_label(x, "hv_scan"))
        self.le_y_label.textChanged.connect(lambda x: self.context.update_y_label(x, "hv_scan"))
        self.le_title.textChanged.connect(lambda x: self.context.update_title(x, "hv_scan"))
        self.le_x_label.returnPressed.connect(self.update_xyt)
        self.le_y_label.returnPressed.connect(self.update_xyt)
        self.le_title.returnPressed.connect(self.update_xyt)
        self.bttngrp1.buttonClicked.connect(self.checkBttn)
        self.bttngrp2.buttonClicked.connect(self.checkBttn)
        self.bttngrp3.buttonClicked.connect(self.checkBttn)
        self.bttngrp4.buttonClicked.connect(self.checkBttn)

    def update_xyt(self):
        self.context.update_xyt("hv_scan")

    def checkBttn(self, button):
        bttn = button.text()
        if bttn == "real space":
            self.context.update_real_space(True)
        elif bttn == "k space":
            self.context.update_real_space(False)
        elif bttn == "across slit normal":
            self.context.update_normal_across_slit(True)
        elif bttn == "across slit off normal":
            self.context.update_normal_across_slit(False)
        elif bttn == "Personal\nStylesheet":
            self.context.update_custom_stylesheet(False)
            self.cbox_stylesheet.setEnabled(True)
        elif bttn == "Custom\nStylesheet":
            self.context.update_custom_stylesheet(True)
            self.cbox_stylesheet.setEnabled(True)
            # TODO: connect pop up toolbar for stylesheet editing here
        elif bttn == "auto":
            self.context.update_auto_normalize(True)
        elif bttn == "custom":
            self.context.update_auto_normalize(False)
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
