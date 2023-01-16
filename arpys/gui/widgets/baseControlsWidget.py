import logging

from gui.widgets.baseControlsWidgetUi import BaseControls_Ui
from PyQt5.QtWidgets import QFrame, QFontDialog, QColorDialog
from PyQt5.QtGui import QFont

log = logging.getLogger(__name__)


class BaseControlsWidget(QFrame, BaseControls_Ui):
    def __init__(self, context, signals, context_type):
        super(BaseControlsWidget, self).__init__()
        self.signals = signals
        self.context = context
        self.context_type = context_type
        self.setupUi(self)
        self.lines = {}
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
        self.le_workfunction.setText(str(self.context.work_function))
        self.le_inner_potential.setText(str(self.context.inner_potential))
        self.le_photon_energy.setText(str(self.context.master_dict["hv"][self.context_type]))
        self.le_xmin.setText(str(self.context.master_dict["x_min"][self.context_type]))
        self.le_xmax.setText(str(self.context.master_dict['x_max'][self.context_type]))
        self.le_ymin.setText(str(self.context.master_dict['y_min'][self.context_type]))
        self.le_ymax.setText(str(self.context.master_dict['y_max'][self.context_type]))
        self.le_aspect_ratio.setText(str(self.context.aspect_ratio))
        self.le_title.setText(str(self.context.master_dict['title'][self.context_type]))
        self.le_x_label.setText(str(self.context.master_dict['x_label'][self.context_type]))
        self.le_y_label.setText(str(self.context.master_dict['y_label'][self.context_type]))
        self.cbox_color_map.setCurrentText(self.context.master_dict['color_map'][self.context_type])
        self.rdbttn_auto_normalization.setChecked(self.context.auto_normalize)
        self.cbox_normalization.setCurrentText(self.context.normalization_type)
        self.bttn_open_edc_tool.pressed.connect(self.start_edc)

    def make_connections(self):
        # QUESTION: does checkVal only emit when enter is pressed? need to
        # decide if I want to have the Apply button or not. Seems unnecessary
        self.le_across_slit_offset.checkVal.connect(self.context.update_axslit_offset)
        self.le_along_slit_offset.checkVal.connect(self.context.update_alslit_offset)
        self.le_azimuth_offset.checkVal.connect(self.context.update_azimuth_offset)
        self.le_workfunction.checkVal.connect(lambda x: self.context.update_work_function(x, self.context_type))
        self.le_inner_potential.checkVal.connect(lambda x: self.context.update_inner_potential(x, self.context_type))
        self.le_photon_energy.checkVal.connect(lambda x: self.context.update_photon_energy(x, self.context_type))
        self.le_xmin.checkVal.connect(lambda x: self.context.update_xmin(x, self.context_type))
        self.le_xmax.checkVal.connect(lambda x: self.context.update_xmax(x, self.context_type))
        self.le_ymin.checkVal.connect(lambda x: self.context.update_ymin(x, self.context_type))
        self.le_ymax.checkVal.connect(lambda x: self.context.update_ymax(x, self.context_type))
        #self.le_aspect_ratio.checkVal.connect(self.context.update_aspect_ratio)
        self.le_x_label.textChanged.connect(lambda x: self.context.update_x_label(x, self.context_type))
        self.le_y_label.textChanged.connect(lambda x: self.context.update_y_label(x, self.context_type))
        self.le_title.textChanged.connect(lambda x: self.context.update_title(x, self.context_type))
        self.le_x_label.returnPressed.connect(self.update_xyt)
        self.le_y_label.returnPressed.connect(self.update_xyt)
        self.le_title.returnPressed.connect(self.update_xyt)
        self.bttngrp1.buttonClicked.connect(self.checkBttn)
        self.bttngrp2.buttonClicked.connect(self.checkBttn)
        self.bttngrp3.buttonClicked.connect(self.checkBttn)
        self.bttngrp4.buttonClicked.connect(self.checkBttn)
        self.bttn_font.pressed.connect(self.choose_font)
        self.bttn_color.pressed.connect(self.choose_color)
        self.le_line_pos.checkVal.connect(self.choose_line_pos)
        self.bttn_add_line.pressed.connect(self.add_line)
        self.cbox_lines.currentIndexChanged.connect(self.set_line_options)
        self.bttn_open_edc_tool.pressed.connect(self.start_edc)
        self.bttn_open_dewarper_tool.pressed.connect(self.start_dewarper)

    def start_edc(self):
        self.context.start_EDC(self.context_type)

    def start_dewarper(self):
        self.context.start_dewarper(self.context_type)

    def update_xyt(self):
        self.context.update_xyt(self.context_type)

    def update_line(self):
        self.context.update_lines(self.lines, self.context_type)

    def choose_font(self):
        font, valid = QFontDialog.getFont()
        if valid:
            self.lbl_font.setFont(font)
            self.lbl_font.setText(font.family())
            self.lines[self.cbox_lines.currentIndex()]['font'] = font
            self.update_line()

    def choose_color(self):
        color = QColorDialog().getColor()
        self.lines[self.cbox_lines.currentIndex()]['color'] = color.name()
        self.lbl_color.setStyleSheet("QWidget { background-color: %s}" % color)
        self.update_line()

    def choose_line_pos(self, p):
        self.lines[self.cbox_lines.currentIndex()]["position"] = p
        self.update_line()

    def build_dict(self, id, vert, font, color, position):
        self.lines[id] = {"vertical": vert, "font": font, "color": color, "position": position}

    def add_line(self):
        name = self.le_line_title.text()
        id = self.cbox_lines.count()
        vert = self.rdbttn_vert.isChecked()
        font = self.lbl_font.font()
        color = self.lbl_color.palette().window().color().name()
        position = float(self.le_line_pos.text())
        self.build_dict(id, vert, font, color, position)
        if vert:
            self.cbox_lines.addItem("V: " + name)
        else:
            self.cbox_lines.addItem("H: " + name)
        self.cbox_lines.setCurrentIndex(id)
        self.update_line()

    def set_line_options(self, id):
        self.bttn_font.setEnabled(True)
        self.bttn_color.setEnabled(True)
        self.le_line_pos.setEnabled(True)
        self.rdbttn_vert.setChecked(self.lines[id]["vertical"])
        self.lbl_font.setFont(self.lines[id]["font"])
        self.lbl_font.setText(self.lbl_font.font().family())
        self.lbl_color.setStyleSheet("QWidget { background-color: %s}" % self.lines[id]['color'])
        self.le_line_pos.setText(str(self.lines[id]['position']))

    def checkBttn(self, button):
        bttn = button.text()
        if bttn == "real space":
            self.context.update_real_space(True, self.context_type)
        elif bttn == "k space":
            self.context.update_real_space(False, self.context_type)
        elif bttn == "across slit normal":
            self.context.update_normal_across_slit(True, self.context_type)
        elif bttn == "across slit off normal":
            self.context.update_normal_across_slit(False, self.context_type)
        elif bttn == "Personal\nStylesheet":
            self.context.update_custom_stylesheet(False, self.context_type)
            self.cbox_stylesheet.setEnabled(True)
        elif bttn == "Custom\nStylesheet":
            self.context.update_custom_stylesheet(True, self.context_type)
            self.cbox_stylesheet.setEnabled(True)
            # TODO: connect pop up toolbar for stylesheet editing here
        elif bttn == "auto":
            self.context.update_auto_normalize(True, self.context_type)
        elif bttn == "custom":
            self.context.update_auto_normalize(False, self.context_type)
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
