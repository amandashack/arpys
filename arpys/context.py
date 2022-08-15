import json
import logging
import os
import threading
from pyimagetool import RegularDataArray

import numpy as np
import yaml

log = logging.getLogger(__name__)
lock = threading.Lock()

# TODO: For any of these repeated variables, they should be a dictionary, send in where the update is coming from,
#  and send signals based on that so that the right tab gets updated when something is changed. Additionally,
#  it would be cool if there was a global changes setting you could toggle so all plots DO change at the same time.


class Context(object):

    def __init__(self, signals):
        self.signals = signals

        self.real_space = True
        self.normal_across_slit = True
        self.across_slit = 0
        self.along_slit = 0
        self.azimuth = 0
        self.custom_stylesheet = False
        self.aspect_ratio = 1.5
        self.color_map = "viridis"
        self.auto_normalize = True
        self.normalization_type = 'linear'
        self.master_dict = {"real_space": {"fermi_map": True, "hv_scan": True, "single": True},
                            "normal_across_slit": {"fermi_map": True, "hv_scan": True, "single": True},
                            "across_slit": {"fermi_map": 0, "hv_scan": 0, "single": 0},
                            "along_slit": {"fermi_map": 0, "hv_scan": 0, "single": 0},
                            "azimuth": {"fermi_map": 0, "hv_scan": 0, "single": 0},
                            "custom_stylesheet": {"fermi_map": False, "hv_scan": False, "single": False},
                            "x_min": {"fermi_map": -10, "hv_scan": -10, "single": -10},
                            "x_max": {"fermi_map": 10, "hv_scan": 10, "single": 10},
                            "y_min": {"fermi_map": -1, "hv_scan": -1, "single": -1},
                            "y_max": {"fermi_map": -1, "hv_scan": -1, "single": -1},
                            "x_label": {"fermi_map": "$k_(x)$ ($\AA^(-1)$)", "hv_scan": "$k_(x)$ ($\AA^(-1)$)",
                                        "single": "$k_(x)$ ($\AA^(-1)$)"},
                            "y_label": {"fermi_map": "Binding Energy", "hv_scan": "Binding Energy",
                                        "single": "Binding Energy"},
                            "title": {"fermi_map": "Fermi Map Cut", "hv_scan": "Photon Energy Scan Cut",
                                      "single": "Single Cut"},
                            "color_map": {"fermi_map": "viridis", "hv_scan": "viridis",
                                      "single": "viridis"}
                            }

        self.hv_reg_data = None
        self.fm_reg_data = None
        self.ss_reg_data = None
        self.hv_xar_data = None
        self.fm_xar_data = None
        self.ss_xar_data = None

    def update_real_space(self, real, scan_type):
        self.real_space = real
        self.signals.updateRealSpace.emit(real, scan_type)

    def update_normal_across_slit(self, normal):
        """defines whether the sample was oriented normal in the across slit direction"""
        self.normal_across_slit = normal

    def update_custom_stylesheet(self, cs):
        """choice between custom or default stylesheet"""
        self.custom_stylesheet = cs

    def update_auto_normalize(self, an):
        """defines how the plot colormap is normalized"""
        self.auto_normalize = an

    def update_axslit_offset(self, axs):
        self.across_slit = axs
        self.signals.axslitOffset.emit(axs)

    def update_alslit_offset(self, als):
        self.along_slit = als
        self.signals.alslitOffset.emit(als)

    def update_azimuth_offset(self, az):
        self.azimuth = az
        self.signals.azimuthOffset.emit(az)

    def update_all_axes(self, scan_type, ax):
        for axis in ax:
            self.master_dict[axis[0]][scan_type] = axis[1]
        self.send_axes(scan_type)

    def send_axes(self, scan_type):
        axs = [self.master_dict["x_min"][scan_type], self.master_dict["x_max"][scan_type],
               self.master_dict["y_min"][scan_type], self.master_dict["y_max"][scan_type]]
        self.signals.axesChanged.emit(scan_type, axs)

    def update_xmin(self, xmin, scan_type):
        self.master_dict["x_min"][scan_type] = xmin
        self.send_axes(scan_type)

    def update_xmax(self, xmax, scan_type):
        self.master_dict["x_max"][scan_type] = xmax
        self.send_axes(scan_type)

    def update_ymin(self, ymin, scan_type):
        self.master_dict["y_min"][scan_type] = ymin
        self.send_axes(scan_type)

    def update_ymax(self, ymax, scan_type):
        self.master_dict["y_max"][scan_type] = ymax
        self.send_axes(scan_type)

    def update_x_label(self, l, scan_type):
        self.master_dict["x_label"][scan_type] = l

    def update_y_label(self, l, scan_type):
        self.master_dict["y_label"][scan_type] = l

    def update_title(self, t, scan_type):
        self.master_dict["title"][scan_type] = t

    def update_xyt(self, scan_type):
        labels = [self.master_dict["x_label"][scan_type], self.master_dict["y_label"][scan_type],
                  self.master_dict["title"][scan_type]]
        self.signals.updateXYTLabel.emit(scan_type, labels)

    def upload_hv_data(self, xar):
        self.hv_xar_data = xar
        self.hv_reg_data = RegularDataArray(xar)
        self.signals.hvData.emit(xar)

    def upload_fm_data(self, xar):
        self.fm_xar_data = xar
        self.fm_reg_data = RegularDataArray(xar)
        self.signals.fmData.emit(xar)

    def upload_ss_data(self, xar):
        self.ss_xar_data = xar
        self.ss_reg_data = RegularDataArray(xar)
        self.signals.ssData.emit(xar)

    def start_dewarper(self, scan_type):
        self.signals.startDewarper.emit(scan_type)
