import logging
import threading
from pyimagetool import RegularDataArray

import numpy as np
import xarray as xr

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
        self.work_function = 4.2
        self.inner_potential = 14
        self.custom_stylesheet = False
        self.aspect_ratio = 1.5
        self.color_map = "viridis"
        self.auto_normalize = True
        self.normalization_type = 'linear'
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        xy = np.meshgrid(x, y, indexing='ij')
        z = np.linspace(-1, 1, 51)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        d = np.sin(np.pi * np.exp(-1 * (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))) * np.cos(np.pi / 2 * xyz[1])
        fm_xar = xr.DataArray(d, coords={"slit": x, 'perp': y, "energy": z}, dims=["slit", "perp", "energy"])
        hv_xar = xr.DataArray(d, coords={"photon_energy": x, 'perp': y, "energy": z},
                              dims=["photon_energy", "perp", "energy"])
        z = np.sin(np.pi * np.exp(-1 * (xy[0] ** 2 + xy[1] ** 2))) * np.cos(np.pi / 2 * xy[1])
        ss_xar = xr.DataArray(z, coords={"slit": x, "energy": y}, dims=["slit", "energy"])
        self.master_dict = {"data": {"fermi_map": fm_xar, "hv_scan": hv_xar, "single": ss_xar},
                            "real_space": {"fermi_map": True, "hv_scan": True, "single": True},
                            "normal_across_slit": {"fermi_map": True, "hv_scan": True, "single": True},
                            "across_slit": {"fermi_map": 0, "hv_scan": 0, "single": 0},
                            "along_slit": {"fermi_map": 0, "hv_scan": 0, "single": 0},
                            "azimuth": {"fermi_map": 0, "hv_scan": 0, "single": 0},
                            "hv": {"fermi_map": 150, "hv_scan": 150, "single": 150},
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
                                      "single": "viridis"},
                            "lines": {"fermi_map": {}, "hv_scan": {}, "single": {}}
                            }

    def update_real_space(self, real, scan_type):
        self.master_dict["real_space"][scan_type] = real
        self.signals.updateRealSpace.emit()

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

    def update_work_function(self, wf, scan_type):
        self.work_function = wf
        self.signals.workFunctionChanged.emit(wf, scan_type)

    def update_inner_potential(self, ip, scan_type):
        self.inner_potential = ip
        self.signals.innerPotentialChanged.emit(ip, scan_type)

    def update_photon_energy(self, hv, scan_type):
        self.master_dict['hv'][scan_type] = hv
        self.signals.hvChanged.emit()

    def update_all_axes(self, scan_type, ax):
        for axis in ax:
            self.master_dict[axis[0]][scan_type] = axis[1]
        self.send_axes(scan_type)

    def send_axes(self, scan_type):
        axs = [self.master_dict["x_min"][scan_type], self.master_dict["x_max"][scan_type],
               self.master_dict["y_min"][scan_type], self.master_dict["y_max"][scan_type]]
        self.signals.axesChanged.emit(axs, scan_type)

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
        self.signals.updateXYTLabel.emit(labels, scan_type)

    def update_lines(self, lines, scan_type):
        self.master_dict["lines"][scan_type] = lines
        self.signals.updateLines.emit(lines, scan_type)

    def upload_data(self, xar, scan_type):
        self.master_dict['data'][scan_type] = xar
        self.signals.updateData.emit(scan_type)

    def start_dewarper(self, scan_type):
        self.signals.startDewarper.emit(scan_type)
