import json
import logging
import os
import threading
from argparse import ArgumentParser
from pathlib import Path

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
        self.custom_stylesheet = True
        self.x_min = -10
        self.x_max = 10
        self.y_min = -10
        self.y_max = 10
        self.aspect_ratio = 1.5
        self.color_map = "viridis"
        self.auto_normalize = True
        self.normalization_type = 'linear'

        self.fm_data = None

    def update_real_space(self, real):
        self.real_space = real
        self.signals.updateRealSpace.emit(real)

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

    def upload_fm_data(self, xar):
        self.fm_data = xar
        self.signals.fmData.emit(xar)
