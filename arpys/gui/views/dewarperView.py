import logging

from gui.widgets.dewarperControls import DewarperControls
from gui.widgets.dewarperImageWidget import DewarperImageWidget
from gui.windows.paramsTableWindow import ParamsTableWindow
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import numpy as np
from plottingTools import PlotWidget
import arpys
from pyimagetool import RegularDataArray

log = logging.getLogger('pydm')
log.setLevel('CRITICAL')


def ke_to_be(hv, spectra):
    # TODO: add option to change work function
    wf = 4.5
    ke = np.array(spectra.energy)
    be = ke - hv + wf
    reassigned = spectra.assign_coords({'energy': be})
    return reassigned


class DewarperView(QWidget):
    def __init__(self, context, signals):
        super(DewarperView, self).__init__()
        self.signals = signals
        self.context = context
        self.camera = ""
        self.mainLayout = QVBoxLayout()
        self.scan_type = "fermi_map"
        self.imageWidget = None
        self.controlWidget = None
        self.table_window = None
        self.create_image_widget()
        self.create_control_widget()
        self.mainLayout.addWidget(self.controlWidget, 20)
        self.mainLayout.addWidget(self.imageWidget, 80)
        self.setLayout(self.mainLayout)
        self.make_connections()

    def make_connections(self):
        self.controlWidget.sb_x_downsample.valueChanged.connect(self.capture_x_downsample)
        self.controlWidget.sb_y_downsample.valueChanged.connect(self.capture_y_downsample)
        self.controlWidget.le_cut.returnPressed.connect(self.capture_cut)
        self.controlWidget.le_slice_range_min.returnPressed.connect(self.capture_emin)
        self.controlWidget.le_slice_range_max.returnPressed.connect(self.capture_emax)
        self.controlWidget.le_a.returnPressed.connect(self.capture_a)
        self.controlWidget.le_b.returnPressed.connect(self.capture_b)
        self.controlWidget.le_power_c.returnPressed.connect(self.capture_c)
        self.controlWidget.le_subtraction_y.returnPressed.connect(self.capture_y)
        self.controlWidget.le_polyfit_order.returnPressed.connect(self.capture_order)
        self.controlWidget.le_polyfit_order.returnPressed.connect(self.capture_threshold)
        self.controlWidget.bttn_convert.pressed.connect(self.convert_to_binding)
        self.controlWidget.bttn_fit_cut.pressed.connect(self.fit_cut)
        self.controlWidget.bttn_preview.pressed.connect(self.view_integral)
        self.controlWidget.bttn_fit_3d.pressed.connect(self.fit_3d)
        self.controlWidget.rdbttn_group.buttonClicked.connect(self.change_plotting)
        self.controlWidget.bttn_pop_out.pressed.connect(self.pop_out)
        self.signals.changeEminText.connect(self.clear_all_fitting)
        self.signals.changeEmaxText.connect(self.clear_all_fitting)

    def capture_cut(self):
        v = float(self.controlWidget.le_cut.text())
        corr_val = min(self.imageWidget.coord1, key=lambda f: abs(f - v))
        self.controlWidget.le_cut.setText(str(corr_val))
        self.imageWidget.cut_val = corr_val
        self.imageWidget.single_ef = None
        self.imageWidget.min_vals = None
        self.imageWidget.update_cut()
        self.imageWidget.handle_plotting(imtool=False)

    def capture_x_downsample(self, v):
        y_ds = self.controlWidget.sb_y_downsample.value()
        try:
            self.imageWidget.xar = self.context.master_dict['data'][self.scan_type].arpes.downsample(
                {self.imageWidget.dim0: v})
            self.imageWidget.xar = self.imageWidget.xar.arpes.downsample({self.imageWidget.dim1: y_ds})
        except ZeroDivisionError:
            self.controlWidget.le_info_bar.setText("got a division by zero error, did you try to downsample to 0?")

        self.set_image_values()
        self.set_control_values()
        self.controlWidget.update_allowed_positions()
        self.imageWidget.cut_val = min(self.imageWidget.coord1,
                                       key=lambda f: abs(f - self.imageWidget.cut_val))
        self.imageWidget.y_edc = min(self.imageWidget.coord0,
                                     key=lambda f: abs(f - self.imageWidget.y_edc))
        self.imageWidget.update_cut()
        self.clear_all_fitting()
        self.imageWidget.handle_plotting()

    def capture_y_downsample(self, v):
        x_ds = self.controlWidget.sb_x_downsample.value()
        try:
            self.imageWidget.xar = self.context.master_dict['data'][self.scan_type].arpes.downsample(
                {self.imageWidget.dim1: v})
            self.imageWidget.xar = self.imageWidget.xar.arpes.downsample({self.imageWidget.dim0: x_ds})
        except ZeroDivisionError:
            self.controlWidget.le_info_bar.setText("got a division by zero error, did you try to downsample to 0?")

        self.set_image_values()
        self.set_control_values()
        self.controlWidget.update_allowed_positions()
        self.imageWidget.cut_val = min(self.imageWidget.coord1,
                                       key=lambda f: abs(f - self.imageWidget.cut_val))
        self.imageWidget.y_edc = min(self.imageWidget.coord0,
                                       key=lambda f: abs(f - self.imageWidget.y_edc))
        self.imageWidget.update_cut()
        self.clear_all_fitting()
        self.imageWidget.handle_plotting()

    def capture_emin(self):
        v = float(self.controlWidget.le_slice_range_min.text())
        v_max = float(self.controlWidget.le_slice_range_max.text())
        corr_val = min(self.imageWidget.coord2, key=lambda f: abs(f - v))
        self.controlWidget.le_slice_range_min.setText(str(corr_val))
        self.controlWidget.le_slice_range_min.valRange(corr_val, v_max)
        self.imageWidget.emin = corr_val
        self.imageWidget.plot_pos_to_line_pos(horizontal=False)
        self.clear_all_fitting()

    def capture_emax(self):
        v = float(self.controlWidget.le_slice_range_max.text())
        v_min = float(self.controlWidget.le_slice_range_min.text())
        corr_val = min(self.imageWidget.coord2, key=lambda f: abs(f - v))
        self.controlWidget.le_slice_range_max.setText(str(corr_val))
        self.controlWidget.le_slice_range_max.valRange(v_min, corr_val)
        self.imageWidget.emax = corr_val
        self.imageWidget.plot_pos_to_line_pos(horizontal=False)
        self.clear_all_fitting()

    def capture_a(self):
        self.imageWidget.a = float(self.controlWidget.le_a.text())
        self.controlWidget.a = float(self.controlWidget.le_a.text())
        self.imageWidget.eloss = None
        self.controlWidget.set_min_eloss()

    def capture_b(self):
        self.imageWidget.b = float(self.controlWidget.le_b.text())
        self.controlWidget.b = float(self.controlWidget.le_b.text())
        self.clear_all_fitting()
        self.controlWidget.set_min_eloss()

    def capture_c(self):
        self.imageWidget.c = float(self.controlWidget.le_power_c.text())
        self.controlWidget.c = float(self.controlWidget.le_power_c.text())
        self.clear_all_fitting()
        self.controlWidget.set_min_eloss()

    def capture_y(self):
        self.imageWidget.y = float(self.controlWidget.le_subtraction_y.text())
        self.controlWidget.y = float(self.controlWidget.le_subtraction_y.text())
        self.clear_all_fitting()
        self.controlWidget.set_min_eloss()

    def capture_order(self):
        self.imageWidget.polyfit_order = float(self.controlWidget.le_polyfit_order.text())
        self.clear_all_fitting()

    def capture_threshold(self):
        self.imageWidget.threshold = float(self.controlWidget.le_threshold.text())
        self.clear_all_fitting()

    def convert_to_binding(self):
        # TODO: This type of thing really needs an undo option, currently you'd have to restart the program
        photon_energy = float(self.controlWidget.le_hv.text())
        self.controlWidget.le_info_bar.setText(f"Converting to Binding energies using hv = {photon_energy}")
        self.imageWidget.xar = ke_to_be(photon_energy, self.context.master_dict['data'][self.scan_type])
        self.context.upload_data(self.imageWidget.xar, self.scan_type)
        self.set_image_values()
        self.set_control_values()
        self.imageWidget.update_cut()
        self.controlWidget.update_allowed_positions()
        self.controlWidget.cb_conv_binding.toggle()
        self.clear_all_fitting()
        self.imageWidget.handle_plotting()

    def fit_cut(self):
        self.controlWidget.le_info_bar.setText("Running fit on current cut.")
        self.clear_all_fitting()
        m = self.imageWidget.fit_cut()
        self.controlWidget.le_info_bar.setText(m)
        self.imageWidget.handle_plotting(imtool=False)

    def view_integral(self):
        self.imageWidget.plot_cum_integral()

    def fit_3d(self):
        self.controlWidget.le_info_bar.setText("Running 3D fit.")
        m = self.imageWidget.fit_3d()
        self.controlWidget.le_info_bar.setText(m)
        self.controlWidget.rdbttn_dewarped.setChecked(True)
        self.change_plotting(1)

    def pop_out(self):
        ident = self.controlWidget.rdbttn_group.checkedId()
        if ident == 0:
            self.imageWidget.dewarped.arpes.plot()
        else:
            self.context.master_dict['data'][self.scan_type].arpes.plot()

    def change_plotting(self, ident):
        ident = self.controlWidget.rdbttn_group.checkedId()
        was = self.imageWidget.plot_dewarp
        NoneType = type(None)
        if ident == 1:
            self.imageWidget.plot_dewarp = False
        if ident == 0:
            if not isinstance(self.imageWidget.dewarped, NoneType):
                self.imageWidget.plot_dewarp = True
            else:
                self.controlWidget.rdbttn_og.setChecked(True)
                self.imageWidget.plot_dewarp = False
        nowis = self.imageWidget.plot_dewarp
        if was != nowis:
            self.imageWidget.handle_plotting(imtool=True)
        else:
            self.controlWidget.le_info_bar.setText("There is not a dewarped spectra yet")

    def create_image_widget(self):
        self.imageWidget = DewarperImageWidget(self.context, self.signals)
        self.set_initial_image_values()

    def create_control_widget(self):
        self.controlWidget = DewarperControls(self.context, self.signals)
        self.set_initial_control_values()

    def update_scan_type(self, st):
        """
        Used to set the scan type before the dewarper is opened so that it
        pulls the correct data.
        Parameters
        ----------
        st: scan type

        Returns
        -------

        """
        self.scan_type = st
        self.imageWidget.st = st
        self.set_initial_image_values()
        self.set_initial_control_values()

    def set_initial_image_values(self):
        self.imageWidget.initialize_data(self.scan_type)
        self.imageWidget.handle_plotting()
        
    def set_image_values(self):
        self.imageWidget.data = RegularDataArray(self.imageWidget.xar)
        self.imageWidget.imagetool = PlotWidget(self.imageWidget.data, layout=1)
        self.imageWidget.coord0 = self.imageWidget.xar[self.imageWidget.dim0].values
        self.imageWidget.coord1 = self.imageWidget.xar[self.imageWidget.dim1].values
        self.imageWidget.coord2 = self.imageWidget.xar[self.imageWidget.dim2].values

    def set_initial_control_values(self):
        self.controlWidget.initialize_data(self.scan_type)
        self.controlWidget.update_allowed_positions()

    def set_control_values(self):
        self.controlWidget.data = self.imageWidget.xar
        self.controlWidget.coord0 = self.imageWidget.xar[self.imageWidget.dim0].values
        self.controlWidget.coord1 = self.imageWidget.xar[self.imageWidget.dim1].values
        self.controlWidget.coord2 = self.imageWidget.xar[self.imageWidget.dim2].values

    def clear_all_fitting(self):
        self.controlWidget.le_info_bar.setText("Clearing all fitting..")
        self.imageWidget.single_ef = None
        self.imageWidget.min_vals = None
        self.imageWidget.eloss = None

