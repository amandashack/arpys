import logging
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QFrame, QHBoxLayout, QVBoxLayout
from gui.widgets.basicWidgets import HLineItem, VLineItem
from PyQt5.QtCore import Qt
import numpy as np
from numpy import add, diff, asarray
import xarray as xr
from plottingTools import PlotCanvas, PlotWidget
from pyimagetool import RegularDataArray
from random import randint
import arpys
import numba


class DewarperImageWidget(QFrame):

    def __init__(self, context, signals):
        super(DewarperImageWidget, self).__init__()
        self.context = context
        self.signals = signals
        self.st = "fermi_map"
        self.single_ef = None
        self.min_vals = None
        self.eloss = None
        self.emin = -1
        self.emax = 1
        self.a = 25
        self.b = .1
        self.c = 4
        self.y = 1.9
        self.polyfit_order = 6
        self.threshold = 1/100

        # temporary/fake data so that you don't have to select a file to test the GUI
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        z = np.linspace(-1, 1, 51)
        xyz = np.meshgrid(x, y, z, indexing='ij')
        d = np.sin(np.pi * np.exp(-1 * (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))) * np.cos(np.pi / 2 * xyz[1])
        self.xar = xr.DataArray(d, coords={"slit": x, 'perp': y, "energy": z}, dims=["slit", "perp", "energy"])
        self.data = RegularDataArray(d, delta=[x[1] - x[0], y[1] - y[0], z[1] - z[0]], coord_min=[x[0], y[0], z[0]])
        self.dewarped = None
        self.dewarped_data = None
        self.cut_val = 0
        self.y_edc = 0
        self.cut = self.xar.sel({"perp": self.cut_val}, method='nearest')
        self.cut_fit = self.cut
        self.plot_dewarp = False
        self.imagetool = PlotWidget(self.data, layout=1)
        self.imagetool_dewarped = None
        self.canvas_cut = PlotCanvas()
        self.canvas_cut_fit = PlotCanvas()
        self.canvas_cut.plot(self.cut)
        self.canvas_cut_fit.plot(self.cut_fit)

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.coord0 = self.xar.slit.values
        self.coord1 = self.xar.perp.values
        self.coord2 = self.xar.energy.values
        self.dim0 = self.xar.dims[0]
        self.dim1 = self.xar.dims[1]
        self.dim2 = self.xar.dims[2]
        self.line_item_hor = HLineItem(self.signals, ident=randint(0, 1000))
        self.line_item_vert_left = VLineItem(self.signals, ident=randint(0, 1000))
        self.line_item_vert_right = VLineItem(self.signals, ident=randint(0, 1000))
        self.connect_scene()

        self.layout_cuts = QHBoxLayout()
        self.layout_imagetool = QHBoxLayout()
        self.layout_cuts.addWidget(self.view)
        self.layout_cuts.addWidget(self.canvas_cut_fit)
        self.layout_imagetool.addWidget(self.imagetool)
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.layout_cuts)
        self.main_layout.addLayout(self.layout_imagetool)
        self.setLayout(self.main_layout)

        self.make_connections()

    def connect_scene(self):
        s = self.canvas_cut.figure.get_size_inches() * self.canvas_cut.figure.dpi
        self.view.setScene(self.scene)
        self.scene.addWidget(self.canvas_cut)
        self.scene.setSceneRect(0, 0, s[0], s[1])
        self.scene.addItem(self.line_item_vert_right)
        self.scene.addItem(self.line_item_vert_left)
        self.scene.addItem(self.line_item_hor) # this could maybe be used for viewing where the minimum is for each edc
        self.capture_scene_change()

    def make_connections(self):
        self.signals.mouseReleased.connect(self.set_emin)
        self.signals.mouseReleased.connect(self.set_emax)
        self.signals.mouseReleased.connect(self.set_yedc)

    def set_yedc(self, ident):
        pass

    def set_emax(self, ident):
        if self.line_item_vert_right.ident == ident:
            x = self.line_item_vert_right.x_pos
            bbox = self.canvas_cut.axes.spines['top'].get_window_extent()
            plot_bbox = [bbox.x0, bbox.x1]
            if x > plot_bbox[1]:
                # self.line_item_vert_left.setPos(plot_bbox[0], 0)
                # set emin to the edge of the data set
                self.emax = self.coord2[-1]
                self.plot_pos_to_line_pos(horizontal=False)
            elif x <= plot_bbox[0]:
                # set emin to be in front of emax
                what_index = self.coord2.tolist().index(self.emin)
                self.emax = self.coord2[what_index + 1]
                self.plot_pos_to_line_pos(horizontal=False)
            else:
                size_range = len(self.coord2)
                r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
                sel_val = min(r, key=lambda f: abs(f - x))
                what_index = r.index(sel_val)
                rel_x = self.coord2[what_index]
                if rel_x <= self.emin:
                    what_index = self.coord2.tolist().index(self.emin)
                    self.emax = self.coord2[what_index + 1]
                    self.plot_pos_to_line_pos(horizontal=False)
                else:
                    self.emax = rel_x
            self.signals.changeEmaxText.emit(str(self.emax))

    def set_emin(self, ident):
        if self.line_item_vert_left.ident == ident:
            x = self.line_item_vert_left.x_pos
            bbox = self.canvas_cut.axes.spines['top'].get_window_extent()
            plot_bbox = [bbox.x0, bbox.x1]
            if x < plot_bbox[0]:
                # set emin to the edge of the data set
                self.emin = self.coord2[0]
                self.plot_pos_to_line_pos(horizontal=False)
            elif x >= plot_bbox[1]:
                # set emin to be in front of emax
                what_index = self.coord2.tolist().index(self.emax)
                self.emin = self.coord2[what_index - 1]
                self.plot_pos_to_line_pos(horizontal=False)
            else:
                size_range = len(self.coord2)
                r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
                sel_val = min(r, key=lambda f: abs(f - x))
                what_index = r.index(sel_val)
                rel_x = self.coord2[what_index]
                if rel_x >= self.emax:
                    what_index = self.coord2.tolist().index(self.emax)
                    self.emin = self.coord2[what_index-1]
                    self.plot_pos_to_line_pos(horizontal=False)
                else:
                    self.emin = rel_x
            self.signals.changeEminText.emit(str(self.emin))

    def generate_eloss(self):
        dslice = self.xar.sel({self.dim2: slice(self.emin, self.emax)})
        dx = abs(dslice[self.dim2].values[1] - dslice[self.dim2].values[0])
        integrated = np.flip(cum_integrate(np.flip(dslice, axis=-1), dx=dx, initial=0), axis=-1)
        spec_integrated = xr.DataArray(integrated, dims=[self.dim0, self.dim1, self.dim2],
                                       coords={self.dim0: dslice[self.dim0].values,
                                               self.dim1: dslice[self.dim1].values,
                                               self.dim2: dslice[self.dim2].values})
        # TODO: this should also be a changeable value (The 1/100)
        for i in range(len(spec_integrated[self.dim0].values)):
            for j in range(len(spec_integrated[self.dim1].values)):
                if spec_integrated[i, j, 0] < np.max(spec_integrated[:, :, 0]) * self.threshold:
                    spec_integrated[i, j, :] = np.nan
        eloss = np.power(np.power(self.a * spec_integrated, self.c) / 2 -
                         xr.ufuncs.log(self.b * spec_integrated) - self.y, 2)
        return eloss

    def fit_cut(self):
        NoneType = type(None)
        if isinstance(self.eloss, NoneType):
            self.eloss = self.generate_eloss()
        self.min_vals = []
        for x in self.eloss[self.dim0].values:
            idx = _nanargmin(self.eloss.sel({self.dim0: x, self.dim1: self.cut_val}).values)
            if idx is np.nan:
                self.min_vals.append(np.nan)
            else:
                self.min_vals.append(self.eloss[self.dim2].values[idx])

        idx = np.isfinite(self.eloss[self.dim0].values) & np.isfinite(self.min_vals)
        p = np.polyfit(self.eloss[self.dim0].values[idx], np.array(self.min_vals)[idx], self.polyfit_order)
        dw_curve = np.poly1d(p)
        self.single_ef = dw_curve(self.cut[self.dim0].values)
        try:
            self.cut_fit = single_dewarp(self.cut, self.single_ef)
            return "successful!"
        except ValueError:
            return "Try a different cut."

    def fit_3d(self):
        NoneType = type(None)
        if isinstance(self.eloss, NoneType):
            self.eloss = self.generate_eloss()
        full_spectra = self.context.master_dict['data'][self.st]
        self.dewarped = dewarp_3D(full_spectra, self.eloss)
        self.dewarped_data = RegularDataArray(self.dewarped)
        self.imagetool_dewarped = PlotWidget(self.dewarped_data, layout=1)
        return "3D dewarp was successful!"

    def initialize_data(self, st):
        """
        Used to initialize the data. This should only be called once when the tool window is opened.
        Subsequently, data is changed by the view and retrieving data from the master dict would
        override changes made by the controller.
        Parameters
        ----------
        st: scan type
        """
        self.st = st
        self.single_ef = None
        self.min_vals = None
        self.eloss = None
        self.xar = self.context.master_dict['data'][st]
        self.data = RegularDataArray(self.xar)
        self.imagetool = PlotWidget(self.data, layout=1)
        self.imagetool_dewarped = None
        self.dim0 = self.xar.dims[0]
        self.dim1 = self.xar.dims[1]
        self.dim2 = self.xar.dims[2]
        self.coord0 = self.xar[self.dim0].values
        self.coord1 = self.xar[self.dim1].values
        self.coord2 = self.xar[self.dim2].values
        self.cut_val = self.coord1[0]
        self.y_edc = self.coord0[0]
        self.emin = self.coord2[0]
        self.emax = self.coord2[-1]
        self.update_cut()
        self.plot_pos_to_line_pos()
        self.plot_pos_to_line_pos(horizontal=False)

    def update_cut(self):
        self.cut = self.xar.sel({self.dim1: self.cut_val}, method='nearest')
        self.cut_fit = self.cut

    def handle_plotting(self, imtool=True):
        self.handle_plotting_cut()
        if imtool:
            self.delete_items_of_layout(self.layout_imagetool)
            if not self.plot_dewarp:
                self.layout_imagetool.addWidget(self.imagetool)
            else:
                self.layout_imagetool.addWidget(self.imagetool_dewarped)

    def handle_plotting_cut(self):
        self.canvas_cut.axes.cla()
        self.canvas_cut.plot(self.cut)
        if all(v is not None for v in [self.single_ef, self.min_vals]):
            self.canvas_cut.axes.plot(self.min_vals, self.cut[self.dim0].values, 'o', color='black')
            self.canvas_cut.draw()
            self.canvas_cut.axes.plot(self.single_ef, self.cut[self.dim0].values, color='green')
            self.canvas_cut.draw()
        self.canvas_cut.plot(self.cut)
        self.canvas_cut.draw()
        self.canvas_cut_fit.axes.cla()
        self.canvas_cut_fit.plot(self.cut_fit)
        self.canvas_cut_fit.draw()

    def delete_items_of_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    self.delete_items_of_layout(item.layout())

    def plot_pos_to_line_pos(self, horizontal=True):
        if horizontal:
            bbox = self.canvas_cut.axes.spines['left'].get_window_extent()
            plot_bbox = [bbox.y0, bbox.y1]
            size_range = len(self.coord0)
            r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
            corr = list(zip(r, self.coord0))
            what_index = self.coord0.tolist().index(self.y_edc)
            bbox_val = corr[what_index][0]
            rel_pos = lambda x: abs(self.scene.sceneRect().height() - x)
            self.line_item_hor.setPos(0, rel_pos(bbox_val))
        else:
            bbox = self.canvas_cut.axes.spines['top'].get_window_extent()
            plot_bbox = [bbox.x0, bbox.x1]
            size_range = len(self.coord2)
            r = np.linspace(plot_bbox[0], plot_bbox[1], size_range).tolist()
            corr = list(zip(r, self.coord2))
            what_index_min = self.coord2.tolist().index(self.emin)
            what_index_max = self.coord2.tolist().index(self.emax)
            bbox_val_min = corr[what_index_min][0]
            bbox_val_max = corr[what_index_max][0]
            self.line_item_vert_left.setPos(bbox_val_min, 0)
            self.line_item_vert_right.setPos(bbox_val_max, 0)

    def plot_cum_integral(self):
        dslice = self.xar.sel({self.dim2: slice(self.emin, self.emax)})
        dx = abs(dslice[self.dim2].values[1] - dslice[self.dim2].values[0])
        integrated = np.flip(cum_integrate(np.flip(dslice, axis=-1), dx=dx, initial=0), axis=-1)
        spec_integrated = xr.DataArray(integrated, dims=[self.dim0, self.dim1, self.dim2],
                                       coords={self.dim0: dslice[self.dim0].values,
                                               self.dim1: dslice[self.dim1].values,
                                               self.dim2: dslice[self.dim2].values})
        spec_integrated.arpes.plot()

    def capture_scene_change(self):
        self.line_item_hor.setLine(0, 0, self.scene.sceneRect().width(), 0)
        self.plot_pos_to_line_pos()
        self.line_item_vert_left.setLine(0, 0, 0, self.scene.sceneRect().height())
        self.line_item_vert_right.setLine(0, 0, 0, self.scene.sceneRect().height())
        self.plot_pos_to_line_pos(horizontal=False)

    def resize_event(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.view.fitInView(self.line_item_vert_left, Qt.KeepAspectRatio)
        self.view.fitInView(self.line_item_hor, Qt.KeepAspectRatio)
        self.view.fitInView(self.line_item_vert_right, Qt.KeepAspectRatio)


"""
These are functions used in the above class that don't need to exist
inside the class itself
"""


@numba.njit
def shift4_numba(arr, num, fill_value=np.nan):
    if num > 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))


import numpy as np
from scipy.linalg import lstsq
from scipy.special import binom

import matplotlib.pyplot as plt


def _get_coeff_idx(coeff):
    idx = np.indices(coeff.shape)
    idx = idx.T.swapaxes(0, 1).reshape((-1, 2))
    return idx


def _scale(x, y):
    # Normalize x and y to avoid huge numbers
    # Mean 0, Variation 1
    offset_x, offset_y = np.mean(x), np.mean(y)
    norm_x, norm_y = np.std(x), np.std(y)
    x = (x - offset_x) / norm_x
    y = (y - offset_y) / norm_y
    return x, y, (norm_x, norm_y), (offset_x, offset_y)


def _unscale(x, y, norm, offset):
    x = x * norm[0] + offset[0]
    y = y * norm[1] + offset[1]
    return x, y


def polyvander2d(x, y, degree):
    A = np.polynomial.polynomial.polyvander2d(x, y, degree)
    return A


def polyscale2d(coeff, scale_x, scale_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    for k, (i, j) in enumerate(idx):
        coeff[i, j] /= scale_x ** i * scale_y ** j
    return coeff


def polyshift2d(coeff, offset_x, offset_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    # Copy coeff because it changes during the loop
    coeff2 = np.copy(coeff)
    for k, m in idx:
        not_the_same = ~((idx[:, 0] == k) & (idx[:, 1] == m))
        above = (idx[:, 0] >= k) & (idx[:, 1] >= m) & not_the_same
        for i, j in idx[above]:
            b = binom(i, k) * binom(j, m)
            sign = (-1) ** ((i - k) + (j - m))
            offset = offset_x ** (i - k) * offset_y ** (j - m)
            coeff[k, m] += sign * b * coeff2[i, j] * offset
    return coeff


def plot2d(x, y, z, coeff):
    # regular grid covering the domain of the data
    if x.size > 500:
        choice = np.random.choice(x.size, size=500, replace=False)
    else:
        choice = slice(None, None, None)
    x, y, z = x[choice], y[choice], z[choice]
    X, Y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), 20), np.linspace(np.min(y), np.max(y), 20)
    )
    Z = np.polynomial.polynomial.polyval2d(X, Y, coeff)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(x, y, z, c="r", s=50)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def _nanargmin(arr, axis=None):
    try:
        return np.nanargmin(arr, axis=axis)
    except ValueError:
        return np.nan


def polyfit2d(x, y, z, degree=1, max_degree=None, scale=True, plot=False):
    """A simple 2D polynomial fit to data x, y, z
    The polynomial can be evaluated with numpy.polynomial.polynomial.polyval2d

    Parameters
    ----------
    x : array[n]
        x coordinates
    y : array[n]
        y coordinates
    z : array[n]
        data values
    degree : {int, 2-tuple}, optional
        degree of the polynomial fit in x and y direction (default: 1)
    max_degree : {int, None}, optional
        if given the maximum combined degree of the coefficients is limited to this value
    scale : bool, optional
        Wether to scale the input arrays x and y to mean 0 and variance 1, to avoid numerical overflows.
        Especially useful at higher degrees. (default: True)
    plot : bool, optional
        wether to plot the fitted surface and data (slow) (default: False)

    Returns
    -------
    coeff : array[degree+1, degree+1]
        the polynomial coefficients in numpy 2d format, i.e. coeff[i, j] for x**i * y**j
    """
    # Flatten input
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    good_idx = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    # Remove masked values
    mask = ~(np.ma.getmask(z) | np.ma.getmask(x) | np.ma.getmask(y))
    x, y, z = x[mask].ravel(), y[mask].ravel(), z[mask].ravel()

    # Scale coordinates to smaller values to avoid numerical problems at larger degrees
    if scale:
        x, y, norm, offset = _scale(x, y)

    if np.isscalar(degree):
        degree = (int(degree), int(degree))
    degree = [int(degree[0]), int(degree[1])]
    coeff = np.zeros((degree[0] + 1, degree[1] + 1))
    idx = _get_coeff_idx(coeff)

    # Calculate elements 1, x, y, x*y, x**2, y**2, ...
    A = polyvander2d(x[good_idx], y[good_idx], degree)

    # We only want the combinations with maximum order COMBINED power
    if max_degree is not None:
        mask = idx[:, 0] + idx[:, 1] <= int(max_degree)
        idx = idx[mask]
        A = A[:, mask]

    # Do the actual least squares fit
    C, *_ = lstsq(A, z[good_idx])

    # Reorder coefficients into numpy compatible 2d array
    for k, (i, j) in enumerate(idx):
        coeff[i, j] = C[k]

    # Reverse the scaling
    if scale:
        coeff = polyscale2d(coeff, *norm, copy=False)
        coeff = polyshift2d(coeff, *offset, copy=False)

    if plot:
        if scale:
            x, y = _unscale(x[good_idx], y[good_idx], norm, offset)
        plot2d(x, y, z[good_idx], coeff)

    return coeff


def single_dewarp(spec, ef_pos):
    ef_min = np.min(ef_pos)
    ef_max = np.max(ef_pos)
    de = spec.coords[spec.dims[1]].values[1] - spec.coords[spec.dims[1]].values[0]
    px_to_remove = int(round((ef_max - ef_min) / de))
    dewarped = np.empty((spec.coords[spec.dims[0]].size, spec.coords[spec.dims[1]].size - px_to_remove))
    for i in range(spec.coords[spec.dims[0]].size):
        rm_from_bottom = int(round((ef_pos[i] - ef_min) / de))
        rm_from_top = spec.coords[spec.dims[1]].size - (px_to_remove - rm_from_bottom)
        dewarped[i,:] = spec.values[i, rm_from_bottom:rm_from_top]
    bottom_energy_offset = int(round((ef_max - ef_min) / de))
    energy = spec.coords[spec.dims[1]].values[bottom_energy_offset:]
    dw_data = xr.DataArray(dewarped, coords={spec.dims[0]: spec.coords[spec.dims[0]].values, spec.dims[1]: energy},
                           dims=[spec.dims[0], spec.dims[1]], attrs=spec.attrs)
    dw_new_ef = dw_data[spec.dims[1]] - ef_max
    dw_data = dw_data.assign_coords({spec.dims[1]: dw_new_ef})
    return dw_data


def dewarp_3D(spec, eloss):
    emin = eloss.idxmin(dim=spec.dims[2])
    eloss_x = emin[spec.dims[0]].values
    eloss_y = emin[spec.dims[1]].values
    im_x, im_y = np.meshgrid(eloss_x, eloss_y)
    xy = np.c_[im_x.flatten(), im_y.flatten()]
    x = xy[:, 0]
    y = xy[:, 1]
    z = emin.values
    # idx = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    # print(idx.tolist())
    # coeff = np.polyfit(eloss[spec.dims[0]].values[idx], np.array(min_vals)[idx], 6)
    coeff = polyfit2d(x, y, z, degree=2, plot=True)

    spec_x = spec[spec.dims[0]].values
    spec_y = spec[spec.dims[1]].values
    im_x, im_y = np.meshgrid(spec_x, spec_y)
    XY = np.c_[im_x.flatten(), im_y.flatten()]
    X = XY[:, 0]
    Y = XY[:, 1]
    Z = np.polynomial.polynomial.polyval2d(X, Y, coeff).reshape(len(spec_x), len(spec_y))
    # plot2d(X, Y, Z, coeff)
    m = np.nanmax(Z)
    de = spec.coords[spec.dims[2]].values[1] - spec.coords[spec.dims[2]].values[0]
    index_shift = ((m - Z) / de).astype(int)
    d = spec.values
    nda = np.empty(spec.values.shape)
    for i in range(len(spec_x)):
        for j in range(len(spec_y)):
            a = shift4_numba(d[i, j, :], index_shift[i, j])
            nda[i, j, :] = a
    new_spec = xr.DataArray(data=nda, dims=[spec.dims[0], spec.dims[1], spec.dims[2]],
                            coords={spec.dims[0]: spec[spec.dims[0]].values, spec.dims[1]: spec[spec.dims[1]].values,
                                    spec.dims[2]: spec[spec.dims[2]].values}, attrs=spec.attrs)
    return new_spec


def cum_integrate(y, x=None, dx=1.0, axis=-1, initial=None):
    # trapezoidal cumulative integration
    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)
    y = asarray(y)
    if x is None:
        d = dx
    else:
        x = asarray(x)
        if x.ndim == 1:
            d = abs(diff(x))
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-d or the "
                    "same as y.")
        else:
            d = abs(diff(x, axis=axis))

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    res = abs(add.accumulate(d * (y[slice1] + y[slice2]) / 2.0, axis))

    if initial is not None:
        if not np.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = np.concatenate([np.ones(shape, dtype=res.dtype) * initial, res],
                             axis=axis)

    return res
