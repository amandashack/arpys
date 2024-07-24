from pyqtgraph.Qt import QtGui, QtWidgets
import sys
import numpy as np
import xarray as xr
import json
import h5py


def update_json(json_file, material, new_entry, scan_type='fermi_map'):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if material not in data[scan_type]:
        data[scan_type][material] = {}

    data[scan_type][material].update(new_entry)

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def save_xarray_to_hdf5(xarr, filepath):
    with h5py.File(filepath, 'w') as f:
        for dim in xarr.dims:
            f.create_dataset(dim, data=xarr[dim].values)
        f.create_dataset('data', data=xarr.values)
        for attr, value in xarr.attrs.items():
            f.attrs[attr] = value


def standardize(ar):
    """
    Makes the spectra intensities between 0 and 1
    :param ar: xarray with data
    :return: xarray standardized to between 0 and 1
    """
    w_max = float(np.nanmax(ar))
    w_min = float(np.nanmin(ar))
    nr_values = (ar - w_min) / (w_max - w_min)
    return nr_values


def normalize_2D(*args):
    spectra = []
    for arg in args:
        dims = arg.dims
        energy = arg[dims[1]]
        kx = arg[dims[0]]
        sum_thing = np.nansum(arg.values)
        area = arg[dims[1]].values.size * arg[dims[0]].values.size
        average = sum_thing / area
        cut_normed = arg / average
        st = standardize(cut_normed)
        spectra.append(xr.DataArray(st, coords={dims[0]: kx, dims[1]: energy},
                                    dims = [dims[0], dims[1]], attrs=arg.attrs))
    return spectra


def normalize_3D(*args):
    scans_normed = []
    for arg in args:
        dims = arg.dims
        energy = arg[dims[2]].values
        kx = arg[dims[0]].values
        ky = arg[dims[1]].values
        sum_thing = np.nansum(arg.values)
        volume = len(kx) * len(ky) * len(energy)
        average = sum_thing / volume
        cut_normed = arg / average
        st = standardize(cut_normed)
        scans_normed.append(xr.DataArray(st, coords={dims[0]: kx, dims[1]: ky, dims[2]: energy},
                                         dims=[dims[0], dims[1], dims[2]], attrs=arg.attrs))
    return scans_normed


def guess_ef(*args):
    # Assuming sending in whole spectra not a crop region
    # but we would like to get a crop region because of the
    # case where the spectra have not been dewarped
    efs = []
    for arg in args:
        spec = arg.copy()
        if spec.ndim > 1:
            for dim_label in spec.dims:
                if dim_label != 'energy':
                    if len(spec[dim_label].values) > 150:
                        factor = int(np.ceil(len(spec[dim_label].values) / 50))
                        spec = spec.arpes.downsample({dim_label: factor})
                    axis = spec[dim_label].values
                    n_points = int(len(axis) / 6)
                    if n_points < 4:
                        n_points = 4
                    pmin = int(len(axis) / 2) - n_points  # center index minus 1/3 spectra to left of center
                    pmax = int(len(axis) / 2) + n_points  # center index plus 1/3 spectra to right of center
                    spec = spec.isel({dim_label: slice(pmin, pmax)})
                    spec = spec.sum(dim_label)
        edc_y = spec.values
        edc_x = spec.coords['energy'].values
        efs.append(edc_x[np.argmin(np.diff(edc_y))])
    return efs


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, plot_func, *args, **kwargs):
        super().__init__()
        self.plot_func = plot_func
        self.args = args
        self.kwargs = kwargs
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        plot_widget = self.plot_func(*self.args, **self.kwargs)
        layout.addWidget(plot_widget)
        self.setLayout(layout)
        self.setWindowTitle('Plot')
        self.show()


def confirm_ef(args, efs, json_file='spectra_info.json'):
    for arg, ef in zip(args, efs):
        print(f"guess_ef found {ef} as the fermi edge")
        plot_imagetool(arg)
        print(f"Is the value {ef} a good value for the Fermi edge? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            confirmed_ef = ef
        else:
            print("What value should it be? (Press Enter to skip)")
            new_ef = input().strip()
            if new_ef == '':
                confirmed_ef = None
            else:
                confirmed_ef = float(new_ef)

        update_json_with_attr(arg.attrs['filepath'], {'ef': confirmed_ef}, json_file)


def update_json_with_attr(filepath, attr, json_file='spectra_info.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Find the entry corresponding to the filepath and update the ef attribute
    for material, files in data['fermi_map'].items():
        if filepath in files:
            files[filepath].update(attr)

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def interpolate_lists(list1, list2, weight=0.5):
    """
    Interpolates between two lists of the same length.

    Parameters:
    - list1: First list of numbers.
    - list2: Second list of numbers.
    - weight: Weight towards the second list (0 <= weight <= 1).

    Returns:
    - A new list with interpolated values.
    """
    # Convert lists to NumPy arrays
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # Calculate interpolated array
    interpolated_arr = arr1 * (1 - weight) + arr2 * weight

    # Convert back to list and return
    return interpolated_arr.tolist()


def plot_imagetool(*args):
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if it does not exist then a QApplication is created
        app = QtWidgets.QApplication([])
    for arg in args:
        window = PlotWindow(arg.arpes.plot)
        app.exec_()
        window.close()
