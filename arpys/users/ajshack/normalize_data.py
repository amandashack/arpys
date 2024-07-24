import numpy as np
import xarray as xr


def standardize(ar):
    """
    Standardizes the spectra intensities between 0 and 1.

    Parameters:
    - ar (xarray.DataArray): Input data array.

    Returns:
    - xarray.DataArray: Standardized data array with values between 0 and 1.
    """
    w_max = float(np.nanmax(ar))
    w_min = float(np.nanmin(ar))
    nr_values = (ar - w_min) / (w_max - w_min)
    return nr_values


def normalize_2D(*args):
    """
    Normalizes 2D spectra by averaging the values and standardizing.

    Parameters:
    - args: Variable length argument list of xarray.DataArray objects.

    Returns:
    - list: List of normalized xarray.DataArray objects.
    """
    spectra = []
    for arg in args:
        if not isinstance(arg, xr.DataArray):
            raise ValueError("All arguments must be xarray DataArray objects.")
        dims = arg.dims
        if len(dims) != 2:
            raise ValueError("Input xarray must have exactly 2 dimensions.")

        sum_thing = np.nansum(arg.values)
        area = arg.size
        average = sum_thing / area
        cut_normed = arg / average
        st = standardize(cut_normed)
        spectra.append(xr.DataArray(st, coords={dims[0]: arg[dims[0]], dims[1]: arg[dims[1]]},
                                    dims=[dims[0], dims[1]], attrs=arg.attrs))
    return spectra


def normalize_3D(*args):
    """
    Normalizes 3D spectra by averaging the values and standardizing.

    Parameters:
    - args: Variable length argument list of xarray.DataArray objects.

    Returns:
    - list: List of normalized xarray.DataArray objects.
    """
    scans_normed = []
    for arg in args:
        if not isinstance(arg, xr.DataArray):
            raise ValueError("All arguments must be xarray DataArray objects.")
        dims = arg.dims
        if len(dims) != 3:
            raise ValueError("Input xarray must have exactly 3 dimensions.")

        sum_thing = np.nansum(arg.values)
        volume = arg.size
        average = sum_thing / volume
        cut_normed = arg / average
        st = standardize(cut_normed)
        scans_normed.append(
            xr.DataArray(st, coords={dims[0]: arg[dims[0]], dims[1]: arg[dims[1]], dims[2]: arg[dims[2]]},
                         dims=[dims[0], dims[1], dims[2]], attrs=arg.attrs))
    return scans_normed