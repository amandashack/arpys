import numpy as np
import xarray as xr
from load_spectra import load_spectra_by_material
from tools import normalize_3D, save_xarray_to_hdf5, update_json
from plotting import plot_all_fermi_surfaces
import json
import os


def mask_fermi_maps(*args, radius=93, x_shift=0, y_shift=0):
    """
    Mask the edges of Fermi maps data and normalize it.

    Parameters:
    - args: xarray DataArray objects to be masked.
    - radius (float): Radius within which data points are kept. Default is 100.
    - x_shift (float): Shift in the x-direction for the mask. Default is 0.
    - y_shift (float): Shift in the y-direction for the mask. Default is 0.

    Returns:
    - masked_data (list): List of masked xarray DataArray objects.
    """
    masked_data = []
    for arg in args:
        if not isinstance(arg, xr.DataArray):
            raise ValueError("All arguments must be xarray DataArray objects.")

        dims = arg.dims
        x_len = len(arg[dims[0]].values)
        y_len = len(arg[dims[1]].values)
        z_len = len(arg[dims[2]].values)

        x = np.linspace(-10, 10, x_len) - x_shift
        y = np.linspace(-10, 10, y_len) - y_shift
        z = np.linspace(-10, 10, z_len)
        x, y, z = np.meshgrid(y, x, z)

        mask = (x)**2 + (y)**2 <= radius
        arg = arg.where(mask)
        masked_data.append(arg)

    return masked_data


def main_mask(json_file, output_dir, material, polarization=None, alignment=None, photon_energy=None, k_conv=None):
    with open(json_file, 'r') as f:
        spectra_info = json.load(f)

    spectra_data = load_spectra_by_material(material, photon_energy=photon_energy, polarization=polarization,
                                            alignment=alignment, k_conv=k_conv)
    spectra_masked = mask_fermi_maps(*spectra_data)

    for mask_spectrum in spectra_masked:
        attrs = mask_spectrum.attrs
        new_filename = f"{material}_{attrs['photon_energy']}_{attrs['alignment']}_{attrs['polarization']}_masked.h5"
        new_filepath = os.path.join(output_dir, new_filename).replace('\\', '/')

        # Save the converted spectra
        save_xarray_to_hdf5(mask_spectrum, new_filepath)

        # Update the JSON
        new_entry = {
            new_filepath: {
                "alignment": attrs["alignment"],
                "polarization": attrs["polarization"],
                "photon_energy": attrs["photon_energy"],
                "k_conv": attrs.get("k_conv", False),
                "masked": True,
                "dewarped": attrs.get("dewarped", False)
            }
        }
        update_json(json_file, material, new_entry)


if __name__ == '__main__':
    json_file = "C:/Users/proxi/Documents/coding/arpys/arpys/users/ajshack/spectra_info.json"
    output_dir = "C:/Users/proxi/Documents/coding/data/fermi_maps"
    #main_mask(json_file, output_dir, "co55tas2", k_conv=True)
    spectra_masked = load_spectra_by_material("co55tas2", alignment="GK", k_conv=True, masked=True)
    #plot_imagetool(*spectra_normed)
    #spectra_masked = mask_fermi_maps(*spectra_data)
    spectra_normed = normalize_3D(*spectra_masked)
    plot_all_fermi_surfaces(*spectra_normed)