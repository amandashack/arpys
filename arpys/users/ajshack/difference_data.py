import numpy as np
import xarray as xr
from load_spectra import load_spectra_by_material
from tools import normalize_3D, save_xarray_to_hdf5, update_json, plot_imagetool
from plotting import plot_all_fermi_surfaces, plot_diff
import json
import os


def difference_calc(specs1, specs2):
    """
    Calculate the difference between two lists of spectra.
    """
    if len(specs1) != len(specs2):
        print("You need to send in matching pairs for each spectra")
        return None
    diff_data = []
    for spec1, spec2 in zip(specs1, specs2):
        if not isinstance(spec1, xr.DataArray) or not isinstance(spec2, xr.DataArray):
            raise ValueError("All arguments must be xarray DataArray objects.")

        # Normalize the spectra
        spec1_norm = spec1 / spec1.max()
        spec2_norm = spec2 / spec2.max()

        # Calculate the difference
        diff_spec = spec2_norm - spec1_norm

        # Check if the difference is empty
        if diff_spec.size == 0:
            print(f"Warning: The difference spectrum for {spec1.attrs['filepath']} is empty.")

        diff_data.append(diff_spec)
    return diff_data


def main_diff(json_file, output_dir, material, alignment=None, photon_energy=None, k_conv=None, masked=None):
    with open(json_file, 'r') as f:
        spectra_info = json.load(f)

    spectra_CR = load_spectra_by_material(material, photon_energy=photon_energy, polarization="CR",
                                                alignment=alignment, k_conv=True, masked=True)
    spectra_CL = load_spectra_by_material(material, photon_energy=photon_energy, polarization="CL",
                                               alignment=alignment, k_conv=True, masked=True)
    spectra_CR_normed = normalize_3D(*spectra_CR)
    spectra_CL_normed = normalize_3D(*spectra_CL)
    diff_spectra = difference_calc(spectra_CR_normed, spectra_CL_normed)

    for diff_spec, spec1 in zip(diff_spectra, spectra_CR_normed):
        attrs = spec1.attrs
        new_filename = f"{material}_{attrs['photon_energy']}_{attrs['alignment']}_diff.h5"
        new_filepath = os.path.join(output_dir, new_filename).replace('\\', '/')
        new_attrs = {
            "alignment": attrs["alignment"],
            "polarization": "diff",
            "photon_energy": attrs["photon_energy"],
            "k_conv": attrs.get("k_conv", False),
            "masked": attrs.get("masked", False),
            "dewarped": attrs.get("dewarped", False)
        }
        diff_spec.attrs = new_attrs
        # Save the difference spectra
        save_xarray_to_hdf5(diff_spec, new_filepath)

        # Update the JSON
        new_entry = {
            new_filepath: {
                "alignment": attrs["alignment"],
                "polarization": "diff",
                "photon_energy": attrs["photon_energy"],
                "k_conv": attrs.get("k_conv", False),
                "masked": attrs.get("masked", False),
                "dewarped": attrs.get("dewarped", False)
            }
        }
        update_json(json_file, material, new_entry)


if __name__ == '__main__':
    json_file = "C:/Users/proxi/Documents/coding/arpys/arpys/users/ajshack/spectra_info.json"
    output_dir = "C:/Users/proxi/Documents/coding/data/fermi_maps"
    #main_diff(json_file, output_dir, "co33tas2", alignment='GM', k_conv=True, masked=True)
    spectra_diff = load_spectra_by_material("co33tas2", polarization="diff",
                                            alignment="GM", k_conv=True, masked=True)
    spectra_CR = load_spectra_by_material("co33tas2", polarization="CR",
                                            alignment="GM", k_conv=True, masked=True)
    spectra_CL = load_spectra_by_material("co33tas2", polarization="CL",
                                          alignment="GM", k_conv=True, masked=True)
    #print(spectra_diff)
    #plot_imagetool(*spectra_CR)
    #spectra_diff_normed = normalize_3D(*spectra_diff)
    #spectra_CR_normed = normalize_3D(*spectra_CR)
    #spectra_CL_normed = normalize_3D(*spectra_CL)
    plot_diff(spectra_diff, spectra_CR, spectra_CL, -0.02, ds=None, color='bwr')