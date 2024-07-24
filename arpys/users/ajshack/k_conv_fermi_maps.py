from load_spectra import load_spectra_by_photon_energy, load_spectra_by_material
from plotting import plot_all_fermi_surfaces
from k_conversion import SpectraConverter
import sys
import os
import h5py
import json
from tools import (normalize_3D, guess_ef, plot_imagetool, confirm_ef, save_xarray_to_hdf5,
                   update_json)


def k_conv_all(*args, ef=None):
    print("K converting all spectra")
    k_conv_spectra = []
    for arg in args:
        hv = arg.attrs['photon_energy']
        wf = arg.attrs['WorkFunction']
        ef = arg.attrs['ef']
        if ef:
            converter = SpectraConverter(arg, hv, wf, ef=ef)
        else:
            converter = SpectraConverter(arg, hv, wf)
        result = converter.map_k_reg()
        k_conv_spectra.append(result)
    return k_conv_spectra


def filter_dataset(*args):
    filtered_data = []
    for arg in args:
        if arg.attrs['ef'] != None:
            filtered_data.append(arg)
    return filtered_data

def main_k_conv(json_file, output_dir, material, polarization=None, alignment=None, photon_energy=None):
    with open(json_file, 'r') as f:
        spectra_info = json.load(f)

    spectra_data = load_spectra_by_material(material, photon_energy=photon_energy, polarization=polarization,
                                            alignment=alignment)
    spectra_data = filter_dataset(*spectra_data)
    spectra_k_conv = k_conv_all(*spectra_data)

    for k_spectrum in spectra_k_conv:
        attrs = k_spectrum.attrs
        new_filename = f"{material}_{attrs['photon_energy']}_{attrs['alignment']}_{attrs['polarization']}_k_conv.h5"
        new_filepath = os.path.join(output_dir, new_filename).replace('\\', '/')

        # Save the converted spectra
        save_xarray_to_hdf5(k_spectrum, new_filepath)

        # Update the JSON
        new_entry = {
            new_filepath: {
                "alignment": attrs["alignment"],
                "polarization": attrs["polarization"],
                "photon_energy": attrs["photon_energy"],
                "k_conv": True,
                "masked": attrs.get("masked", False),
                "dewarped": attrs.get("dewarped", False)
            }
        }
        update_json(json_file, material, new_entry)



if __name__ == '__main__':
    json_file = "C:/Users/proxi/Documents/coding/arpys/arpys/users/ajshack/spectra_info.json"
    output_dir = "C:/Users/proxi/Documents/coding/data/fermi_maps"
    #spectra_data = load_spectra_by_material("co55tas2")#, polarization="CL", alignment="GM", k_conv=False)
    #efs = guess_ef(*spectra_data)
    #confirm_ef(spectra_data, efs, json_file)
    main_k_conv(json_file, output_dir, "co55tas2")#, polarization="CR", alignment='GM')
    #spectra_data = load_spectra_by_material("co33tas2", polarization="CL",
    #                                        alignment="GM", k_conv=True)
    #spectra_normed = normalize_3D(*spectra_data)
    #plot_imagetool(*spectra_normed)
    #plot_all_fermi_surfaces(*spectra_normed)
