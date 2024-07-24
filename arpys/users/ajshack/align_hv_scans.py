from load_spectra import load_hv_scan_by_material
from tools import plot_imagetool, interpolate_lists, guess_ef, update_json, save_xarray_to_hdf5
import xarray as xr
import numpy as np
import os


def adjust_ef(scan):
    adjusted_cuts = []
    dims = scan.dims
    # step axis is in photon energy
    step_axis = scan[dims[0]]
    for val in step_axis:
        cut = scan.sel({dims[0]: val})
        print(f"What is a good index for the Fermi edge? (Press Enter to skip)")
        new_ef = input().strip()
        if new_ef == '':
            confirmed_ef = None
            continue
        else:
            confirmed_ef = int(new_ef)
        ef = cut
        new_be = cut[dims[2]].values - confirmed_ef
        newcoords = {dims[2]: new_be, dims[1]: cut[dims[1]]}
        cut_new = xr.DataArray(cut.values, dims=dims, coords=newcoords)
        adjusted_cuts.append(cut_new)

    aligned_eks = []
    first_ek = adjusted_cuts[0]
    aligned_eks.append(first_ek)
    for i in np.arange(1, len(adjusted_cuts)):
        interped = adjusted_cuts[i].interp_like(first_ek)
        aligned_eks.append(interped)
    aligned_photonE_scan = xr.concat(aligned_eks, dims[0])
    aligned_photonE_scan = aligned_photonE_scan.assign_coords(coords={dims[0]: scan[dims[0]]})
    aligned_photonE_scan.attrs = scan.attrs
    return aligned_photonE_scan


def main_dewarp(json_file, output_dir, material, alignment=None, polarization=None, photon_energy=None):
    # only 1 at a time
    spectra_data = load_hv_scan_by_material(material, photon_energy=photon_energy, alignment=alignment,
                                             polarization=polarization)
    spectra_dewarp = adjust_ef(spectra_data[0])

    attrs = spectra_dewarp.attrs
    new_filename = f"hvscan_{material}_{attrs['photon_energy'][0]}_{attrs['photon_energy'][1]}_{attrs['alignment']}_dewarp.h5"
    new_filepath = os.path.join(output_dir, new_filename).replace('\\', '/')
    new_attrs = {
        "alignment": attrs["alignment"],
        "polarization": "diff",
        "photon_energy": attrs["photon_energy"],
        "k_conv": attrs.get("k_conv", False),
        "masked": attrs.get("masked", False),
        "dewarped": attrs.get("dewarped", False)
    }
    spectra_dewarp.attrs = new_attrs
    # Save the difference spectra
    save_xarray_to_hdf5(spectra_dewarp, new_filepath)

    # Update the JSON
    new_entry = {
        new_filepath: {
            "alignment": attrs["alignment"],
            "polarization": attrs["polarization"],
            "photon_energy": attrs["photon_energy"],
            "k_conv": attrs.get("k_conv", False),
            "masked": attrs.get("masked", False),
            "dewarped": True,
            "beam": attrs["beam"],
        }
    }
    update_json(json_file, material, new_entry, scan_type='hv_scan')


def combine_spectra(spec1, spec2):
    hv_70_120_en = spec1.energy.values.tolist()
    hv_30_70_en = spec2.energy.values.tolist()

    # Calculate the spacing in the second array
    spacing_differences = [hv_30_70_en[i] - hv_30_70_en[i - 1] for i in range(1, len(hv_30_70_en))]
    average_spacing = sum(spacing_differences) / len(spacing_differences)

    # Determine the number of values to add to the first array
    num_values_to_add = abs(int((hv_30_70_en[0] - hv_70_120_en[0]) / average_spacing))

    # Generate new values to add to the beginning of the first array
    new_values = [hv_70_120_en[0] - ((i + 1) * average_spacing) for i in range(num_values_to_add)]
    print(list(reversed(new_values)))

    # Adjust the first array
    adjusted_70_120 = list(reversed(new_values)) + hv_70_120_en[:-num_values_to_add]

    # Show the results
    print(len(adjusted_70_120), hv_30_70_en[0], hv_30_70_en[-1], adjusted_70_120[0], adjusted_70_120[-1])


if __name__ == '__main__':
    json_file = "C:/Users/proxi/Documents/coding/arpys/arpys/users/ajshack/spectra_info.json"
    output_dir = "C:/Users/proxi/Documents/coding/data/hv_scans"
    main_dewarp(json_file, output_dir, "co33tas2", polarization="CL", alignment="GK", photon_energy=[60, 170])
    #spectra_data = load_hv_scan_by_material("co33tas2", polarization="CL", alignment="GK", photon_energy=[60, 170])
    #plot_imagetool(*spectra_data)