import json
import os
from arpys.loaders.ssrl import load_ssrl_52
import arpys
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from restructure import fix_array
from tools import plot_imagetool
import h5py
import xarray as xr
import numpy as np
import math


# Load the JSON file
with open('spectra_info.json', 'r') as json_file:
    spectra_info = json.load(json_file)

# Define the base path
base_path = "C:/Users/proxi/Documents/coding/data/"


# Function to generate unique names for the datasets
def generate_unique_name(prefix, index):
    return f"{prefix}_{index}"


def load_ssrl_52_photonEscan(filename):
    conv = {'X': 'x', 'Z': 'z', 'ThetaX': 'slit', 'ThetaY': 'perp', 'Theta Y': 'perp', 'Kinetic Energy': 'energy'}
    f = h5py.File(filename, 'r')
    # 3d dataset, kinetic energy, angle, photon energy
    counts = np.array(f['Data']['Count'])
    print(counts.shape)
    I0 = np.abs(np.array(f['MapInfo']['Beamline:I0']))

    xaxis_offsets = np.array(f['MapInfo']['Measurement:XAxis:Offset'])
    xaxis_maxs = np.array(f['MapInfo']['Measurement:XAxis:Maximum'])
    xaxis_size = counts.shape[0]
    try:
        yaxis_offsets = np.array(f['MapInfo']['Measurement:YAxis:Offset'])
        yaxis_deltas = np.array(f['MapInfo']['Measurement:YAxis:Delta'])
        yaxis_size = counts.shape[1]
    except KeyError:
        yaxis_size = counts.shape[1]
        yaxis_offsets = np.repeat(f['Data']['Axes1'].attrs['Offset'], yaxis_size)
        yaxis_deltas = np.repeat(f['Data']['Axes1'].attrs['Delta'], yaxis_size)

    if (type(f['Data']['Axes2'].attrs['Offset']) is str):
        zaxis_coord = np.array(f['MapInfo']['Beamline:energy'])
        print(zaxis_coord)
        zaxis_size = len(zaxis_coord)
    else:
        zaxis_offset = f['Data']['Axes2'].attrs['Offset']
        zaxis_delta = f['Data']['Axes2'].attrs['Delta']
        zaxis_size = counts.shape[2]
        zaxis_max = zaxis_size * zaxis_delta + zaxis_offset
        zaxis_coord = np.linspace(zaxis_offset, zaxis_max, num=zaxis_size)

    photon_energy_scan_dataarrays = []

    # Slice by slice along z (photon energy)
    for photon_energy_slice in np.arange(len(zaxis_coord.tolist())):
        ekslice = counts[:, :, photon_energy_slice] / I0[photon_energy_slice]
        kinetic_coords = np.linspace(xaxis_offsets[photon_energy_slice], xaxis_maxs[photon_energy_slice],
                                     num=xaxis_size)
        angle_coords = np.arange(yaxis_size) * yaxis_deltas[photon_energy_slice] + yaxis_offsets[photon_energy_slice]
        dims = ('energy', 'slit')
        coords = {'energy': kinetic_coords, 'slit': angle_coords}
        ekslice_dataarray = xr.DataArray(ekslice, dims=dims, coords=coords)

        # Cut down on window to find ef with initial guess, will always need tuning if mono drifts too much...
        photon_energy = zaxis_coord[photon_energy_slice]
        workfunc = 4.465
        efguess = math.floor(photon_energy - workfunc)
        binding_coords = kinetic_coords - efguess

        newcoords = {'energy': binding_coords, 'slit': angle_coords}
        ekslice_binding = xr.DataArray(ekslice, dims=dims, coords=newcoords)
        photon_energy_scan_dataarrays.append(ekslice_binding)

    aligned_eks = []
    first_ek = photon_energy_scan_dataarrays[0]
    aligned_eks.append(first_ek)
    for i in np.arange(1, len(photon_energy_scan_dataarrays)):
        interped = photon_energy_scan_dataarrays[i].interp_like(first_ek)
        aligned_eks.append(interped)

    attrs = dict(f['Beamline'].attrs).copy()
    attrs.update(dict(f['Manipulator'].attrs))
    attrs.update(dict(f['Measurement'].attrs))
    attrs.update(dict(f['Temperature'].attrs))
    attrs.update(dict(f['UserSettings'].attrs))
    attrs.update(dict(f['UserSettings']['AnalyserSlit'].attrs))

    aligned_photonE_scan = xr.concat(aligned_eks, 'photon_energy')
    aligned_photonE_scan = aligned_photonE_scan.assign_coords(coords={'photon_energy': zaxis_coord[:]})
    aligned_photonE_scan.attrs = attrs
    return aligned_photonE_scan


# Function to add the new attribute based on material name
def add_material_attribute(data, material):
    if '33' in material:
        data.attrs['material_value'] = 0.33
    elif '55' in material:
        data.attrs['material_value'] = 0.55
    else:
        data.attrs['material_value'] = 0
    return data


def load_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        # Extract data
        data = f['data'][:]

        # Extract dimensions
        keys = list(f.keys())
        keys.remove('data')  # Remove the 'data' key

        # Determine the correct order of dimensions by matching lengths
        axes = {key: f[key][:] for key in keys}
        dims = [None] * len(data.shape)
        coords = {}

        for key, axis in axes.items():
            axis_len = axis.shape[0]
            for i, dim_len in enumerate(data.shape):
                if axis_len == dim_len:
                    dims[i] = key
                    coords[key] = axis
                    break

        # Check if all dimensions are assigned
        if None in dims:
            raise ValueError("Could not match all dimensions with the data shape.")

        # Extract attributes
        attrs = dict(f.attrs)

    # Create DataArray
    data_array = xr.DataArray(
        data,
        dims=dims,
        coords=coords,
        attrs=attrs
    )

    return data_array


def load_dataset(fp, scan_type="fermi_map"):
    if scan_type == "fermi_map" or scan_type == "fermi_map_k":
        try:
            dataset = load_ssrl_52(fp)
            dataset = fix_array(dataset, scan_type)
        except KeyError:
            dataset = load_h5(fp)
            dataset = fix_array(dataset, scan_type)
        return dataset
    else:
        try:
            dataset = load_ssrl_52_photonEscan(fp)
            dataset = fix_array(dataset, scan_type)
        except KeyError:
            dataset = load_h5(fp)
            dataset = fix_array(dataset, scan_type)
        return dataset


# Function to load spectra based on photon energy
def load_spectra_by_photon_energy(photon_energy, polarization=None, alignment=None, material=None):
    spectra_data = []
    materials_to_search = [material] if material else spectra_info['fermi_map'].keys()

    for material in materials_to_search:
        if material not in spectra_info['fermi_map']:
            print("That material name was not found in the json.")
            continue
        for file_path, attributes in spectra_info['fermi_map'][material].items():
            if (attributes['photon_energy'] == photon_energy) and \
                    (polarization is None or attributes['polarization'] == polarization) and \
                    (alignment is None or attributes['alignment'] == alignment):
                dataset = load_dataset(file_path)
                dataset = fix_array(dataset, 'fermi_map')
                dataset.attrs.update(attributes)
                dataset.attrs.update({"filepath": file_path})
                add_material_attribute(dataset, material)
                spectra_data.append(dataset)
    return spectra_data


def load_spectra_by_material(material, photon_energy=None, polarization=None, alignment=None,
                             k_conv=False, masked=False, dewarped=False):
    spectra_data = []
    if material in spectra_info['fermi_map']:
        for file_path, attributes in spectra_info['fermi_map'][material].items():
            if ((photon_energy is None or attributes['photon_energy'] == photon_energy) and
                    (polarization is None or attributes['polarization'] == polarization) and
                    (alignment is None or attributes['alignment'] == alignment) and
                    (attributes['k_conv'] == k_conv) and
                    (attributes['masked'] == masked) and
                    (attributes['dewarped'] == dewarped)):
                # select between k converted or not so that fix_array is correct
                if k_conv:
                    dataset = load_dataset(file_path, scan_type='fermi_map_k')
                else:
                    dataset = load_dataset(file_path, scan_type='fermi_map')
                dataset.attrs.update(attributes)
                dataset.attrs.update({"filepath": file_path})
                add_material_attribute(dataset, material)
                spectra_data.append(dataset)
    else:
        print("That material name was not found in the json.")
    return spectra_data


def load_hv_scan_by_material(material, photon_energy=None, polarization=None, alignment=None,
                             k_conv=False, masked=False, dewarped=False, beam="SSRL"):
    spectra_data = []
    if material in spectra_info['hv_scan']:
        for file_path, attributes in spectra_info['hv_scan'][material].items():
            if ((photon_energy is None or attributes['photon_energy'] == photon_energy) and
                    (polarization is None or attributes['polarization'] == polarization) and
                    (alignment is None or attributes['alignment'] == alignment) and
                    (attributes['k_conv'] == k_conv) and
                    (attributes['masked'] == masked) and
                    (attributes['dewarped'] == dewarped) and
                    (attributes['beam'] == beam)):
                if beam == "SSRL":
                    if k_conv:
                        dataset = load_dataset(file_path, scan_type='hv_scan')
                    else:
                        dataset = load_dataset(file_path, scan_type='hv_scan')
                else:
                    # this is where we put loading for APS
                    pass
                dataset.attrs.update(attributes)
                dataset.attrs.update({"filepath": file_path})
                add_material_attribute(dataset, material)
                spectra_data.append(dataset)
    else:
        print("That material name was not found in the json.")
    return spectra_data


if __name__ == '__main__':
    #spectra_data = load_hv_scan_by_material("co33tas2", polarization="CL", alignment="GK", photon_energy=[60, 170])
    spectra_data = load_spectra_by_material("co33tas2", photon_energy=None, polarization=None, alignment=None,
                                            k_conv=False, masked=False, dewarped=False)
    #print(spectra_data)
    plot_imagetool(*spectra_data)
