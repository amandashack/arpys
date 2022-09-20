import arpys
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
import math
import h5py
from arpys.loaders.maestro import load_maestro_fits_hvscan, align_binding, read_maestro_fits_attrs
# from arpys.loaders.ssrl import load_ssrl_52_photonEscan
from lmfit.models import ThermalDistributionModel, LinearModel, ConstantModel, PolynomialModel
from PyQt5.QtWidgets import QApplication
from scipy.interpolate import RegularGridInterpolator
from pyimagetool import ImageTool

"""
spectrum.attrs - empty dict
spectrum.coords - the coordinate values and names. 'energy': -1.583 - 0.2577, 'slit': -17.9 to 21.7, 
                  and 'photon_energy': 62 - 150
spectrum.data - the 3D array that holds the data
spectrum.dims - the coordinate names/dimensions. 'photon_energy, 'slit', and 'energy'
spectrum['energy'] - 
"""


def load_ssrl_52_photonEscan(filename):
    conv = {'X': 'x', 'Z': 'z', 'ThetaX': 'slit', 'ThetaY': 'perp', 'Theta Y': 'perp', 'Kinetic Energy': 'energy'}
    f = h5py.File(filename, 'r')
    # 3d dataset, kinetic energy, angle, photon energy
    counts = np.array(f['Data']['Count'])
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
        zaxis_coord = f['MapInfo']['Beamline:energy']
        zaxis_size = len(zaxis_coord)
    else:
        zaxis_offset = f['Data']['Axes2'].attrs['Offset']
        zaxis_delta = f['Data']['Axes2'].attrs['Delta']
        zaxis_size = counts.shape[2]
        zaxis_max = zaxis_size * zaxis_delta + zaxis_offset
        zaxis_coord = np.linspace(zaxis_offset, zaxis_max, num=zaxis_size)

    photon_energy_scan_dataarrays = []

    # Slice by slice along z (photon energy)
    for photon_energy_slice in np.arange(zaxis_size):
        ekslice = counts[:, :, photon_energy_slice] / I0[photon_energy_slice]
        kinetic_coords = np.linspace(xaxis_offsets[photon_energy_slice], xaxis_maxs[photon_energy_slice],
                                     num=xaxis_size)
        angle_coords = np.arange(yaxis_size) * yaxis_deltas[photon_energy_slice] + yaxis_offsets[photon_energy_slice]
        dims = ('energy', 'slit')
        coords = {'energy': kinetic_coords, 'slit': angle_coords}
        ekslice_dataarray = xr.DataArray(ekslice, dims=dims, coords=coords)

        # Cut down on window to find ef with initial guess, will always need tuning if mono drifts too much...
        photon_energy = zaxis_coord[photon_energy_slice]
        workfunc = 4.365
        efguess = photon_energy - workfunc
        maxkinetic = np.nanmax(kinetic_coords)
        effinder = ekslice_dataarray.sel({'energy': slice(efguess - 1.0, maxkinetic)})
        ef = effinder.arpes.guess_ef()
        binding_coords = kinetic_coords - ef

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
    aligned_photonE_scan = aligned_photonE_scan.assign_coords(coords={'photon_energy': zaxis_coord})
    aligned_photonE_scan = aligned_photonE_scan.assign_attrs(attrs=attrs)
    return aligned_photonE_scan


def np_transpose(xar, tr):
    """Transpose the RegularSpacedData
    :param xar: starting xarray
    :param tr: list of the new transposed order
    """
    coords = {}
    dims = []
    for i in tr:
        name = list(xar.dims)[i]
        coords[name] = xar[name].values
        dims.append(list(xar.dims)[i])
    return xr.DataArray(np.transpose(xar.data, tr), dims=dims, coords=coords)

"""
def fix_array(ar, scan_type):
    
    #make your array uniform based on what type of scan it is
    #:param ar: input xarray
    #:param scan_type: the scan type can be "hv_scan"
    #:return: the fixed array
    
    if scan_type == "hv_scan":
        photon_energy = ar.photon_energy.values
        slit = ar.slit.values
        energy = ar.energy.values
        size_new = [len(photon_energy), len(slit), len(energy)]
        size_ar = list(ar.values.shape)
        tr = [size_ar.index(i) for i in size_new]
        return np_transpose(ar, tr)
    if scan_type == "fermi_map":
        slit = ar.slit.values
        perp = ar.perp.values
        energy = ar.energy.values
        size_new = [len(slit), len(perp), len(energy)]
        size_ar = list(ar.values.shape)
        tr = [size_ar.index(i) for i in size_new]
        return np_transpose(ar, tr)
"""


def fix_array(ar, scan_type):
    """
    make your array uniform based on what type of scan it is
    :param ar: input xarray
    :param scan_type: the scan type can be "hv_scan"
    :return: the fixed array
    """

    def np_transpose(xar, tr):
        """Transpose the RegularSpacedData
        :param xar: starting xarray
        :param tr: list of the new transposed order
        """
        coords = {}
        dims = []
        for i in tr:
            name = list(xar.dims)[i]
            coords[name] = xar[name].values
            dims.append(list(xar.dims)[i])
        return xr.DataArray(np.transpose(xar.data, tr), dims=dims, coords=coords)

    if scan_type == "hv_scan":
        photon_energy = ar.photon_energy.values
        slit = ar.slit.values
        energy = ar.energy.values
        size_new = [len(photon_energy), len(slit), len(energy)]
        size_ar = list(ar.values.shape)
        tr = [size_ar.index(i) for i in size_new]
        print(size_new, size_ar, tr)
        return np_transpose(ar, tr)
    if scan_type == "fermi_map":
        slit = ar.slit.values
        perp = ar.perp.values
        energy = ar.energy.values
        size_new = [len(slit), len(perp), len(energy)]
        size_ar = list(ar.values.shape)
        if size_new == size_ar:
            size_new = ar.coords.keys()
            before = list(ar.dims)
            tr = [before.index(i) for i in size_new]
        else:
            tr = [size_ar.index(i) for i in size_new]
        return np_transpose(ar, tr)
    if scan_type == "single":
        slit = ar.slit.values
        energy = ar.energy.values
        size_new = [len(slit), len(energy)]
        size_ar = list(ar.values.shape)
        tr = [size_ar.index(i) for i in size_new]
        return np_transpose(ar, tr)


def k_forward_normal(ke, theta, v0):
    """
    convert to kx and kz
    :param ke: kinetic energy data
    :param theta: what slit angle to convert at
    :param v0: inner potential
    :return: kx, kz
    """
    rad = math.radians(theta)
    kx = 0.512 * np.sqrt(ke) * np.sin(rad)
    kz = 0.512 * np.sqrt(ke * (np.cos(rad))**2 + v0)
    return kx, kz


def k_forward_normal_partial(ke, theta):
    """
    convert to kx
    :param ke: kinetic energy data
    :param theta: what slit angle to convert at
    :return:
    """
    rad = math.radians(theta)
    kx = 0.512 * np.sqrt(ke) * np.sin(rad)
    return kx


def k_reverse_normal(kx, kz, v0, wf, be):
    """
    convert back to real space
    :param kx:
    :param kz:
    :param v0:
    :param wf:
    :param be:
    :return:
    """
    a = 0.512
    ke = (kz/a)**2 + (kx/a)**2 - v0
    theta = (180/np.pi) * np.arcsin(kx/(a*np.sqrt(ke)))
    hv = ke + wf - be
    return hv, theta


def k_reverse_normal_partial(kx, hv, be, wf):
    """
    convert back to real space
    :param kx:
    :param hv:
    :param be:
    :param wf:
    :return:
    """
    a = 0.512
    ke = hv + be - wf
    theta = (180 / np.pi) * np.arcsin(kx / (a * np.sqrt(ke)))
    return theta


def normalize_scan(scan, d, key):
    """
    break scan into each photon energy and normalize to integrate intensity
    :param scan:
    :param x0: lower slit position
    :param x1: upper slit position
    :param e0: lower energy range
    :param e1: upper energy range
    :return:
    """
    dims = list(d.keys())
    cuts = []
    for val in scan[key]:
        cut = scan.sel({key: val}, method='nearest')
        cut_cr = cut.sel({dims[0]: slice(d[dims[0]][0], d[dims[0]][1])}).sel(
            {dims[1]: slice(d[dims[1]][0], d[dims[1]][1])})
        sum_thing = cut_cr.sum(dims[0])
        sum_thing2 = np.sum(sum_thing.values.astype('int64'))
        area = cut_cr[dims[1]].values.size * cut_cr[dims[0]].values.size
        average = sum_thing2 / area
        cut_normed = cut / average
        st = standardize(cut_normed)
        cuts.append(st)

    scan_normed = xr.concat(cuts, key)
    scan_normed = scan_normed.assign_coords({key: scan[key]})
    scan_normed = fix_array(scan_normed, scan_type="fermi_map")
    return scan_normed


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


def find_spacing(mm, hv, ke, slit, inner_potential=14, num_x=None, num_z=None, partial=False):
    if partial:
        if num_x is None:
            middle_ke = int(len(ke) / 2)
            middle_slit = int(len(slit) / 2)
            kx_ur = k_forward_normal_partial(ke[middle_ke], slit[middle_slit])
            kx_ll = k_forward_normal_partial(ke[middle_ke - 1], slit[middle_slit - 1])
            dkx = kx_ur - kx_ll
            num_x = abs(int((mm[1] - mm[0]) / dkx))
            print(num_x, len(slit))
            return num_x
        else:
            return num_x
    elif not partial:
        if num_x is None and num_z is None:
            middle_ke = int(len(ke) / 2)
            middle_slit = int(len(slit) / 2)
            kx_ur, kz_ur = k_forward_normal(ke[middle_ke],
                                            slit[middle_slit], inner_potential)
            kx_ll, kz_ll = k_forward_normal(ke[middle_ke - 1],
                                            slit[middle_slit - 1], inner_potential)
            dkx = kx_ur - kx_ll
            dkz = kz_ur - kz_ll
            print("\n\n\nThese are the spacing: ", dkx, dkz)
            num_x = abs(int((mm[0][1] - mm[0][0]) / dkx))
            num_z = abs(int((mm[1][1] - mm[1][0]) / dkz))
            print(num_x, num_z, len(slit), len(hv))
        elif num_x or num_z:
            if num_x is None:
                num_x = slit.size
            elif num_z is None:
                num_z = hv.size
        else:
            print("There is something wrong in choosing your "
                  "spacing for reg grid kz conversion")
    return num_x, num_z


def convert_2d_normal_emission(scans, be=0, inner_potential=14, wf=4.2, num_x=None, num_z=None):
    """
    :param scans: xarray of the photon energy scan
    :param be: binding energy to select to look at a kz cut
    :param inner_potential: inner potential
    :param wf: work function
    :param num_x: number of points to interpolate in the x direction, setting
                  to None means it will do a kz conversion to find the spacing
    :param num_z: number of points to interpolate in the z direction, setting
                  to None means it will do a kz conversion to find the spacing
    :return: xarray with an iso-energy cut kz converted
    """
    energy_iso = scans.sel({"energy": be}, method='nearest')
    photon_energy_iso = energy_iso['photon_energy'].values
    slit = energy_iso['slit'].values
    ke = photon_energy_iso - wf + be
    print("\nlsdjfpsodjf: ", scans.attrs, "\nfinally youre here: ", scans.coords,
          "\nthis is the data: ", scans.data, "\nthis is the dims: ",
          scans.dims, "\nphoton energy values: ", energy_iso['photon_energy'].values,
          '\nbinding energy 0: ', energy_iso, '\nke: ', ke, '\nslit: ', slit)
    interp_object = RegularGridInterpolator((photon_energy_iso, slit),
                                            energy_iso.values, bounds_error=False)
    kx_ur, kz_ur = k_forward_normal(np.nanmax(ke),
                                    np.nanmax(slit), inner_potential)
    kx_ul, kz_ul = k_forward_normal(np.nanmax(ke),
                                    np.nanmin(slit), inner_potential)
    kx_um, kz_um = k_forward_normal(np.nanmax(ke),
                                    0, inner_potential)
    kx_lr, kz_lr = k_forward_normal(np.nanmin(ke),
                                    np.nanmax(slit), inner_potential)
    kx_ll, kz_ll = k_forward_normal(np.nanmin(ke),
                                    np.nanmin(slit), inner_potential)
    xx = [kx_ur, kx_ul, kx_um, kx_lr, kx_ll]
    zz = [kz_ur, kz_ul, kz_um, kz_lr, kz_ll]
    min_x = min(xx)
    max_x = max(xx)
    min_z = min(zz)
    max_z = max(zz)
    mm = [[min_x, max_x], [min_z, max_z]]
    num_x, num_z = find_spacing(mm, photon_energy_iso, ke, slit, inner_potential, num_x, num_z, partial=False)
    print('\n\nnumber spacing: ', num_x, num_z, slit.size, photon_energy_iso.size)
    kx_new = np.sort(np.linspace(min_x, max_x, num=num_x, endpoint=True))
    kz_new = np.sort(np.linspace(min_z, max_z, num=num_z, endpoint=True))
    kxx, kzz = np.meshgrid(kx_new, kz_new, indexing='ij', sparse=False)
    hv, theta = k_reverse_normal(kxx, kzz, inner_potential, wf, be)
    points_stacked = np.stack((hv.reshape(-1, order='C'), theta.reshape(-1, order='C')))
    interp_out = interp_object(points_stacked.T)
    interp_out = interp_out.reshape((hv.shape[0], hv.shape[1]), order='C')
    return xr.DataArray(interp_out, dims=['kx', 'kz'], coords={'kx': kx_new, 'kz': kz_new})


def convert_3d_normal_emission(scans, inner_potential=14, wf=4.2, num_x=None, num_z=None):
    """
    :param scans: xarray of the photon energy scan
    :param inner_potential: inner potential
    :param wf: work function
    :param num_x: number of points to interpolate in the x direction, setting
                  to None means it will do a kz conversion to find the spacing
    :param num_z: number of points to interpolate in the z direction, setting
                  to None means it will do a kz conversion to find the spacing
    :return: xarray with an iso-energy cut kz converted
    """
    binding_energy = scans.energy.values
    photon_energy = scans.photon_energy.values
    slit = scans.slit.values
    max_ke1 = np.nanmax(photon_energy) - wf + np.nanmax(binding_energy)
    max_ke2 = np.nanmin(photon_energy) - wf + np.nanmax(binding_energy)
    min_ke = np.nanmin(photon_energy) - wf + np.nanmin(binding_energy)
    print("\nfinally youre here: ", scans.coords,
          "\nthis is the data: ", scans, "\nthis is the dims: ",
          scans.dims, "\nbinding energy values: ", binding_energy.shape, '\nslit: ', slit.shape,
          "\nscan shape: ", scans.shape, "\nphoton energy: ", photon_energy.shape)
    interp_object = RegularGridInterpolator((photon_energy, slit, binding_energy),
                                            scans.data, bounds_error=False)
    kx_ur, kz_ur = k_forward_normal(max_ke1,
                                    np.nanmax(slit), inner_potential)
    kx_ul, kz_ul = k_forward_normal(max_ke1,
                                    np.nanmin(slit), inner_potential)
    kx_um, kz_um = k_forward_normal(max_ke1,
                                    0, inner_potential)
    kx_lr_maxke, kz_lr_maxke = k_forward_normal(max_ke2,
                                                np.nanmax(slit), inner_potential)
    kx_ll_maxke, kz_ll_maxke = k_forward_normal(max_ke2,
                                                np.nanmin(slit), inner_potential)
    kx_lr_minke, kz_lr_minke = k_forward_normal(min_ke,
                                                np.nanmax(slit), inner_potential)
    kx_ll_minke, kz_ll_minke = k_forward_normal(min_ke,
                                                np.nanmin(slit), inner_potential)
    xx = [kx_ur, kx_ul, kx_um, kx_lr_maxke, kx_ll_maxke, kx_lr_minke, kx_ll_minke]
    zz = [kz_ur, kz_ul, kz_um, kz_lr_maxke, kz_ll_maxke, kz_lr_minke, kz_ll_minke]
    min_x = min(xx)
    max_x = max(xx)
    min_z = min(zz)
    max_z = max(zz)
    mm = [[min_x, max_x], [min_z, max_z]]
    be = np.nanmax(binding_energy)
    energy_iso = scans.sel({"energy": be}, method='nearest')
    photon_energy_iso = energy_iso['photon_energy'].values
    ke_iso = photon_energy_iso - wf - be
    num_x, num_z = find_spacing(mm, photon_energy_iso, ke_iso, slit, inner_potential, num_x, num_z, partial=False)
    kx_new = np.sort(np.linspace(min_x, max_x, num=num_x, endpoint=True))
    kz_new = np.sort(np.linspace(min_z, max_z, num=num_z, endpoint=True))
    be_new = np.sort(np.linspace(np.nanmin(binding_energy), np.nanmax(binding_energy),
                                 num=len(binding_energy), endpoint=True))
    kxx, kzz, be_grid = np.meshgrid(kx_new, kz_new, be_new, indexing='ij', sparse=False)
    hv, theta = k_reverse_normal(kxx, kzz, inner_potential, wf, be_grid)
    points_stacked = np.stack((hv.reshape(-1, order='C'), theta.reshape(-1, order='C'),
                              be_grid.reshape(-1, order='C')))
    interp_out = interp_object(points_stacked.T)
    interp_out = interp_out.reshape((hv.shape[0], theta.shape[1], be_grid.shape[2]), order='C')
    return xr.DataArray(interp_out, dims=['kx', 'kz', 'energy'], coords={'kx': kx_new,
                                                                         'kz': kz_new,
                                                                         'energy': be_new})


def convert_partial_3d_normal_emission(scans, inner_potential=14, wf=4.2, num_x=None, num_z=None):
    """
    :param scans: xarray of the photon energy scan
    :param inner_potential: inner potential
    :param wf: work function
    :param num_x: number of points to interpolate in the x direction, setting
                  to None means it will do a kz conversion to find the spacing
    :param num_z: number of points to interpolate in the z direction, setting
                  to None means it will do a kz conversion to find the spacing
    :return: xarray with an iso-energy cut kz converted
    """
    binding_energy = scans.energy.values
    photon_energy = scans.photon_energy.values
    slit = scans.slit.values
    max_ke1 = np.nanmax(photon_energy) - wf + np.nanmax(binding_energy)
    max_ke2 = np.nanmin(photon_energy) - wf + np.nanmax(binding_energy)
    min_ke = np.nanmin(photon_energy) - wf + np.nanmin(binding_energy)
    interp_object = RegularGridInterpolator((photon_energy, slit, binding_energy),
                                            scans.data, bounds_error=False)
    kx_ur = k_forward_normal_partial(max_ke1, np.nanmax(slit))
    kx_ul = k_forward_normal_partial(max_ke1, np.nanmin(slit))
    kx_um = k_forward_normal_partial(max_ke1, 0)
    kx_lr_maxke = k_forward_normal_partial(max_ke2, np.nanmax(slit))
    kx_ll_maxke = k_forward_normal_partial(max_ke2, np.nanmin(slit))
    kx_lr_minke = k_forward_normal_partial(min_ke, np.nanmax(slit))
    kx_ll_minke = k_forward_normal_partial(min_ke, np.nanmin(slit))
    xx = [kx_ur, kx_ul, kx_um, kx_lr_maxke, kx_ll_maxke, kx_lr_minke, kx_ll_minke]
    min_x = min(xx)
    max_x = max(xx)
    mm = [min_x, max_x]
    be = np.nanmax(binding_energy)
    energy_iso = scans.sel({"energy": be}, method='nearest')
    photon_energy_iso = energy_iso['photon_energy'].values
    ke_iso = photon_energy_iso - wf - be
    num_x = find_spacing(mm, photon_energy_iso, ke_iso, slit, inner_potential, num_x, num_z, partial=True)
    kx_new = np.sort(np.linspace(min_x, max_x, num=num_x, endpoint=True))
    hv, kxx, be = np.meshgrid(photon_energy, kx_new, binding_energy,
                              indexing='ij', sparse=False)
    theta = k_reverse_normal_partial(kxx, hv, be, wf)
    points_stacked = np.stack((hv.reshape(-1, order='C'), theta.reshape(-1, order='C'),
                               be.reshape(-1, order='C')))
    interp_out = interp_object(points_stacked.T)
    interp_out = interp_out.reshape((hv.shape[0], theta.shape[1], be.shape[2]), order='C')
    return xr.DataArray(interp_out, dims=['photon_energy', 'kx', 'energy'],
                        coords={'photon_energy': photon_energy,
                                'kx': kx_new,
                                'energy': binding_energy})


def generate_fit(fp):
    """
    Generate the fit for one EDC given a specified window
    :param edc:
    :param window_min:
    :param window_max:
    :param fp: fit parameters for setting values in this function
    :return:
    """
    fermi_func = ThermalDistributionModel(prefix='fermi_', form='fermi')
    params_ = fermi_func.make_params()
    temp = float(fp[0][0])  # Temperature of map
    k = 8.617333e-5  # Boltzmann in ev/K
    params_['fermi_kt'].set(value=k * temp, vary=bool(fp[1][0]), max=float(fp[1][1]), min=float(fp[1][2]))
    params_['fermi_center'].set(value=float(fp[2][0]), max=float(fp[2][1]), vary=bool(fp[2][2]))
    params_['fermi_amplitude'].set(value=float(fp[3][0]), vary=bool(fp[3][1]))

    linear_back = LinearModel(prefix='linear_')
    params_.update(linear_back.make_params())
    params_['linear_slope'].set(value=float(fp[4][0]), max=float(fp[4][1]), vary=bool(fp[4][2]))
    params_['linear_intercept'].set(value=float(fp[5][0]), vary=bool(fp[5][1]))

    constant = ConstantModel(prefix='constant_')
    params_.update(constant.make_params())
    params_['constant_c'].set(value=float(fp[6][0]), max=float(fp[6][1]))

    full_model = linear_back * fermi_func + constant
    return full_model, params_


def ef_guess_for_edc(scan, dim_key, e0, e1, offset=0, ndown=2, scan_info=["hv_scan", 14]):
    """
    :param scan: the full 3D scan
    :param dim_key: the key to use for looping
    :param e0: bottom of the energy range to select
    :param e1: top of the energy range to select
    :param offset: amount to offset the dx range by
    :param ndown: number of points to use to downsample
    :param scan_info: a list with the type of scan and either
    the spacing or the radius
    :return:
    """
    r = None
    if scan_info[0] == "hv_scan":
        dx = scan_info[1]
    elif scan_info[0] == "fermi_map":
        r = scan_info[1]
    ef_guess = []
    for val in scan[dim_key]:
        if r:
            dx = math.sqrt((r)**2 - (val)**2)
        im = scan.sel({dim_key: val}, method='nearest')
        im_cr = im.sel({'energy': slice(e0, e1)}).sel({'slit': slice(-dx + offset, dx + offset)})
        im_cr_ds = im_cr.arpes.downsample({'energy': ndown})

        edc = im_cr_ds.sum('slit')
        ef_est = edc.arpes.guess_ef()
        ef_guess.append(ef_est)
    return ef_guess


def my_dewarp(spectra, ef_pos):
    """
    dewarping for a given spectra by passing
    in the fermi level curve
    :param spectra:
    :param ef_pos:
    :return:
    """
    #  ef_pos = dewarp(spectra.coords['slit'].values)
    ef_min = np.min(ef_pos)
    ef_max = np.max(ef_pos)
    de = spectra.coords['energy'].values[1] - spectra.coords['energy'].values[0]
    px_to_remove = int(round((ef_max - ef_min) / de))
    dewarped = np.empty((spectra.coords['slit'].size, spectra.coords['energy'].size - px_to_remove))
    for i in range(spectra.coords['slit'].size):
        rm_from_bottom = int(round((ef_pos[i] - ef_min) / de))
        rm_from_top = spectra.coords['energy'].size - (px_to_remove - rm_from_bottom)
        dewarped[i, :] = spectra.values[i, rm_from_bottom:rm_from_top]
    bottom_energy_offset = int(round((ef_max - ef_min) / de))
    energy = spectra.coords['energy'].values[bottom_energy_offset:]
    dw_data = xr.DataArray(dewarped, coords={'energy': energy, 'slit': spectra.coords['slit'].values},
                           dims=['slit', 'energy'], attrs=spectra.attrs)
    dw_new_ef = dw_data['energy'] - ef_max
    dw_data = dw_data.assign_coords({'energy': dw_new_ef})
    return dw_data


def dewarp_2d(spectrum_2d, e0, e1, x1, x2, ds, de, fermi_model, fermi_params, r2_threshold):
    """
    dewarps the spectrum fermi level for a selected photon energy by using the generated
    fermi model. First a list of ef guesses are made and then a dewarp curve is generated using polyfit
    and then sent into my_dewarp.
    :param spectrum_2d: scans.sel({'photon_energy': hv}, method='nearest')
    :param e0:
    :param e1:
    :param x1:
    :param x2:
    :param ds:
    :param de:
    :param fermi_model:
    :param fermi_params:
    :param r2_threshold:
    :return:
    """
    spectrum_2d_crop = spectrum_2d.sel({'energy': slice(e0, e1)}).sel({'slit': slice(x1, x2)})
    # is there a reason we don't send both of these in at the same time? Also, why was ds so much larger than de?
    spectrum_2d_crop_downsample_slit = spectrum_2d_crop.arpes.downsample({'slit': ds})
    spectrum_2d_crop_downsample = spectrum_2d_crop_downsample_slit.arpes.downsample({'energy': de})
    angle = spectrum_2d['slit']
    angle_downsample = spectrum_2d_crop_downsample['slit']
    ene_downsample = spectrum_2d_crop_downsample['energy']
    init_params = fermi_params
    ef_downsample = []
    ef_sigma = []
    i = 0
    for theta in angle_downsample:
        edc_xr = spectrum_2d_crop_downsample.sel({'slit': theta}, method='nearest')
        edc_vals = edc_xr.values
        fit_result = fermi_model.fit(edc_vals, fermi_params, x=ene_downsample)
        fermi_params = fit_result.params
        ef_value = fermi_params['fermi_center'].value
        ef_error = fermi_params['fermi_center'].stderr
        fit_result_points = fermi_model.eval(fermi_params, x=ene_downsample)
        r2 = 1 - (fit_result.residual.var() / np.var(fit_result_points))
        if r2 < r2_threshold:
            ef_value = np.NaN
            ef_error = np.NaN
            fermi_params = init_params

        else:
            i += 1

        ef_downsample.append(ef_value)
        ef_sigma.append(ef_error)

    if i < 5:
        print('i was less than 5')
        ef_downsample = []
        ef_sigma = []
        for theta in angle_downsample:
            edc_xr = spectrum_2d_crop_downsample.sel({'slit': theta}, method='nearest')
            ef_value = edc_xr.arpes.guess_ef()
            ef_downsample.append(ef_value)
            ef_sigma.append(1)

    aa = np.array(angle_downsample)
    ee = np.array(ef_downsample)
    clean_ef = np.isfinite(aa) & np.isfinite(ee)
    ww = 1.0 / ((np.array(ef_sigma)) ** 2)
    p = np.polyfit(aa[clean_ef], ee[clean_ef], 2, w=ww)
    dw_curve = np.poly1d(p)
    # dw_curve= arpys.Arpes.make_dewarp_curve(aa[clean_ef], ee[clean_ef])
    # modify this to add weights to the fit error bars
    ef = dw_curve(angle.values)
    spectrum_2d_dw = my_dewarp(spectrum_2d, ef)
    return (spectrum_2d_dw, angle_downsample, ef_downsample, ef,
            spectrum_2d_crop_downsample, ef_sigma)


def run_dewarp(scans, ef_guess, model, params, iter_key, offset=0, ndown=2,
               scan_info=['hv_scan', 15], threshold=95):
    """

    :param scans:
    :param ef_guess:
    :param model:
    :param params:
    :param iter_key:
    :param offset:
    :param ndown:
    :param scan_info:
    :param threshold:
    :return:

    """
    r = None
    dx = 0
    if scan_info[0] == "hv_scan":
        dx = scan_info[1]
    elif scan_info[0] == "fermi_map":
        r = scan_info[1]
    scans_dewarped = []
    i = 0
    # scans_downsample_iter_key = scans.arpes.downsample({iter_key: 5})
    for val in scans[iter_key]:
        print(val)
        if r:
            dx = math.sqrt((r)**2 - (val)**2)
        if not dx:
            print("dx was never assigned so setting the x spacing to 1")
            dx = 1
        ef_ini = ef_guess[i]
        params['fermi_center'].set(value=ef_ini, max=0.15, vary=True)
        im_2d = scans.sel({iter_key: val}, method='nearest')
        im_2d_dw = dewarp_2d(im_2d, ef_ini - 0.2, ef_ini + 0.2, -dx + offset, dx + offset,
                             50, ndown, model, params, threshold)
        scans_dewarped.append(im_2d_dw[0])
        i += 1
    dewarp_interp = [scans_dewarped[0]]
    for scan_no in np.arange(1, len(scans_dewarped)):
        dewarp_interp.append(scans_dewarped[scan_no].interp_like(scans_dewarped[0]))
    _out = xr.concat(dewarp_interp, iter_key)
    _out = _out.assign_coords({iter_key: scans[iter_key]})
    #_out = normalize_scan(_out, -10, 10, -0.3, 0.1)
    #fix_array(_out, scan_type='hv_scan')
    return _out



def run_edc_plot(edc):
    fig1, ax1 = plt.subplots()
    edc.plot(x='energy', ax=ax1)
    ax1.axvline(-5e-2, color='black')
    fig1.patch.set_facecolor('white')
    fig1.patch.set_alpha(0.95)
    plt.show()


def run_plot(cp):
    app = QApplication([])
    imtool = cp.arpes.plot()
    app.exec_()


def run_2d_plot(pl):
    figa, axa = plt.subplots()
    pl.plot(x='kx', y='kz', ax=axa, add_colorbar=False)
    axa.set_title("")
    axa.set_xlabel('Photon Energy (eV)')
    axa.set_ylabel('Binding Energy (eV)')
    figa.set_size_inches(15, 9)
    plt.show()
