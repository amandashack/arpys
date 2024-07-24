import numpy as np
import xarray as xr


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
        return xr.DataArray(np.transpose(xar.data, tr), dims=dims, coords=coords, attrs = xar.attrs)
    if scan_type == 'hv_scan':
        photon_energy = ar.photon_energy.values
        slit = ar.slit.values
        energy = ar.energy.values
        size_new = [len(photon_energy), len(slit), len(energy)]
        size_ar = list(ar.values.shape)
        tr = [size_ar.index(i) for i in size_new]
        return np_transpose(ar, tr)
    if scan_type == 'fermi_map':
        slit = ar.slit.values
        perp = ar.perp.values
        energy = ar.energy.values
        size_new = [len(slit), len(perp), len(energy)]
        size_ar = list(ar.values.shape)
        tr = [size_ar.index(i) for i in size_new]
        return np_transpose(ar, tr)
    if scan_type == 'fermi_map_k':
        kx = ar.kx.values
        ky = ar.ky.values
        binding = ar.binding.values
        size_new = [len(kx), len(ky), len(binding)]
        size_ar = list(ar.values.shape)
        tr = [size_ar.index(i) for i in size_new]
        return np_transpose(ar, tr)
    if scan_type == 'single':
        slit = ar.slit.values
        energy = ar.energy.values
        size_new = [len(slit), len(energy)]
        size_ar = list(ar.values.shape)
        tr = [size_ar.index(i) for i in size_new]
        return np_transpose(ar, tr)


def ke_to_be(hv, spectra, wf=4.5):
    wf = wf
    ke = np.array(spectra.energy)
    be = ke + wf - hv
    reassigned = spectra.assign_coords({'energy': be})
    return reassigned


def be_to_ke(hv, spectra, wf=4.2):
    be = np.array(spectra.energy)
    ke = hv - wf + be
    reassigned = spectra.assign_coords({'energy': ke})
    return reassigned

