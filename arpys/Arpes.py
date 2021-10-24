import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from pyimagetool import ImageTool
from multiprocessing import Process


def requires_ef(func):
    def func_wrapper(*args, **kwargs):
        if getattr(args[0], 'ef') is None:
            raise AttributeError("E_F is not defined yet.")
        return func(*args, **kwargs)

    return func_wrapper


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Source: https://stackoverflow.com/a/29042041
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def fermi_fcn_bg(x, ef, kBT, y0, *coeffs):
    """
    Fermi function with a polynomial background term.
    coeffs example:
    coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
    this notation is consistent with that provided by numpy.polyfit
    """
    p = np.array(coeffs)
    dos = (p[np.newaxis, :] * x[:, np.newaxis] ** (np.arange(len(p))[::-1])).sum(axis=1)
    return dos / (np.exp((x - ef) / kBT) + 1) + y0


def fermi_fcn_linear_bg(x, ef, kBT, y0, a, b):
    return (a + b * x) / (np.exp((x - ef) / kBT) + 1) + y0
# This is wrong
def phi2k(phi, ke):
    return 0.512 * np.sqrt(ke) * np.sin(np.radians(phi))
# The is wrong
def k2phi(k, ke):
    return np.degrees(np.arcsin(k/(0.512 * np.sqrt(ke))))


# slit angle is alpha, phi0 is offset to normal emission, theta0 is perpendicular angle (distance to normal emission)
# slit orientation values 0=vertical slit, 1=horizontal slit, 2= deflectors
# Transforms angle along the slit to k along the slit
def alpha2k(alpha, ke, theta0=0, phi0=0, slit_orientation=0):
    # Vertical Slit
    if slit_orientation == 0:
        k = -1 * 0.512 * np.sqrt(ke) * (np.sin((np.pi / 180) * (alpha + phi0)) / 2.0 + np.sin((np.pi / 180)
                                        * (alpha - phi0)) / 2.0 - np.cos((np.pi / 180) * theta0) *
                                        (np.sin((np.pi / 180) * (alpha + phi0)) / 2.0 -
                                        np.sin((np.pi / 180) * (alpha - phi0)) / 2.0))
        return k
    # Horizontal Slit
    elif slit_orientation == 1:
        k = 0.512 * np.sqrt(ke) * np.sin((np.pi / 180) * (alpha - phi0))
        return k
    # Deflectors
    elif slit_orientation == 2:
        k = 0.512 * np.sqrt(ke) * ((alpha - phi0) / (np.sqrt((alpha - phi0) ** 2 + theta0**2))) * \
             np.sin((np.pi / 180) * np.sqrt((alpha - phi0) ** 2 + theta0**2))
        return k
    else:
        print('big error my guy')


# Finds the associated ky from beta angle (angle across the slit) and alpha angle, theta0 is perpendicular
# angle offset to normal
# emission, phi0 is the offset to normal emission parallel to the slit.
# slit orientation values 0=vertical slit, 1=horizontal slit, 2=deflectors
def beta2k(beta, alpha, ke, theta0=0, phi0=0, slit_orientation=0):
    # Vertical Slit
    if slit_orientation == 0:
        k = -1 * 0.512 * np.sqrt(ke) * np.sin(np.radians(beta-theta0)) * np.cos(np.radians(alpha-phi0))
        return k
    if slit_orientation == 1:
        k = 0.512 * np.sqrt(ke) * np.sin(np.radians(beta-theta0)) * np.cos(np.radians(alpha-phi0))
        return k
    if slit_orientation == 2:
        k = -1 * 0.512 * np.sqrt(ke) * ((beta - phi0) / np.sqrt((alpha - phi0)**2 + (beta - theta0)**2)) * \
            np.sin(np.radians(np.sqrt((alpha-phi0)**2 + (beta-theta0)**2)))


# "Inverse Mapping Functions" from Ishida, Shin paper. Calculates angular sector from momentum sector.
# Calculates "kz" from kx, ky and energy. This is kz momentum of electron, not kz in the crystal (No inner potential)
# alpha is the angle along the slit, kx is momentum along the slit, ky is momentum across the slit
# phi0 is the angular offset to normal emission along the slit
def k2alpha(kx, ky, ke, phi0=0, slit_orientation=0):
    k = 0.512 * np.sqrt(ke)
    kz = np.sqrt(k**2 - kx**2 - ky**2)
    if slit_orientation == 0:
        alpha = np.degrees(np.arcsin((-np.cos(np.radians(phi0))*kx + np.sin(np.radians(phi0))*kz)/k))
        return alpha
    if slit_orientation == 1:
        alpha = np.degrees(np.arcsin((np.sin(np.radians(phi0))*np.sqrt(k**2 - kx**2) + np.cos(phi0)*kx)/k))
        return alpha
    elif slit_orientation ==2:
        alpha = (kx / np.sqrt(k**2 - kz**2)) * np.degrees(np.arccos(kz/k))
        return alpha
    else:
        print("what's the slit orientation?")


def k2beta(kx, ky, ke, phi0=0, slit_orientation=0):
    k = 0.512 * np.sqrt(ke)
    kz = np.sqrt(k**2 - kx**2 - ky**2)
    if slit_orientation == 0:
        beta = np.degrees(np.arctan((-1*ky)/(np.sin(np.radians(phi0)) * ky + np.cos(np.radians(phi0))*kz)))
        return beta
    if slit_orientation == 1:
        beta = np.degrees(np.arctan(ky/kz))
        return beta
    if slit_orientation == 2:
        beta = (-1 * ky / np.sqrt(k**2 - kz**2)) * np.degrees(np.arccos(kz/k))
        return beta
@xr.register_dataarray_accessor("arpes")
class Arpes:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.is_kinetic = True
        self.ef = None
        self.it = None

    # Because this function uses an irregularly spaced k-mesh, we can use the "forward" k-conversion
    # from angle space directly to k-space. K-space conversion depends on the orientation of the slit
    # so the user must select which orientation the experiment was using. This could in practice be automated
    # by setting an attribute in the associated beamline loader, but beamlines like MAESTRO can use both slit
    # orientations, so it's up to the user to set correctly. This version also drops support for azimuth's that
    # are misaligned. slit_orientation values 0=vertical slit, 1=horizontal slit, 2=deflectors
    @requires_ef
    def map_isoenergy_k_irreg(self, ke=None, be=None, binwidth=1, phi0=0, theta0=0, slit_orientation=0):
        if ke is None:
            ke = be + self.ef
        iso_e = self._obj.arpes.sel_kinetic(ke - binwidth, ke + binwidth).sum('energy')
        alpha, T = np.meshgrid(iso_e.arpes.slit, iso_e.arpes.perp, indexing='ij')

        # Vertical Slit K Conversion, kx along slit
        if slit_orientation == 0:
            kx = -1 * 0.512 * np.sqrt(ke) * (np.sin((np.pi / 180)*(alpha + phi0))/2.0 + np.sin((np.pi / 180)
                                             * (alpha - phi0))/2.0 -
                                             np.cos((np.pi / 180) * (T - theta0)) *
                                             (np.sin((np.pi / 180) * (alpha + phi0))/2.0 -
                                             np.sin((np.pi / 180) * (alpha - phi0))/2.0))

            ky = -1 * 0.512 * np.sqrt(ke) * np.cos(np.pi / 180 * (alpha-phi0)) * np.sin(np.pi / 180 * (T - theta0))
            iso_e = iso_e.assign_coords({'kx': (('slit', 'perp'), kx), 'ky': (('slit', 'perp'), ky)})
        # Horizontal Slit K Conversion, kx along slit
        elif slit_orientation == 1:
            kx = 0.512 * np.sqrt(ke) * np.sin((np.pi / 180) * (alpha - phi0))
            ky = 0.512 * np.sqrt(ke) * np.sin((np.pi / 180) * (T - theta0)) * np.cos((np.pi / 180) * (alpha - phi0))
            iso_e = iso_e.assign_coords({'kx': (('slit', 'perp'), kx), 'ky': (('slit', 'perp'), ky)})
        # Deflector K Conversion, kx along slit, orientation of slit shouldn't matter... (overall minus sign difference
        # in kx, this just changes the direction of k, doesn't mess with magnitudes)
        elif slit_orientation == 2:
            kx = 0.512 * np.sqrt(ke) * ((alpha - phi0)/(np.sqrt((alpha-phi0)**2 + (T-theta0)**2))) * \
                 np.sin((np.pi / 180) * np.sqrt((alpha-phi0)**2 + (T-theta0)**2))
            ky = -1 * 0.512 * np.sqrt(ke) * ((T - theta0)/(np.sqrt((alpha - phi0)**2 + (T - theta0)**2)))\
                 * np.sin((np.pi / 180) * np.sqrt((alpha - phi0)**2 + (T - theta0)**2))

            iso_e = iso_e.assign_coords({'kx': (('slit', 'perp'), kx), 'ky': (('slit', 'perp'), ky)})

        return iso_e

    # Uses an irregularly spaced k-mesh, very quick, good for just plotting.
    # If you want a regularly spaced grid, use spectra_k_reg() to get a regularly spaced rectilinear grid
    # in be vs. k (This is slower)
    # Uses forward k-conversion due to irregularly spaced k-mesh, needs theta_offset  for vertical slit which is
    # assumed to be the offset perpendicular to the slit from normal emission.
    # slit_orientation values 0=vertical slit, 1=horizontal slit, 2=deflector(single cuts)
    @requires_ef
    def spectra_k_irreg(self, phi0=0, theta_offset=0, slit_orientation=0):
        KE, alpha = np.meshgrid(self._obj.arpes.energy, self._obj.arpes.slit, indexing='ij')

        # Vertical Slit
        if slit_orientation == 0:
            kx = -1 * 0.512 * np.sqrt(KE) * (np.sin((np.pi / 180)*(alpha + phi0))/2.0 + np.sin((np.pi / 180)
                                             * (alpha - phi0))/2.0 -
                                             np.cos((np.pi / 180) * theta_offset) *
                                             (np.sin((np.pi / 180) * (alpha + phi0))/2.0 -
                                             np.sin((np.pi / 180) * (alpha - phi0))/2.0))
            self._obj = self._obj.assign_coords(
                {'kx': (('energy', 'slit'), kx), 'binding': (('energy', 'slit'), KE - self.ef)})
        # Horizontal Slit
        elif slit_orientation == 1:
            kx = 0.512 * np.sqrt(KE) * np.sin((np.pi / 180) * (alpha - phi0))
            self._obj = self._obj.assign_coords(
                {'kx': (('energy', 'slit'), kx), 'binding': (('energy', 'slit'), KE - self.ef)})
        # Deflector Cut
        elif slit_orientation == 2:
            kx = 0.512 * np.sqrt(KE) * ((alpha-phi0)/(np.sqrt((alpha-phi0)**2 + theta_offset**2))) * np.sin(
                (np.pi / 180)*(np.sqrt((alpha-phi0)**2 + theta_offset**2)))
            self._obj = self._obj.assign_coords(
                {'kx': (('energy', 'slit'), kx), 'binding': (('energy', 'slit'), KE - self.ef)})

        return self._obj

    # This is much slower and relies on scipy RegularGridInterpolator and utilizes linear interpolation
    # to produce the rectilinear energy vs. k grid. This may not be strictly mathematically exact in preservation
    # of spectral weight through the transformation. This is roughly equivalent to "TransformToK" in our Igor code
    # Also assumes that the xarray has been constructed with dimensions energy vs. slit not slit vs. energy
    @requires_ef
    def spectra_k_reg(self, phi0):
        copy = self._obj.copy()
        interp_object = RegularGridInterpolator((copy.energy.values, copy.slit.values),
                                                copy.values, bounds_error=False, fill_value=0)
        lowk = phi2k(np.nanmin(copy.slit.values) - phi0, np.nanmax(copy.energy.values))
        highk = phi2k(np.nanmax(copy.slit.values) - phi0, np.nanmax(copy.energy.values))
        numk = copy.slit.size

        lowe = np.nanmin(copy.energy.values)
        highe = np.nanmax(copy.energy.values)
        nume = copy.energy.size

        kx = np.linspace(lowk, highk, num=numk, endpoint=True)
        ke = np.linspace(lowe, highe, num=nume, endpoint=True)
        be = ke - self.ef

        output = np.empty(copy.shape)

        i = 0
        j = 0
        for energy in ke:
            for k in kx:
                counts = interp_object([energy, k2phi(k + phi2k(phi0, energy), energy)])
                output[i, j] = counts
                j += 1
            j = 0
            i += 1
        return xr.DataArray(output,dims=['binding','kx'],coords={'binding':be,'kx':kx},attrs=copy.attrs)

    # This thing is monstrous, please put me out of my misery
    @requires_ef
    def spectra_k_reg_2(self, phi0=0, theta0=0, slit_orientation=0):
        copy = self._obj.copy()

        # Embedding the cut into a higher dimensional space (3d for 2d spectra)
        # Padding spectra with Theta+-0.1
        empty_space = np.zeros(copy.values.shape)
        empty_xr = xr.DataArray(empty_space, dims=copy.dims, coords=copy.coords, attrs=copy.attrs)
        higher_dimensioned_spectra = xr.concat([empty_xr,copy,empty_xr], 'theta')
        higher_dim_thetas = [theta0-0.1, theta0, theta0+0.1]
        higher_dimensioned_spectra = higher_dimensioned_spectra.transpose("energy", "slit", "theta")
        interp_object = RegularGridInterpolator((higher_dimensioned_spectra.energy.values,
                                                 higher_dimensioned_spectra.slit.values, higher_dim_thetas),
                                                higher_dimensioned_spectra.values, bounds_error=False, fill_value=0)
        lowkx = alpha2k(np.nanmin(higher_dimensioned_spectra.slit.values),
                        np.nanmax(higher_dimensioned_spectra.energy.values), theta0=theta0, phi0=phi0,
                        slit_orientation=slit_orientation)

        highkx = alpha2k(np.nanmax(higher_dimensioned_spectra.slit.values),
                         np.nanmax(higher_dimensioned_spectra.energy.values),
                         theta0=theta0, phi0=phi0, slit_orientation=slit_orientation)
        numkx = higher_dimensioned_spectra.slit.size
        lowky = beta2k(theta0, np.nanmax(np.abs(higher_dimensioned_spectra.slit.values)),
                       np.nanmax(higher_dimensioned_spectra.energy.values),
                       theta0=0, phi0=phi0, slit_orientation=slit_orientation)
        highky = beta2k(theta0, np.nanmin(np.abs(higher_dimensioned_spectra.slit.values)),
                        np.nanmax(higher_dimensioned_spectra.energy.values),
                        theta0=0, phi0=phi0, slit_orientation=slit_orientation)
        # What even is this
        numky = 10

        lowe = np.nanmin(higher_dimensioned_spectra.energy.values)
        highe = np.nanmax(higher_dimensioned_spectra.energy.values)
        nume = higher_dimensioned_spectra.energy.size

        energy_array = np.linspace(lowe, highe, num=nume, endpoint=True)
        binding_array = energy_array - self.ef
        kx_array = np.linspace(lowkx, highkx, num=numkx, endpoint=True)
        ky_array = np.linspace(lowky, highky, num=numky, endpoint=True)

        # DO IT! JUST DO IT! DONT LET YOUR DREAMS BE DREAMS!
        print("We doin this?")
        interpolation_points = []
        for energy in energy_array:
            for kx in kx_array:
                for ky in ky_array:
                    interpolation_points.append([energy, k2alpha(kx, ky, energy, phi0=phi0,
                                                                 slit_orientation=slit_orientation),
                                                 k2beta(kx, ky, energy, phi0=phi0, slit_orientation=slit_orientation)])

        interpolation_points = np.array(interpolation_points)
        print(interpolation_points.shape)
        print("points array generated, now callling interpolation object")
        flat_interpolated = interp_object(interpolation_points)
        print("interpolation finished, now reshaping etc.")
        reshaped_interpolated = np.reshape(flat_interpolated,
                                           (energy_array.size, kx_array.size, ky_array.size), order='C')
        print(reshaped_interpolated.shape)
        # regridded_higherdimension = np.empty((energy_array.size, kx_array.size, ky_array.size))
        # i = 0
        # j = 0
        # k = 0
        # for energy in energy_array:
        #     for kx in kx_array:
        #         for ky in ky_array:
        #             regridded_higherdimension[i, j, k] = interp_object([energy, k2alpha(kx, ky, energy,
        #                                                                                 phi0=phi0,
        #                                                                 slit_orientation=slit_orientation),
        #                                                                 k2beta(kx, ky, energy, phi0=phi0,
        #                                                                        slit_orientation=slit_orientation)])
        #             k+=1
        #         j+=1
        #         k=0
        #     i+=1
        #     j=0

        high_dim_xr = xr.DataArray(reshaped_interpolated, dims=['binding', 'kx', 'ky'],
                                   coords={'binding': binding_array, 'kx': kx_array, 'ky': ky_array}, attrs=copy.attrs)
        flattened = high_dim_xr.sum('ky')
        return flattened

    # Kz maps should always be in binding energy, will need to shift off using a fixed work-function to recover
    # kinetic energy for k conversion
    def map_isoenergy_kz_k_irreg(self, be=0, workfunc=4.2, binwidth=0.1, phi0=0, inner_potential=15):
        iso_e = self._obj.arpes.sel_kinetic(be-binwidth, be+binwidth).sum('energy')
        T, F = np.meshgrid(iso_e.arpes.photon_energy, iso_e.arpes.slit, indexing='ij')
        kx = 0.512 * np.sqrt(be + T - workfunc) * np.sin(np.pi / 180 * (F - phi0))
        kz = 0.512 * np.sqrt((be + T - workfunc)*np.cos(np.pi/180 * (F-phi0))**2 + inner_potential)

        iso_e = iso_e.assign_coords({'kx':(('photon_energy','slit'), kx), 'kz':(('photon_energy', 'slit'), kz)})
        return iso_e

    # Currently can only take spectra at gamma point (in angle space) before k-converting
    def spectra_kz_k_irreg(self, binwidth=1, workfunc=4.2, phi0=0, inner_potential=15):
        cut = self._obj.arpes.sel_slit(phi0-binwidth, phi0+binwidth).sum('slit')
        F, BE = np.meshgrid(self._obj.arpes.photon_energy, self._obj.arpes.energy, indexing='ij')
        kz = 0.512 * np.sqrt((F + BE - workfunc) + inner_potential)

        cut = cut.assign_coords({'kz': (('photon_energy', 'energy'), kz)})
        return cut

    @requires_ef
    def cut_from_map_slit_k_irreg(self, theta0=0, phi0=0):
        cut = self._obj.sel({'perp':theta0}, method='nearest')
        KE, F = np.meshgrid(cut.arpes.energy, cut.arpes.slit, indexing='ij')
        kx = 0.512 * np.sqrt(KE) * np.sin(np.pi / 180 * (F - phi0))
        cut = cut.assign_coords({'kx': (('energy','slit'), kx), 'binding': (('energy','slit'), KE - self.ef)})
        return cut

    @requires_ef
    def cut_from_map_perp_k_irreg(self, theta0=0, phi0=0):
        cut = self._obj.sel({'slit': phi0}, method='nearest')
        KE, F = np.meshgrid(cut.arpes.energy, cut.arpes.perp, indexing='ij')
        kx = 0.512 * np.sqrt(KE) * np.sin(np.pi / 180 * (F - theta0))
        cut = cut.assign_coords({'kx': (('energy', 'perp'), kx), 'binding': (('energy', 'perp'), KE - self.ef)})
        return cut

    def normalize(self):
        self._obj.values = (self._obj.values - np.min(self._obj.values)) / (
                    np.max(self._obj.values) - np.min(self._obj.values))

    def set_gamma(self, slit=None, perp=None):
        if slit is not None:
            self._obj.coords['slit'].values -= slit
        if perp is not None:
            self._obj.coords['perp'].values -= perp

    def guess_high_symmetry(self):
        x = self._obj.values
        x_r = np.flip(x)
        cor = np.correlate(x, x_r, mode='full')
        idx = (np.argmax(cor) + 1) / 2
        return np.interp(idx, np.arange(self._obj.coords['slit'].size), self._obj.coords['slit'].values)

    def plot_spectra(self, kspace=False):
        if kspace:
            self._obj.plot(x='kx', y='binding')
        else:
            self._obj.plot(x='slit', y='energy')

    def plot_map(self, kspace=False):
        if self._obj.ndim == 2:
            if 'slit' in self._obj.coords and 'perp' in self._obj.coords:
                if kspace:
                    if 'kx' not in self._obj.coords or 'ky' not in self._obj.coords:
                        raise ValueError('You have not computed kspace yet!')
                    self._obj.plot(x='kx', y='ky')
                else:
                    self._obj.plot(x='slit', y='perp')
        else:
            raise ValueError('Your data is more than two dimensions. Cannot plot a map.')

    @requires_ef
    def waterfall(self, spacing=0.2, linear=True, kspace=False, fig=None, ax=None, **lineparams):
        if fig is None and ax is None:
            fig, ax = plt.subplots(1)
        edc_range = (np.max(self._obj, axis=self._obj.dims.index('slit')).values -
                     np.min(self._obj, axis=self._obj.dims.index('slit')).values)
        if linear:
            spacing = spacing * np.max(edc_range)
        offset = 0
        for i in range(self._obj['energy'].size):
            mdc = self._obj.isel({'energy': i}).values + offset
            if kspace:
                ax.plot(self._obj['kx'].isel({'energy': i}).values, mdc, **lineparams)
            else:
                ax.plot(self._obj.coords['slit'].values, mdc, **lineparams)
            if linear:
                offset += spacing
            else:
                offset += spacing * edc_range[i]
        return fig, ax

    @property
    def kinetic(self):
        return self._obj.coords['energy'].values

    @requires_ef
    @property
    def binding(self):
        return self._obj.coords['energy'].values - self.ef

    @property
    def energy(self):
        return self._obj.coords['energy'].values

    @property
    def slit(self):
        return self._obj.coords['slit'].values

    @property
    def perp(self):
        return self._obj.coords['perp'].values

    @property
    def photon_energy(self):
        return self._obj.coords['photon_energy'].values

    @requires_ef
    def sel_binding(self, *args):
        if len(args) == 2:
            return self._obj.sel({'energy': slice(args[0] + self.ef, args[1] + self.ef)})
        else:
            raise ValueError('binding only accepts min and max')

    def sel_kinetic(self, *args):
        if len(args) == 2:
            return self._obj.sel({'energy': slice(args[0], args[1])})
        else:
            return self._obj.sel({'energy': args[0]})

    def sel_slit(self, *args):
        if len(args) > 1:
            return self._obj.sel({'slit': slice(args[0], args[1])})
        else:
            return self._obj.sel({'slit': args[0]})

    def sel_perp(self, *args):
        if len(args) > 1:
            return self._obj.sel({'perp': slice(args[0], args[1])})
        else:
            return self._obj.sel({'perp': args[0]})

    def guess_ef(self):
        edc = self._obj.copy()
        if edc.ndim > 1:
            for dim_label in edc.dims:
                if dim_label != 'energy':
                    edc = edc.sum(dim_label)
        if edc.size > 150:
            factor = int(np.ceil(edc.size / 100))
            edc = edc.arpes.downsample({'energy': factor})
        edc_y = edc.values
        edc_x = edc.coords['energy'].values
        return edc_x[np.argmin(np.diff(edc_y))]

    @staticmethod
    def guess_fermi_params(edc, bg_order=1):
        """
        downsamples the data so that it contains at most 150 points
        estimate Ef by minimizing the first derivative
        fit the data below Ef to a polynomial of order bg_order
        guess that the resolution is 4.4 meV (TODO: make this more robust by finding width of peak in derivative)
        estimate y0 by suggesting the average of the final few values
        """
        if edc.size > 150:
            factor = int(np.ceil(edc.size / 100))
            edc = edc.arpes.downsample({'energy': factor})
        edc_y = edc.values
        edc_x = edc.coords['energy'].values
        i = np.argmin(np.diff(edc_y))
        ef = edc_x[i]
        p = np.polyfit(edc_x[:(i - 5)], edc_y[:(i - 5)], bg_order)
        kBT = 0.001
        y0 = np.mean(edc_y[-5:])
        return (ef, kBT, y0) + tuple(p)

    @staticmethod
    def fermi_fit(edc, bg_order=1):
        return curve_fit(fermi_fcn_bg, edc.arpes.energy, edc.values, p0=Arpes.guess_fermi_params(edc, bg_order))

    @staticmethod
    def print_fit_params(popt):
        formatter = 'Ef: {0} (eV)\n10-90: {1} (meV)\ny0: {2}\n'
        for i in range(3, len(popt)):
            formatter += 'bg x**{0}: '.format(len(popt) - i - 1)
            formatter += '{' + str(i) + '}\n'
        vals = list(popt)
        vals[1] *= 4400
        print(formatter.format(*vals))

    @staticmethod
    def dewarp_curve(p, x):
        if len(p) == 1:
            return p[0]
        elif len(p) == 2:
            return p[0] * x + p[1]
        elif len(p) == 3:
            return p[0] * x ** 2 + p[1] * x + p[2]
        elif len(p) == 4:
            return p[0] * x ** 3 + p[1] * x ** 2 + p[2] * x + p[3]

    @staticmethod
    def make_dewarp_curve(angles, ef):
        p = np.polyfit(angles, ef, deg=2)
        return partial(Arpes.dewarp_curve, p)

    @staticmethod
    def dewarp_spectra(spectra, dewarp):
        ef_pos = dewarp(spectra.coords['slit'].values)
        ef_min = np.min(ef_pos)
        ef_max = np.max(ef_pos)
        de = spectra.coords['energy'].values[1] - spectra.coords['energy'].values[0]
        px_to_remove = int(round((ef_max - ef_min) / de))
        dewarped = np.empty((spectra.coords['energy'].size - px_to_remove, spectra.coords['slit'].size))
        for i in range(spectra.coords['slit'].size):
            rm_from_bottom = int(round((ef_pos[i] - ef_min) / de))
            rm_from_top = spectra.coords['energy'].size - (px_to_remove - rm_from_bottom)
            dewarped[:, i] = spectra.values[rm_from_bottom:rm_from_top, i]
        bottom_energy_offset = int(round((ef_max - ef_min) / de))
        energy = spectra.coords['energy'].values[bottom_energy_offset:]
        return xr.DataArray(dewarped, coords={'energy': energy, 'slit': spectra.coords['slit'].values},
                            dims=['energy', 'slit'], attrs=spectra.attrs)

    @staticmethod
    def create_dewarp(spectra):
        N = spectra.coords['slit'].size
        slit_angles = spectra.coords['slit'].values
        ef_pos = np.empty(N)
        for i in range(N):
            edc = spectra.isel({'slit': i}).values
            energy = spectra.coords['energy'].values
            params = Arpes.guess_fermi_params(spectra.isel({'slit': i}), 2)
            params, pcov = curve_fit(fermi_fcn_bg, energy, edc, p0=params)
            ef_pos[i] = params[0]
        dewarp = Arpes.make_dewarp_curve(slit_angles, ef_pos)
        return dewarp

    def downsample(self, dsf, operation='mean'):
        """
        dsf is a dict of dims and the amount to downsample by (i.e. 2, 3, 4, etc)
        operation can be sum or mean
        downsampled coordinates are always averaged
        """
        dat = self._obj.copy()
        new_shape = list(dat.shape)
        new_coords = {key: dat[key].values for key in dat.dims}
        dat_mat = dat.values
        for label, ds in dsf.items():
            i = dat.dims.index(label)
            extra = new_shape[i] % ds
            if extra == 0:
                extra = None
            else:
                extra *= -1
            new_shape[i] = new_shape[i] // ds
            new_coords[label] = new_coords[label][:extra].reshape((-1, ds)).mean(axis=1)
            slicer = [slice(None, None)] * dat_mat.ndim
            slicer[i] = slice(None, extra)
            dat_mat = dat_mat[tuple(slicer)]
        mat = bin_ndarray(dat_mat, new_shape, operation=operation)
        return xr.DataArray(mat, coords=new_coords, dims=dat.dims, attrs=dat.attrs)

    def plot(self, layout=ImageTool.LayoutComplete):
        self.it = ImageTool(self._obj, layout=layout)
        self.it.show()
        return self.it
