from scipy.interpolate import RegularGridInterpolator
import numpy as np
import arpys
import xarray as xr
import time


class SpectraConverter:
    def __init__(self, obj, hv, wf, ef=None):
        self._obj = obj
        self.hv = hv  # Photon energy
        self.wf = wf  # Work function
        if not ef:
            self.ef = self.hv - self.wf
        else:
            self.ef = ef

    def spectra_k_reg(self, phi0=0):
        copy = self._obj.copy()
        dims = copy.dims
        interp_object = RegularGridInterpolator(
            (copy[dims[0]].values, copy[dims[1]].values),
            copy.values,
            bounds_error=False,
            fill_value=0
        )

        lowk = arpys.phi2k(np.nanmin(copy[dims[0]].values) - phi0, np.nanmax(copy[dims[1]].values))
        highk = arpys.phi2k(np.nanmax(copy[dims[0]].values) - phi0, np.nanmax(copy[dims[1]].values))
        numk = copy[dims[0]].size

        lowe = np.nanmin(copy[dims[1]].values)
        highe = np.nanmax(copy[dims[1]].values)
        nume = copy[dims[1]].size

        kx = np.linspace(lowk, highk, num=numk, endpoint=True)
        ke = np.linspace(lowe, highe, num=nume, endpoint=True)
        be = ke - self.ef

        kx_mesh, ke_mesh = np.meshgrid(kx, ke, indexing='ij')
        phi_mesh = arpys.k2phi(kx_mesh + arpys.phi2k(phi0, ke_mesh), ke_mesh)

        coords = np.stack([ke_mesh.ravel(), phi_mesh.ravel()], axis=-1)
        interpolated_values = interp_object(coords).reshape(numk, nume)

        return xr.DataArray(interpolated_values, dims=['kx', 'binding'],
                            coords={'kx': kx, 'binding': be}, attrs=copy.attrs)

    def map_k_reg(self, phi0=0, theta0=0, azimuth=0, slit_orientation=0):
        copy = self._obj.copy()
        dims = copy.dims  # ('slit', 'perp', 'energy')

        interp_object = RegularGridInterpolator(
            (copy[dims[0]].values, copy[dims[1]].values, copy[dims[2]].values),
            copy.values,
            bounds_error=False,
            fill_value=0
        )

        max_energy = np.nanmax(copy[dims[2]].values)
        slit_vals = copy[dims[0]].values
        perp_vals = copy[dims[1]].values

        kxmin, _ = SpectraConverter.forward_k_conversion(max_energy, np.nanmin(slit_vals), 0, phi0=phi0, theta0=theta0,
                                              azimuth=azimuth, slit_orientation=slit_orientation)
        kxmax, _ = SpectraConverter.forward_k_conversion(max_energy, np.nanmax(slit_vals), 0, phi0=phi0, theta0=theta0,
                                              azimuth=azimuth, slit_orientation=slit_orientation)
        _, kymin = SpectraConverter.forward_k_conversion(max_energy, 0, np.nanmin(perp_vals), phi0=phi0, theta0=theta0,
                                              azimuth=azimuth, slit_orientation=slit_orientation)
        _, kymax = SpectraConverter.forward_k_conversion(max_energy, 0, np.nanmax(perp_vals), phi0=phi0, theta0=theta0,
                                              azimuth=azimuth, slit_orientation=slit_orientation)

        kx_new = np.sort(np.linspace(kxmin, kxmax, num=copy[dims[0]].size, endpoint=True))
        ky_new = np.sort(np.linspace(kymin, kymax, num=copy[dims[1]].size, endpoint=True))
        energy_new = np.linspace(np.nanmin(copy[dims[2]].values), np.nanmax(copy[dims[2]].values),
                                 num=copy[dims[2]].size,endpoint=True)

        kxx, kyy, energy_grid = np.meshgrid(kx_new, ky_new, energy_new, indexing='ij', sparse=False)

        t0 = time.time()
        alpha, beta, energy = SpectraConverter.reverse_k_conversion(energy_grid, kxx, kyy, phi0=phi0, theta0=theta0,
                                                         azimuth=azimuth, slit_orientation=slit_orientation)
        t1 = time.time()
        print("Generation of interpolation points time elapsed = {:.3f}s".format(t1 - t0))

        points_stacked = np.stack((alpha.ravel(order='C'), beta.ravel(order='C'), energy.ravel(order='C')), axis=-1)

        t2 = time.time()
        interpolation_output = interp_object(points_stacked)
        t3 = time.time()
        print("Interpolation time elapsed = {:.3f}s".format(t3 - t2))

        interpolation_reshaped = interpolation_output.reshape((kx_new.size, ky_new.size, energy_new.size), order='C')
        return xr.DataArray(interpolation_reshaped, dims=['kx', 'ky', 'binding'],
                            coords={'kx': kx_new, 'ky': ky_new, 'binding': energy_new - self.ef}, attrs=copy.attrs)


    @staticmethod
    def forward_k_conversion(ke, alpha, beta, phi0=0, theta0=0, azimuth=0, slit_orientation=0):
        # Vertical Slit K Conversion, kx along slit
        if slit_orientation == 0:
            kx = 0.512 * np.sqrt(ke) * ((np.sin(np.radians(azimuth))*np.sin(np.radians(beta-theta0)) +
                                         np.cos(np.radians(azimuth))*np.sin(np.radians(phi0)) *
                                         np.cos(np.radians(beta-theta0)))*np.cos(np.radians(alpha)) -
                                        np.cos(np.radians(azimuth))*np.cos(np.radians(phi0)) *
                                        np.sin(np.radians(alpha)))

            ky = 0.512 * np.sqrt(ke) * ((-1*np.cos(np.radians(azimuth)) * np.sin(np.radians(beta-theta0)) +
                                         np.sin(np.radians(azimuth))*np.sin(np.radians(phi0)) *
                                         np.cos(np.radians(beta-theta0)))*np.cos(np.radians(alpha)) -
                                        np.sin(np.radians(azimuth))*np.cos(np.radians(phi0)) *
                                        np.sin(np.radians(alpha)))
        # Horizontal Slit K Conversion, kx along slit (swap kx and ky from Ishida paper)
        elif slit_orientation == 1:
            kx = 0.512 * np.sqrt(ke) * ((-1*np.cos(np.radians(azimuth))*np.sin(np.radians(phi0)) +
                                         np.sin(np.radians(azimuth))*np.sin(np.radians(beta-theta0)) *
                                         np.cos(np.radians(phi0))) * np.cos(np.radians(alpha)) +
                                        (np.cos(np.radians(azimuth)) * np.cos(np.radians(phi0)) +
                                         np.sin(np.radians(azimuth)) * np.sin(np.radians(beta-theta0)) *
                                         np.sin(np.radians(phi0))) * np.sin(np.radians(alpha)))
            ky = 0.512 * np.sqrt(ke) * ((np.sin(np.radians(azimuth)) * np.sin(np.radians(phi0)) +
                                         np.cos(np.radians(azimuth))*np.sin(np.radians(beta-theta0)) *
                                         np.cos(np.radians(phi0))) * np.cos(np.radians(alpha)) -
                                        (np.sin(np.radians(azimuth)) * np.cos(np.radians(phi0)) -
                                         np.cos(np.radians(azimuth)) * np.sin(np.radians(beta-theta0)) *
                                         np.sin(np.radians(phi0))) * np.sin(np.radians(alpha)))
        # Vertical Slit Deflector K Conversion, kx along slit
        elif slit_orientation == 2:
            trot = SpectraConverter.calc_rotation_matrix(phi0=phi0, theta0=theta0, azimuth=0)
            eta = np.sqrt(np.radians(alpha)**2 + np.radians(beta)**2)

            kx = 0.512 * np.sqrt(ke) * ((-1 * np.radians(alpha) * trot[0,0] * np.sinc(eta / np.pi)) +
                                        (-1 * np.radians(beta) * trot[0,1] * np.sinc(eta / np.pi)) +
                                        trot[0,2] * np.cos(eta))
            ky = 0.512 * np.sqrt(ke) * ((-1 * np.radians(alpha) * trot[1,0] * np.sinc(eta / np.pi)) +
                                        (-1 * np.radians(beta) * trot[1,1] * np.sinc(eta / np.pi)) +
                                        trot[1,2] * np.cos(eta))

        # Horizontal Slit Deflector K Conversion
        elif slit_orientation == 3:
            trot = SpectraConverter.calc_rotation_matrix(phi0=phi0, theta0=theta0, azimuth=0)
            eta = np.sqrt(np.radians(alpha) ** 2 + np.radians(beta) ** 2)
            kx = 0.512 * np.sqrt(ke) * ((-1 * np.radians(beta) * trot[1,0] * np.sinc(eta / np.pi)) +
                                        (np.radians(alpha) * trot[1,1] * np.sinc(eta / np.pi)) +
                                        (trot[1,2] * np.cos(eta)))

            ky = 0.512 * np.sqrt(ke) * (-1 * np.radians(beta) * trot[0,0] * np.sinc(eta / np.pi) +
                                        (np.radians(alpha) * trot[0,1] * np.sinc(eta / np.pi)) +
                                        (trot[0,2] * np.cos(eta)))
        else:
            kx = None
            ky = None
            print('Slit Orientation not set correctly')

        return kx, ky

    @staticmethod
    def reverse_k_conversion(ke, kx, ky, phi0=0, theta0=0, azimuth=0, slit_orientation=0):
        k = 0.512 * np.sqrt(ke)
        kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2)

        # Vertical Slit
        if slit_orientation == 0:
            alpha = np.degrees(np.arcsin((np.sin(np.radians(phi0)) * kz - (np.cos(np.radians(phi0)) *
                                                                           (np.cos(np.radians(azimuth)) * kx +
                                                                            np.sin(np.radians(azimuth)) * ky))) / k))
            beta = theta0 + np.degrees(np.arctan((np.sin(np.radians(azimuth)) * kx - np.cos(np.radians(azimuth)) * ky) /
                                                 (np.sin(np.radians(phi0)) * np.cos(np.radians(azimuth)) * kx +
                                                  np.sin(np.radians(phi0)) * np.sin(np.radians(azimuth)) * ky +
                                                  np.cos(np.radians(phi0)) * kz)))
        # Horizontal Slit - I've switched kx and ky from the Ishida paper to keep kx along the slit,
        # and ky across the slit
        elif slit_orientation == 1:
            alpha = np.degrees(
                np.arcsin((np.sin(np.radians(phi0)) * np.sqrt(k ** 2 - (np.sin(np.radians(azimuth)) * ky -
                                                                        np.cos(np.radians(azimuth)) * kx) ** 2) -
                           np.cos(np.radians(phi0)) * (np.sin(np.radians(azimuth)) * ky -
                                                       np.cos(np.radians(azimuth)) * kx)) / k))
            beta = theta0 + np.degrees(np.arctan((np.cos(np.radians(azimuth)) * ky + np.sin(np.radians(azimuth)) * kx)
                                                 / kz))
        # Vertical Slit Deflectors
        elif slit_orientation == 2:
            # trot_inv = Arpes.calc_inverse_rotation_matrix(phi0=phi0, theta0=theta0, azimuth=azimuth)
            trot = SpectraConverter.calc_rotation_matrix(phi0=phi0, theta0=theta0, azimuth=azimuth)
            trot_inv = trot.T
            kx_bar = trot_inv[0, 0] * kx + trot_inv[0, 1] * ky + trot_inv[0, 2] * kz
            ky_bar = trot_inv[1, 0] * kx + trot_inv[1, 1] * ky + trot_inv[1, 2] * kz
            kz_bar = trot_inv[2, 0] * kx + trot_inv[2, 1] * ky + trot_inv[2, 2] * kz

            alpha = np.degrees(((-1 * kx_bar) / np.sqrt(k ** 2 - kz_bar ** 2)) * np.arccos(kz_bar / k))
            beta = np.degrees(((-1 * ky_bar) / np.sqrt(k ** 2 - kz_bar ** 2)) * np.arccos(kz_bar / k))



        # Horizontal Slit Deflectors switching ky -> kx from the Ishida paper to keep kx along the slit
        elif slit_orientation == 3:
            trot_inv = SpectraConverter.calc_inverse_rotation_matrix(phi0=phi0, theta0=theta0, azimuth=azimuth)
            alpha = np.degrees(np.arccos((trot_inv[2, 0] * ky + trot_inv[2, 1] * kx + trot_inv[2, 2] * kz) / k) *
                               ((trot_inv[1, 0] * ky + trot_inv[1, 1] * kx + trot_inv[1, 2] * kz) /
                                np.sqrt(
                                    k ** 2 - (trot_inv[2, 0] * ky + trot_inv[2, 1] * kx + trot_inv[2, 2] * kz) ** 2)))
            beta = -1 * np.degrees(np.arccos((trot_inv[2, 0] * ky + trot_inv[2, 1] * kx + trot_inv[2, 2] * kz) / k) *
                                   ((trot_inv[0, 0] * ky + trot_inv[0, 1] * kx + trot_inv[0, 2] * kz) /
                                    np.sqrt(k ** 2 - (
                                                trot_inv[2, 0] * ky + trot_inv[2, 1] * kx + trot_inv[2, 2] * kz) ** 2)))
        else:
            alpha = None
            beta = None
            print('slit_orientation not set properly')

        return alpha, beta, ke,

    @staticmethod
    def calc_rotation_matrix(phi0=0, theta0=0, azimuth=0):
        trot = np.array([[np.cos(np.radians(azimuth)) * np.cos(np.radians(phi0)), np.cos(np.radians(azimuth)) *
                          np.sin(np.radians(theta0)) * np.sin(np.radians(phi0)) - np.sin(np.radians(azimuth)) *
                          np.cos(np.radians(theta0)), np.cos(np.radians(azimuth)) * np.cos(np.radians(theta0)) *
                          np.sin(np.radians(phi0)) + np.sin(np.radians(azimuth)) * np.sin(np.radians(theta0))],
                         [np.sin(np.radians(azimuth)) * np.cos(np.radians(phi0)), np.cos(np.radians(azimuth)) *
                          np.cos(np.radians(theta0)) + np.sin(np.radians(azimuth)) * np.sin(np.radians(phi0)) *
                          np.sin(np.radians(theta0)), np.sin(np.radians(azimuth)) * np.sin(np.radians(phi0)) *
                          np.cos(np.radians(theta0)) - np.cos(np.radians(azimuth)) * np.sin(np.radians(theta0))],
                         [-1 * np.sin(np.radians(phi0)), np.cos(np.radians(phi0)) * np.sin(np.radians(theta0)),
                          np.cos(np.radians(phi0)) * np.cos(np.radians(theta0))]])
        return trot

    @staticmethod
    def calc_inverse_rotation_matrix(phi0=0, theta0=0, azimuth=0):
        trot_inv = np.array([[np.cos(np.radians(phi0)) * np.cos(np.radians(azimuth)), np.cos(np.radians(phi0)) *
                             np.sin(np.radians(azimuth)), -1*np.sin(np.radians(phi0))],
                             [np.sin(np.radians(theta0)) * np.sin(np.radians(phi0)) * np.cos(np.radians(azimuth)) -
                              np.cos(np.radians(theta0)) * np.sin(np.radians(azimuth)),
                              np.sin(np.radians(theta0)) * np.sin(np.radians(phi0)) * np.sin(np.radians(azimuth)) +
                              np.cos(np.radians(theta0)) * np.cos(np.radians(azimuth)),
                              np.sin(np.radians(theta0)) * np.cos(np.radians(phi0))],
                             [np.cos(np.radians(theta0)) * np.sin(np.radians(phi0)) * np.cos(np.radians(azimuth)) +
                              np.sin(np.radians(theta0)) * np.sin(np.radians(azimuth)),
                              np.cos(np.radians(theta0)) * np.sin(np.radians(phi0)) * np.sin(np.radians(azimuth)) -
                              np.sin(np.radians(theta0)) * np.cos(np.radians(azimuth)), np.cos(np.radians(theta0)) *
                              np.cos(np.radians(phi0))]])
        return trot_inv