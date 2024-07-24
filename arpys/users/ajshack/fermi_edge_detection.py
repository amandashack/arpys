from load_spectra import load_spectra_by_material
from tools import plot_imagetool
from mask_data import mask_fermi_maps
from restructure import ke_to_be
from normalize_data import normalize_3D
import numpy as np
from scipy.optimize import curve_fit
from lmfit.models import ThermalDistributionModel, LinearModel, ConstantModel
import random
import matplotlib.pyplot as plt
import numba
import xarray as xr


@numba.njit
def shift4_numba(arr, num, fill_value=np.nan):
    if num > 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))[:len(arr)]
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))[:len(arr)]


def flatten_data(data, params):
    x_coords = data.coords['slit'].values
    y_coords = data.coords['perp'].values
    z_coords = data.coords['energy'].values

    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing='ij')
    z_fit = poly2d((x_mesh, y_mesh), *params)

    ef_max = np.nanmax(z_fit)
    de = z_coords[1] - z_coords[0]
    index_shift = ((ef_max - z_fit) / (2*de)).astype(int)

    d = data.values
    nda = np.empty(d.shape)

    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            nda[i, j, :] = shift4_numba(d[i, j, :], index_shift[i, j])

    new_spec = xr.DataArray(data=nda, dims=["slit", "perp", "energy"],
                            coords={"slit": x_coords, "perp": y_coords, "energy": z_coords},
                            attrs=data.attrs)
    return new_spec


def poly2d(xy, a, b, c, d, e, f):
    x, y = xy
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f


def fit_2d_polynomial(fermi_centers, uncertainties=None):
    x = np.array([p[0] for p in fermi_centers])
    y = np.array([p[1] for p in fermi_centers])
    z = np.array([p[2] for p in fermi_centers])
    if uncertainties:
        weights = 1 / np.array(uncertainties)  # Using inverse of uncertainties as weights
        params, _ = curve_fit(poly2d, (x, y), z, sigma=weights)
    else:
        params, _ = curve_fit(poly2d, (x, y), z)

    return params


def fit_fermi_function(data, points):
    fermi_centers = []
    uncertainties = []

    for x, y in points:
        spectrum = data.sel(slit=x, perp=y).values
        energy = data.coords['energy'].values

        # Define the model
        fermifunc = ThermalDistributionModel(prefix='fermi_', form='fermi')
        params = fermifunc.make_params()
        T = 13.6  # Temperature of map
        k = 8.617333e-5  # Boltzmann in eV/K
        params['fermi_kt'].set(value=k * T, vary=True, max=0.03, min=0.001)
        params['fermi_center'].set(value=-0.015, max=0.15, vary=True)
        params['fermi_amplitude'].set(value=1, vary=False)

        linearback = LinearModel(prefix='linear_')
        params.update(linearback.make_params())
        params['linear_slope'].set(value=0, max=50, vary=True)
        params['linear_intercept'].set(value=1, vary=True)

        constant = ConstantModel(prefix='constant_')
        params.update(constant.make_params())
        params['constant_c'].set(value=-1, max=1)

        model = linearback * fermifunc + constant

        # Fit the model to the data
        result = model.fit(spectrum, params, x=energy)

        # Extract the 'fermi_center' value and its uncertainty
        fermi_center = result.params['fermi_center'].value
        uncertainty = result.params['fermi_center'].stderr

        if uncertainty is not None:
            fermi_centers.append((x, y, fermi_center))
            uncertainties.append(uncertainty)

    return fermi_centers, uncertainties


def select_valid_random_points(data, n):
    x_coords = data.coords['slit'].values
    y_coords = data.coords['perp'].values
    selected_points = []

    while len(selected_points) < n:
        x = random.choice(x_coords)
        y = random.choice(y_coords)
        spectrum = data.sel(slit=x, perp=y).values

        # Check if the spectrum contains NaNs
        if not np.isnan(spectrum).any():
            selected_points.append((x, y))

    return selected_points



spectra_data = load_spectra_by_material("co33tas2", photon_energy=76, polarization='LH', alignment=None,
                                            k_conv=False, masked=False, dewarped=False)
data_masked = mask_fermi_maps(*spectra_data)
data_be = [ke_to_be(76, data_masked[0], wf=4.45)]
data_normed = normalize_3D(*data_be)
plot_imagetool(*data_normed)
# Select random points
random_points = select_valid_random_points(data_normed[0], n=1000)

# Fit Fermi function to random points
fermi_centers, uncertainties = fit_fermi_function(data_normed[0], random_points)

# Fit 2D polynomial to the fermi centers
params = fit_2d_polynomial(fermi_centers)

# Flatten the data using the fitted polynomial
flattened_data = flatten_data(data_normed[0], params)
plot_imagetool(flattened_data)

# Extract the points for plotting
x = np.array([p[0] for p in fermi_centers])
y = np.array([p[1] for p in fermi_centers])
z = np.array([p[2] for p in fermi_centers])

# Create meshgrid for plotting the polynomial fit surface
x_coords = data_normed[0].coords['slit'].values
y_coords = data_normed[0].coords['perp'].values
x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing='ij')
z_fit = poly2d((x_mesh, y_mesh), *params)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the fitted points
scatter = ax.scatter(x, y, z, color='r', label='Fermi Fit Points')

# Plot the polynomial fit surface
surface = ax.plot_surface(x_mesh, y_mesh, z_fit, color='b', alpha=0.5)

# Labels and title
ax.set_xlabel('Slit')
ax.set_ylabel('Perp')
ax.set_zlabel('Fermi Center')
ax.set_title('Fermi Fit Points and 2D Polynomial Fit')

# Adding a manual legend
scatter_proxy = plt.Line2D([0], [0], linestyle="none", marker='o', color='r')
surface_proxy = plt.Line2D([0], [0], linestyle="none", marker='s', color='b')

ax.legend([scatter_proxy, surface_proxy], ['Fermi Fit Points', '2D Polynomial Fit'])

plt.show()
