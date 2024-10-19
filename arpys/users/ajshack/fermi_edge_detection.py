import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from load_spectra import load_spectra_by_material
from tools import plot_imagetool
from mask_data import mask_fermi_maps
from restructure import ke_to_be
from normalize_data import normalize_3D
from scipy.optimize import curve_fit
from scipy import ndimage
import xarray as xr
import numba
import random
import arpys


# Shift function with numba for efficiency
@numba.njit
def shift4_numba(arr, num, fill_value=np.nan):
    if num > 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))[:len(arr)]
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))[:len(arr)]


# Edge detection and filtering
def preprocess_image(xar_slice):
    image = (xar_slice.values * 255).astype('uint8')
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    return smoothed


def detect_edges(image):
    v = np.median(image)
    sigma = 0.2
    lower = int(max(150, (1.0 - sigma) * v))
    upper = int(min(200, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    return edges


def cluster_edges(edges):
    coords = np.column_stack(np.where(edges > 0))
    clustering = DBSCAN(eps=2, min_samples=10).fit(coords)
    labels = clustering.labels_
    return labels, coords


def filter_leading_edge(coords, exclude_top=5, exclude_bottom=5):
    # Convert coords to an xarray Dataset
    ds_coords = xr.Dataset({'x': ('points', coords[:, 1]),
                            'y': ('points', coords[:, 0])})

    # Group by y and get the leading edge by selecting the minimum x per group
    leading_edge = ds_coords.groupby('y').reduce(np.max, dim='points')

    # Get y-values and apply the top and bottom exclusion
    y_values = leading_edge['y'].values
    valid_idx = (y_values >= np.min(y_values) + exclude_top) & (y_values <= np.max(y_values) - exclude_bottom)

    # Select only the valid rows based on the filter
    leading_edge_filtered = leading_edge.sel(y=leading_edge['y'].values[valid_idx])

    # Return the filtered leading edge coordinates
    leading_edge_coords = np.column_stack((leading_edge_filtered['y'].values, leading_edge_filtered['x'].values))

    return leading_edge_coords

# Polynomial fitting
def poly2d(xy, a, b, c, d, e, f):
    x, y = xy
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f


def fit_2d_polynomial(fermi_centers, uncertainties=None):
    x = np.array([p[0] for p in fermi_centers])
    y = np.array([p[1] for p in fermi_centers])
    z = np.array([p[2] for p in fermi_centers])
    if uncertainties:
        weights = 1 / np.array(uncertainties)
        params, _ = curve_fit(poly2d, (x, y), z, sigma=weights)
    else:
        params, _ = curve_fit(poly2d, (x, y), z)
    return params


def flatten_data(data, params):
    x_coords = data.coords['slit'].values
    y_coords = data.coords['perp'].values
    z_coords = data.coords['energy'].values

    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing='ij')
    z_fit = poly2d((x_mesh, y_mesh), *params)

    ef_max = np.nanmax(z_fit)
    de = z_coords[1] - z_coords[0]
    index_shift = ((ef_max - z_fit) / de).astype(int)

    d = data.values
    nda = np.empty(d.shape)

    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            nda[i, j, :] = shift4_numba(d[i, j, :], index_shift[i, j])

    new_spec = xr.DataArray(data=nda, dims=["slit", "perp", "energy"],
                            coords={"slit": x_coords, "perp": y_coords, "energy": z_coords},
                            attrs=data.attrs)
    return new_spec


# Select random points from edge
def select_valid_random_points(edge_points, n):
    selected_points = []
    while len(selected_points) < n:
        point = random.choice(edge_points)
        selected_points.append(point)
    return selected_points


def plot_edges(image, coords):
    plt.imshow(image, cmap='gray')
    plt.scatter(coords[:, 1], coords[:, 0], c='red', s=1)
    plt.tight_layout()
    plt.figaspect(1)
    plt.show()


if __name__ == '__main__':
    # Main execution workflow
    spectra_data = load_spectra_by_material("co33tas2", photon_energy=76, polarization='LH', alignment=None,
                                            k_conv=False, masked=False, dewarped=False)
    data_masked = mask_fermi_maps(*spectra_data)
    data_be = [ke_to_be(76, data_masked[0], wf=4.45)]
    data_normed = normalize_3D(*data_be)
    data = data_be[0].arpes.downsample({'perp': 2})

    # Step 1: Apply preprocessing and edge detection
    edge_points = []
    for slit in data.slit.values:
        slice_2d = data.sel({'slit': slit})
        if not np.isnan(slice_2d).all():
            preprocessed = preprocess_image(slice_2d)
            edges = detect_edges(preprocessed)
            labels, coords = cluster_edges(edges)

            # Filter for the leading edge
            # Example: Exclude the first 10 rows and the last 10 rows
            leading_edge_coords = filter_leading_edge(coords, exclude_top=5, exclude_bottom=5)
            # plot_edges(slice_2d, leading_edge_coords)
            # Add the (slit, perp, energy) values to edge_points
            for (perp_idx, energy_idx) in leading_edge_coords:
                perp = data.coords['perp'].values[int(perp_idx)]
                energy = data.coords['energy'].values[int(energy_idx)]
                edge_points.append((slit, perp, energy))  # (slit, perp, energy)
    edge_points = np.array(edge_points)

    # Step 2: Select random points from leading edge
    random_points = select_valid_random_points(edge_points, n=500)

    # Visualize the selected points to ensure they are well-distributed
    slit_vals = [p[0] for p in random_points]
    perp_vals = [p[1] for p in random_points]
    energy_vals = [p[2] for p in random_points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(slit_vals, perp_vals, energy_vals, c='r', marker='o')
    ax.set_xlabel('Slit')
    ax.set_ylabel('Perp')
    ax.set_zlabel('Energy')
    plt.show()

    # Step 3: Fit 2D polynomial to random points
    params = fit_2d_polynomial(random_points)

    # Step 3: Fit 2D polynomial to random points
    params = fit_2d_polynomial(random_points)

    # Visualize the fitted polynomial surface
    x_vals = np.linspace(min(slit_vals), max(slit_vals), 100)
    y_vals = np.linspace(min(perp_vals), max(perp_vals), 100)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    z_fit = poly2d((x_mesh, y_mesh), *params)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(slit_vals, perp_vals, energy_vals, c='r', marker='o', label='Selected Points')
    ax.plot_surface(x_mesh, y_mesh, z_fit, color='b', alpha=0.5, label='Fitted Polynomial')
    ax.set_xlabel('Slit')
    ax.set_ylabel('Perp')
    ax.set_zlabel('Energy')
    plt.show()

    # Step 4: Shift the data using the fitted polynomial
    flattened_data = flatten_data(data_be[0], params)

    # Plot or analyze flattened data
    plot_imagetool(flattened_data)
