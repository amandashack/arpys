import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from load_spectra import load_spectra_by_material
from fermi_edge_detection import (preprocess_image, detect_edges, filter_leading_edge, cluster_edges,
                                  shift4_numba, select_valid_random_points)
import xarray as xr
from tools import plot_imagetool
from mask_data import mask_fermi_maps
from restructure import ke_to_be
from normalize_data import normalize_3D

# Main execution workflow
spectra_data = load_spectra_by_material("co33tas2", photon_energy=76, polarization='LH', alignment=None,
                                        k_conv=False, masked=False, dewarped=False)
data_masked = mask_fermi_maps(*spectra_data)
data_be = [ke_to_be(76, data_masked[0], wf=4.45)]
data_normed = normalize_3D(*data_be)
data = data_be[0].arpes.downsample({'energy': 5})

print("Step 1: Apply preprocessing and edge detection")

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
        #plot_edges(slice_2d, leading_edge_coords)
        # Add the (slit, perp, energy) values to edge_points
        for (perp_idx, energy_idx) in leading_edge_coords:
            perp = data.coords['perp'].values[int(perp_idx)]
            energy = data.coords['energy'].values[int(energy_idx)]
            edge_points.append((slit, perp, energy))  # (slit, perp, energy)

edge_points = np.array(edge_points)

print("Step 2: Select random points from leading edge")
#random_points = select_valid_random_points(edge_points, n=500)

# Prepare your edge points data
#edge_points = np.array(random_points)
X = edge_points[:, :2]  # (slit, perp) as input
y = edge_points[:, 2]   # energy as the target

print("Split the data into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Build a simple feedforward neural network using TensorFlow/Keras")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),  # Input: (slit, perp)
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output: energy
])

# Set a custom learning rate
learning_rate = 0.01  # Adjust the learning rate as needed

# Create the optimizer with the custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=32)

# Use the trained model to predict the dewarped energy values
predicted_energy = model.predict(X)

# Apply the dewarping to the dataset
def apply_dewarping(data, model):
    x_coords = data.coords['slit'].values
    y_coords = data.coords['perp'].values
    z_coords = data.coords['energy'].values

    X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords, indexing='ij')
    inputs = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])

    # Predict energy corrections using the trained model
    predicted_energy = model.predict(inputs).reshape(X_mesh.shape)

    ef_max = np.nanmax(predicted_energy)
    de = z_coords[1] - z_coords[0]
    index_shift = ((ef_max - predicted_energy) / de).astype(int)

    d = data.values
    nda = np.empty(d.shape)

    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            nda[i, j, :] = shift4_numba(d[i, j, :], index_shift[i, j])

    new_spec = xr.DataArray(data=nda, dims=["slit", "perp", "energy"],
                            coords={"slit": x_coords, "perp": y_coords, "energy": z_coords},
                            attrs=data.attrs)
    return new_spec

print("Apply the dewarping")
flattened_data = apply_dewarping(data_be[0], model)

# Plot or analyze flattened data
plot_imagetool(flattened_data)
