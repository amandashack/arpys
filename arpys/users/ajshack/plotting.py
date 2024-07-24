import matplotlib.pyplot as plt
import math
import numpy as np
from load_spectra import load_spectra_by_photon_energy


def plot_all_fermi_surfaces(*args, energy=-0.02, color='Blues', **kwargs):
    num_plots = len(args)
    num_rows = math.ceil(num_plots / 2)
    fig, axes = plt.subplots(num_rows, 2, figsize=(8, 3 * num_rows))
    fig.canvas.manager.window.move(0, 0)
    ni = 0
    for i, (arg, ax) in enumerate(zip(args, axes.flatten())):
        dims = list(arg.dims)
        alignment = arg.attrs.get('alignment', 'N/A')
        polarization = arg.attrs.get('polarization', 'N/A')
        sample_name = arg.attrs.get('material_value', 'N/A')
        photon_energy = arg.attrs.get('photon_energy', 'N/A')

        # Downsample if needed
        if "downsample" in kwargs.keys():
            arg_ds = arg.arpes.downsample({dims[2]: kwargs["downsample"]})
            arg_cut = arg_ds.sel({dims[2]: energy}, method='nearest')
        else:
            arg_cut = arg.sel({dims[2]: energy}, method='nearest')

        # Plot with or without colorbar
        if "add_colorbar" in kwargs.keys():
            arg_cut.plot(x=dims[0], y=dims[1], ax=ax, cmap=color, add_colorbar=kwargs['add_colorbar'])
        else:
            arg_cut.plot(x=dims[0], y=dims[1], ax=ax, cmap=color, add_colorbar=False)

        # Add the information text
        info_text = (f"Sample: {sample_name}\nhv: {photon_energy} eV\nAlignment: {alignment}\n"
                      f"Polarization: {polarization}\nBinding Energy: {energy}")
        ax.text(0.95, 0.05, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                horizontalalignment='right')

        # Turn off the title for each axis
        ax.set_title('')
        ni += 1
    for j in range(ni, num_rows * 2):
        fig.delaxes(axes.flatten()[j])
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_difference_ntimes(spec1, spec2, energy_range, n_plots, ds, color='bwr', **kwargs):
    dims = spec1.dims
    energies = np.linspace(energy_range[0], energy_range[1], num=n_plots)
    fig, axes = plt.subplots(len(energies.tolist()), 3, figsize=(12, 4 * n_plots))
    fig.canvas.manager.window.move(0, 0)

    spec1_ds = spec1.arpes.downsample({dims[2]: ds})
    spec2_ds = spec2.arpes.downsample({dims[2]: ds})
    diff_spec = (spec2 / spec2.max()) - (spec1 / spec1.max())
    diff_spec_ds = diff_spec.arpes.downsample({dims[2]: ds})

    alignment1 = spec1.attrs.get('alignment', 'N/A')
    sample_name1 = spec1.attrs.get('material_value', 'N/A')
    photon_energy1 = spec1.attrs.get('photon_energy', 'N/A')
    polarization1 = spec1.attrs.get('polarization', 'N/A')

    alignment2 = spec2.attrs.get('alignment', 'N/A')
    sample_name2 = spec2.attrs.get('material_value', 'N/A')
    photon_energy2 = spec2.attrs.get('photon_energy', 'N/A')
    polarization2 = spec2.attrs.get('polarization', 'N/A')

    for i, energy in enumerate(energies):
        diff_cut = diff_spec_ds.sel({dims[2]: energy}, method="nearest")
        spec1_cut = spec1_ds.sel({dims[2]: energy}, method="nearest")
        spec2_cut = spec2_ds.sel({dims[2]: energy}, method="nearest")

        if "add_colorbar" in kwargs.keys():
            diff_cut.plot(x=dims[0], y=dims[1], ax=axes[i][0], cmap=color, add_colorbar=kwargs['add_colorbar'])
            spec1_cut.plot(x=dims[0], y=dims[1], ax=axes[i][1], cmap='Blues', add_colorbar=kwargs['add_colorbar'])
            spec2_cut.plot(x=dims[0], y=dims[1], ax=axes[i][2], cmap='Blues', add_colorbar=kwargs['add_colorbar'])
        else:
            diff_cut.plot(x=dims[0], y=dims[1], ax=axes[i][0], cmap=color, add_colorbar=False)
            spec1_cut.plot(x=dims[0], y=dims[1], ax=axes[i][1], cmap='Blues', add_colorbar=False)
            spec2_cut.plot(x=dims[0], y=dims[1], ax=axes[i][2], cmap='Blues', add_colorbar=False)

        # Add the information text
        info_text1 = (f"Sample: {sample_name1}\nhv: {photon_energy1} eV\nAlignment: {alignment1}\n"
                      f"Polarization: {polarization1}\nBinding Energy: {energy}")
        axes[i][1].text(0.95, 0.05, info_text1, transform=axes[i][1].transAxes, fontsize=10, verticalalignment='bottom',
                        horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

        axes[i][0].set_title('')
        axes[i][1].set_title('')
        axes[i][2].set_title('')

        info_text2 = (f"Sample: {sample_name2}\nhv: {photon_energy2} eV\nAlignment: {alignment2}\n"
                      f"Polarization: {polarization2}\nBinding Energy: {energy}")
        axes[i][2].text(0.95, 0.05, info_text2, transform=axes[i][2].transAxes, fontsize=10, verticalalignment='bottom',
                        horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def plot_diff(diff_specs, specs1, specs2, energy, ds=None, color='bwr', **kwargs):
    fig, axes = plt.subplots(len(diff_specs), 3, figsize=(10, 2 * len(diff_specs)))
    fig.canvas.manager.window.move(0, 0)

    for i, (diff_spec, spec1, spec2) in enumerate(zip(diff_specs, specs1, specs2)):
        dims = spec1.dims
        if ds:
            spec1 = spec1.arpes.downsample({dims[2]: ds})
            spec2 = spec2.arpes.downsample({dims[2]: ds})
            diff_spec = diff_spec.arpes.downsample({dims[2]: ds})

        alignment1 = spec1.attrs.get('alignment', 'N/A')
        sample_name1 = spec1.attrs.get('material_value', 'N/A')
        photon_energy1 = spec1.attrs.get('photon_energy', 'N/A')
        polarization1 = spec1.attrs.get('polarization', 'N/A')

        alignment2 = spec2.attrs.get('alignment', 'N/A')
        sample_name2 = spec2.attrs.get('material_value', 'N/A')
        photon_energy2 = spec2.attrs.get('photon_energy', 'N/A')
        polarization2 = spec2.attrs.get('polarization', 'N/A')

        diff_cut = diff_spec.sel({dims[2]: energy}, method="nearest")
        spec1_cut = spec1.sel({dims[2]: energy}, method="nearest")
        spec2_cut = spec2.sel({dims[2]: energy}, method="nearest")

        if "add_colorbar" in kwargs.keys():
            diff_cut.plot(x=dims[0], y=dims[1], ax=axes[i][0], cmap=color, add_colorbar=kwargs['add_colorbar'])
            spec1_cut.plot(x=dims[0], y=dims[1], ax=axes[i][1], cmap='Blues', add_colorbar=kwargs['add_colorbar'])
            spec2_cut.plot(x=dims[0], y=dims[1], ax=axes[i][2], cmap='Blues', add_colorbar=kwargs['add_colorbar'])
        else:
            diff_cut.plot(x=dims[0], y=dims[1], ax=axes[i][0], cmap=color, add_colorbar=False)
            spec1_cut.plot(x=dims[0], y=dims[1], ax=axes[i][1], cmap='Blues', add_colorbar=False)
            spec2_cut.plot(x=dims[0], y=dims[1], ax=axes[i][2], cmap='Blues', add_colorbar=False)

        # Add the information text
        info_text1 = (f"Sample: {sample_name1}\nhv: {photon_energy1} eV\nAlignment: {alignment1}\n"
                      f"Polarization: {polarization1}\nBinding Energy: {energy}")
        axes[i][1].text(0.95, 0.05, info_text1, transform=axes[i][1].transAxes, fontsize=10, verticalalignment='bottom',
                        horizontalalignment='right')
        info_text2 = (f"Sample: {sample_name2}\nhv: {photon_energy2} eV\nAlignment: {alignment2}\n"
                      f"Polarization: {polarization2}\nBinding Energy: {energy}")
        axes[i][2].text(0.95, 0.05, info_text2, transform=axes[i][2].transAxes, fontsize=10, verticalalignment='bottom',
                        horizontalalignment='right')
        axes[i][0].set_title('')
        axes[i][1].set_title('')
        axes[i][2].set_title('')

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


if __name__ == '__main__':

    # Example usage: Load spectra with photon energy of 86
    photon_energy = 90
    spectra_data = load_spectra_by_photon_energy(photon_energy)

    #plot_all_fermi_surfaces(*list(spectra_data.values()))
    for spectra in spectra_data.values():
        if spectra.attrs['material_value'] == 0.33:
            if spectra.attrs['polarization'] == "CR":
                CR = spectra
            elif spectra.attrs['polarization'] == "CL":
                CL = spectra
    plot_difference_ntimes(CR, CL, [0.0, -0.1], 5, 1)
