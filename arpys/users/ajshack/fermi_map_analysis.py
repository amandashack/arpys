from load_spectra import load_spectra_by_material
from plotting import plot_all_fermi_surfaces


if __name__ == '__main__':
    spectra_data = load_spectra_by_material("co33tas2", k_conv=True)
    plot_all_fermi_surfaces(*spectra_data)