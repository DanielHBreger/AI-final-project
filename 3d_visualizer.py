import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
import numpy as np
import pyvista as pv

def main():
    # Create root window for file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask user to select CSV file
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        initialdir=r"c:\Users\danib\OneDrive\Documents\Technion\AI final project\icedrive-dl-182bd",
        filetypes=[("CSV files", "*.csv")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    # Load the CSV
    try:
        # Read the CSV, skipping the comment line and setting header manually
        df = pd.read_csv(file_path, sep=r'\s+', header=None, skiprows=1)
        # Set column names based on the header
        df.columns = ['ix', 'iy', 'iz', 'nH', 'nH2', 'T', 'vx', 'vy', 'vz', 'nHp', 'ext', 'fh2', 'bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr']
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Get columns except position columns
    position_cols = ['ix', 'iy', 'iz']
    available_cols = [col for col in df.columns if col not in position_cols]

    if not available_cols:
        print("No data columns found.")
        return

    # Columns to visualize in 3x2 grid
    columns = ['nH', 'nH2', 'nHp', 'T', 'ext', 'fh2']

    # Create 3D visualization using volume rendering with GPU acceleration
    # Reshape data into 3D grid (32x32x32 after downsampling)
    grid_size = 32
    scale_factor = 1

    # Extract position indices
    ix_vals = df['ix'].values.astype(int)
    iy_vals = df['iy'].values.astype(int)
    iz_vals = df['iz'].values.astype(int)

    # Plotter with 3x2 subplots
    plotter = pv.Plotter(shape=(3, 2))

    for i, chosen_col in enumerate(columns):
        plotter.subplot(i // 2, i % 2)

        # Fill volume_data
        volume_data = np.zeros((129, 129, 129))
        volume_data[ix_vals, iy_vals, iz_vals] = df[chosen_col].values
        volume_data_log_10 = np.log10(volume_data + 1e-10)

        # Downsample
        volume_data_log_10 = volume_data_log_10[1:129:scale_factor, 1:129:scale_factor, 1:129:scale_factor]

        # Create grid
        grid = pv.ImageData()
        grid.dimensions = np.array(volume_data_log_10.shape) + 1
        grid.origin = (1, 1, 1)
        grid.spacing = (scale_factor, scale_factor, scale_factor)
        grid.cell_data['values'] = volume_data_log_10.flatten(order='F')

        vmin = np.nanmin(volume_data_log_10)
        vmax = np.nanmax(volume_data_log_10)
        clim = (vmin, vmax)
        opacity = [0, 0, 0.1 * (vmax - vmin) / 10, 0.3 * (vmax - vmin) / 10, 1]

        plotter.add_volume(grid, scalars='values', cmap='magma', opacity=opacity, clim=clim)
        plotter.add_text(chosen_col, position='upper_left')

    plotter.show()

if __name__ == "__main__":
    main()