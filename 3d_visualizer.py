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
        initialdir=r".",
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

    # Create column selection dialog
    selected_column = [None]

    def select_column():
        dialog = tk.Toplevel(root)
        dialog.title("Select Column")
        dialog.geometry(f"300x{min(600, 120 + len(available_cols)*25)}")
        tk.Label(dialog, text="Choose a column to visualize:").pack(pady=10)
        var = tk.StringVar(value=available_cols[0])
        for col in available_cols:
            tk.Radiobutton(dialog, text=col, variable=var, value=col).pack(anchor='w', padx=20)
        def ok():
            selected_column[0] = var.get()
            dialog.destroy()
        tk.Button(dialog, text="OK", command=ok).pack(pady=10)
        dialog.wait_window()

    select_column()
    chosen_col = selected_column[0] or 'nH'

    # Create 3D visualization using volume rendering with GPU acceleration
    # Reshape data into 3D grid (32x32x32 after downsampling)
    grid_size = 32
    scale_factor = 1

    # Extract position indices
    ix_vals = df['ix'].values.astype(int)
    iy_vals = df['iy'].values.astype(int)
    iz_vals = df['iz'].values.astype(int)

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

    # Single plotter
    plotter = pv.Plotter()
    plotter.window_size = (800, 600)
    plotter.background_color = 'black'

    # Add volume
    plotter.add_volume(grid, scalars='values', cmap='magma', opacity='sigmoid', clim=clim)

    # Add text labels
    plotter.add_text("3D Volume Visualization", position='upper_right', font_size=14, color='white')
    plotter.add_text(f"Column: {chosen_col}", position='upper_left', font_size=12, color='white')

    # Add axes
    plotter.add_axes(color='white')

    # Set isometric view
    plotter.view_isometric()

    plotter.show()

if __name__ == "__main__":
    main()