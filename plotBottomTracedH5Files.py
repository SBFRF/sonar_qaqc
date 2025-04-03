# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:05:50 2025

@author: RDCHLJEB
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize tkinter root (hide main window)
root = tk.Tk()
root.withdraw()

# Set the directory and file pattern
labeled_dir = r'D:/gitRepos/sonar_qaqc/labeledData/20240724_sonarRaw/'
filetypes = [("H5 files", "*.h5")]

# Open a file selection dialog for the user to choose one H5 file
selected_file = filedialog.askopenfilename(initialdir=labeled_dir,
                                           title="Select an H5 file for plotting",
                                           filetypes=filetypes)

if not selected_file:
    messagebox.showinfo("No File Selected", "No file was selected. Exiting script.")
    exit()

# Open the selected file and retrieve the required datasets
with h5py.File(selected_file, 'r') as f:
    if "profile_data" in f and "depth_line_by_time_idx" in f:
        profile_data_slice = f["profile_data"][:]
        depth_line = f["depth_line_by_time_idx"][:]
    elif "profile_data_slice" in f and "depth_line_by_slice_idx" in f:
        profile_data_slice = f["profile_data_slice"][:]
        depth_line = f["depth_line_by_slice_idx"][:]
    else:
        messagebox.showerror("Missing Dataset", 
                             "The selected file does not contain the required datasets: either "
                             "'profile_data' and 'depth_line_by_time_idx' or 'profile_data_slice' and 'depth_line_by_slice_idx'.")
        exit()

# Create a pcolormesh plot of the profile_data_slice
fig, ax = plt.subplots(figsize=(12, 8))
c = ax.pcolormesh(profile_data_slice, cmap='plasma')

# Overlay the depth line in cyan
# Assumes depth_line is a 2-column array: first column is x (horizontal), second is y (depth)
ax.plot(depth_line[:, 0], depth_line[:, 1], color='cyan', linewidth=2, label="Saved Depth Line")

ax.set_title(os.path.basename(selected_file))
ax.legend()

plt.show()