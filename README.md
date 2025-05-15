# bottomTracerGUI

**bottomTracer** is a GUI application for manually and semi-automatically annotating sonar profile data stored in HDF5 files. It provides a visual interface to trace, edit, and save depth lines (the seabed) based on sonar scans, supporting both automated factory lines and manual corrections.

---

## 🔧 Features

- **Chunked Data Processing**: Efficient handling of large sonar datasets by processing in user-defined slices.
- **Multiple Annotation Modes**:
  - **Factory Mode**: Overlay with automatic depth lines (`Ping Depth` or `Smooth Depth`).
  - **Manual Mode**: Freehand tracing using mouse input.
  - **Editing Mode**: Modify factory lines with clean-up tools and custom edits.
- **Interactive Navigation**:
  - Move through slices with `Next`, `Previous`, or jump to a specific slice.
  - Update and zoom the Y-axis (depth range) for better annotation precision.
- **Clean-Up Tools**:
  - Use lasso selections to remove outliers or invalid points.
  - Annotate omissions using right-click red drawing.
- **Data Export**:
  - Save annotations as `.png` images and `.h5` files per slice.
  - Update a consolidated HDF5 file (`qaqc_depth_line`) with all edits.
- **Secondary Depth Axis**:
  - Dual Y-axis support showing bin # and actual depth (in meters).

---

## 🧠 How It Works

### Load and Visualize
- Load an HDF5 sonar data file.
- View sonar backscatter (`profile_data`) as a color image.
- Select chunk size for slicing time series.

### Annotate
- Enable manual tracing or edit factory lines.
- Use green lines for valid edits, red for omissions.
- Interpolate sparse input into continuous depth lines.

### Save & Export
- Save each slice's annotation as:
  - `.png` image (stored in `/qcPlots`)
  - `.h5` depth data (stored in `/qcdData`)
- Update a global `qaqc_depth_line` dataset in both:
  - The raw sonar HDF5 file
  - A dedicated whole-record HDF5 file

---

## 🗂 Output Structure

    project_directory/
    ├── raw_data_file.h5
    ├── qcPlots/
    │   └── bottomTraced_{start}-{end}.png
    ├── qcdData/
    │   └── bottomTraced_{start}-{end}.h5
    └── *_bottomTraced_wholeRecord.h5

---

## 🧪 Use Case

This tool is designed for validating or correcting automatic seabed detections to develope training datasets for machine algorithms that will be designed for use in senarios that require consistent depth line detection.

By combining machine-generated suggestions with manual expert edits, **bottomTracer** ensures high-quality bottom detection across sonar data samples.

---

## 🖥️ Requirements

- Python 3.x
- `tkinter`
- `matplotlib`
- `h5py`
- `numpy`
- `Pillow`

---

## 🚀 Getting Started

1. Run the script:
   `bash python bottomTracerGUI_V5.py`
  
2. Use the GUI to:
    - Load a sonar HDF5 file
    - Annotate and edit depth lines
    - Save the annotated output

---

## ✍️ Authors

Developed by Jeremy E. Braun, Kara Koetje, and Spicer Bak of the USACE Engineer Research and Development Center Coastal and Hydraulics Laboratory Field Research Facility.
