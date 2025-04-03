import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageGrab

"""HDF5Annotator – A GUI tool for annotating HDF5 file slices.

There are two main modes:

  • Factory Mode ("Ping Depth" or "Smooth Depth"):
      - Default mode when a file is loaded.
      - Displays an automatic factory depth line (blue) that can be edited.
      - Buttons available: "Save {Toggle} Line" and "Edit {Toggle} Line".
      - In editing mode the factory line is shown with 25% opacity.
        Editing mode then provides "Clear Annotations", "Apply Edits", and "Save Edited Depth Line".
        
  • Manual Tracing Mode ("Off"):
      - Enables manual tracing by left-clicking and dragging (green).
      - Buttons available: "Clear Annotations", "Apply Traced Line", and "Save Manual Depth Line".

After saving any output (PNG/H5), the next slice is automatically loaded and the controls reset."""

class HDF5Annotator:
    def __init__(self, root):
        self.root = root
        self.root.title("HDF5 Annotator")
        
        # Initialize state variables
        self.depth_option = tk.StringVar(value="Ping Depth")  # Default toggle option
        self.manual_line_saved = False
        self.edit_mode = False  # False: factory mode; True: editing mode
        self.draw_mode = "green"  # Current drawing color/mode
        self.edited_line = None   # Stores merged edited line after "Apply Edits"
        
        # -------------------- Top Menu Frame --------------------
        self.menu_frame = tk.Frame(root, borderwidth=2, relief="groove")
        self.menu_frame.pack(side="top", fill="x", padx=10, pady=10)
        for col in range(4):
            self.menu_frame.grid_columnconfigure(col, weight=1)
        tk.Label(self.menu_frame, text="           Chunk Size:").grid(row=0, column=1, padx=0, pady=5)
        self.chunk_size_entry = tk.Entry(self.menu_frame, width=10, justify="center")
        self.chunk_size = 1000
        self.chunk_size_entry.insert(0, str(self.chunk_size))
        self.chunk_size_entry.grid(row=0, column=2, padx=0, pady=5)
        tk.Label(self.menu_frame, text="Output Directory:").grid(row=1, column=1, padx=10, pady=5)
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        self.output_dir_entry = tk.Entry(self.menu_frame, textvariable=self.output_dir_var, width=25, justify="center")
        self.output_dir_entry.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.browse_button = tk.Button(self.menu_frame, text="Browse", command=self.choose_output_directory)
        self.browse_button.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.load_button = tk.Button(self.menu_frame, text="Load HDF5 File", command=self.load_file, width=30)
        self.load_button.grid(row=2, column=0, padx=0, pady=5, columnspan=4)
        
        # -------------------- Annotation Frame & Canvas --------------------
        self.annotation_frame = tk.Frame(root)
        self.fig, self.ax = plt.subplots(figsize=(18, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.annotation_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()
        
        # -------------------- Jump Navigation --------------------
        self.jump_frame = tk.Frame(self.annotation_frame)
        self.jump_frame.pack(pady=5)
        tk.Label(self.jump_frame, text="Slice #").grid(row=0, column=0, padx=5)
        self.prev_button_jump = tk.Button(self.jump_frame, text="<<", command=self.prev_slice)
        self.prev_button_jump.grid(row=0, column=1, padx=5)
        self.jump_slice_var = tk.StringVar(value="1")
        self.jump_entry = tk.Entry(self.jump_frame, textvariable=self.jump_slice_var, width=5, justify="center")
        self.jump_entry.grid(row=0, column=2, padx=5)
        self.next_button_jump = tk.Button(self.jump_frame, text=">>", command=self.next_slice)
        self.next_button_jump.grid(row=0, column=3, padx=5)
        self.jump_button = tk.Button(self.jump_frame, text="Jump To Slice", command=self.jump_to_slice)
        self.jump_button.grid(row=0, column=4, padx=5)
        
        # -------------------- Depth Options --------------------
        self.depth_frame = tk.Frame(self.annotation_frame)
        self.depth_frame.pack(pady=5)
        tk.Label(self.depth_frame, text="Toggle Factory Line:").pack(side="left", padx=5)
        tk.Radiobutton(self.depth_frame, text="Ping", variable=self.depth_option, value="Ping Depth",
                    command=self.update_display).pack(side="left", padx=5)
        tk.Radiobutton(self.depth_frame, text="Smooth", variable=self.depth_option, value="Smooth Depth",
                    command=self.update_display).pack(side="left", padx=5)
        tk.Radiobutton(self.depth_frame, text="Off", variable=self.depth_option, value="Off",
                    command=self.update_display).pack(side="left", padx=5)
        
        # -------------------- Control Buttons --------------------
        self.button_frame = tk.Frame(self.annotation_frame)
        self.button_frame.pack(fill="both", expand=True)
        # In manual tracing ("Off") mode: Clear, Apply Traced Line, Save Manual Depth Line.
        self.clear_button = tk.Button(self.button_frame, text="Clear Annotations", command=self.clear_annotations)
        self.clear_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.save_button = tk.Button(self.button_frame, text="", state="normal")
        self.save_button.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.save_depth_button = tk.Button(self.button_frame, text="", state="normal")
        self.save_depth_button.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        # For factory mode the clear button is hidden.
        self.clear_button.grid_remove()
        self.quit_button = tk.Button(self.button_frame, text="Quit", command=self.quit_gui)
        self.quit_button.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        for col in range(3):
            self.button_frame.grid_columnconfigure(col, weight=1)
        
        # -------------------- Internal Variables --------------------
        self.file_path = None
        self.base_name = None
        self.whole_record_file = None
        self.idx_start = 0
        self.image = None
        self.smooth_depth = None
        self.length_mm = None
        self.this_ping_depth_m = None
        self.bin_size = None
        self.smooth_depth_img = None
        self.this_ping_depth_img = None
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.coordinates = []  # List of (canvas_x, canvas_y, data_x, data_y, draw_mode)
        self.image_for_saving = None
        self.total_slices = None
        self.total_time = None
        self.applied_line = None
        self.slice_number = None
        self.slice_length = None

    # -------------------- Utility Methods --------------------
    def choose_output_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)

    def load_file(self):
        try:
            self.chunk_size = int(self.chunk_size_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Chunk size must be an integer.")
            return
        self.file_path = filedialog.askopenfilename(initialdir=os.getcwd(),
                                                    filetypes=[("HDF5 files", "*.h5 *.hdf5")])
        if not self.file_path:
            return
        self.base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        selected_dir = self.output_dir_var.get()
        self.output_folder = os.path.join(selected_dir, self.base_name)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.whole_record_file = os.path.join(self.output_folder, f"_{self.base_name}_bottomTraced_wholeRecord.h5")
        self.menu_frame.pack_forget()
        self.annotation_frame.pack(fill="both", expand=True)
        self.idx_start = 0
        # Reset toggle to default ("Ping Depth") and re-enable radio buttons.
        self.depth_option.set("Ping Depth")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        self.process_next_chunk()

    def process_next_chunk(self):
        if not self.file_path:
            return
        with h5py.File(self.file_path, "r") as f:
            self.total_time = f["time"].shape[0]
            self.total_slices = int(np.ceil(self.total_time / self.chunk_size))
            self.slice_number = self.idx_start // self.chunk_size + 1
            end_idx = min(self.idx_start + self.chunk_size, self.total_time)
            idx = slice(self.idx_start, end_idx)
            self.image = f["profile_data"][:, idx]
            if "smooth_depth_m" in f:
                self.smooth_depth = f["smooth_depth_m"][idx]
                if "length_mm" in f:
                    self.length_mm = f["length_mm"][idx]
                    self.bin_size = (self.length_mm[0] / 1000.0) / self.image.shape[0]
                else:
                    self.length_mm = None
                    self.bin_size = None
                if "this_ping_depth_m" in f:
                    self.this_ping_depth_m = f["this_ping_depth_m"][idx]
                else:
                    self.this_ping_depth_m = None
            else:
                self.smooth_depth = None
                self.length_mm = None
                self.this_ping_depth_m = None
                self.bin_size = None
        if self.bin_size is not None:
            self.smooth_depth_img = self.smooth_depth / self.bin_size if self.smooth_depth is not None else None
            self.this_ping_depth_img = self.this_ping_depth_m / self.bin_size if self.this_ping_depth_m is not None else None
        else:
            self.smooth_depth_img = None
            self.this_ping_depth_img = None
        self.edit_mode = False
        self.update_display()

    def update_display(self):
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        # Determine the current slice length from the image width.
        self.slice_length = self.image.shape[1]
        # Set the x-axis to the actual slice length.
        self.ax.set_xlim(0, self.slice_length)
        self.ax.set_ylim(0, self.image.shape[0])
        if self.depth_option.get() != "Off" and ((self.smooth_depth_img is not None) or (self.this_ping_depth_img is not None)):
            self.plot_depth()
        # Remove any existing figure texts and add slice info.
        for txt in self.fig.texts[:]:
            txt.remove()
        idxS = self.idx_start
        idxE = min(self.idx_start + self.chunk_size - 1, self.total_time - 1)
        self.fig.text(0.01, 0.98, f"Slice #{self.slice_number} of {self.total_slices}\nTime Indices: {idxS} - {idxE}",
                    horizontalalignment="left", verticalalignment="top", fontsize=12, color="black")
        self.canvas.draw()
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.enable_annotation()
        self.fig.canvas.draw()
        # Capture canvas image for saving.
        image_array = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.image_for_saving = Image.fromarray(image_array)
        # In factory mode, hide clear button; in manual or editing mode, show it.
        if self.depth_option.get() != "Off" and not self.edit_mode:
            self.clear_button.grid_remove()
        self.update_button_states()
        self.jump_slice_var.set(str(self.slice_number))

    def update_button_states(self):
        if self.depth_option.get() == "Off":
            # Manual tracing mode: show clear, apply traced, and save manual buttons.
            self.clear_button.grid()  # Make sure clear button is visible.
            self.clear_button.config(text="Clear Annotations", command=self.clear_annotations, state="normal")
            self.save_button.config(text="Apply Traced Line", command=self.apply_traced_line, state="normal")
            self.save_depth_button.config(text="Save Manual Depth Line", command=self.save_data, state="disabled")
        else:
            if not self.edit_mode:
                # Factory mode: hide clear button; show save and edit buttons.
                self.clear_button.grid_remove()
                self.save_button.config(text=f"Save {self.depth_option.get()} Line", command=self.save_depth_line, state="normal")
                self.save_depth_button.config(text=f"Edit {self.depth_option.get()} Line", command=self.enter_editing_mode, state="normal")
            else:
                # Editing mode: show clear, apply edits, and save edited buttons.
                self.clear_button.grid()
                self.clear_button.config(text="Clear Annotations", command=self.clear_edit_mode, state="normal")
                self.save_button.config(text="Apply Edits", command=self.apply_edits, state="normal")
                self.save_depth_button.config(text="Save Edited Depth Line", command=self.save_edited_depth_line, state="disabled")
        self.next_button_jump.config(state="normal" if self.slice_number < self.total_slices else "disabled")
        self.prev_button_jump.config(state="normal" if self.slice_number > 1 else "disabled")

    def plot_depth(self):
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            return
        # Use the actual slice length for the x-axis.
        x = np.arange(0, self.slice_length)
        if data is not None:
            alpha_val = 1.0 if not self.edit_mode else 0.35
            self.ax.plot(x, data[:self.slice_length], color="blue", linewidth=2, alpha=alpha_val, label=self.depth_option.get())
            self.ax.legend(loc="upper right")
            self.canvas.draw()

    def prev_slice(self):
        self.clear_annotations()
        self.manual_line_saved = False
        self.edit_mode = False
        # Reset toggle to default ("Ping Depth") and re-enable radio buttons.
        self.depth_option.set("Ping Depth")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        self.idx_start = self.idx_start - self.chunk_size
        self.process_next_chunk()

    def next_slice(self):
        self.clear_annotations()
        self.manual_line_saved = False
        self.edit_mode = False
        # Reset toggle to default.
        self.depth_option.set("Ping Depth")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        self.idx_start = self.idx_start + self.chunk_size
        self.process_next_chunk()

    def jump_to_slice(self):
        try:
            target_slice = int(self.jump_slice_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Slice number must be an integer.")
            return
        if self.total_slices is None:
            messagebox.showerror("Error", "No file loaded.")
            return
        if target_slice < 1 or target_slice > self.total_slices:
            messagebox.showerror("Invalid Slice", f"Slice number must be between 1 and {self.total_slices}.")
            return
        self.idx_start = (target_slice - 1) * self.chunk_size
        # Reset toggle to default.
        self.depth_option.set("Ping Depth")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        self.process_next_chunk()

    def quit_gui(self):
        self.root.destroy()

    # -------------------- Annotation Binding and Helpers --------------------
    def unbind_all_events(self):
        # Helper method to unbind all annotation-related events.
        events = ["<Button-1>", "<B1-Motion>", "<Button-3>", "<B3-Motion>", "<ButtonRelease-1>", "<ButtonRelease-3>"]
        for event in events:
            self.canvas_widget.unbind(event)

    def enable_annotation(self):
        if not self.manual_line_saved:
            if self.depth_option.get() == "Off":
                self.canvas_widget.bind("<Button-1>", self.start_tracing)
                self.canvas_widget.bind("<B1-Motion>", self.trace_line)
                self.canvas_widget.bind("<Button-3>", self.stop_tracing)
            elif self.edit_mode:
                self.canvas_widget.bind("<Button-1>", self.start_tracing_editing_green)
                self.canvas_widget.bind("<B1-Motion>", self.trace_line_editing_green)
                self.canvas_widget.bind("<Button-3>", self.start_tracing_editing_red)
                self.canvas_widget.bind("<B3-Motion>", self.trace_line_editing_red)
                self.canvas_widget.bind("<ButtonRelease-1>", self.stop_tracing)
                self.canvas_widget.bind("<ButtonRelease-3>", self.stop_tracing)
            else:
                self.unbind_all_events()
        else:
            self.unbind_all_events()

    def canvas_to_data(self, x, y):
        # Convert canvas coordinates to data coordinates.
        x_offset = self.canvas_widget.winfo_rootx() - self.root.winfo_rootx()
        y_offset = self.canvas_widget.winfo_rooty() - self.root.winfo_rooty()
        fig_x = x - x_offset
        fig_y = y - y_offset
        fig_y = self.canvas_widget.winfo_height() - fig_y
        data_x, data_y = self.ax.transData.inverted().transform((fig_x, fig_y))
        return data_x, data_y

    # -------------------- Manual Tracing Methods --------------------
    def start_tracing(self, event):
        self.tracing = True
        self.last_x, self.last_y = event.x, event.y
        self.coordinates.append((event.x, event.y, *self.canvas_to_data(event.x, event.y), "green"))
        
    def trace_line(self, event):
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            if not (x_min <= data_x <= x_max and y_min <= data_y <= y_max):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line(self.last_x, self.last_y, event.x, event.y,
                                            fill="green", width=2, tags="annotation")
            if not self.coordinates or (event.x, event.y) != (self.coordinates[-1][0], self.coordinates[-1][1]):
                self.coordinates.append((event.x, event.y, data_x, data_y, "green"))
            self.last_x, self.last_y = event.x, event.y

    def stop_tracing(self, event):
        self.tracing = False

    # -------------------- Editing Tracing Methods --------------------
    def start_tracing_editing_green(self, event):
        self.tracing = True
        self.last_x, self.last_y = event.x, event.y
        x_data, y_data = self.canvas_to_data(event.x, event.y)
        self.coordinates.append((event.x, event.y, x_data, y_data, "green"))

    def trace_line_editing_green(self, event):
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            if not (self.ax.get_xlim()[0] <= data_x <= self.ax.get_xlim()[1] and
                    self.ax.get_ylim()[0] <= data_y <= self.ax.get_ylim()[1]):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line(self.last_x, self.last_y, event.x, event.y,
                                            fill="green", width=2, tags="annotation")
            if not self.coordinates or (event.x, event.y) != (self.coordinates[-1][0], self.coordinates[-1][1]):
                self.coordinates.append((event.x, event.y, data_x, data_y, "green"))
            self.last_x, self.last_y = event.x, event.y

    def start_tracing_editing_red(self, event):
        self.tracing = True
        self.last_x, self.last_y = event.x, event.y
        x_data, y_data = self.canvas_to_data(event.x, event.y)
        self.coordinates.append((event.x, event.y, x_data, y_data, "red"))

    def trace_line_editing_red(self, event):
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            if not (self.ax.get_xlim()[0] <= data_x <= self.ax.get_xlim()[1] and
                    self.ax.get_ylim()[0] <= data_y <= self.ax.get_ylim()[1]):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line(self.last_x, self.last_y, event.x, event.y,
                                            fill="red", width=2, tags="annotation")
            if not self.coordinates or (event.x, event.y) != (self.coordinates[-1][0], self.coordinates[-1][1]):
                self.coordinates.append((event.x, event.y, data_x, data_y, "red"))
            self.last_x, self.last_y = event.x, event.y

    def interpolate_coordinates_by_color(self, color):
        # Interpolate traced coordinates for a specific drawing color.
        x_coords = np.arange(self.chunk_size)
        filtered = [pt for pt in self.coordinates if pt[4] == color]
        if not filtered:
            return np.column_stack((x_coords, np.full(self.chunk_size, np.nan)))
        pts = sorted([(pt[2], pt[3]) for pt in filtered], key=lambda p: p[0])
        y_interp = np.full(self.chunk_size, np.nan)
        for x in x_coords:
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i+1]
                if (x1 - x0) <= 2.0 and x0 <= x <= x1:
                    y_interp[x] = np.interp(x, [x0, x1], [y0, y1])
                    break
        return np.column_stack((x_coords, y_interp))

    def interpolate_coordinates(self):
        # Interpolate all traced coordinates (regardless of color).
        x_coords = np.arange(self.chunk_size)
        if not self.coordinates:
            return np.column_stack((x_coords, np.full(self.chunk_size, np.nan)))
        pts = sorted([(pt[2], pt[3]) for pt in self.coordinates], key=lambda p: p[0])
        y_interp = np.full(self.chunk_size, np.nan)
        for x in x_coords:
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i+1]
                if (x1 - x0) <= 2.0 and x0 <= x <= x1:
                    y_interp[x] = np.interp(x, [x0, x1], [y0, y1])
                    break
        return np.column_stack((x_coords, y_interp))

    def clear_annotations(self):
        # Clear all drawn annotations and reset tracing state.
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.canvas_widget.delete("annotation")
        if self.applied_line:
            self.update_display()

    def clear_edit_mode(self):
        """Clear editing annotations and replot the factory line at 25% opacity."""
        self.clear_annotations()
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            data = None
        if data is not None:
            x = np.arange(0, self.chunk_size)
            self.ax.plot(x, data, color="blue", linewidth=2, alpha=0.35, label=self.depth_option.get())
        self.ax.set_title("Editing Mode Enabled:\nLeft click to draw edits (green), Right click for NaN out (red).")
        self.ax.legend(loc="upper right")
        self.canvas.draw()
        # Rebind editing events.
        self.unbind_all_events()
        self.canvas_widget.bind("<Button-1>", self.start_tracing_editing_green)
        self.canvas_widget.bind("<B1-Motion>", self.trace_line_editing_green)
        self.canvas_widget.bind("<Button-3>", self.start_tracing_editing_red)
        self.canvas_widget.bind("<B3-Motion>", self.trace_line_editing_red)
        self.canvas_widget.bind("<ButtonRelease-1>", self.stop_tracing)
        self.canvas_widget.bind("<ButtonRelease-3>", self.stop_tracing)
        self.update_button_states()

    # -------------------- Editing Mode Methods --------------------
    def enter_editing_mode(self):
        """Enter editing mode: lower factory line opacity, update title, lock toggle options, and bind editing events."""
        self.edit_mode = True
        self.draw_mode = "green"
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.set_xlim(0, self.slice_length)
        self.ax.set_ylim(0, self.image.shape[0])
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            data = None
        if data is not None:
            x = np.arange(0, self.slice_length)
            self.ax.plot(x, data, color="blue", linewidth=2, alpha=0.35, label=self.depth_option.get())
        self.ax.set_title("Editing Mode Enabled:\nLeft click to draw edits (green), Right click for NaN out (red).")
        self.ax.legend(loc="upper right")
        self.canvas.draw()
        # Disable the toggle radio buttons during editing.
        for child in self.depth_frame.winfo_children():
            child.config(state="disabled")
        self.unbind_all_events()
        self.canvas_widget.bind("<Button-1>", self.start_tracing_editing_green)
        self.canvas_widget.bind("<B1-Motion>", self.trace_line_editing_green)
        self.canvas_widget.bind("<Button-3>", self.start_tracing_editing_red)
        self.canvas_widget.bind("<B3-Motion>", self.trace_line_editing_red)
        self.canvas_widget.bind("<ButtonRelease-1>", self.stop_tracing)
        self.canvas_widget.bind("<ButtonRelease-3>", self.stop_tracing)
        self.clear_button.grid()
        self.clear_button.config(text="Clear Annotations", command=self.clear_edit_mode, state="normal")
        self.save_button.config(text="Apply Edits", command=self.apply_edits, state="normal")
        self.save_depth_button.config(text="Save Edited Depth Line", command=self.save_edited_depth_line, state="disabled")

    def apply_edits(self):
        """
        Merge the factory depth line with drawn edits.
        The green edits override the factory line, while red edits force NaN.
        The merged line is replotted in blue and stored.
        """
        if self.depth_option.get() == "Smooth Depth":
            factory_line = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            factory_line = self.this_ping_depth_img
        else:
            messagebox.showerror("Error", "No factory line data available!")
            return
        auto_slice = factory_line[:self.slice_length].copy()
        green_edit = self.interpolate_coordinates_by_color("green")[:self.slice_length, 1]
        red_edit = self.interpolate_coordinates_by_color("red")[:self.slice_length, 1]
        merged = np.copy(auto_slice)
        for i in range(self.slice_length):
            if not np.isnan(red_edit[i]):
                merged[i] = np.nan
            elif not np.isnan(green_edit[i]):
                merged[i] = green_edit[i]
        self.edited_line = merged
        x_vals = np.arange(self.slice_length)
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.plot(x_vals, merged, color="blue", linewidth=2, label=self.depth_option.get())
        self.ax.legend(loc="upper right")
        self.ax.set_title("")
        self.canvas.draw()
        self.clear_annotations()
        self.unbind_all_events()
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="normal")

    def apply_traced_line(self):
        """
        Merge the manually traced (green) line.
        The traced line is replotted in blue, drawing is disabled, and toggle options are locked.
        In manual tracing mode, if no annotations were made, the entire slice will be set to NaN.
        """
        if not self.coordinates:
            messagebox.showinfo("No annotations", "No annotations found.\nThis will result in the whole slice being NaN'd out.")
            traced_values = np.full(self.slice_length, np.nan)
        else:
            traced_values = self.interpolate_coordinates_by_color("green")[:self.slice_length, 1]
        self.canvas_widget.delete("annotation")
        x_vals = np.arange(self.slice_length)
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.plot(x_vals, traced_values, color="blue", linewidth=2, label="Manual Depth Line")
        self.ax.legend(loc="upper right")
        self.ax.set_title("")
        self.canvas.draw()
        for child in self.depth_frame.winfo_children():
            child.config(state="disabled")
        self.unbind_all_events()
        self.applied_line = True
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="normal")

    def save_edited_depth_line(self):
        """
        Save the edited depth line (resulting from applied edits).
        The merged line is re-plotted in cyan and both a PNG and HDF5 file are saved.
        After saving, the next slice is automatically loaded.
        """
        if self.edited_line is None:
            messagebox.showerror("Error", "No edited depth line available!")
            return
        merged = self.edited_line
        idxS = self.idx_start
        x_vals = np.arange(self.slice_length)
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.plot(x_vals, merged, color="cyan", linewidth=2, label="Saved Depth Line")
        self.ax.legend(loc="upper right")
        self.ax.set_title("")
        self.canvas.draw()
        self.root.update()
        default_filename = f"{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png"
        x0 = self.root.winfo_rootx() + self.canvas_widget.winfo_x()
        y0 = self.root.winfo_rooty() + self.canvas_widget.winfo_y()
        x1 = x0 + self.canvas_widget.winfo_width()
        y1 = y0 + self.canvas_widget.winfo_height()
        self.image_for_saving = ImageGrab.grab((x0, y0, x1, y1))
        save_path = filedialog.asksaveasfilename(initialdir=self.output_folder,
                                                initialfile=default_filename,
                                                defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
        if save_path:
            self.image_for_saving.save(save_path)
            print(f"Image saved: {save_path}")
            h5_path = save_path.replace(".png", ".h5")
            with h5py.File(h5_path, "w") as hf:
                merged_data = np.column_stack((x_vals, merged))
                hf.create_dataset("depth_line_by_slice_idx", data=merged_data)
                time_indices = np.arange(idxS, idxS + self.slice_length)
                hf.create_dataset("depth_line_by_time_idx", data=np.column_stack((time_indices, merged)))
                # Save the profile_data for this slice
                hf.create_dataset("profile_data_slice", data=self.image)
            print(f"Edited depth line saved: {h5_path}")
            self.update_whole_record(np.arange(idxS, idxS + self.slice_length), merged)
        self.unbind_all_events()
        self.slice_saved()

    def save_data(self):
        """
        Save the manually drawn depth line.
        The traced line is re-plotted in cyan and both a PNG and HDF5 file are saved.
        After saving, the next slice is automatically loaded.
        If no manual annotations are found, the entire slice is saved as NaN.
        """
        self.clear_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="disabled")
        if self.image is None:
            messagebox.showerror("Error", "No image loaded to save.")
            return
        idxS = self.idx_start
        default_filename = f"{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png"
        # Get interpolated values even if no annotations exist (will be all NaN)
        interp_values = self.interpolate_coordinates()[:self.slice_length, :]
        valid = ~np.isnan(interp_values[:, 1])
        self.clear_annotations()
        if np.any(valid):
            indices = np.where(valid)[0]
            splits = np.where(np.diff(indices) != 1)[0] + 1
            segments = np.split(indices, splits)
            first = True
            for seg in segments:
                if len(seg) >= 2:
                    if first:
                        self.ax.plot(interp_values[seg, 0], interp_values[seg, 1],
                                    color="cyan", linewidth=2, label="Saved Depth Line")
                        first = False
                    else:
                        self.ax.plot(interp_values[seg, 0], interp_values[seg, 1],
                                    color="cyan", linewidth=2)
        else:
            # Plot a full NaN line (optional, may not be visible)
            self.ax.plot(np.arange(self.slice_length), np.full(self.slice_length, np.nan),
                        color="cyan", linewidth=2, label="Saved Depth Line")
        self.ax.legend(loc="upper right")
        # Remove any manual depth line artifacts.
        for line in self.ax.get_lines():
            if line.get_label() == "Manual Depth Line":
                line.remove()
        self.canvas.draw()
        self.root.update()
        self.image_for_saving = Image.fromarray(np.array(self.fig.canvas.renderer.buffer_rgba()))
        save_path = filedialog.asksaveasfilename(initialdir=self.output_folder,
                                                initialfile=default_filename,
                                                defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
        if save_path:
            self.image_for_saving.save(save_path)
            print(f"Image saved: {save_path}")
            h5_path = save_path.replace(".png", ".h5")
            with h5py.File(h5_path, "w") as hf:
                hf.create_dataset("depth_line_by_slice_idx", data=interp_values)
                time_indices = np.arange(idxS, idxS + self.slice_length)
                hf.create_dataset("depth_line_by_time_idx", data=np.column_stack((time_indices, interp_values[:, 1])))
                # Save the profile_data for this slice
                hf.create_dataset("profile_data_slice", data=self.image)
            print(f"Manual depth line saved: {h5_path}")
            self.update_whole_record(np.arange(idxS, idxS + self.slice_length), interp_values[:, 1])
        self.manual_line_saved = True
        self.unbind_all_events()
        self.slice_saved()

    def enable_nan_out_drawing(self):
        """Enable red drawing mode for NaN-ing out values."""
        self.draw_mode = "red"
        messagebox.showinfo("NaN Out Mode", "Drawing enabled in red. Draw over sections to NaN out those values.")

    def save_depth_line(self):
        self.clear_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="disabled")
        if self.depth_option.get() not in ["Smooth Depth", "Ping Depth"]:
            messagebox.showerror("Error", "No depth line selected!")
            return
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            data = None
        if data is None:
            messagebox.showerror("Error", "Depth data not available!")
            return
        idxS = self.idx_start
        x_coords = np.arange(0, self.slice_length)
        depth_coords = data[:self.slice_length]
        # Clear the axes and replot the image and depth line in cyan.
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.plot(x_coords, depth_coords, color="cyan", linewidth=2, label="Saved Depth Line")
        self.ax.legend(loc="upper right")
        self.ax.set_title("")
        self.canvas.draw()
        self.root.update()
        default_filename = f"{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png"
        x0 = self.root.winfo_rootx() + self.canvas_widget.winfo_x()
        y0 = self.root.winfo_rooty() + self.canvas_widget.winfo_y()
        x1 = x0 + self.canvas_widget.winfo_width()
        y1 = y0 + self.canvas_widget.winfo_height()
        self.image_for_saving = ImageGrab.grab((x0, y0, x1, y1))
        save_path = filedialog.asksaveasfilename(initialdir=self.output_folder,
                                                initialfile=default_filename,
                                                defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
        if save_path:
            self.image_for_saving.save(save_path)
            print(f"Image saved: {save_path}")
            h5_path = save_path.replace(".png", ".h5")
            with h5py.File(h5_path, "w") as hf:
                merged_data = np.column_stack((x_coords, depth_coords))
                hf.create_dataset("depth_line_by_slice_idx", data=merged_data)
                time_indices = np.arange(idxS, idxS + self.slice_length)
                hf.create_dataset("depth_line_by_time_idx", data=np.column_stack((time_indices, depth_coords)))
                # Save the profile_data for this slice
                hf.create_dataset("profile_data_slice", data=self.image)
            print(f"Depth line saved: {h5_path}")
            self.update_whole_record(np.arange(idxS, idxS + self.slice_length), depth_coords)
        self.manual_line_saved = True
        self.unbind_all_events()
        self.slice_saved()

    def slice_saved(self):
        if self.slice_number != self.total_slices:
            self.next_slice()
        else:
            response = messagebox.askyesno("All slices annotated", 
                                        "All slices have been annotated. Are you done editing?")
            if response:
                self.root.destroy()
                print("Bottom tracing program complete!")
                return
            
    def update_whole_record(self, time_indices, depth_values):
        # Open (or create) the whole-record file in append mode to avoid locking issues.
        with h5py.File(self.whole_record_file, "a", locking=False) as hf:
            # If "profile_data" does not exist, load it from the raw input file and create the dataset.
            if "profile_data" not in hf:
                with h5py.File(self.file_path, "r", locking=False) as raw_f:
                    full_profile_data = raw_f["profile_data"][:]
                hf.create_dataset("profile_data", data=full_profile_data)
            
            # If the "depth_line_by_time_idx" dataset does not exist, create it.
            if "depth_line_by_time_idx" not in hf:
                full_data = np.column_stack((np.arange(self.total_time), np.full(self.total_time, np.nan)))
                hf.create_dataset("depth_line_by_time_idx", data=full_data, maxshape=(self.total_time, 2))
            else:
                # Otherwise, update the existing dataset with new depth values.
                dset = hf["depth_line_by_time_idx"]
                for i, idx in enumerate(time_indices):
                    dset[idx, 0] = idx
                    dset[idx, 1] = depth_values[i]
        print(f'Whole record updated: {self.whole_record_file}')
                 
if __name__ == "__main__":
    root = tk.Tk()
    app = HDF5Annotator(root)
    root.mainloop()
