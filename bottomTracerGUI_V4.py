import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageGrab
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

"""bottomTracer: A GUI tool for identifying and annotating the bottom line in raw sonar HDF5 file data slices.
Modes:
  • Factory Mode ("Ping Depth" or "Smooth Depth"): Displays an automatic factory depth line (blue) for editing.
  • Manual Tracing Mode ("Off"): Enables manual tracing using left-click and drag (green).
Outputs are saved as PNG and HDF5 files, and the next slice is loaded automatically.
"""

class bottomTracer:
    def __init__(self, root):
        self.root = root
        self.root.title("HDF5 Annotator")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_gui)

        # Initialize state variables
        self.depth_option = tk.StringVar(value="Ping Depth")
        self.manual_line_saved = False
        self.edit_mode = False
        self.draw_mode = "green"
        self.edited_line = None

        # Top Menu Frame
        self.menu_frame = tk.Frame(root, borderwidth=2, relief="groove")
        self.menu_frame.pack(side="top", fill="x", padx=10, pady=10)
        for col in range(4):
            self.menu_frame.grid_columnconfigure(col, weight=1)
        tk.Label(self.menu_frame, text="Chunk Size:").grid(row=0, column=1, padx=0, pady=5)
        self.chunk_size_entry = tk.Entry(self.menu_frame, width=10, justify="center")
        self.chunk_size = 250
        self.chunk_size_entry.insert(0, str(self.chunk_size))
        self.chunk_size_entry.grid(row=0, column=2, padx=0, pady=5)
        tk.Label(self.menu_frame, text="Input File:").grid(row=1, column=1, padx=10, pady=5)
        self.input_file_path = tk.StringVar(value=os.getcwd())
        self.input_dir_entry = tk.Entry(self.menu_frame, textvariable=self.input_file_path, width=25, justify="center")
        self.input_dir_entry.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.browseInput_Button = tk.Button(self.menu_frame, text="Browse", command=self.choose_input_file)
        self.browseInput_Button.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.load_button = tk.Button(self.menu_frame, text="Start Labeling", command=self.load_file, width=30)
        self.load_button.grid(row=3, column=0, padx=0, pady=5, columnspan=4)

        # Annotation Frame & Canvas
        self.annotation_frame = tk.Frame(root)
        self.fig, self.ax = plt.subplots(figsize=(18, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.annotation_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()
        self.canvas_widget.bind("<Destroy>", self.on_canvas_destroy)

        # Jump Navigation
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

        # Y Axis Limits
        self.y_axis_frame = tk.Frame(self.annotation_frame)
        self.y_axis_frame.pack(pady=5)
        tk.Label(self.y_axis_frame, text="Y Axis Limits:").pack(side="left", padx=5)
        self.ymin_entry = tk.Entry(self.y_axis_frame, width=10, justify="center")
        self.ymin_entry.pack(side="left", padx=5)
        tk.Label(self.y_axis_frame, text="-").pack(side="left", padx=5)
        self.ymax_entry = tk.Entry(self.y_axis_frame, width=10, justify="center")
        self.ymax_entry.pack(side="left", padx=5)
        self.y_update_button = tk.Button(self.y_axis_frame, text="Update", command=self.update_y_axis_limits)
        self.y_update_button.pack(side="left", padx=5)

        # Depth Options
        self.depth_frame = tk.Frame(self.annotation_frame)
        self.depth_frame.pack(pady=5)
        tk.Label(self.depth_frame, text="Toggle Factory Line:").grid(row=0, column=0)
        tk.Radiobutton(self.depth_frame, text="Ping Depth",
                    variable=self.depth_option, value="Ping Depth",
                    command=self.update_display).grid(row=0, column=1)
        tk.Radiobutton(self.depth_frame, text="Smooth Depth",
                    variable=self.depth_option, value="Smooth Depth",
                    command=self.update_display).grid(row=0, column=2)
        tk.Radiobutton(self.depth_frame, text="Off",
                    variable=self.depth_option, value="Off",
                    command=self.update_display).grid(row=0, column=3)

        self.nan_button = tk.Button(self.depth_frame, text="Omit Whole Slice",command=self.apply_traced_line)
        self.nan_button.grid(row=0, column=4, padx=(20,0))

        # Control Buttons
        self.button_frame = tk.Frame(self.annotation_frame)
        self.button_frame.pack(fill="both", expand=True)
        self.clear_button = tk.Button(self.button_frame, text="Clear Annotations", command=self.clear_annotations)
        self.clear_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.save_button = tk.Button(self.button_frame, text="", state="normal")
        self.save_button.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.save_depth_button = tk.Button(self.button_frame, text="", state="normal")
        self.save_depth_button.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        self.lasso_button = tk.Button(self.button_frame, text="", command=self.activate_lasso, state="disabled")
        self.lasso_button.grid(row=0, column=3, padx=10, pady=5, sticky="ew")
        self.quit_button = tk.Button(self.button_frame, text="Quit", command=self.quit_gui)
        self.quit_button.grid(row=1, column=0, columnspan=4, padx=10, pady=5, sticky="ew")
        for col in range(4):
            self.button_frame.grid_columnconfigure(col, weight=1)

        # Internal Variables
        self.base_name = None
        self.whole_record_file = None
        self.idx_start = 0
        self.data_blanking_distance_cm = 5
        self.image = None
        self.smooth_depth = None
        self.length_mm = None
        self.this_ping_depth_m = None
        self.bin_size = None
        self.smooth_depth_img = None
        self.this_ping_depth_img = None
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.coordinates = []
        self.image_for_saving = None
        self.total_slices = None
        self.total_time = None
        self.applied_line = None
        self.slice_number = None
        self.slice_length = None
        self.ymin = None
        self.ymax = None
        self.lasso = None
        self.lasso_selected = np.zeros(0,dtype=bool)
        
    def on_canvas_destroy(self, event):
        if not self.canvas_widget.winfo_exists():
            self.quit_gui()

    def choose_input_file(self):
        file_path = filedialog.askopenfilename(initialdir=os.getcwd(),
                                               filetypes=[("HDF5 files", "*.h5 *.hdf5")])
        if file_path:
            self.input_file_path.set(file_path)

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
        self.input_file_path = self.input_file_path.get()
        input_file = os.path.basename(self.input_file_path)
        self.base_name = os.path.splitext(input_file)[0]
        
        # always use the raw‐data folder as output folder
        raw_dir = os.path.dirname(self.input_file_path)
        self.output_folder = raw_dir
        # whole‐record HDF5 stays in the raw data directory
        self.whole_record_file = os.path.join(raw_dir, f"{self.base_name}_bottomTraced_wholeRecord.h5")
        self.menu_frame.pack_forget()
        self.annotation_frame.pack(fill="both", expand=True)
        self.idx_start = 0
        self.depth_option.set("Ping Depth")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        for child in self.y_axis_frame.winfo_children():
            child.config(state="normal")
        self.process_next_chunk()

    def process_next_chunk(self):
        if not self.input_file_path:
            return
        with h5py.File(self.input_file_path, "r") as f:
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
                self.this_ping_depth_m = f["this_ping_depth_m"][idx] if "this_ping_depth_m" in f else None
            else:
                self.smooth_depth = None
                self.length_mm = None
                self.this_ping_depth_m = None
                self.bin_size = None
        if self.bin_size is not None:
            self.smooth_depth_img = self.smooth_depth / self.bin_size if self.smooth_depth is not None else None
            self.this_ping_depth_img = self.this_ping_depth_m / self.bin_size if self.this_ping_depth_m is not None else None
            # Mask out shallow values
            if self.smooth_depth_img is not None:
                self.smooth_depth_img[self.smooth_depth < (self.data_blanking_distance_cm/100)] = np.nan
            if self.this_ping_depth_img is not None:
                self.this_ping_depth_img[self.this_ping_depth_m < (self.data_blanking_distance_cm/100)] = np.nan
        else:
            self.smooth_depth_img = None
            self.this_ping_depth_img = None
        self.edit_mode = False
        self.update_display()

    def update_display(self):
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.slice_length = self.image.shape[1]
        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)
        self.ax.set_xlim(0, self.slice_length)
        self.ax.set_ylim(0, self.image.shape[0])
        self.sync_y_axis_entries()
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        if self.depth_option.get() != "Off" and (self.smooth_depth_img is not None or self.this_ping_depth_img is not None):
            self.plot_depth()
        for txt in self.fig.texts[:]:
            txt.remove()
        idxS = self.idx_start
        idxE = min(self.idx_start + self.chunk_size - 1, self.total_time - 1)
        self.fig.text(0.01, 0.98, f"Slice #{self.slice_number} of {self.total_slices}\nTime Indices: {idxS} - {idxE}",
                      horizontalalignment="left", verticalalignment="top", fontsize=12, color="black")
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.enable_annotation()
        self.fig.canvas.draw()
        image_array = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.image_for_saving = Image.fromarray(image_array)
        if self.depth_option.get() != "Off" and not self.edit_mode:
            self.clear_button.grid_remove()
        self.update_button_states()
        self.jump_slice_var.set(str(self.slice_number))

    def add_secondary_y_axis(self):
        for ax in self.fig.axes:
            if ax is not self.ax and ax.get_label() == "secondary_y":
                self.fig.delaxes(ax)
        
        if self.bin_size is not None:
            ax2 = self.ax.twinx()
            ax2.set_ylim(self.ax.get_ylim()[0] * self.bin_size, self.ax.get_ylim()[1] * self.bin_size)
            ax2.set_ylabel("Depth Range (m)", fontsize=15)
            ax2.tick_params(axis='y', labelsize=14)
            ax2.set_label("secondary_y")  # Set label to identify later

    def update_button_states(self):
        if self.depth_option.get() == "Off":
            self.clear_button.grid()
            self.clear_button.config(text="Clear Annotations", command=self.clear_annotations, state="normal")
            self.save_button.config(text="Apply Traced Line", command=self.apply_traced_line, state="normal")
            self.save_depth_button.config(text="Save Manual Depth Line", command=self.save_data, state="disabled")
            self.lasso_button.config(text="", command=None, state="disabled")
            self.ax.set_title("Manual Tracing Mode Enabled:\nLeft click to trace the depth line in green.", fontsize=16)
        else:
            if not self.edit_mode:
                self.clear_button.grid_remove()
                self.save_button.config(text=f"Save {self.depth_option.get()} Line", command=self.save_depth_line, state="normal")
                self.save_depth_button.config(text=f"Edit {self.depth_option.get()} Line", command=self.enter_editing_mode, state="normal")
                self.lasso_button.config(text=f"Clean {self.depth_option.get()} Line", command=self.activate_lasso, state="normal")
            else:
                self.clear_button.grid()
                self.clear_button.config(text="Clear Annotations", command=self.clear_edit_mode, state="normal")
                self.save_button.config(text="Apply Edits", command=self.apply_edits, state="normal")
                self.save_depth_button.config(text="Save Edited Depth Line", command=self.save_edited_depth_line, state="disabled")
                self.lasso_button.config(text="", command=None, state="disabled")
        self.next_button_jump.config(state="normal" if self.slice_number < self.total_slices else "disabled")
        self.prev_button_jump.config(state="normal" if self.slice_number > 1 else "disabled")
        self.canvas.draw()
        
    def update_y_axis_limits(self):
        try:
            self.ymin = float(self.ymin_entry.get())
            self.ymax = float(self.ymax_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values for Y Axis Limits.")
            return
        num_results = self.image.shape[0] if self.image is not None else 0
        if self.ymin < 0 or self.ymax < 0 or self.ymin > num_results or self.ymax > num_results:
            messagebox.showerror("Invalid Input", f"Values must be between 0 and {num_results}.")
            return
        if self.ymin > self.ymax:
            messagebox.showerror("Invalid Input", "Lower limit cannot be greater than upper limit.")
            return
        if (self.ymax - self.ymin) < 100:
            messagebox.showerror("Invalid Input", "The difference between the limits must be at least 100.")
            return
        self.ax.set_ylim(self.ymin, self.ymax)
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.sync_y_axis_entries()

    def sync_y_axis_entries(self):
        ymin, ymax = self.ax.get_ylim()
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        self.ymin_entry.config(state="normal")
        self.ymax_entry.config(state="normal")
        self.ymin_entry.delete(0, tk.END)
        self.ymin_entry.insert(0, str(int(ymin)))
        self.ymax_entry.delete(0, tk.END)
        self.ymax_entry.insert(0, str(int(ymax)))
        self.add_secondary_y_axis()

    def plot_depth(self):
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            return
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
        self.depth_option.set("Ping Depth")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        for child in self.y_axis_frame.winfo_children():
            child.config(state="normal")
        self.idx_start -= self.chunk_size
        self.process_next_chunk()

    def next_slice(self):
        self.clear_annotations()
        self.manual_line_saved = False
        self.edit_mode = False
        self.depth_option.set("Ping Depth")
        self.y_update_button.config(state="normal")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        for child in self.y_axis_frame.winfo_children():
            child.config(state="normal")
        self.idx_start += self.chunk_size
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
        self.depth_option.set("Ping Depth")
        for child in self.depth_frame.winfo_children():
            child.config(state="normal")
        for child in self.y_axis_frame.winfo_children():
            child.config(state="normal")
        self.process_next_chunk()

    def quit_gui(self):
        sys.exit()

    def unbind_all_events(self):
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
        x_offset = self.canvas_widget.winfo_rootx() - self.root.winfo_rootx()
        y_offset = self.canvas_widget.winfo_rooty() - self.root.winfo_rooty()
        fig_x = x - x_offset
        fig_y = y - y_offset
        fig_y = self.canvas_widget.winfo_height() - fig_y
        data_x, data_y = self.ax.transData.inverted().transform((fig_x, fig_y))
        return data_x, data_y

    def start_tracing(self, event):
        self.tracing = True
        for child in self.y_axis_frame.winfo_children():
            child.config(state="disabled")
        self.last_x, self.last_y = event.x, event.y
        self.ymin_entry.config(state="disabled")
        self.ymax_entry.config(state="disabled")
        self.y_update_button.config(state="disabled")
        self.coordinates.append((event.x, event.y, *self.canvas_to_data(event.x, event.y), "green"))
        if self.depth_option.get() == "Off":
            self.save_button.config(text="Apply Traced Line", command=self.apply_traced_line, state="normal")

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

    def start_tracing_editing_green(self, event):
        self.tracing = True
        for child in self.y_axis_frame.winfo_children():
            child.config(state="disabled")
        self.last_x, self.last_y = event.x, event.y
        self.ymin_entry.config(state="disabled")
        self.ymax_entry.config(state="disabled")
        self.y_update_button.config(state="disabled")
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
        for child in self.y_axis_frame.winfo_children():
            child.config(state="disabled")
        self.last_x, self.last_y = event.x, event.y
        self.ymin_entry.config(state="disabled")
        self.ymax_entry.config(state="disabled")
        self.y_update_button.config(state="disabled")
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
        x_coords = np.arange(self.chunk_size)
        filtered = [pt for pt in self.coordinates if pt[4] == color]
        if not filtered:
            return np.column_stack((x_coords, np.full(self.chunk_size, np.nan)))
        pts = sorted([(pt[2], pt[3]) for pt in filtered], key=lambda p: p[0])
        y_interp = np.full(self.chunk_size, np.nan)
        for x in x_coords:
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                if (x1 - x0) <= 2.0 and x0 <= x <= x1:
                    y_interp[x] = np.interp(x, [x0, x1], [y0, y1])
                    break
        return np.column_stack((x_coords, y_interp))

    def interpolate_coordinates(self):
        x_coords = np.arange(self.chunk_size)
        if not self.coordinates:
            return np.column_stack((x_coords, np.full(self.slice_length, float(-999))))
        pts = sorted([(pt[2], pt[3]) for pt in self.coordinates], key=lambda p: p[0])
        y_interp = np.full(self.chunk_size, np.nan)
        for x in x_coords:
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                if (x1 - x0) <= 2.0 and x0 <= x <= x1:
                    y_interp[x] = np.interp(x, [x0, x1], [y0, y1])
                    break
        return np.column_stack((x_coords, y_interp))

    def clear_annotations(self):
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.canvas_widget.delete("annotation")
        if self.applied_line:
            self.update_display()

    def clear_edit_mode(self):
        self.clear_annotations()
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.set_ylim(self.ymin, self.ymax)
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            data = None
        if data is not None:
            x = np.arange(0, self.chunk_size)
            self.ax.plot(x, data, color="blue", linewidth=2, alpha=0.35, label=self.depth_option.get())
        self.ax.set_title("Editing Mode Enabled:\nLeft click to draw edits (green), Right click to omit data (red).", fontsize=16)
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        self.canvas.draw()
        self.unbind_all_events()
        self.canvas_widget.bind("<Button-1>", self.start_tracing_editing_green)
        self.canvas_widget.bind("<B1-Motion>", self.trace_line_editing_green)
        self.canvas_widget.bind("<Button-3>", self.start_tracing_editing_red)
        self.canvas_widget.bind("<B3-Motion>", self.trace_line_editing_red)
        self.canvas_widget.bind("<ButtonRelease-1>", self.stop_tracing)
        self.canvas_widget.bind("<ButtonRelease-3>", self.stop_tracing)
        self.update_button_states()

    def activate_lasso(self):
        if self.edit_mode or self.depth_option.get() == "Off":
            messagebox.showwarning("Unavailable", "Lasso cleaning is only available before editing Ping/Smooth Depth lines.")
            return

        if self.depth_option.get() == "Ping Depth":
            self.depth_data = self.this_ping_depth_img[:self.slice_length]
        else:
            self.depth_data = self.smooth_depth_img[:self.slice_length]

        self.points = np.column_stack((np.arange(self.slice_length), self.depth_data))

        def onselect(verts):
            path = Path(verts)
            selected = path.contains_points(self.points)
            for i, sel in enumerate(selected):
                if sel and not np.isnan(self.points[i, 1]):
                    self.depth_data[i] = np.nan  # immediately remove from depth line
            self.points[:, 1] = self.depth_data
            for line in self.ax.lines[:]:
                line.remove()
            valid_mask = ~np.isnan(self.points[:, 1])
            self.ax.plot(self.points[valid_mask, 0], self.points[valid_mask, 1], 'bo', markersize=4, label=self.depth_option.get())
            self.canvas.draw()
            self.start_lasso()  # restart lasso selector 
            
        def finish_lasso():
            if self.lasso:
                self.lasso.disconnect_events()
                self.lasso = None
            for pt in self.coordinates:
                if pt[4] == "red":
                    x = int(pt[2])
                    self.depth_data[x] = np.nan
            self.apply_edits()
            self.lasso_button.config(text=f"Clean {self.depth_option.get()} Line", command=self.activate_lasso)

        def start_lasso():
            if self.lasso:
                self.lasso.disconnect_events()
            self.ax.set_title("Lasso Tool Active: Draw around points to remove. Press 'Finish Cleaning' when done.", fontsize=16)
            self.lasso = LassoSelector(self.ax,onselect,props=dict(color='red'))
            self.lasso.set_active(True)
            self.canvas.draw()

        self.start_lasso = start_lasso
        self.finish_lasso = finish_lasso

        # Convert depth line to scatter points
        for line in self.ax.lines[:]:
            if line.get_label() == self.depth_option.get():
                line.remove()
        self.ax.plot(self.points[:, 0], self.points[:, 1], 'bo', markersize=4, label=self.depth_option.get())

        self.lasso_button.config(text="Finish Cleaning", command=self.finish_lasso)
        self.start_lasso()

  
    def enter_editing_mode(self):
        self.enable_annotation()
        self.edit_mode = True
        self.draw_mode = "green"
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        if self.ymin is not None and self.ymax is not None:
            self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            data = None
        if data is not None:
            x = np.arange(0, self.slice_length)
            self.ax.plot(x, data, color="blue", linewidth=2, alpha=0.35, label=self.depth_option.get())
        self.ax.set_title("Editing Mode Enabled:\nLeft click to draw edits (green), Right click to omit data (red).", fontsize=16)
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        self.sync_y_axis_entries()
        self.add_secondary_y_axis()
        self.canvas.draw()
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
        self.lasso_button.config(state="normal" if self.edit_mode else "disabled")

    def apply_edits(self):
        num_results = self.image.shape[0]
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
        # self.ax.clear()
        # self.ax.pcolormesh(self.image, cmap="plasma")
        # self.ax.plot(x_vals, merged, color="blue", linewidth=2, label=self.depth_option.get())
        # self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)
        # self.ax.legend(loc="upper right")
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        # split merged line into original vs. manual points
        # green_edit is the user‐drawn override (NaN where untouched)
        green_mask = ~np.isnan(green_edit)          # where user drew new points
        orig_mask  = np.isnan(green_edit)           # untouched, factory points
        # plot factory/original depth‐line points in blue
        orig_x = x_vals[orig_mask]
        orig_y = merged[orig_mask]
        segments = np.split(np.column_stack((orig_x, orig_y)), np.where(np.diff(orig_x) > 1)[0] + 1)
        for i, seg in enumerate(segments):
            if len(seg) > 0:
                self.ax.plot(seg[:, 0], seg[:, 1],
                            linestyle='-', color="blue", linewidth=2,
                            label="Original Depth Line" if i == 0 else "_nolegend_")
        # plot overridden points in green
        manual_x = x_vals[green_mask]
        manual_y = merged[green_mask]
        segments = np.split(np.column_stack((manual_x, manual_y)), np.where(np.diff(manual_x) > 1)[0] + 1)
        for i, seg in enumerate(segments):
            if len(seg) > 0:
                self.ax.plot(seg[:, 0], seg[:, 1],
                            linestyle='-', color="green", linewidth=2,
                            label="Manual Edits" if i == 0 else "_nolegend_")
        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)
        self.ax.legend(loc="upper right")
        self.ax.set_title("")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        self.ax.set_ylim(0, num_results)
        self.sync_y_axis_entries()
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.clear_annotations()
        self.unbind_all_events()
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="normal")

    def apply_traced_line(self):
        if not self.coordinates:
            self.canvas_widget.delete("annotation")
            self.ax.clear()
            self.ax.pcolormesh(self.image, cmap="plasma")
            self.ax.set_xlim(0, self.slice_length)
            self.ax.set_ylim(0, self.image.shape[0])
            self.ax.set_xlabel("Ping Count", fontsize=15)
            self.ax.set_ylabel("Bin #", fontsize=15)
            self.ax.tick_params(axis='x', labelsize=14)
            self.ax.tick_params(axis='y', labelsize=14)
            self.canvas.draw()
            self.save_data()
            return
        traced_values = self.interpolate_coordinates_by_color("green")[:self.slice_length, 1]
        self.canvas_widget.delete("annotation")
        x_vals = np.arange(self.slice_length)
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.plot(x_vals, traced_values, color="blue", linewidth=2, label="Manual Depth Line")
        self.ax.legend(loc="upper right")
        self.ax.set_title("")
        self.ax.set_xlabel("Time (s)", fontsize = 15)
        self.ax.set_ylabel("Bin #", fontsize = 15)
        self.ax.tick_params(axis='x', labelsize=14)  
        self.ax.tick_params(axis='y', labelsize=14)
        self.add_secondary_y_axis()
        self.canvas.draw()
        for child in self.depth_frame.winfo_children():
            child.config(state="disabled")
        self.unbind_all_events()
        self.applied_line = True
        self.save_button.config(state="disabled")
        self.save_depth_button.config(text="Save Manual Depth Line", command=self.save_data, state="normal")

    def save_edited_depth_line(self):
        if self.edited_line is None:
            messagebox.showerror("Error", "No edited depth line available!")
            return
        num_results = self.image.shape[0]
        self.ax.set_ylim(0, num_results)
        self.canvas.draw()
        merged = self.edited_line
        idxS = self.idx_start
        x_vals = np.arange(self.slice_length)
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.plot(x_vals, merged, color="cyan", linewidth=2, label="Saved Depth Line")
        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)
        self.ax.set_xlim(0, self.slice_length)
        self.ax.set_ylim(0, self.image.shape[0])
        self.ax.legend(loc="upper right")
        self.ax.set_title("")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.root.update()
        default_filename = f"{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png"
        x0 = self.root.winfo_rootx() + self.canvas_widget.winfo_x()
        y0 = self.root.winfo_rooty() + self.canvas_widget.winfo_y()
        x1 = x0 + self.canvas_widget.winfo_width()
        y1 = y0 + self.canvas_widget.winfo_height()
        self.image_for_saving = ImageGrab.grab((x0, y0, x1, y1))
        x0 = self.root.winfo_rootx() + self.canvas_widget.winfo_x()
        y0 = self.root.winfo_rooty() + self.canvas_widget.winfo_y()
        x1 = x0 + self.canvas_widget.winfo_width()
        y1 = y0 + self.canvas_widget.winfo_height()
        self.image_for_saving = ImageGrab.grab((x0, y0, x1, y1))
        raw_dir = os.path.dirname(self.input_file_path)
        qcplots = os.path.join(raw_dir, "qcPlots")
        os.makedirs(qcplots, exist_ok=True)
        png_path = os.path.join(qcplots, default_filename)
        self.image_for_saving.save(png_path)
        print(f"Image saved: {os.path.normpath(png_path)}")
        qcddata = os.path.join(raw_dir, "qcdData")
        os.makedirs(qcddata, exist_ok=True)
        h5_name = default_filename.replace(".png", ".h5")
        h5_path = os.path.join(qcddata, h5_name)
        with h5py.File(h5_path, "w") as hf:
            merged_data = np.column_stack((x_vals, merged))
            hf.create_dataset("depth_line_by_slice_idx", data=merged_data)
            time_indices = np.arange(idxS, idxS + self.slice_length)
            hf.create_dataset("depth_line_by_time_idx",data=np.column_stack((time_indices, merged)))
            hf.create_dataset("profile_data_slice", data=self.image)
        print(f"Depth line saved: {os.path.normpath(h5_path)}")
        self.update_whole_record(np.arange(idxS, idxS + self.slice_length), merged)
        self.unbind_all_events()
        self.slice_saved()

    def save_data(self):
        self.clear_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="disabled")
        num_results = self.image.shape[0]
        self.ax.set_ylim(0, num_results)
        idxS = self.idx_start
        default_filename = f"{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png"
        self.ax.set_title(default_filename, fontsize=16)
        self.canvas.draw()
        self.sync_y_axis_entries()
        if self.image is None:
            messagebox.showerror("Error", "No image loaded to save.")
            return
        
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
                        self.ax.plot(interp_values[seg, 0], interp_values[seg, 1],color="cyan", linewidth=2, label="Saved Depth Line")
                        first = False
                    else:
                        self.ax.plot(interp_values[seg, 0], interp_values[seg, 1], color="cyan", linewidth=2)
        else:
            self.ax.plot(np.arange(self.slice_length), np.full(self.slice_length, float(-999)),color="cyan", linewidth=2, label="Saved Depth Line")
        self.ax.legend(loc="upper right")
        for line in self.ax.get_lines():
            if line.get_label() == "Manual Depth Line":
                line.remove()
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.root.update()
        self.image_for_saving = Image.fromarray(np.array(self.fig.canvas.renderer.buffer_rgba()))
        raw_dir = os.path.dirname(self.input_file_path)
        qcplots = os.path.join(raw_dir, "qcPlots")
        os.makedirs(qcplots, exist_ok=True)
        png_path = os.path.join(qcplots, default_filename)
        self.image_for_saving.save(png_path)
        print(f"Image saved: {os.path.normpath(png_path)}")
        qcddata = os.path.join(raw_dir, "qcdData")
        os.makedirs(qcddata, exist_ok=True)
        h5_name = default_filename.replace(".png", ".h5")
        h5_path = os.path.join(qcddata, h5_name)
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("depth_line_by_slice_idx", data=interp_values)
            time_indices = np.arange(idxS, idxS + self.slice_length)
            hf.create_dataset("depth_line_by_time_idx",data=np.column_stack((time_indices, interp_values[:, 1])))
            hf.create_dataset("profile_data_slice", data=self.image)
        print(f"Manual depth line saved: {os.path.normpath(h5_path)}")
        self.update_whole_record(np.arange(idxS, idxS + self.slice_length), interp_values[:, 1])
        self.manual_line_saved = True
        self.unbind_all_events()
        self.slice_saved()

    def save_depth_line(self):
        self.clear_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="disabled")
        num_results = self.image.shape[0]
        self.ax.set_ylim(0, num_results)
        self.canvas.draw()
        self.sync_y_axis_entries()
        if self.depth_option.get() not in ["Smooth Depth", "Ping Depth"]:
            messagebox.showerror("Error", "No depth line selected!")
            return
        data = self.smooth_depth_img if self.depth_option.get() == "Smooth Depth" else self.this_ping_depth_img
        if data is None:
            messagebox.showerror("Error", "Depth data not available!")
            return
        idxS = self.idx_start
        x_coords = np.arange(0, self.slice_length)
        depth_coords = data[:self.slice_length]
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap="plasma")
        self.ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)
        self.ax.plot(x_coords, depth_coords, color="cyan", linewidth=2, label="Saved Depth Line")
        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("Ping Count", fontsize=15)
        self.ax.set_ylabel("Bin #", fontsize=15)
        self.ax.tick_params(axis='x', labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)
        default_filename = f"{self.base_name}_bottomTraced_{idxS}-{idxS + self.slice_length - 1}.png"
        self.ax.set_title(default_filename, fontsize=16)
        self.add_secondary_y_axis()
        self.canvas.draw()
        self.root.update()
        x0 = self.root.winfo_rootx() + self.canvas_widget.winfo_x()
        y0 = self.root.winfo_rooty() + self.canvas_widget.winfo_y()
        x1 = x0 + self.canvas_widget.winfo_width()
        y1 = y0 + self.canvas_widget.winfo_height()
        self.image_for_saving = ImageGrab.grab((x0, y0, x1, y1))
        raw_dir = os.path.dirname(self.input_file_path)
        qcplots = os.path.join(raw_dir, "qcPlots")
        os.makedirs(qcplots, exist_ok=True)
        png_path = os.path.join(qcplots, default_filename)
        self.image_for_saving.save(png_path)
        print(f"Image saved: {os.path.normpath(png_path)}")
        qcddata = os.path.join(raw_dir, "qcdData")
        os.makedirs(qcddata, exist_ok=True)
        h5_name = default_filename.replace(".png", ".h5")
        h5_path = os.path.join(qcddata, h5_name)
        with h5py.File(h5_path, "w") as hf:
            merged_data = np.column_stack((x_coords, depth_coords))
            hf.create_dataset("depth_line_by_slice_idx", data=merged_data)
            time_indices = np.arange(idxS, idxS + self.slice_length)
            hf.create_dataset("depth_line_by_time_idx",data=np.column_stack((time_indices, depth_coords)))
            hf.create_dataset("profile_data_slice", data=self.image)
        print(f"Depth line saved: {os.path.normpath(h5_path)}")
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
                self.quit_gui()
                print("Bottom tracing program complete!")
                return

    def update_whole_record(self, time_indices, depth_values):
        with h5py.File(self.whole_record_file, "a", locking=False) as hf:
            if "qaqc_depth_line" not in hf:
                full_data = np.column_stack((np.arange(self.total_time), np.full(self.total_time, float(-999))))
                hf.create_dataset("qaqc_depth_line", data=full_data, maxshape=(self.total_time, 2))
            dset = hf["qaqc_depth_line"]
            for i, idx in enumerate(time_indices):
                dset[idx, 0] = idx
                dset[idx, 1] = depth_values[i]
        print(f'Whole record updated: {os.path.normpath(self.whole_record_file)}')
            
if __name__ == "__main__":
    root = tk.Tk()
    app = bottomTracer(root)
    root.mainloop()
    print(app)