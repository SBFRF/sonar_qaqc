import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageGrab

class HDF5Annotator:
    def __init__(self, root):
        self.root = root
        self.root.title("HDF5 Annotator")
        
        # New: Depth option variable with default value "Smooth"
        self.depth_option = tk.StringVar(value="Smooth")
        
        # Initialize toggle flag: True uses smooth_depth_img, False uses this_ping_depth_img.
        self.use_smooth_depth_img = True

        # Initial menu frame with only the "Load HDF5 File" button
        self.menu_frame = tk.Frame(root)
        self.menu_frame.pack(fill="both", expand=True)
        self.load_button = tk.Button(self.menu_frame, text="Load HDF5 File", command=self.load_file)
        self.load_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.menu_frame.grid_columnconfigure(0, weight=1)
        self.menu_frame.grid_rowconfigure(0, weight=1)
        
        # Annotation frame (hidden until a file is loaded)
        self.annotation_frame = tk.Frame(root)
        
        # Create the Matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(12,8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.annotation_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()
        
        # New Limits Frame for updating axis limits
        self.chunk_size = 1000  # Used to load data slices
        self.limits_frame = tk.Frame(self.annotation_frame)
        self.limits_frame.pack(pady=5)
        # Upper X Limit (lower x is fixed at 0)
        tk.Label(self.limits_frame, text="Upper X Limit:").pack(side="left")
        self.x_limit_entry = tk.Entry(self.limits_frame, width=10)
        self.x_limit_entry.insert(0, "1000")  # Default value
        self.x_limit_entry.pack(side="left", padx=5)
        # Lower Y Limit
        tk.Label(self.limits_frame, text="Lower Y Limit:").pack(side="left")
        self.y_lower_entry = tk.Entry(self.limits_frame, width=10)
        self.y_lower_entry.insert(0, "0")  # Default value
        self.y_lower_entry.pack(side="left", padx=5)
        # Upper Y Limit – default will be updated to num_results from the file
        tk.Label(self.limits_frame, text="Upper Y Limit:").pack(side="left")
        self.y_upper_entry = tk.Entry(self.limits_frame, width=10)
        self.y_upper_entry.insert(0, str(self.chunk_size))  # Placeholder; will be updated
        self.y_upper_entry.pack(side="left", padx=5)
        # Button to update axis limits
        tk.Button(self.limits_frame, text="Update Limits", command=self.update_limits).pack(side="left", padx=5)
        
        # New Depth Options Frame for selecting depth option
        self.depth_frame = tk.Frame(self.annotation_frame)
        self.depth_frame.pack(pady=5)
        tk.Label(self.depth_frame, text="Toggle Depth Option:").pack(side="left", padx=5)
        tk.Radiobutton(self.depth_frame, text="Smooth", variable=self.depth_option, value="Smooth Depth",
                       command=self.update_display).pack(side="left", padx=5)
        tk.Radiobutton(self.depth_frame, text="Ping", variable=self.depth_option, value="Ping Depth",
                       command=self.update_display).pack(side="left", padx=5)
        tk.Radiobutton(self.depth_frame, text="Off", variable=self.depth_option, value="Off",
                       command=self.update_display).pack(side="left", padx=5)
        
        # Frame for control buttons
        self.button_frame = tk.Frame(self.annotation_frame)
        self.button_frame.pack(fill="both", expand=True)
        
        # First row of buttons
        self.clear_button = tk.Button(self.button_frame, text="Clear Annotations", command=self.clear_annotations)
        self.clear_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.save_button = tk.Button(self.button_frame, text="Save Image & Annotations", command=self.save_data)
        self.save_button.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        # Second row of buttons
        self.prev_button = tk.Button(self.button_frame, text="Previous Slice", command=self.prev_slice)
        self.prev_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.next_button = tk.Button(self.button_frame, text="Next Slice", command=self.next_slice)
        self.next_button.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.quit_button = tk.Button(self.button_frame, text="Quit", command=self.quit_gui)
        self.quit_button.grid(row=1, column=2, padx=10, pady=5, sticky="ew")
        
        # Configure grid weights for uniform button sizes
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure(1, weight=1)
        
        # Initialize variables
        self.file_path = None
        self.idx_start = 0
        self.image = None
        self.smooth_depth = None  # Will hold the smooth_depth_m variable (if present)
        self.length_mm = None     # Holds the length_mm variable (if present)
        self.this_ping_depth_m = None  # Holds the this_ping_depth_m variable (if present)
        self.bin_size = None      # Bin size (in meters per row)
        self.smooth_depth_img = None  # smooth_depth_m divided by bin_size
        self.this_ping_depth_img = None  # this_ping_depth_m divided by bin_size
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.coordinates = []
        self.image_for_saving = None
        # User-specified axis limits (if updated)
        self.user_x_max = None
        self.user_y_min = None
        self.user_y_max = None

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5 *.hdf5")])
        if not self.file_path:
            return
        self.menu_frame.pack_forget()
        self.annotation_frame.pack(fill="both", expand=True)
        self.idx_start = 0  # Reset index for new file
        self.process_next_chunk()

    def update_limits(self):
        try:
            new_x_max = float(self.x_limit_entry.get())
            new_y_min = float(self.y_lower_entry.get())
            new_y_max = float(self.y_upper_entry.get())
            if new_x_max <= 0:
                raise ValueError("Upper X limit must be positive.")
            if new_x_max > 1000:
                raise ValueError("Upper X limit cannot exceed 1000.")
            if new_y_max <= new_y_min:
                raise ValueError("Upper Y limit must be greater than Lower Y limit.")
            if self.image is not None and new_y_max > self.image.shape[0]:
                raise ValueError(f"Upper Y limit cannot exceed {self.image.shape[0]} (the number of rows in the dataset).")
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        self.user_x_max = new_x_max
        self.user_y_min = new_y_min
        self.user_y_max = new_y_max
        self.ax.set_xlim(0, self.user_x_max)
        self.ax.set_ylim(self.user_y_min, self.user_y_max)
        self.canvas.draw()

    def process_next_chunk(self):
        if not self.file_path:
            return
        with h5py.File(self.file_path, 'r') as f:
            if self.idx_start >= f['time'].shape[0]:
                print("All data processed.")
                return
            idx = slice(self.idx_start, min(self.idx_start + self.chunk_size, f['time'].shape[0]))
            self.image = f['profile_data'][:, idx]
            if 'smooth_depth_m' in f:
                self.smooth_depth = f['smooth_depth_m'][idx]
                # Load length_mm if available and compute bin_size (convert mm to m)
                if 'length_mm' in f:
                    self.length_mm = f['length_mm'][idx]
                    # Assume length_mm is constant for the slice; convert first element from mm to m,
                    # then divide by number of rows (num_results)
                    self.bin_size = (self.length_mm[0] / 1000.0) / self.image.shape[0]
                else:
                    self.length_mm = None
                    self.bin_size = None
                # Load this_ping_depth_m if available
                if 'this_ping_depth_m' in f:
                    self.this_ping_depth_m = f['this_ping_depth_m'][idx]
                else:
                    self.this_ping_depth_m = None
            else:
                self.smooth_depth = None
                self.length_mm = None
                self.this_ping_depth_m = None
                self.bin_size = None
        # Calculate new variables if bin_size is available
        if self.bin_size is not None:
            self.smooth_depth_img = self.smooth_depth / self.bin_size if self.smooth_depth is not None else None
            self.this_ping_depth_img = self.this_ping_depth_m / self.bin_size if self.this_ping_depth_m is not None else None
        else:
            self.smooth_depth_img = None
            self.this_ping_depth_img = None
        self.update_display()

    def update_display(self):
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap='plasma')
        self.ax.set_title("Click and drag to trace. Start and stop as needed.")
        # Set axis limits using user-specified limits if available; otherwise defaults.
        if self.user_x_max is not None:
            self.ax.set_xlim(0, self.user_x_max)
        else:
            self.ax.set_xlim(0, self.chunk_size)
        if self.user_y_min is not None and self.user_y_max is not None:
            self.ax.set_ylim(self.user_y_min, self.user_y_max)
        else:
            self.ax.set_ylim(0, self.image.shape[0])
            # Update the upper Y limit entry to match the dataset's number of rows.
            self.y_upper_entry.delete(0, tk.END)
            self.y_upper_entry.insert(0, str(self.image.shape[0]))
        # Instead of plotting depth unconditionally, check the selection:
        if self.depth_option.get() != "Off" and ((self.smooth_depth_img is not None) or (self.this_ping_depth_img is not None)):
            self.plot_depth()
        self.canvas.draw()
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.enable_annotation()
        self.fig.canvas.draw()
        image_array = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.image_for_saving = Image.fromarray(image_array)

    def plot_depth(self):
        if self.depth_option.get() == "Smooth":
            data = self.smooth_depth_img
            label = "Smooth Depth"
        elif self.depth_option.get() == "Ping":
            data = self.this_ping_depth_img
            label = "Ping Depth"
        else:
            return  # "Off" selected: do nothing.
        # Use an integer range from 0 to chunk_size - 1 as x-axis.
        x = np.arange(0, self.chunk_size)
        if data is not None:
            self.ax.plot(x, data, color='cyan', linewidth=2, label=label)
            self.ax.legend(loc='upper right')
            self.canvas.draw()

    def prev_slice(self):
        self.clear_annotations()
        if self.idx_start - self.chunk_size >= 0:
            self.idx_start -= self.chunk_size
            # Reset axis limits to default values:
            self.user_x_max = None
            self.user_y_min = None
            self.user_y_max = None
            self.process_next_chunk()
                
    def next_slice(self):
        self.clear_annotations()
        self.idx_start += self.chunk_size
        # Reset axis limits to default values:
        self.user_x_max = None
        self.user_y_min = None
        self.user_y_max = None
        self.process_next_chunk()

    def quit_gui(self):
        self.root.destroy()
        
    def enable_annotation(self):
        self.canvas_widget.bind("<Button-1>", self.start_tracing)
        self.canvas_widget.bind("<B1-Motion>", self.trace_line)
        self.canvas_widget.bind("<Button-3>", self.stop_tracing)
        
    def start_tracing(self, event):
        self.tracing = True
        self.last_x, self.last_y = event.x, event.y
        self.coordinates.append((event.x, event.y, *self.canvas_to_data(event.x, event.y)))
        
    def canvas_to_data(self, x, y):
        x_offset = self.canvas_widget.winfo_rootx() - self.root.winfo_rootx()
        y_offset = self.canvas_widget.winfo_rooty() - self.root.winfo_rooty()
        fig_x = x - x_offset
        fig_y = y - y_offset
        fig_y = self.canvas_widget.winfo_height() - fig_y  
        data_x, data_y = self.ax.transData.inverted().transform((fig_x, fig_y))
        return data_x, data_y
    
    def trace_line(self, event):
        if self.tracing:
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            if not (x_min <= data_x <= x_max and y_min <= data_y <= y_max):
                self.stop_tracing(event)
                return
            self.canvas_widget.create_line((self.last_x, self.last_y, event.x, event.y),
                                             fill='red', width=2, tags='annotation')
            print(f"Tkinter: ({event.x}, {event.y}), Data: ({data_x:.2f}, {data_y:.2f})")
            if not self.coordinates or (event.x, event.y) != (self.coordinates[-1][0], self.coordinates[-1][1]):
                self.coordinates.append((event.x, event.y, data_x, data_y))
            self.last_x, self.last_y = event.x, event.y

    def stop_tracing(self, event):
        self.tracing = False

    def gap_filler(self):
        self.nan_coords = [(float('nan'), float('nan'))] * self.chunk_size
        inv = self.ax.transData.inverted()
        for i, (x, y) in enumerate(self.coordinates):
            data_x, data_y = inv.transform((x, y))
            if int(data_x) < self.chunk_size:
                self.nan_coords[int(data_x)] = (data_x, data_y)
        return self.nan_coords

    def clear_annotations(self):
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.canvas_widget.delete("annotation")

    def save_data(self):
        global coords
        if self.image is None or not self.coordinates:
            print("No annotations to save.")
            return
        x0 = self.root.winfo_rootx() + self.canvas_widget.winfo_x()
        y0 = self.root.winfo_rooty() + self.canvas_widget.winfo_y()
        x1 = x0 + self.canvas_widget.winfo_width()
        y1 = y0 + self.canvas_widget.winfo_height()
        self.image_for_saving = ImageGrab.grab((x0, y0, x1, y1))
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")])
        if save_path:
            self.image_for_saving.save(save_path)
            print(f"Image saved: {save_path}")
            # Instead of saving a CSV, save the coordinate data as an HDF5 file.
            h5_path = save_path.replace('.png', '.h5')
            with h5py.File(h5_path, 'w') as hf:
                # Save the coordinates as a dataset. Each row: [pixel_x, pixel_y, data_x, data_y]
                coords = np.array(self.coordinates)
                hf.create_dataset("annotations", data=coords)
            print(f"Data coordinates saved: {h5_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HDF5Annotator(root)
    root.mainloop()