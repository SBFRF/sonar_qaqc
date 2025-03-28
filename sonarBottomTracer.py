import os
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
        
        # Set default depth option to "Smooth Depth" so that the Smooth radio button is selected
        self.depth_option = tk.StringVar(value="Smooth Depth")
        
        # Create a menu frame with a fixed height.
        self.menu_frame = tk.Frame(root, height=150)
        self.menu_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.menu_frame.pack_propagate(False)
        
        # Use grid in the menu frame: first row: chunk size label and entry.
        tk.Label(self.menu_frame, text="Chunk Size:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.chunk_size_entry = tk.Entry(self.menu_frame, width=10)
        self.chunk_size = 1000  # default chunk size
        self.chunk_size_entry.insert(0, str(self.chunk_size))
        self.chunk_size_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        # Second row: center the "Load HDF5 File" button.
        self.load_button = tk.Button(self.menu_frame, text="Load HDF5 File", command=self.load_file)
        self.load_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        # Annotation frame (hidden until a file is loaded)
        self.annotation_frame = tk.Frame(root)
        # (Will be packed when a file is loaded)
        
        # Create the Matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(18,12))  # Recommended sizes: 12x8 or 18x12
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.annotation_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()
        
        # Depth Options Frame for selecting depth option
        self.depth_frame = tk.Frame(self.annotation_frame)
        self.depth_frame.pack(pady=5)
        tk.Label(self.depth_frame, text="Toggle Factory Line:").pack(side="left", padx=5)
        tk.Radiobutton(self.depth_frame, text="Smooth", variable=self.depth_option, value="Smooth Depth",
                       command=self.update_display).pack(side="left", padx=5)
        tk.Radiobutton(self.depth_frame, text="Ping", variable=self.depth_option, value="Ping Depth",
                       command=self.update_display).pack(side="left", padx=5)
        tk.Radiobutton(self.depth_frame, text="Off", variable=self.depth_option, value="Off",
                       command=self.update_display).pack(side="left", padx=5)
        
        # Frame for control buttons
        self.button_frame = tk.Frame(self.annotation_frame)
        self.button_frame.pack(fill="both", expand=True)
        self.clear_button = tk.Button(self.button_frame, text="Clear Annotations", command=self.clear_annotations)
        self.clear_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.save_button = tk.Button(self.button_frame, text="Save Manual Depth Line", command=self.save_data)
        self.save_button.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.save_depth_button = tk.Button(self.button_frame, text="Save Automatic Depth Line", command=self.save_depth_line)
        self.save_depth_button.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        self.prev_button = tk.Button(self.button_frame, text="Previous Slice", command=self.prev_slice)
        self.prev_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.next_button = tk.Button(self.button_frame, text="Next Slice", command=self.next_slice)
        self.next_button.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.quit_button = tk.Button(self.button_frame, text="Quit", command=self.quit_gui)
        self.quit_button.grid(row=1, column=2, padx=10, pady=5, sticky="ew")
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure(1, weight=1)
        
        # Initialize other variables.
        self.file_path = None
        self.idx_start = 0
        self.image = None
        self.smooth_depth = None      # Holds smooth_depth_m if available
        self.length_mm = None         # Holds length_mm if available
        self.this_ping_depth_m = None # Holds this_ping_depth_m if available
        self.bin_size = None          # Bin size (in meters per row)
        self.smooth_depth_img = None  # smooth_depth_m divided by bin_size
        self.this_ping_depth_img = None  # this_ping_depth_m divided by bin_size
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.coordinates = []
        self.image_for_saving = None
        self.total_slices = None
        self.total_time = None

    def load_file(self):
        try:
            self.chunk_size = int(self.chunk_size_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Chunk size must be an integer.")
            return

        self.file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5 *.hdf5")])
        if not self.file_path:
            return
        self.menu_frame.pack_forget()
        self.annotation_frame.pack(fill="both", expand=True)
        self.idx_start = 0
        self.process_next_chunk()

    def process_next_chunk(self):
        if not self.file_path:
            return
        with h5py.File(self.file_path, 'r') as f:
            self.total_time = f['time'].shape[0]
            self.total_slices = int(np.ceil(self.total_time / self.chunk_size))
            if self.idx_start >= self.total_time:
                print("All data processed.")
                return
            idx = slice(self.idx_start, min(self.idx_start + self.chunk_size, self.total_time))
            self.image = f['profile_data'][:, idx]
            if 'smooth_depth_m' in f:
                self.smooth_depth = f['smooth_depth_m'][idx]
                if 'length_mm' in f:
                    self.length_mm = f['length_mm'][idx]
                    self.bin_size = (self.length_mm[0] / 1000.0) / self.image.shape[0]
                else:
                    self.length_mm = None
                    self.bin_size = None
                if 'this_ping_depth_m' in f:
                    self.this_ping_depth_m = f['this_ping_depth_m'][idx]
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
        self.update_display()

    def update_display(self):
        self.ax.clear()
        self.ax.pcolormesh(self.image, cmap='plasma')
        self.ax.set_title("Click and drag to trace. Start and stop as needed.\n(Factory Line must be toggled off.)")
        self.ax.yaxis.set_ticks_position('both')
        self.ax.tick_params(axis='y', labelleft=True, labelright=True)
        self.ax.set_xlim(0, self.chunk_size)
        self.ax.set_ylim(0, self.image.shape[0])
        if self.depth_option.get() != "Off" and ((self.smooth_depth_img is not None) or (self.this_ping_depth_img is not None)):
            self.plot_depth()
        # Remove any existing figure texts using a copy of the list.
        for txt in self.fig.texts[:]:
            txt.remove()
        slice_number = self.idx_start // self.chunk_size + 1
        idxS = self.idx_start
        idxE = min(self.idx_start + self.chunk_size - 1, self.total_time - 1) if self.total_time is not None else self.idx_start + self.chunk_size - 1
        self.fig.text(0.01, 0.98, f"Slice #{slice_number} of {self.total_slices}\nTime Indices: {idxS} - {idxE}",
                      horizontalalignment="left", verticalalignment="top", fontsize=12, color="black")
        self.canvas.draw()
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.enable_annotation()
        self.fig.canvas.draw()
        image_array = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.image_for_saving = Image.fromarray(image_array)
        self.update_button_states()

    def update_button_states(self):
        if self.depth_option.get() == "Off":
            self.save_button.config(state="normal")
            self.clear_button.config(state="normal")
            self.save_depth_button.config(state="disabled")
        else:
            self.save_button.config(state="disabled")
            self.clear_button.config(state="disabled")
            self.save_depth_button.config(state="normal")

    def plot_depth(self):
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
        else:
            return
        x = np.arange(0, self.chunk_size)
        if data is not None:
            self.ax.plot(x, data, color='cyan', linewidth=2, label=self.depth_option.get())
            self.ax.legend(loc='upper right')
            self.canvas.draw()

    def prev_slice(self):
        self.clear_annotations()
        self.depth_option.set("Smooth Depth")
        if self.idx_start - self.chunk_size < 0:
            self.idx_start = 0
        else:
            self.idx_start -= self.chunk_size
        self.process_next_chunk()
                
    def next_slice(self):
        self.clear_annotations()
        with h5py.File(self.file_path, 'r') as f:
            total_time = f['time'].shape[0]
        if self.idx_start + self.chunk_size < total_time:
            self.idx_start += self.chunk_size
        else:
            self.idx_start = total_time - self.chunk_size
        self.depth_option.set("Smooth Depth")
        self.process_next_chunk()

    def quit_gui(self):
        self.root.destroy()
        
    def enable_annotation(self):
        if self.depth_option.get() == "Off":
            self.canvas_widget.bind("<Button-1>", self.start_tracing)
            self.canvas_widget.bind("<B1-Motion>", self.trace_line)
            self.canvas_widget.bind("<Button-3>", self.stop_tracing)
        else:
            self.canvas_widget.unbind("<Button-1>")
            self.canvas_widget.unbind("<B1-Motion>")
            self.canvas_widget.unbind("<Button-3>")
        
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
            # print(f"Tkinter: ({event.x}, {event.y}), Data: ({data_x:.2f}, {data_y:.2f})")
            if not self.coordinates or (event.x, event.y) != (self.coordinates[-1][0], self.coordinates[-1][1]):
                self.coordinates.append((event.x, event.y, data_x, data_y))
            self.last_x, self.last_y = event.x, event.y

    def stop_tracing(self, event):
        self.tracing = False

    def interpolate_coordinates(self):
        x_coords = np.arange(self.chunk_size)
        if not self.coordinates:
            return np.column_stack((x_coords, np.full(self.chunk_size, np.nan)))
        pts = sorted([(pt[2], pt[3]) for pt in self.coordinates], key=lambda p: p[0])
        y_interp = np.full(self.chunk_size, np.nan)
        gap_threshold = 2.0
        for x in x_coords:
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i+1]
                if (x1 - x0) <= gap_threshold and x0 <= x <= x1:
                    y_interp[x] = np.interp(x, [x0, x1], [y0, y1])
                    break
        return np.column_stack((x_coords, y_interp))

    def clear_annotations(self):
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.canvas_widget.delete("annotation")

    def save_data(self):
        self.clear_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="disabled")
        
        if self.image is None or not self.coordinates:
            print("No annotations to save.")
            return
        base = os.path.splitext(os.path.basename(self.file_path))[0]
        idxS = self.idx_start
        idxE = self.idx_start + (self.chunk_size - 1)
        default_filename = f"{base}_bottomTraced_{idxS}-{idxE}.png"
        
        # Compute interpolated values then clear annotations.
        interp_values = self.interpolate_coordinates()
        self.clear_annotations()
        
        valid = ~np.isnan(interp_values[:, 1])
        if np.any(valid):
            indices = np.where(valid)[0]
            splits = np.where(np.diff(indices) != 1)[0] + 1
            segments = np.split(indices, splits)
            first = True
            for seg in segments:
                if len(seg) >= 2:
                    if first:
                        self.ax.plot(interp_values[seg, 0], interp_values[seg, 1],
                                     color='blue', linewidth=2, label="Saved Manual Depth Line [Manual]")
                        first = False
                    else:
                        self.ax.plot(interp_values[seg, 0], interp_values[seg, 1],
                                     color='blue', linewidth=2)
        self.ax.legend(loc='upper right')
        self.canvas.draw()
        
        x0 = self.root.winfo_rootx() + self.canvas_widget.winfo_x()
        y0 = self.root.winfo_rooty() + self.canvas_widget.winfo_y()
        x1 = x0 + self.canvas_widget.winfo_width()
        y1 = y0 + self.canvas_widget.winfo_height()
        self.image_for_saving = ImageGrab.grab((x0, y0, x1, y1))
        
        save_path = filedialog.asksaveasfilename(initialfile=default_filename,
                                                 defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")])
        if save_path:
            self.image_for_saving.save(save_path)
            print(f"Image saved: {save_path}")
            h5_path = save_path.replace('.png', '.h5')
            with h5py.File(h5_path, 'w') as hf:
                hf.create_dataset("depth_line", data=interp_values)
            print(f"Manual depth line saved: {h5_path}")

    def save_depth_line(self):
        self.clear_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.save_depth_button.config(state="disabled")
        
        if self.depth_option.get() not in ["Smooth Depth", "Ping Depth"]:
            messagebox.showerror("Error", "No depth line selected!")
            return
        if self.depth_option.get() == "Smooth Depth":
            data = self.smooth_depth_img
            legend_label = "Saved Automatic Depth Line [Smooth]"
        elif self.depth_option.get() == "Ping Depth":
            data = self.this_ping_depth_img
            legend_label = "Saved Automatic Depth Line [Ping]"
        else:
            data = None

        if data is None:
            messagebox.showerror("Error", "Depth data not available!")
            return
        
        base = os.path.splitext(os.path.basename(self.file_path))[0]
        idxS = self.idx_start
        idxE = self.idx_start + (self.chunk_size - 1)
        default_filename = f"{base}_bottomTraced_{idxS}-{idxE}.png"
        
        x_coords = np.arange(0, self.chunk_size)
        depth_coords = np.column_stack((x_coords, data))
        
        # Remove any existing plotted lines (only lines, not pcolormesh)
        for line in self.ax.get_lines():
            line.remove()
        self.ax.plot(x_coords, data, color='blue', linewidth=2, label=legend_label)
        self.ax.legend(loc='upper right')
        self.canvas.draw()
        
        x0 = self.root.winfo_rootx() + self.canvas_widget.winfo_x()
        y0 = self.root.winfo_rooty() + self.canvas_widget.winfo_y()
        x1 = x0 + self.canvas_widget.winfo_width()
        y1 = y0 + self.canvas_widget.winfo_height()
        self.image_for_saving = ImageGrab.grab((x0, y0, x1, y1))
        
        save_path = filedialog.asksaveasfilename(initialfile=default_filename,
                                                 defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")])
        if save_path:
            self.image_for_saving.save(save_path)
            print(f"Image saved: {save_path}")
            h5_path = save_path.replace('.png', '.h5')
            with h5py.File(h5_path, 'w') as hf:
                hf.create_dataset("depth_line", data=depth_coords)
            print(f"Depth line saved: {h5_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HDF5Annotator(root)
    root.mainloop()
