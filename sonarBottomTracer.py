# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:16:32 2025

@author: RDCHLJEB

Last checked 03/14/2025
"""
#%%
import h5py
# import csv
# import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageGrab

class HDF5Annotator:
    def __init__(self, root):
        self.root = root
        self.root.title("HDF5 Annotator")

        # Initial menu frame
        self.menu_frame = tk.Frame(root)
        self.menu_frame.pack(fill="both", expand=True)
        
        # Set up buttons in the menu frame using grid
        self.load_button = tk.Button(self.menu_frame, text="Load HDF5 File", command=self.load_file)
        self.load_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        self.chunk_size_button = tk.Button(self.menu_frame, text="Set Chunk Size", command=self.set_chunk_size)
        self.chunk_size_button.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Ensure rows and columns expand correctly if necessary
        self.menu_frame.grid_columnconfigure(0, weight=1)
        self.menu_frame.grid_rowconfigure(0, weight=1)
        
        # Annotation frame (hidden at first)
        self.annotation_frame = tk.Frame(root)
        
        self.fig, self.ax = plt.subplots(figsize=(12,8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.annotation_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()
        
        # Create a new frame for buttons in two rows and grid them
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
        
        # Make all 3 columns the same width
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        
        # Ensure rows also expand correctly if necessary
        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_rowconfigure(1, weight=1)
        
        # Initialize variables
        self.file_path = None
        self.chunk_size = 1000  # Default chunk size
        self.idx_start = 0
        self.image = None
        self.lines = []
        self.image_time = None
        self.range = None
        self.current_path = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.coordinates = []
        self.photo = None
        self.image_for_saving = None

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5 *.hdf5")])
        if not self.file_path:
            return
        
        self.menu_frame.pack_forget()
        self.annotation_frame.pack()
        
        self.idx_start = 0  # Reset index for new file
        self.process_next_chunk()

    def set_chunk_size(self):
        size = simpledialog.askinteger("Chunk Size", "Enter preferred chunk size:", minvalue=1, maxvalue=1000)
        if size:
            self.chunk_size = size

    def process_next_chunk(self):
        if not self.file_path:
            return
        
        with h5py.File(self.file_path, 'r') as f:
            if self.idx_start >= f['time'].shape[0]:
                print("All data processed.")
                return
            
            idx = slice(self.idx_start, min(self.idx_start + self.chunk_size, f['time'].shape[0]))
            self.image_time = f['time'][idx]
            self.range = f['range_m'][...]
            self.image = f['profile_data'][:, idx]
        
        self.update_display()

    def update_display(self):
        self.ax.clear()
        im = self.ax.pcolormesh(self.image, cmap='plasma')
        self.fig.colorbar(im, ax=self.ax)
        self.ax.set_title("Click and drag to trace. Start and stop as needed.")
        self.canvas.draw()
        self.coordinates = []
        self.tracing = False
        self.last_x, self.last_y = None, None
        self.enable_annotation()
        # self.enable_zoom()  # Activate zooming
        self.fig.canvas.draw()
        # Save figure as image
        self.fig.canvas.draw()
        image_array = np.array(self.fig.canvas.renderer.buffer_rgba())  # Get image as array
        self.image_for_saving = Image.fromarray(image_array)  # Convert to PIL Image
        
    def prev_slice(self):
        if self.idx_start - self.chunk_size >= 0:
            self.idx_start -= self.chunk_size
            self.process_next_chunk()
    
    def next_slice(self):
        self.idx_start += self.chunk_size
        self.process_next_chunk()

    def quit_gui(self):
        self.root.destroy()
        
    def enable_annotation(self):
        self.canvas_widget.bind("<Button-1>", self.start_tracing)
        self.canvas_widget.bind("<B1-Motion>", self.trace_line)
        self.canvas_widget.bind("<Button-3>", self.stop_tracing)
        
    def start_tracing(self, event):
        """Start tracing the line in image coordinates but also convert to data coordinates."""
        self.tracing = True
        self.last_x, self.last_y = event.x, event.y
        self.coordinates.append((event.x, event.y, *self.canvas_to_data(event.x, event.y)))  # Save both sets of coordinates

        
    def canvas_to_data(self, x, y):
        """Convert Tkinter canvas coordinates to Matplotlib data coordinates."""
        
        # Get Tkinter widget position
        x_offset = self.canvas_widget.winfo_rootx() - self.root.winfo_rootx()
        y_offset = self.canvas_widget.winfo_rooty() - self.root.winfo_rooty()
    
        # Convert event (canvas) coordinates to figure coordinates
        fig_x = x - x_offset
        fig_y = y - y_offset  # Keep this unchanged for now
    
        # Flip y-coordinate to match Matplotlib's bottom-up system
        fig_y = self.canvas_widget.winfo_height() - fig_y  
    
        # Convert figure coordinates to data coordinates
        data_x, data_y = self.ax.transData.inverted().transform((fig_x, fig_y))

        return data_x, data_y
    
    def trace_line(self, event):
        """Trace line while storing accurate data coordinates, ensuring it stays within figure bounds."""
        if self.tracing:
            # Convert event (pixel) coordinates to data coordinates
            data_x, data_y = self.canvas_to_data(event.x, event.y)
            
            # Check if the point is inside the axes' limits
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            
            if not (x_min <= data_x <= x_max and y_min <= data_y <= y_max):
                return  # Ignore drawing if outside limits
    
            # Draw in Tkinter canvas
            self.canvas_widget.create_line((self.last_x, self.last_y, event.x, event.y), fill='red', width=2)
    
            # Debug print to verify correct transformation
            print(f"Tkinter: ({event.x}, {event.y}), Data: ({data_x:.2f}, {data_y:.2f})")
    
            # Store both pixel and data coordinates
            if not self.coordinates or (event.x, event.y) != (self.coordinates[-1][0], self.coordinates[-1][1]):
                self.coordinates.append((event.x, event.y, data_x, data_y))
    
            self.last_x, self.last_y = event.x, event.y

    def stop_tracing(self, event):
        """Stop tracing the line"""
        self.tracing = False

    def gap_filler(self):
        """Convert and store coordinates in data space."""
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
        if self.image is not None:
            self.update_display()
            self.canvas_widget.delete("all")

    def save_data(self):
        """Save annotations as image and data coordinates as CSV."""
        if self.image is None or not self.coordinates:
            print("No annotations to save.")
            return

        # Save Image
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

            csv_path = save_path.replace('.png', '.csv')
            with open(csv_path, 'w') as f:
                f.write("data_x,data_y\n")
                for _, _, dx, dy in self.coordinates:
                    f.write(f"{dx},{dy}\n")
            print(f"Data coordinates saved: {csv_path}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()
        return self.coordinates, self.nan_coords  
          
if __name__ == "__main__":
    root = tk.Tk()
    app = HDF5Annotator(root)
    root.mainloop()
    
    
    
    
    
    
    
 #%%  
# def enable_zoom(self):
#     """Bind zooming functionality to mouse scroll with reduced lag."""
#     self.canvas_widget.bind("<MouseWheel>", self.zoom)  # Windows/macOS
#     self.canvas_widget.bind("<Button-4>", self.zoom)    # Linux (scroll up)
#     self.canvas_widget.bind("<Button-5>", self.zoom)    # Linux (scroll down)

#     # Store the original limits for resetting
#     self.original_xlim = self.ax.get_xlim()
#     self.original_ylim = self.ax.get_ylim()
    
#     # For reducing lag
#     self.last_zoom_time = time.time()
#     self.zoom_delay = 0.2  # Minimum time between zoom updates (20ms)

# def zoom(self, event):
#     """Zoom in or out using mouse scroll with real-time response."""
#     current_time = time.time()
    
#     # Skip excessive redraws (limit updates to 50 FPS)
#     if current_time - self.last_zoom_time < self.zoom_delay:
#         return
#     self.last_zoom_time = current_time

#     zoom_factor = 1.5  # Adjust zoom intensity

#     # Get current limits
#     x_min, x_max = self.ax.get_xlim()
#     y_min, y_max = self.ax.get_ylim()

#     # Compute the center of the current view
#     x_center = (x_min + x_max) / 2
#     y_center = (y_min + y_max) / 2

#     # Adjust limits based on scroll direction
#     if event.delta > 0 or event.num == 4:  # Scroll up (Zoom in)
#         new_xlim = ((x_min - x_center) / zoom_factor + x_center,
#                     (x_max - x_center) / zoom_factor + x_center)
#         new_ylim = ((y_min - y_center) / zoom_factor + y_center,
#                     (y_max - y_center) / zoom_factor + y_center)
#     elif event.delta < 0 or event.num == 5:  # Scroll down (Zoom out)
#         new_xlim = ((x_min - x_center) * zoom_factor + x_center,
#                     (x_max - x_center) * zoom_factor + x_center)
#         new_ylim = ((y_min - y_center) * zoom_factor + y_center,
#                     (y_max - y_center) * zoom_factor + y_center)

#         # Ensure we don’t zoom out past the original limits
#         new_xlim = (max(new_xlim[0], self.original_xlim[0]), min(new_xlim[1], self.original_xlim[1]))
#         new_ylim = (max(new_ylim[0], self.original_ylim[0]), min(new_ylim[1], self.original_ylim[1]))

#     # Apply new limits and use `blit` for faster rendering
#     self.ax.set_xlim(new_xlim)
#     self.ax.set_ylim(new_ylim)

#     self.canvas.draw_idle()  # Uses Matplotlib's optimized rendering