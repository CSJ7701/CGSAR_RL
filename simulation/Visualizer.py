from typing import Optional
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as anim
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import logging

from application.logger import Logger

plt.set_loglevel(level='warning')
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
conv_logger = logging.getLogger('h5py')
conv_logger.setLevel(logging.WARNING)
logger = Logger(__name__).get()

class Visualizer:
    def __init__(self, hdf5_file: str):
        self.data_file = hdf5_file
        self.fig, self.ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
        self.ax.coastlines()
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray')
        self.current_step = 1
        self.wind_interpolator_u = None
        self.wind_interpolator_v = None

        with h5py.File(self.data_file, "r") as f:
            self.total_steps = len(f.keys())  # Count number of steps stored

        self.trackline = None
        self.trackline_success = None
        self.real_victim = None
        self.real_victim_index = None
        self.real_victim_plot = None
        self.victims = None
        self.heatmap_img = None

        logger.info({
            "message": "\033[32mVisualizer initialized.\033[0m",
            "event": "visualizer_object_created",
            "data": {"total_steps": self.total_steps, "h5py_file": self.data_file}
        })

    
    def _load_step_data(self, step_number):
        with h5py.File(self.data_file, "r") as file:
            step_group = file[f"step_{step_number}"]

            if not step_group:
                logger.critical({
                    'message': f"Step {step_number} does not exist.",
                    'event': "step_data_error",
                    'data': {
                        'step': step_number
                                 }
                                })
                raise ValueError(f"Step {step_number} does not exist.")

            current = step_group['current']
            depth = step_group['depth']
            wind = step_group['wind']
            victims = step_group['victims']
            
            self.lat = current['latitude'][:]
            self.lon = current['longitude'][:]
            self.uo = current['uo'][:]
            self.vo = current['vo'][:]
            self.deptho = depth['deptho'][:]
            
            wind_lat = wind['latitude'][:]
            wind_lon = wind['longitude'][:]
            uw = wind['eastward_wind'][:]
            vw = wind['northward_wind'][:]

            self._create_interpolators(wind_lat, wind_lon, uw, vw)

            if 'victim_positions' in victims:
                self.victim_positions = victims['victim_positions'][:]
                self.heatmap = victims['heatmap'][:]
                self.x_edges = victims['heatmap_lon_bin'][:]
                self.y_edges = victims['heatmap_lat_bin'][:]
            else:
                self.victim_positions = np.empty((0,2))

    def _load_trackline(self, trackline: dict[str, list], success: bool = False):
        self.trackline = trackline
        self.trackline_success = success

    def _load_real_victim(self, index: int):
        """Store the real person's location for visualization."""
        if not self.real_victim_index:
            self.real_victim_index = index
            
        if not hasattr(self, 'victim_positions') or self.victim_positions.size == 0:
            logger.warning("No victim positions available.")
            return

        # Assumes index is valid. Also assumes order of victims is constant between cutter and visualizer.
        self.real_victim = np.array(self.victim_positions[index]).flatten()

    def _create_interpolators(self, wind_lat, wind_lon, uw, vw):
        self.wind_interpolator_u = RegularGridInterpolator((wind_lat, wind_lon), uw, bounds_error=False, fill_value=0)
        self.wind_interpolator_v = RegularGridInterpolator((wind_lat, wind_lon), vw, bounds_error=False, fill_value=0)

    def _purple_transparent_colormap(self):
        """Creates a colormap that is transparent for low values."""

        # Generate colors for our custom colormap
        colors = [(0.85, 0.8, 1.0, 0.3)]  # Start with transparent
        color_stops = [
            (0.0, (0.85, 0.8, 1.0)), # Light lilac
            (0.3, (0.7, 0.4, 0.9)), # Medium Purple
            (0.6, (0.5, 0.0, 0.8)), # Deep Purple
            (1.0, (0.3, 0.0, 0.5))  # Dark violet
        ]

        for i in range(1, 256):
            t=i/255.0
            for j in range(len(color_stops)-1):
                if t <= color_stops[j+1][0]:
                    t_local = (t-color_stops[j][0]) / (color_stops[j+1][0] - color_stops[j][0])
                    c1 = color_stops[j][1]
                    c2 = color_stops[j+1][1]
                    r = c1[0] + (c2[0] - c1[0]) * t_local
                    g = c1[1] + (c2[1] - c1[1]) * t_local
                    b = c1[2] + (c2[2] - c1[2]) * t_local
                    colors.append((r,g,b,t))
                    break
        return mcolors.LinearSegmentedColormap.from_list('custom_purple', colors)

    def _mpl_transparent_colormap(self):
        """Creates a colormap that is tranparent for low values, built off a built-in colormap."""
        base_cmap = plt.cm.get_cmap('plasma')

        # List of colors with transparency scaling with value
        colors = []
        for i in range(256):
            t = i/255.0
            rgba = base_cmap(i)
            alpha = t**0.5
            colors.append((rgba[0], rgba[1], rgba[2], alpha))

        return mcolors.LinearSegmentedColormap.from_list('transparent_builtin', colors)
    
    def _heatmap(self):
        """Creates a numpy histogram. This should move somewhere else later."""
        particle_lons = self.victim_positions[:,1]
        particle_lats = self.victim_positions[:,0]

        lon_bins = np.linspace(self.lon.min(), self.lon.max(), 5*len(self.lon))
        lat_bins = np.linspace(self.lat.min(), self.lat.max(), 5*len(self.lat))

        heatmap, x_edges, y_edges = np.histogram2d(
            particle_lons, particle_lats,
            bins=[lon_bins, lat_bins]
        )

        #heatmap = heatmap + 1e-10

        # Normalize to [0,1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap.T, lon_bins, lat_bins

    def plot(self, step):
        self._load_step_data(step)
        if self.real_victim_index is not None:
            self._load_real_victim(self.real_victim_index)

        #heatmap, x_edges, y_edges = self._heatmap()
        
        if not self.wind_interpolator_v or not self.wind_interpolator_u:
            logger.critical({
                'message': "Wind interpolators not initialized.",
                'event': "plot_error"
            })
            raise ValueError("Wind interpolators not initialized.")
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        #uw_grid = self.wind_interpolator_u((lat_grid, lon_grid))
        #vw_grid = self.wind_interpolator_v((lat_grid, lon_grid))

        # Depth countour
        depth_min, depth_max = np.nanmin(self.deptho), np.nanmax(self.deptho)
        self.depth_contour = self.ax.contourf(lon_grid, lat_grid, self.deptho, levels = np.linspace(depth_min, depth_max, 20), cmap='Blues', alpha=0.7)

        # Heatmap
        if self.heatmap.size != 0:
            custom_cmap = self._mpl_transparent_colormap()
            self.heatmap_img = self.ax.pcolormesh(
                self.x_edges, self.y_edges, self.heatmap,
                cmap='plasma',
                norm='log',
                shading='flat',
                alpha=0.8,
                zorder=1,
            )
        
        # Vectors
        self.currents = self.ax.quiver(lon_grid, lat_grid, self.uo, self.vo, color='teal', alpha=0.4, label='Currents')
        #self.winds = self.ax.quiver(lon_grid, lat_grid, uw_grid, vw_grid, color='green', alpha=0.7, label='Wind')

        # Victims
        if self.victim_positions.size != 0:
            self.victims = self.ax.scatter(self.victim_positions[:,1], self.victim_positions[:,0], color='dimgray', marker='o', label='Victims', s=0.5, alpha=0.0)

        # Trackline
        if self.trackline:
            start_lat, start_lon = self.trackline["start"][0]
            if "1" in self.trackline:
                step1 = self.trackline["1"]
                step1_lats = [coord[0] for coord in step1]
                step1_lons = [coord[1] for coord in step1]
                self.trackline_plot, = self.ax.plot(
                    [start_lon] + step1_lons,
                    [start_lat] + step1_lats,
                    marker='x', linestyle='-', color='b')
            else:
                self.trackline_plot, = self.ax.plot(
                    [start_lon], [start_lat],
                    marker='x', linestyle='-', color='b')

        if hasattr(self, 'real_victim') and self.real_victim is not None:
            self.real_victim_plot = self.ax.scatter(
                self.real_victim[1], self.real_victim[0],
                color='yellow', marker='*', label='Real Victim', s=30, edgecolors='black', linewidth=0.5
            )

        self.ax.set_title(f"Surface Currents and Wind at Step {step}")
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_xlim(self.lon.min(), self.lon.max())
        self.ax.set_ylim(self.lat.min(), self.lat.max())

        #self.fig.colorbar(self.depth_contour, ax=self.ax, label='Depth (m)')

    def update(self, frame):
        self._load_step_data(frame+1)
        self.ax.set_title(f"Surface Currents and Wind at Step {frame+1}")

        if not self.wind_interpolator_v or not self.wind_interpolator_u:
            logger.critical({
                'message': "Wind interpolators not initialized.",
                'event': "plot_update_error"
            })
            raise ValueError("Wind interpolators not initialized.")
        #lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        #uw_grid = self.wind_interpolator_u((lat_grid, lon_grid))
        #vw_grid = self.wind_interpolator_v((lat_grid, lon_grid))
        
        self.currents.set_UVC(self.uo, self.vo)
        #self.winds.set_UVC(uw_grid, vw_grid)
        if self.victims:
            self.victims.set_offsets(self.victim_positions[:, [1,0]])

        if self.heatmap_img:
            heatmap,_,_= self._heatmap()
            self.heatmap_img.set_array(heatmap.ravel())

        if self.real_victim_plot:
            self.real_victim = self.victim_positions[self.real_victim_index]
            self.real_victim_plot.set_offsets([self.real_victim[1], self.real_victim[0]])

        if self.trackline:
            steps = sorted(int(k) for k in self.trackline.keys() if k!="start")
            valid_steps = [str(s) for s in steps if s <= frame+1]
            lats = [coord[0] for coord in self.trackline["start"]]
            lons = [coord[1] for coord in self.trackline["start"]]
            for s in valid_steps:
                if s in self.trackline:
                    lats.extend(coord[0] for coord in self.trackline[s])
                    lons.extend(coord[1] for coord in self.trackline[s])
            self.trackline_plot.set_data(lons, lats)

        #return self.currents, self.winds, self.victims, self.heatmap_img
        return self.currents, self.victims, self.heatmap_img, self.real_victim_plot
    
    def show(self):
        plt.show()

    def run(self, show: bool = False, file: Optional[str] = None):
        self.plot(1)
        if self.trackline:
            steps = [int(k) for k in self.trackline.keys() if k != "start"]
            max_frame = max(steps) + 2
            if max_frame > self.total_steps:
                max_frame = self.total_steps
        else:
            max_frame = self.total_steps
        self.ani = anim.FuncAnimation(self.fig, self.update, frames=max_frame, interval = 100, blit=False)

        if show:
            logger.info({
                "message": "\033[32mDisplaying plot...\033[0m",
                "event": "plot_display"
                })
            plt.show()
        else:
            save_path = file if file else "./test.mp4"
            self.ani.save(save_path, writer=anim.FFMpegWriter())
            logger.info({
                "message": f"Animation saved to \033[32m{save_path}\033[0m",
                "event": "plot_save",
                "data": {"file": save_path}
            })
