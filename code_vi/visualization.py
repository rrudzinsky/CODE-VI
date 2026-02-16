import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from .materials import MaterialLib

class Draw:
    """
    Static helper class for rendering the optical system using Matplotlib.
    Features: Dual-Panel Interactive Dashboard, PFT Analysis, Fixed Axes, and High-Res Interpolated Cloud.
    """
    
    # Cache for the structured mesh topology
    _cached_triangles = None
    _cached_ray_count = 0

    @staticmethod
    def interactive_session(manager, tracer, **kwargs):
        # 1. Validation
        if tracer.rays.empty:
            print("Warning: RayTrace data is empty. Run the simulation first.")
            return
            
        n_sources = tracer.rays['source_id'].nunique() if 'source_id' in tracer.rays.columns else 0
        n_time_steps = len(tracer.snapshots)
        
        # 2. Pre-calculate Global Bounds
        include_beam = kwargs.get('draw_beam_arrow', False)
        # Use the new helper to get SQUARE limits to prevent aspect ratio crushing
        global_xlim, global_ylim = Draw._get_global_bounds(manager, tracer, include_beam=include_beam)

        # 3. Create Figure
        plt.ioff()
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), layout='constrained', gridspec_kw={'width_ratios': [1.5, 1]})
        ax_global, ax_pulse = axes
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        plt.ion()

        current_view = {'xlim': global_xlim, 'ylim': global_ylim}
        first_draw = [True]

        # Clear cache
        Draw._cached_triangles = None

        # 4. Define Update Logic
        def update_dashboard(source_idx, time_idx, show_cloud):
            if not first_draw[0]:
                current_view['xlim'] = ax_global.get_xlim()
                current_view['ylim'] = ax_global.get_ylim()

            ax_global.clear()
            ax_pulse.clear()
            
            filter_val = None if source_idx == -1 else source_idx
            
            # --- MODE 1: INTEGRATED VIEW ---
            if time_idx == -1:
                Draw.draw_system(manager, tracer=tracer, ax=ax_global, source_id_filter=filter_val, show_plot=False, **kwargs)
                ax_global.set_title(f"Integrated Path (Source #{filter_val if filter_val is not None else 'All'})")
                
                if filter_val is not None:
                    Draw._plot_pft_history(ax_pulse, tracer, filter_val)
                else:
                    ax_pulse.text(0.5, 0.5, "Select a single Source ID\nto see PFT Evolution", ha='center', va='center', color='gray')
                    ax_pulse.set_xlim(0, 100); ax_pulse.set_ylim(-10, 10); ax_pulse.axis('on')

            # --- MODE 2: SNAPSHOT VIEW ---
            else:
                snap = tracer.snapshots[time_idx]
                t_current = snap['t']
                
                # Draw Geometry
                Draw.draw_system(manager, ax=ax_global, tracer=None, show_plot=False, **kwargs) 
                
                # Draw Cloud
                if show_cloud:
                    Draw._draw_continuous_cloud(ax_global, snap, tracer, filter_val)
                else:
                    Draw._draw_photon_cloud_global(ax_global, snap, tracer, filter_val)
                    
                ax_global.set_title(f"Snapshot @ t={t_current:.1f} ps")

                # Right Panel (Analysis)
                Draw._analyze_pulse_shape(ax_pulse, snap, tracer, filter_val)

            # Restore View
            ax_global.set_xlim(current_view['xlim'])
            ax_global.set_ylim(current_view['ylim'])
            # Use 'box' to force the plot to fill the axes box while keeping aspect equal
            ax_global.set_aspect('equal', adjustable='box')
            ax_pulse.relim(); ax_pulse.autoscale_view()
            
            first_draw[0] = False
            fig.canvas.draw_idle()

        # 5. Controls
        s_source = widgets.IntSlider(min=-1, max=n_sources-1, step=1, value=0, description='Source ID:')
        s_time = widgets.IntSlider(min=-1, max=n_time_steps-1, step=1, value=-1, description='Time Step:')
        c_cloud = widgets.Checkbox(value=False, description='Show Interpolated Cloud')
        
        ui = widgets.interactive(update_dashboard, source_idx=s_source, time_idx=s_time, show_cloud=c_cloud)
        
        display(widgets.VBox([
            widgets.HBox([s_source, s_time]),
            c_cloud,
            fig.canvas
        ]))

    # ------------------------------------------------------------------
    #   STRUCTURED CLOUD RENDERING
    # ------------------------------------------------------------------

    @staticmethod
    def _get_structured_topology(tracer):
        if Draw._cached_triangles is not None and Draw._cached_ray_count == len(tracer.rays):
            return Draw._cached_triangles

        df = tracer.rays.copy()
        counts = df['source_id'].value_counts()
        rays_per_source = counts.mode()[0] if counts.nunique() > 1 else counts.iloc[0]
        n_sources = df['source_id'].nunique()
        
        triangles = []
        for s in range(n_sources - 1):
            base_curr = s * rays_per_source
            base_next = (s + 1) * rays_per_source
            for r in range(rays_per_source - 1):
                p00, p01 = base_curr + r, base_curr + r + 1  
                p10, p11 = base_next + r, base_next + r + 1  
                triangles.append([p00, p10, p11])
                triangles.append([p00, p11, p01])
                
        Draw._cached_triangles = np.array(triangles)
        Draw._cached_ray_count = len(tracer.rays)
        return Draw._cached_triangles

    @staticmethod
    def _draw_continuous_cloud(ax, snap, tracer, source_filter):
        if len(snap['ids']) < 3: return
        
        all_x = tracer.rays['x'].values.copy()
        all_y = tracer.rays['y'].values.copy()
        active_ids = snap['ids']
        all_x[active_ids] = snap['x']
        all_y[active_ids] = snap['y']
        
        try:
            triangles = Draw._get_structured_topology(tracer)
        except:
            return 
            
        # Masking
        is_active = np.zeros(len(tracer.rays), dtype=bool)
        is_active[active_ids] = True
        
        if source_filter is not None:
            is_visible_source = (tracer.rays['source_id'].values == source_filter)
            is_active = is_active & is_visible_source
            
        mask = ~ (is_active[triangles[:,0]] & is_active[triangles[:,1]] & is_active[triangles[:,2]])

        # Triangulation
        triang = mtri.Triangulation(all_x, all_y, triangles=triangles)
        triang.set_mask(mask)
        
        wls = tracer.rays['wavelength'].values
        norm = mcolors.Normalize(vmin=wls.min(), vmax=wls.max())
        cmap = cm.get_cmap('jet')
        
        ax.tripcolor(triang, wls, norm=norm, cmap=cmap, shading='gouraud', alpha=0.8)

        # Add velocity
        if np.any(is_active):
            id_to_source = tracer.rays['source_id'].values
            snap_global_ids = snap['ids']
            if source_filter is not None:
                subset_mask = (id_to_source[snap_global_ids] == source_filter)
            else:
                subset_mask = np.ones(len(snap_global_ids), dtype=bool)
            
            if np.any(subset_mask):
                sx = snap['x'][subset_mask]
                sy = snap['y'][subset_mask]
                svx = snap['vx'][subset_mask]
                svy = snap['vy'][subset_mask]
                cx, cy = np.mean(sx), np.mean(sy)
                mvx, mvy = np.mean(svx), np.mean(svy)
                ax.quiver(cx, cy, mvx, mvy, color='black', scale=20, width=0.005, zorder=11, alpha=0.8)

    # ------------------------------------------------------------------
    #   DISCRETE & HELPERS
    # ------------------------------------------------------------------
    
    @staticmethod
    def _get_global_bounds(manager, tracer, padding=0.1, include_beam=False):
        """
        Calculates a SQUARED bounding box containing all elements.
        Ensures the X and Y ranges are equal so 'equal' aspect ratio doesn't shrink the plot.
        """
        xs, ys = [], []
        # Optics
        for opt in manager.elements:
            r = opt.diameter / 2
            xs.extend([opt.x_center - r, opt.x_center + r])
            ys.extend([opt.y_center - r, opt.y_center + r])
        # Rays
        if not tracer.rays.empty:
            xs.extend([tracer.rays['x'].min(), tracer.rays['x'].max()])
            ys.extend([tracer.rays['y'].min(), tracer.rays['y'].max()])
        # Beam
        if include_beam:
            xs.extend([0, 25.4]); ys.extend([-10, 5])
        
        if not xs: return (0, 100), (0, 100)
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        # Calculate ranges
        width = xmax - xmin
        height = ymax - ymin
        
        # Determine the larger dimension to make it square
        max_dim = max(width, height)
        
        # Center points
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        
        # Create square bounds with padding
        half_span = (max_dim / 2) * (1 + padding)
        
        return (cx - half_span, cx + half_span), (cy - half_span, cy + half_span)

    @staticmethod
    def _draw_photon_cloud_global(ax, snap, tracer, source_filter):
        if len(snap['ids']) == 0: return
        id_to_source = tracer.rays['source_id'].values
        all_wls = tracer.rays['wavelength'].values
        global_ids = snap['ids']
        if source_filter is not None:
            mask = (id_to_source[global_ids] == source_filter)
            indices = np.where(mask)[0]
        else:
            indices = range(len(global_ids))   
        if len(indices) == 0: return
        
        active_ids = global_ids[indices]
        x = snap['x'][indices]
        y = snap['y'][indices]
        vx = snap['vx'][indices]
        vy = snap['vy'][indices]
        wls = all_wls[active_ids]
        norm = mcolors.Normalize(vmin=all_wls.min(), vmax=all_wls.max())
        cmap = cm.get_cmap('jet')
        colors = cmap(norm(wls))
        
        ax.scatter(x, y, c=colors, s=15, alpha=0.9, zorder=10, edgecolors='none')
        
        active_sources = id_to_source[active_ids]
        unique_active_sources = np.unique(active_sources)
        for sid in unique_active_sources:
            s_mask = (active_sources == sid)
            cx = np.mean(x[s_mask])
            cy = np.mean(y[s_mask])
            mvx = np.mean(vx[s_mask])
            mvy = np.mean(vy[s_mask])
            ax.quiver(cx, cy, mvx, mvy, color='black', scale=20, width=0.005, zorder=11, alpha=0.6)

    @staticmethod
    def _analyze_pulse_shape(ax, snap, tracer, source_filter):
        """
        Dual-Mode Analysis:
        1. Single Source: Plots PFT (Time vs Transverse Position).
        2. All Sources (Integrated): Plots Phase Space (Time vs Wavelength) + Histogram.
        """
        id_to_source = tracer.rays['source_id'].values
        global_ids = snap['ids']
        
        # --- 1. FILTER DATA ---
        if source_filter is not None:
            mask = (id_to_source[global_ids] == source_filter)
            mode = "PFT"
        else:
            mask = np.ones(len(global_ids), dtype=bool)
            mode = "PHASE_SPACE"
            
        indices = np.where(mask)[0]
        if len(indices) < 5: 
            ax.text(0.5, 0.5, "No active rays", ha='center', va='center')
            return

        # Extract Raw Data
        x, y = snap['x'][indices], snap['y'][indices]
        vx, vy = snap['vx'][indices], snap['vy'][indices]
        wls = tracer.rays['wavelength'][global_ids[indices]].values
        # Note: In ray_trace.py, t_total is Absolute Time. 
        # For a pulse duration, we need Relative Arrival Time at the current plane.
        # We project the ray forward/backward to a common plane perpendicular to the average velocity.
        
        avg_vx, avg_vy = np.mean(vx), np.mean(vy)
        theta = np.arctan2(avg_vy, avg_vx)
        
        # Centroid
        cx, cy = np.mean(x), np.mean(y)
        dx, dy = x - cx, y - cy
        
        # Longitudinal coordinate (z_prime) and Transverse (x_prime)
        z_prime = dx * np.cos(theta) + dy * np.sin(theta)
        x_prime = -dx * np.sin(theta) + dy * np.cos(theta)
        
        c_mm_ps = 0.29979
        # Relative time delay: if z_prime is positive (ahead), it arrived "earlier" (negative delay relative to center)
        # But usually we map "Space" to "Time" via t = z/c.
        dt_fs = -(z_prime / c_mm_ps) * 1000.0 
        
        # --- MODE 1: SINGLE SOURCE (PFT ANALYSIS) ---
        if mode == "PFT":
            ax.scatter(x_prime, dt_fs, c=wls, cmap='jet', alpha=0.6, s=15, edgecolors='none')
            try:
                # Linear Fit for Tilt
                coeffs = np.polyfit(x_prime, dt_fs, 1)
                fit_y = np.polyval(coeffs, x_prime)
                ax.plot(x_prime, fit_y, 'k--', lw=1, alpha=0.5)
                ax.text(0.05, 0.95, f"Tilt: {coeffs[0]:.1f} fs/mm", 
                       transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.7))
            except: pass
            
            ax.set_xlabel("Transverse Position [mm]")
            ax.set_ylabel("Relative Time [fs]")
            ax.set_title(f"Pulse Front Tilt (Source #{source_filter})")

        # --- MODE 2: ALL SOURCES (COMPRESSION DIAGNOSTIC) ---
        else:
            # Main Plot: Time vs Wavelength (The "Chirp" Plot)
            # If this is a diagonal line -> Chirped. If flat -> Compressed.
            
            # 1. RMS Duration Calculation
            rms_duration = np.std(dt_fs)
            
            # Scatter Plot
            sc = ax.scatter(wls, dt_fs, c='blue', alpha=0.3, s=5, label='Rays')
            
            # Add a trend line (Chirp)
            try:
                coeffs = np.polyfit(wls, dt_fs, 2)
                fit_w = np.linspace(min(wls), max(wls), 50)
                fit_t = np.polyval(coeffs, fit_w)
                ax.plot(fit_w, fit_t, 'r-', lw=1.5, alpha=0.8, label='Chirp')
            except: pass

            # Stats Overlay
            stats = (f"RMS Duration: {rms_duration:.1f} fs\n"
                     f"Total Span: {max(dt_fs)-min(dt_fs):.0f} fs")
            
            ax.text(0.05, 0.95, stats, transform=ax.transAxes, va='top', 
                   bbox=dict(facecolor='white', edgecolor='red', alpha=0.9, pad=3))
            
            ax.set_xlabel(r"Wavelength [mm]")
            ax.set_ylabel("Relative Delay [fs]")
            ax.set_title("Longitudinal Phase Space (Chirp)")
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Optional: Histogram on side (conceptually)
            # For now, just the phase space is dense enough information.

    @staticmethod
    def _plot_pft_history(ax, tracer, source_filter):
        times, tilts = [], []
        id_to_source = tracer.rays['source_id'].values
        c_mm_ps = 0.29979
        for snap in tracer.snapshots:
            global_ids = snap['ids']
            mask = (id_to_source[global_ids] == source_filter)
            indices = np.where(mask)[0]
            if len(indices) < 5: continue 
            x, y = snap['x'][indices], snap['y'][indices]
            vx, vy = snap['vx'][indices], snap['vy'][indices]
            cx, cy = np.mean(x), np.mean(y)
            theta = np.arctan2(np.mean(vy), np.mean(vx))
            dx, dy = x - cx, y - cy
            z_prime = dx * np.cos(theta) + dy * np.sin(theta)
            x_prime = -dx * np.sin(theta) + dy * np.cos(theta)
            time_delay_fs = (z_prime / c_mm_ps) * 1000.0
            try:
                coeffs = np.polyfit(x_prime, time_delay_fs, 1) 
                tilts.append(coeffs[0])
                times.append(snap['t'])
            except: pass
        ax.plot(times, tilts, 'b.-')
        ax.set_xlabel("Simulation Time [ps]")
        ax.set_ylabel("Pulse Front Tilt [fs/mm]")
        ax.set_title("PFT Evolution History")
        ax.grid(True)
        ax.axhline(0, color='black', linewidth=1, linestyle='--')

    @staticmethod
    def draw_system(manager, ax=None, tracer=None, draw_labels=True, show_skeleton=True, show_curvature=True, show_focus=True, auto_dimension=True, draw_beam_arrow=False, show_intersection_points=False, source_id_filter=None, show_plot=True, dimension_overrides=None, label_overrides=None):
        if ax is None: 
            fig, ax = plt.subplots(figsize=(8, 6), layout='constrained', dpi=120)
        if dimension_overrides is None: dimension_overrides = {}
        if label_overrides is None: label_overrides = {}
        ax.grid(False) 
        for opt in manager.elements:
            mat_color = MaterialLib.get_color(opt.material)
            Draw._draw_optic_shape(ax, opt, mat_color)
            if draw_labels:
                manual_shift = label_overrides.get(opt.name, (0, 0))
                Draw._add_smart_label(ax, opt, manual_shift)
            if show_skeleton: Draw._draw_skeleton_dots(ax, opt)
            if opt.optic_type == "Mirror" and opt.RFL is not None: Draw._draw_oap_vectors(ax, opt, show_focus)
            else:
                if show_curvature: Draw._draw_curvature_spokes(ax, opt)
                if show_focus: Draw._draw_focal_spokes(ax, opt)
        if auto_dimension and len(manager.elements) > 1:
            for i in range(len(manager.elements) - 1):
                opt_a = manager.elements[i]
                opt_b = manager.elements[i+1]
                gap_key = f"{opt_a.name}->{opt_b.name}"
                offset = dimension_overrides.get(gap_key, 25)
                if offset is not None: Draw.draw_dimension(ax, opt_a, opt_b, offset=offset)
        if draw_beam_arrow: Draw.draw_electron_beam(ax, angle=0.0)
        if tracer:
            if tracer.rays.empty and hasattr(tracer, '_sync_to_dataframe'): tracer._sync_to_dataframe()
            mappable = Draw.draw_rays(ax, tracer, lw=0.8, alpha=0.5, source_filter=source_id_filter)
            Draw.draw_wavefronts(ax, tracer, color='black', alpha=0.6, lw=0.8, source_filter=source_id_filter)
            if show_intersection_points: Draw._draw_intersection_points(ax, tracer, source_filter=source_id_filter)
            Draw.add_colorbar(ax, mappable)
            title_suffix = ""
            if source_id_filter is not None: title_suffix = f" (Source #{source_id_filter})"
            Draw.finalize(ax, title=f"SPR Compressor{title_suffix}", show=show_plot)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Global X [mm]"); ax.set_ylabel("Global Y [mm]")
        return ax

    @staticmethod
    def add_colorbar(ax, mappable, label="Wavelength [um]"):
        if mappable and ax.figure:
            if not any(hasattr(a, 'get_label') and a.get_label() == '<colorbar>' for a in ax.figure.axes):
                 ax.figure.colorbar(mappable, ax=ax, label=label, shrink=0.8, aspect=30)

    @staticmethod
    def finalize(ax, title="SPR Compressor", show=True):
        if title: ax.set_title(title, fontsize=14, pad=10)
        if show: plt.show()
    
    # --- GEOMETRY HELPERS (Unchanged) ---
    @staticmethod
    def draw_wavefronts(ax, tracer, color='black', alpha=0.6, lw=0.8, source_filter=None):
        if not tracer.snapshots: return
        if 'source_id' not in tracer.rays.columns: return 
        for snap in tracer.snapshots:
            x_arr, y_arr, ids = snap['x'], snap['y'], snap['ids']
            if len(ids) == 0: continue
            df_snap = pd.DataFrame({'x': x_arr, 'y': y_arr, 'id': ids})
            id_to_source = tracer.rays['source_id'].to_dict()
            df_snap['source_id'] = df_snap['id'].map(id_to_source)
            for s_id, group in df_snap.groupby('source_id'):
                if source_filter is not None and s_id != source_filter: continue
                group = group.sort_values('id')
                ax.plot(group['x'], group['y'], linestyle='--', color=color, alpha=alpha, lw=lw, zorder=5)

    @staticmethod
    def _draw_intersection_points(ax, tracer, source_filter=None):
        xs, ys = [], []
        for ray_idx, history in tracer.history.items():
            if source_filter is not None:
                if tracer.rays.at[ray_idx, 'source_id'] != source_filter: continue
            if len(history) > 1:
                intermediates = history[1:] 
                for px, py in intermediates: xs.append(px); ys.append(py)
        if xs: ax.scatter(xs, ys, color='black', s=10, zorder=4, alpha=0.8)

    @staticmethod
    def draw_rays(ax, tracer, cmap_name='jet', alpha=0.6, lw=1.0, source_filter=None):
        wls = tracer.rays['wavelength'].values
        norm = mcolors.Normalize(vmin=wls.min(), vmax=wls.max())
        cmap = cm.get_cmap(cmap_name)
        lines, colors = [], []
        for idx, history_points in tracer.history.items():
            if source_filter is not None:
                if tracer.rays.at[idx, 'source_id'] != source_filter: continue
            points = list(history_points)
            current_x, current_y = tracer.rays.at[idx, 'x'], tracer.rays.at[idx, 'y']
            last_x, last_y = points[-1]
            if (current_x != last_x) or (current_y != last_y): points.append((current_x, current_y))
            if len(points) < 2: continue
            for i in range(len(points)-1):
                lines.append([points[i], points[i+1]])
                colors.append(cmap(norm(tracer.rays.at[idx, 'wavelength'])))
        lc = LineCollection(lines, colors=colors, linewidths=lw, alpha=alpha, zorder=1)
        ax.add_collection(lc)
        return cm.ScalarMappable(norm=norm, cmap=cmap)

    @staticmethod
    def draw_dimension(ax, opt_a, opt_b, offset=25):
        def get_anchor(opt, is_source):
            if opt.optic_type == "Prism":
                candidates = []
                for f in ["FaceA", "FaceB", "FaceC"]:
                    if f in opt.coords: candidates.append((opt.coords[f]["vertex_x"], opt.coords[f]["vertex_y"]))
                ref_x, ref_y = (opt_a.x_center, opt_a.y_center) if not is_source else (opt_b.x_center, opt_b.y_center)
                if candidates:
                    dists = [(c[0]-ref_x)**2 + (c[1]-ref_y)**2 for c in candidates]
                    return candidates[np.argmin(dists)]
            if opt.optic_type in ["Mirror", "Grating", "Source", "Detector"]:
                if "Front" in opt.coords: return opt.coords["Front"]["vertex_x"], opt.coords["Front"]["vertex_y"]
            if opt.optic_type == "Lens":
                surfs = [s for s in ["Front", "Back", "Center"] if s in opt.coords]
                ref_x, ref_y = (opt_b.x_center, opt_b.y_center) if is_source else (opt_a.x_center, opt_a.y_center)
                best_pt, min_dist = (opt.x_center, opt.y_center), np.inf
                for s in surfs:
                    d = opt.coords[s]
                    px, py = d["vertex_x"], d["vertex_y"]
                    dist = (px-ref_x)**2 + (py-ref_y)**2
                    if dist < min_dist: min_dist = dist; best_pt = (px, py)
                return best_pt
            return opt.x_center, opt.y_center
        x1, y1 = get_anchor(opt_a, is_source=True)
        x2, y2 = get_anchor(opt_b, is_source=False)
        angle = np.arctan2(y2 - y1, x2 - x1)
        nx, ny = -np.sin(angle), np.cos(angle)
        lx1, ly1 = x1 + offset * nx, y1 + offset * ny
        lx2, ly2 = x2 + offset * nx, y2 + offset * ny
        ax.plot([x1, lx1], [y1, ly1], color='blue', lw=0.8, ls='--', alpha=0.4, clip_on=True)
        ax.plot([x2, lx2], [y2, ly2], color='blue', lw=0.8, ls='--', alpha=0.4, clip_on=True)
        ax.plot([lx1, lx2], [ly1, ly2], color='blue', lw=1.2, clip_on=True, zorder=10)
        dx, dy = lx2 - lx1, ly2 - ly1
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 1e-6: return
        ux, uy = dx/mag, dy/mag
        ax.quiver([lx1, lx2], [ly1, ly2], [-ux, ux], [-uy, uy], color='blue', scale=15, scale_units='height', width=0.004, headwidth=4, headlength=5, pivot='tip', zorder=11)
        tx, ty = (lx1+lx2)/2, (ly1+ly2)/2 + 4*ny
        ax.text(tx, ty, f"{mag:.2f} mm", color='blue', fontsize=8, fontweight='bold', ha='center', va='center', clip_on=True, zorder=12, bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=2))

    @staticmethod
    def _add_smart_label(ax, opt, manual_shift):
        theta = np.deg2rad(opt.orientation_angle)
        nx, ny = np.cos(theta), np.sin(theta)
        tx, ty = -np.sin(theta), np.cos(theta)
        dist = 10.0 
        if opt.optic_type == "Lens": lx, ly = opt.x_center + (opt.diameter/2 + dist)*tx, opt.y_center + (opt.diameter/2 + dist)*ty
        elif opt.optic_type in ["Mirror", "Grating"]: lx, ly = opt.x_center - (opt.center_thickness + dist)*nx, opt.y_center - (opt.center_thickness + dist)*ny
        elif opt.optic_type == "Prism": lx, ly = opt.x_center + (opt.side_length/2 + dist)*ty, opt.y_center + (opt.side_length/2 + dist)*tx
        else: lx, ly = opt.x_center + dist, opt.y_center + dist
        lx += manual_shift[0]; ly += manual_shift[1]
        ax.text(lx, ly, f"{opt.name}\nØ{opt.diameter}mm", fontsize=8, fontweight='bold', ha='center', va='center', clip_on=True, zorder=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

    @staticmethod
    def draw_electron_beam(ax, length=25.4, start_x=0.0, start_y=0.0, angle=0.0):
        rad = np.deg2rad(angle)
        end_x, end_y = start_x + length * np.cos(rad), start_y + length * np.sin(rad)
        ax.plot([start_x, end_x], [start_y, end_y], color='purple', lw=2, clip_on=True, zorder=9)
        ax.plot(end_x, end_y, marker='>', color='purple', markersize=8, clip_on=True, zorder=9)
        nx, ny = np.sin(rad), -np.cos(rad)
        tx, ty = (start_x + end_x)/2 + 4*nx, (start_y + end_y)/2 + 4*ny
        ax.text(tx, ty, "e- Beam", color='purple', rotation=angle, ha='center', va='center', fontweight='bold', clip_on=True, zorder=10)
        ax.plot(tx, ty, alpha=0)

    @staticmethod
    def _draw_oap_vectors(ax, opt, show_focus):
        if not show_focus or opt.RFL is None: return
        theta_mech = np.deg2rad(opt.orientation_angle)
        bend_total = np.deg2rad(opt.reflection_angle)
        is_left = (getattr(opt, 'handedness', 'Right') == 'Left') or (opt.off_axis_distance < 0)
        sign = 1 if is_left else -1
        theta_refl = theta_mech + sign * bend_total
        fx, fy = opt.x_center + opt.RFL * np.cos(theta_refl), opt.y_center + opt.RFL * np.sin(theta_refl)
        ax.plot(fx, fy, '*', color='gold', markeredgecolor='darkgoldenrod', ms=12, zorder=6)
        if "Front" in opt.coords:
            d = opt.coords["Front"]
            p_top = (d["aperture_x"][0], d["aperture_y"][0]) if "aperture_x" in d else (d["skeleton_x"][0], d["skeleton_y"][0])
            p_bot = (d["aperture_x"][1], d["aperture_y"][1]) if "aperture_x" in d else (d["skeleton_x"][-1], d["skeleton_y"][-1])
            ax.plot([p_top[0], fx], [p_top[1], fy], '--', color='gold', lw=0.8, alpha=0.6)
            ax.plot([p_bot[0], fx], [p_bot[1], fy], '--', color='gold', lw=0.8, alpha=0.6)
            ax.add_patch(Polygon(np.array([[fx, fy], p_top, p_bot]), closed=True, facecolor='gold', alpha=0.2, edgecolor='none', zorder=0))
        vx, vy = np.cos(theta_refl), np.sin(theta_refl)
        ax.text(fx - vy*8, fy + vx*8, f"RFL={opt.RFL:.0f}mm", color='darkgoldenrod', fontsize=7, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    @staticmethod
    def _draw_optic_shape(ax, opt, mat_color):
        if opt.optic_type == "Lens":
            if "Front" in opt.coords and "Back" in opt.coords:
                xs = np.concatenate([opt.coords["Front"]["skeleton_x"], opt.coords["Back"]["skeleton_x"][::-1]])
                ys = np.concatenate([opt.coords["Front"]["skeleton_y"], opt.coords["Back"]["skeleton_y"][::-1]])
                ax.add_patch(Polygon(np.column_stack([xs, ys]), closed=True, facecolor=mat_color, edgecolor='black', linewidth=1.5, alpha=0.5))
        elif opt.optic_type in ["Mirror", "Grating"]:
            if all(s in opt.coords for s in ["Front", "Interface", "Back"]):
                xs_sub = np.concatenate([opt.coords["Interface"]["skeleton_x"], opt.coords["Back"]["skeleton_x"][::-1]])
                ys_sub = np.concatenate([opt.coords["Interface"]["skeleton_y"], opt.coords["Back"]["skeleton_y"][::-1]])
                ax.add_patch(Polygon(np.column_stack([xs_sub, ys_sub]), closed=True, facecolor='#404040', edgecolor='black', linewidth=1.5))
                coat_color = "lightsteelblue" if opt.optic_type == "Grating" else mat_color
                xs_coat = np.concatenate([opt.coords["Front"]["skeleton_x"], opt.coords["Interface"]["skeleton_x"][::-1]])
                ys_coat = np.concatenate([opt.coords["Front"]["skeleton_y"], opt.coords["Interface"]["skeleton_y"][::-1]])
                ax.add_patch(Polygon(np.column_stack([xs_coat, ys_coat]), closed=True, facecolor=coat_color, edgecolor='black', linewidth=1.0))
        elif opt.optic_type == "Prism":
            if "Vertices" in opt.coords:
                ax.add_patch(Polygon(np.column_stack([opt.coords["Vertices"]["x"], opt.coords["Vertices"]["y"]]), closed=True, facecolor=mat_color, edgecolor='black', linewidth=1.5, alpha=0.5))
        elif opt.optic_type == "Detector":
            if "Front" in opt.coords: ax.plot(opt.coords["Front"]["skeleton_x"], opt.coords["Front"]["skeleton_y"], color="red", linewidth=4)
        elif opt.optic_type == "Source":
            if "Front" in opt.coords:
                ax.plot(opt.coords["Front"]["skeleton_x"], opt.coords["Front"]["skeleton_y"], color="lime", linewidth=2)
                ax.plot(opt.x_center, opt.y_center, 'x', color='red')

    @staticmethod
    def _draw_skeleton_dots(ax, opt):
        allowed_surfaces = []
        if opt.optic_type == "Lens": allowed_surfaces = ["Front", "Back"]
        elif opt.optic_type in ["Mirror", "Grating", "Detector", "Source"]: allowed_surfaces = ["Front"] 
        elif opt.optic_type == "Prism": allowed_surfaces = ["FaceA", "FaceB", "FaceC"]
        for surf_name in allowed_surfaces:
            if surf_name in opt.coords:
                data = opt.coords[surf_name]
                ax.plot(data["skeleton_x"][[0, -1]], data["skeleton_y"][[0, -1]], 'o', color='green', ms=3, zorder=5)
                ax.plot(data["vertex_x"], data["vertex_y"], '+', color='black', ms=6, zorder=5)
                if "aperture_x" in data: ax.plot(data["aperture_x"], data["aperture_y"], '.', color='red', ms=4, zorder=5)
        if opt.optic_type == "Prism" and "Vertices" in opt.coords:
            v = opt.coords["Vertices"]
            ax.plot(v["x"], v["y"], 'o', color='green', ms=3, zorder=5)
            ax.plot(opt.x_center, opt.y_center, '+', color='black', ms=6, zorder=5)

    @staticmethod
    def _draw_curvature_spokes(ax, optic):
        theta = np.deg2rad(optic.orientation_angle)
        allowed_surfaces = ["Front", "Back"] if optic.optic_type == "Lens" else ["Front"]
        def draw(surf_key, R_val):
            if surf_key not in allowed_surfaces or surf_key not in optic.coords: return
            if np.isinf(R_val) or R_val == 0: return
            d = optic.coords[surf_key]
            cx, cy = d["vertex_x"] + R_val * np.cos(theta), d["vertex_y"] + R_val * np.sin(theta)
            ax.plot(cx, cy, 'x', color='darkred', ms=6, alpha=0.6)
            for ex, ey in zip(d["skeleton_x"][[0, -1]], d["skeleton_y"][[0, -1]]):
                ax.plot([cx, ex], [cy, ey], linestyle='--', color='red', lw=0.5, alpha=0.3)
            lx, ly = d["vertex_x"] + (R_val * 0.5) * np.cos(theta), d["vertex_y"] + (R_val * 0.5) * np.sin(theta)
            ax.text(lx, ly, f"R={R_val:.0f}mm", color='darkred', fontsize=7, ha='center', va='center', fontweight='bold', clip_on=True, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
        draw("Front", optic.R1); draw("Back", optic.R2)

    @staticmethod
    def _draw_focal_spokes(ax, optic):
        if optic.optic_type not in ["Mirror"] or optic.RFL is not None: return
        theta = np.deg2rad(optic.orientation_angle)
        nx, ny = np.cos(theta), np.sin(theta)
        def draw_focus_helper(fx, fy, f_val, surf_name_for_edges, label_offset_dir=1):
            ax.plot(fx, fy, '*', color='gold', markeredgecolor='darkgoldenrod', ms=12, zorder=6)
            if surf_name_for_edges in optic.coords:
                d = optic.coords[surf_name_for_edges]
                p_top = (d["aperture_x"][0], d["aperture_y"][0]) if "aperture_x" in d else (d["skeleton_x"][0], d["skeleton_y"][0])
                p_bot = (d["aperture_x"][1], d["aperture_y"][1]) if "aperture_x" in d else (d["skeleton_x"][-1], d["skeleton_y"][-1])
                ax.plot([p_top[0], fx], [p_top[1], fy], '--', color='gold', lw=0.8, alpha=0.6)
                ax.plot([p_bot[0], fx], [p_bot[1], fy], '--', color='gold', lw=0.8, alpha=0.6)
                ax.add_patch(Polygon(np.array([[fx, fy], p_top, p_bot]), closed=True, facecolor='gold', alpha=0.2, edgecolor='none', zorder=0))
            ox, oy = -ny, nx
            lx, ly = fx + ox * 8.0 * label_offset_dir, fy + oy * 8.0 * label_offset_dir
            ax.text(lx, ly, f"f={abs(f_val):.0f}mm", color='darkgoldenrod', fontsize=7, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
        if optic.optic_type == "Mirror" and not np.isinf(optic.R1):
            f = optic.R1 / 2.0
            draw_focus_helper(optic.x_center + f * nx, optic.y_center + f * ny, f, "Front", label_offset_dir=1)