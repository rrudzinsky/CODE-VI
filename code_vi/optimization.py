import numpy as np
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
from .ray_trace import RayTracer

def _plot_results(x_data, y_data, best_x, title, xlabel, ylabel):
    """Helper to generate clean, toolbar-free plots."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x_data, y_data, '.-', color='C0')
    ax.axvline(best_x, color='r', linestyle='--', label='Optimal Position')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True); ax.legend(); fig.tight_layout()
    if hasattr(fig.canvas, 'toolbar_visible'):
        fig.canvas.toolbar_visible = fig.canvas.header_visible = fig.canvas.footer_visible = False
    plt.show()

def optimize_lens_collimation(manager, lens_name="Lens 1", guess_y=None, search_window=15.0, steps=30, show_plot=True):
    print(f"--- Optimizing {lens_name} for Collimation ---")
    lens = next(el for el in manager.elements if el.name == lens_name)
    guess_y = lens.y_center if guess_y is None else guess_y
        
    test_ys = np.linspace(guess_y - search_window, guess_y + search_window, steps)
    divergences = []
    best_y, min_div = guess_y, np.inf
    
    for y in test_ys:
        lens.y_center = y
        manager._calculate_surface_geometry(lens) 
        
        tracer = RayTracer(manager)
        with redirect_stdout(io.StringIO()):
            tracer.generate_smart_spr_source(
                n_sources=1, rays_per_source=30, target_optic_name=lens_name, 
                grating_search_bounds=(12.0, 13.0), acceptance_angle_range=(70, 110), 
                grating_period=10.0, beam_energy=0.99
            )
        
        for t in np.arange(0, 1500, 50.0): tracer.run_time_step(t, 50.0)
        
        snap = tracer.snapshots[-1]
        if len(snap['ids']) > 5:
            div = np.std(np.arctan2(snap['vy'], snap['vx']))
            divergences.append(div)
            if div < min_div: best_y, min_div = y, div
        else:
            divergences.append(np.nan)
            
    print(f"✅ Best Collimation Position: y = {best_y:.3f} mm (Divergence: {min_div:.6f} rad)")
    if show_plot: _plot_results(test_ys, divergences, best_y, f"{lens_name} Collimation", f"{lens_name} y_center (mm)", "Beam Divergence (rad)")
    
    lens.y_center = best_y
    manager._calculate_surface_geometry(lens) 
    return best_y

def optimize_telecentric_spacing(manager, lens_name="Lens 2", guess_y=None, search_window=15.0, steps=20, show_plot=True):
    print(f"--- Optimizing {lens_name} for Telecentric Spacing ---")
    lens = next(el for el in manager.elements if el.name == lens_name)
    guess_y = lens.y_center if guess_y is None else guess_y
        
    test_ys = np.linspace(guess_y - search_window, guess_y + search_window, steps)
    errors = []
    best_y, min_error = guess_y, np.inf
    
    for y in test_ys:
        lens.y_center = y
        manager._calculate_surface_geometry(lens)
        
        tracer = RayTracer(manager)
        with redirect_stdout(io.StringIO()):
            tracer.generate_smart_spr_source(
                n_sources=3, rays_per_source=20, target_optic_name=lens_name, 
                grating_search_bounds=(0.0, 25.4), acceptance_angle_range=(70, 110), 
                grating_period=10.0, beam_energy=0.99
            )
        
        for t in np.arange(0, 3500, 50.0): tracer.run_time_step(t, 50.0)
        
        snap = tracer.snapshots[-1]
        if len(snap['ids']) > 5:
            src_ids = tracer._source_id[snap['ids']]
            mean_vxs = [np.mean(snap['vx'][src_ids == src]) for src in np.unique(src_ids)]
            error = np.std(mean_vxs) + np.abs(np.mean(mean_vxs))
            errors.append(error)
            
            if error < min_error: best_y, min_error = y, error
        else:
            errors.append(np.nan)
            
    print(f"✅ Best Telecentric Position: y = {best_y:.3f} mm (Error: {min_error:.6f})")
    if show_plot: _plot_results(test_ys, errors, best_y, f"{lens_name} Telecentric Spacing", f"{lens_name} y_center (mm)", "Telecentric Error")
    
    lens.y_center = best_y
    manager._calculate_surface_geometry(lens) 
    return best_y

def optimize_focal_plane(manager, guess_y=None, search_window=15.0, steps=40, show_plot=True):
    print("--- Hunting for Exact Image Plane (Focus) ---")
    if guess_y is None:
        g1 = next((el for el in manager.elements if el.name == "Grating 1"), None)
        guess_y = g1.y_center if g1 else 1000.0

    tracer = RayTracer(manager)
    with redirect_stdout(io.StringIO()):
        tracer.generate_smart_spr_source(
            n_sources=3, rays_per_source=20, target_optic_name="Lens 2", 
            grating_search_bounds=(0, 25.4), acceptance_angle_range=(70, 110), 
            grating_period=10.0, beam_energy=0.99
        )
        
    for t in np.arange(0, 4500, 50.0): tracer.run_time_step(t, 50.0)
    
    snap = tracer.snapshots[-1]
    if len(snap['ids']) < 5: return guess_y
        
    x0, y0, vx, vy = snap['x'], snap['y'], snap['vx'], snap['vy']
    test_planes = np.linspace(guess_y - search_window, guess_y + search_window, steps)
    spot_sizes = []
    best_plane, min_spot = guess_y, np.inf
    
    for plane_y in test_planes:
        projected_x = x0 + vx * ((plane_y - y0) / vy)
        spot_size = np.std(projected_x)
        spot_sizes.append(spot_size)
        if spot_size < min_spot: best_plane, min_spot = plane_y, spot_size

    print(f"✅ Best Focal Plane: y = {best_plane:.3f} mm (RMS Spot Size: {min_spot:.4f} mm)")
    if show_plot: _plot_results(test_planes, spot_sizes, best_plane, "Focal Plane Location Mapping", "Y Position (mm)", "RMS Spot Size (mm)")
    
    return best_plane

def optimize_compressor_gratings(manager, g2_name="Grating 2", focal_plane_base=1000.0, angle_steps=5, offset_steps=5):
    print("--- Hunting for Higher-Order Dispersion Null Point ---")
    g1 = next(el for el in manager.elements if el.name == "Grating 1")
    g2 = next(el for el in manager.elements if el.name == g2_name)
    
    angles, offsets = np.linspace(-2.0, 2.0, angle_steps), np.linspace(-5.0, 5.0, offset_steps)
    best_score, best_params = np.inf, (0, 0)
    
    for offset in offsets:
        for angle in angles:
            g1.y_center, g2.y_center, g2.orientation_angle = focal_plane_base + offset, (focal_plane_base - 25.0) + offset, angle
            manager._calculate_surface_geometry(g1); manager._calculate_surface_geometry(g2)
            
            tracer = RayTracer(manager)
            with redirect_stdout(io.StringIO()):
                tracer.generate_smart_spr_source(
                    n_sources=3, rays_per_source=10, target_optic_name="Lens 2", 
                    grating_search_bounds=(0, 25.4), acceptance_angle_range=(70, 110), 
                    grating_period=10.0, beam_energy=0.99
                )
            for t in np.arange(0, 5000, 100): tracer.run_time_step(t, 100)
            
            snap = tracer.snapshots[-1]
            if len(snap['ids']) < 5: continue
            
            vx, vy = snap['vx'], snap['vy']
            ray_angles = np.arctan2(vy, vx)
            spatial_spread = np.std(ray_angles)
            
            x, y, theta = snap['x'], snap['y'], np.mean(ray_angles)
            dx, dy = x - np.mean(x), y - np.mean(y)
            z_prime = dx * np.cos(theta) + dy * np.sin(theta)
            dt_fs = -(z_prime / 0.29979) * 1000.0
            
            try: chirp_slope, _ = np.polyfit(tracer.rays['wavelength'].iloc[snap['ids']].values, dt_fs, 1)
            except: continue
                
            score = (spatial_spread * 1000) + abs(chirp_slope)
            if score < best_score: best_score, best_params = score, (angle, offset)

    print(f"✅ Best Configuration -> Grating 2 Angle: {best_params[0]:5.2f}°, Offset: {best_params[1]:5.2f}mm")
    
    g1.y_center, g2.y_center, g2.orientation_angle = focal_plane_base + best_params[1], (focal_plane_base - 25.0) + best_params[1], best_params[0]
    manager._calculate_surface_geometry(g1); manager._calculate_surface_geometry(g2)
    return best_params