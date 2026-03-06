import numpy as np
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
from scipy.optimize import minimize_scalar
from .ray_trace import RayTracer

def _plot_results(x_data, y_data, best_x, title, xlabel, ylabel):
    """Absolute basic plotting function."""
    plt.figure(figsize=(6, 3))
    plt.plot(x_data, y_data, '.-', color='C0')
    plt.axvline(best_x, color='r', linestyle='--', label=f'Exact Opt: {best_x:.3f}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def optimize_lens_collimation(manager, lens_name="Lens 1", guess_y=None, search_window=15.0, steps=30, show_plot=True):
    print(f"--- Optimizing {lens_name} for Collimation ---")
    lens = next(el for el in manager.elements if el.name == lens_name)
    guess_y = lens.y_center if guess_y is None else guess_y
        
    def objective(y):
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
            return np.std(np.arctan2(snap['vy'], snap['vx']))
        return 1e6

    test_ys = np.linspace(guess_y - search_window, guess_y + search_window, steps)
    grid_divs = [objective(y) for y in test_ys]
    
    best_discrete_idx = np.argmin(grid_divs)
    step_size = test_ys[1] - test_ys[0]
    bounds = (test_ys[best_discrete_idx] - step_size, test_ys[best_discrete_idx] + step_size)
    
    res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-4})
    best_y = res.x
    min_div = res.fun
            
    print(f"✅ Exact Collimation Position: y = {best_y:.4f} mm (Divergence: {min_div:.6f} rad)")
    if show_plot: 
        _plot_results(test_ys, grid_divs, best_y, f"{lens_name} Collimation", f"{lens_name} y_center (mm)", "Beam Divergence (rad)")
    
    lens.y_center = best_y
    manager._calculate_surface_geometry(lens) 
    return best_y

def optimize_telecentric_spacing(manager, lens_name="Lens 2", guess_y=None, search_window=15.0, steps=20, show_plot=True):
    print(f"--- Optimizing {lens_name} for Telecentric Spacing ---")
    lens = next(el for el in manager.elements if el.name == lens_name)
    guess_y = lens.y_center if guess_y is None else guess_y
        
    def objective(y):
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
            return np.std(mean_vxs) + np.abs(np.mean(mean_vxs))
        return 1e6

    test_ys = np.linspace(guess_y - search_window, guess_y + search_window, steps)
    grid_errors = [objective(y) for y in test_ys]
    
    best_discrete_idx = np.argmin(grid_errors)
    step_size = test_ys[1] - test_ys[0]
    bounds = (test_ys[best_discrete_idx] - step_size, test_ys[best_discrete_idx] + step_size)
    
    res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-4})
    best_y = res.x
    min_error = res.fun
            
    print(f"✅ Exact Telecentric Position: y = {best_y:.4f} mm (Error: {min_error:.6f})")
    if show_plot: 
        _plot_results(test_ys, grid_errors, best_y, f"{lens_name} Telecentric Spacing", f"{lens_name} y_center (mm)", "Telecentric Error")
    
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
    
    def objective(plane_y):
        projected_x = x0 + vx * ((plane_y - y0) / vy)
        return np.std(projected_x)

    test_planes = np.linspace(guess_y - search_window, guess_y + search_window, steps)
    spot_sizes = [objective(y) for y in test_planes]
    
    best_discrete_idx = np.argmin(spot_sizes)
    step_size = test_planes[1] - test_planes[0]
    bounds = (test_planes[best_discrete_idx] - step_size, test_planes[best_discrete_idx] + step_size)
    
    res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-4})
    best_plane = res.x
    min_spot = res.fun

    print(f"✅ Exact Focal Plane: y = {best_plane:.4f} mm (RMS Spot Size: {min_spot:.4f} mm)")
    if show_plot: 
        _plot_results(test_planes, spot_sizes, best_plane, "Focal Plane Location Mapping", "Y Position (mm)", "RMS Spot Size (mm)")
    
    return best_plane

def plot_field_curvature(manager, lens2_name="Lens 2", search_window=50.0, steps=200):
    print("--- Mapping Focal Surface (Field Curvature) ---")
    
    l2 = next(el for el in manager.elements if el.name == lens2_name)
    approx_focus_y = l2.y_center + 250.0 
    
    tracer = RayTracer(manager)
    with redirect_stdout(io.StringIO()):
        tracer.generate_smart_spr_source(
            n_sources=5, rays_per_source=30, target_optic_name=lens2_name, 
            grating_search_bounds=(0, 25.4), acceptance_angle_range=(70, 110), 
            grating_period=10.0, beam_energy=0.99
        )
    
    for t in np.arange(0, 4500, 50.0): 
        tracer.run_time_step(t, 50.0)
        
    snap = tracer.snapshots[-1]
    
    x0, y0 = snap['x'], snap['y']
    vx, vy = snap['vx'], snap['vy']
    src_ids = tracer._source_id[snap['ids']]
    
    focal_points_x = []
    focal_points_y = []
    
    test_planes = np.linspace(approx_focus_y - search_window, approx_focus_y + search_window, steps)
    unique_sources = np.unique(src_ids)
    
    for src in unique_sources:
        mask = (src_ids == src)
        src_x0, src_y0 = x0[mask], y0[mask]
        src_vx, src_vy = vx[mask], vy[mask]
        
        min_spot = np.inf
        best_y = approx_focus_y
        best_x = 0.0
        
        for plane_y in test_planes:
            projected_x = src_x0 + src_vx * ((plane_y - src_y0) / src_vy)
            spot_size = np.std(projected_x)
            
            if spot_size < min_spot:
                min_spot = spot_size
                best_y = plane_y
                best_x = np.mean(projected_x)
                
        focal_points_x.append(best_x)
        focal_points_y.append(best_y)
        print(f"Source {src}: Focus at X={best_x:.2f}, Y={best_y:.2f} (Spot RMS={min_spot:.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(focal_points_x, focal_points_y, 'o-', color='crimson', markersize=8, linewidth=2, label='Focal Surface')
    
    if len(focal_points_x) > 2:
        z = np.polyfit(focal_points_x, focal_points_y, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(focal_points_x), max(focal_points_x), 100)
        plt.plot(x_smooth, p(x_smooth), 'r--', alpha=0.5, label='Petzval Arc Fit')

    plt.axhline(np.mean(focal_points_y), color='k', linestyle='--', alpha=0.3, label='Average Focal Plane')
    plt.title("Field Curvature: Focus Location per Source")
    plt.xlabel("Transverse Position X (mm)")
    plt.ylabel("Longitudinal Focal Position Y (mm)")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis() 
    plt.show()
    
    return focal_points_x, focal_points_y

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