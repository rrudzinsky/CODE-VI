import numpy as np
import matplotlib.pyplot as plt
from .ray_trace import RayTracer  # <--- Updated to relative import

def optimize_lens1_collimation(manager, lens1_name="Lens 1", guess_y=250.0, search_window=5.0, steps=30, show_plot=True):
    """
    Finds the exact y_center for Lens 1 that minimizes angular divergence (perfect collimation).
    """
    print(f"--- Optimizing {lens1_name} for Collimation ---")
    lens1 = next(el for el in manager.elements if el.name == lens1_name)
    
    test_ys = np.linspace(guess_y - search_window, guess_y + search_window, steps)
    divergences = []
    
    best_y = guess_y
    min_div = np.inf
    
    for y in test_ys:
        lens1.y_center = y
        
        tracer = RayTracer(manager)
        tracer.generate_smart_spr_source(
            n_sources=1, rays_per_source=30, target_optic_name=lens1_name, 
            grating_search_bounds=(12.0, 13.0), acceptance_angle_range=(70, 110), 
            grating_period=10.0, beam_energy=0.99
        )
        
        for t in np.arange(0, 1500, 50.0): tracer.run_time_step(t, 50.0)
        
        snap = tracer.snapshots[-1]
        if len(snap['ids']) > 5:
            angles = np.arctan2(snap['vy'], snap['vx'])
            divergence = np.std(angles)
            divergences.append(divergence)
            
            if divergence < min_div:
                min_div = divergence
                best_y = y
        else:
            divergences.append(np.nan)
            
    print(f"✅ Best Collimation Position: y = {best_y:.3f} mm (Divergence: {min_div:.6f} rad)")
    
    if show_plot:
        plt.figure(figsize=(6,3))
        plt.plot(test_ys, divergences, 'b.-')
        plt.axvline(best_y, color='r', linestyle='--', label='Optimal Position')
        plt.title(f"{lens1_name} Collimation Optimization")
        plt.xlabel("Lens 1 y_center (mm)")
        plt.ylabel("Beam Divergence (rad)")
        plt.grid(True); plt.legend(); plt.show()
    
    lens1.y_center = best_y
    return best_y

def optimize_focal_plane(manager, guess_y=1000.0, search_window=10.0, steps=40, show_plot=True):
    """
    Finds the exact y-coordinate where the spatial spread (spot size) is minimized.
    """
    print("--- Hunting for Exact Image Plane (Focus) ---")
    
    tracer = RayTracer(manager)
    tracer.generate_smart_spr_source(
        n_sources=3, rays_per_source=20, target_optic_name="Lens 2", 
        grating_search_bounds=(0, 25.4), acceptance_angle_range=(70, 110), 
        grating_period=10.0, beam_energy=0.99
    )
    for t in np.arange(0, 4500, 50.0): tracer.run_time_step(t, 50.0)
    
    snap = tracer.snapshots[-1]
    if len(snap['ids']) < 5:
        print("Error: Rays did not reach the focal area.")
        return guess_y
        
    x0, y0 = snap['x'], snap['y']
    vx, vy = snap['vx'], snap['vy']
    
    test_planes = np.linspace(guess_y - search_window, guess_y + search_window, steps)
    spot_sizes = []
    
    best_plane = guess_y
    min_spot = np.inf
    
    for plane_y in test_planes:
        dy = plane_y - y0
        dt = dy / vy
        projected_x = x0 + vx * dt
        
        spot_size = np.std(projected_x)
        spot_sizes.append(spot_size)
        
        if spot_size < min_spot:
            min_spot = spot_size
            best_plane = plane_y

    print(f"✅ Best Focal Plane: y = {best_plane:.3f} mm (RMS Spot Size: {min_spot:.4f} mm)")
    
    if show_plot:
        plt.figure(figsize=(6,3))
        plt.plot(test_planes, spot_sizes, 'g.-')
        plt.axvline(best_plane, color='r', linestyle='--', label='Optimal Focus')
        plt.title("Focal Plane Optimization")
        plt.xlabel("Y Position (mm)")
        plt.ylabel("RMS Spot Size (mm)")
        plt.grid(True); plt.legend(); plt.show()
    
    return best_plane

def optimize_compressor_gratings(manager, g2_name="Grating 2", focal_plane_base=1000.0):
    """
    2D Grid Search to find the optimal combination of Grating 2 Angle and 
    Focal Plane Offset.
    """
    print("--- Hunting for Higher-Order Dispersion Null Point ---")
    
    g1 = next(el for el in manager.elements if el.name == "Grating 1")
    g2 = next(el for el in manager.elements if el.name == g2_name)
    
    angles = np.linspace(-2.0, 2.0, 10)
    offsets = np.linspace(-5.0, 5.0, 10)
    
    best_score = np.inf
    best_params = (0, 0)
    
    for offset in offsets:
        for angle in angles:
            g1.y_center = focal_plane_base + offset
            g2.y_center = (focal_plane_base - 25.0) + offset
            g2.orientation_angle = angle
            
            tracer = RayTracer(manager)
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
            
            x, y = snap['x'], snap['y']
            theta = np.mean(ray_angles)
            dx, dy = x - np.mean(x), y - np.mean(y)
            z_prime = dx * np.cos(theta) + dy * np.sin(theta)
            dt_fs = -(z_prime / 0.29979) * 1000.0
            
            wls = tracer.rays['wavelength'].iloc[snap['ids']].values
            try:
                chirp_slope, _ = np.polyfit(wls, dt_fs, 1)
            except:
                continue
                
            score = (spatial_spread * 1000) + abs(chirp_slope)
            
            if score < best_score:
                best_score = score
                best_params = (angle, offset)

    print(f"✅ Best Configuration -> Grating 2 Angle: {best_params[0]:5.2f}°, Offset: {best_params[1]:5.2f}mm")
    
    g1.y_center = focal_plane_base + best_params[1]
    g2.y_center = (focal_plane_base - 25.0) + best_params[1]
    g2.orientation_angle = best_params[0]
    
    return best_params