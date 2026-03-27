import numpy as np
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
from scipy.optimize import minimize_scalar, minimize
from IPython.display import display, Image
from .ray_trace import RayTracer

def _plot_results(x_data, y_data, best_x, title, xlabel, ylabel):
    """Generates a static PNG image to completely bypass Jupyter's interactive backend."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x_data, y_data, '.-', color='C0')
    ax.axvline(best_x, color='r', linestyle='--', label=f'Exact Opt: {best_x:.3f}')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    display(Image(data=buf.getvalue()))

def _estimate_focal_length(lens):
    """Estimate thin-lens focal length from lens radii and material (ZnSe n≈2.4)."""
    n = 2.4  
    inv_R1 = 0.0 if np.isinf(lens.R1) else 1.0 / lens.R1
    inv_R2 = 0.0 if np.isinf(lens.R2) else 1.0 / lens.R2
    phi = (n - 1.0) * (inv_R1 - inv_R2)  
    if abs(phi) < 1e-12:
        return 1000.0  
    return abs(1.0 / phi)

def _inject_fixed_rays(tracer, state, n):
    """Re-inject a saved ray state into a fresh tracer for massive speed gains."""
    tracer._x = state['x'].copy()
    tracer._y = state['y'].copy()
    tracer._vx = state['vx'].copy()
    tracer._vy = state['vy'].copy()
    tracer._t_total = state['t'].copy()
    tracer._wavelength = state['wl'].copy()
    tracer._active = state['active'].copy()
    tracer._source_id = state['src'].copy()
    tracer._current_optic_idx = state['optic_idx'].copy()
    tracer._current_surf_in_optic = state['surf'].copy()
    tracer._current_n = state['n'].copy()
    for r in range(n):
        tracer.history[r] = [(state['x'][r], state['y'][r])]
    tracer.snapshots = []

def _save_ray_state(tracer):
    """Save the current ray state from a tracer."""
    return {
        'x': tracer._x.copy(), 'y': tracer._y.copy(),
        'vx': tracer._vx.copy(), 'vy': tracer._vy.copy(),
        't': tracer._t_total.copy(), 'wl': tracer._wavelength.copy(),
        'active': tracer._active.copy(), 'src': tracer._source_id.copy(),
        'optic_idx': tracer._current_optic_idx.copy(),
        'surf': tracer._current_surf_in_optic.copy(),
        'n': tracer._current_n.copy(),
    }

def _trace_through_lens(tracer, lens_idx):
    """Run time steps and capture velocities. Returns a boolean mask of rays that hit."""
    sim_time, dt = 0.0, 50.0
    n_rays = len(tracer._x)
    
    result_vx = tracer._vx.copy()
    result_vy = tracer._vy.copy()
    captured_mask = np.zeros(n_rays, dtype=bool)

    for _ in range(100):
        tracer.run_time_step(sim_time, dt)
        sim_time += dt

        for i in range(n_rays):
            if (tracer._active[i] 
                and tracer._current_optic_idx[i] > lens_idx 
                and not captured_mask[i]):
                result_vx[i] = tracer._vx[i]
                result_vy[i] = tracer._vy[i]
                captured_mask[i] = True

        active = tracer._active
        if np.any(active):
            if np.all(tracer._current_optic_idx[active] > lens_idx):
                break
        else:
            break

    return result_vx, result_vy, captured_mask

def _collimation_metric(vx, vy, src_ids, captured_mask):
    """Pure collimation score. ONLY evaluates rays that successfully passed through the lens."""
    n_captured = np.sum(captured_mask)
    if n_captured < 5:
        return 1e6  
        
    valid_vx = vx[captured_mask]
    valid_vy = vy[captured_mask]
    valid_src_ids = src_ids[captured_mask]
    
    ray_angles = np.arctan2(valid_vy, valid_vx)
    
    divergences = []
    for src in np.unique(valid_src_ids):
        mask = (valid_src_ids == src)
        if np.sum(mask) > 1:
            divergences.append(np.std(ray_angles[mask]))
            
    if not divergences:
        return 1e6
        
    return np.mean(divergences)

def optimize_lens_collimation(manager, lens_name="Lens 1", guess_y=None, search_window=50.0, steps=30, show_plot=True):
    print(f"--- Optimizing {lens_name} for Collimation ---")
    lens = next(el for el in manager.elements if el.name == lens_name)
    lens_idx = next(i for i, el in enumerate(manager.elements) if el.name == lens_name)
    
    center_y = lens.y_center if guess_y is None else guess_y
    original_y = lens.y_center
    
    lens.y_center = center_y
    manager._calculate_surface_geometry(lens)
    
    tracer_init = RayTracer(manager)
    with redirect_stdout(io.StringIO()):
        tracer_init.generate_smart_spr_source(
            n_sources=5, rays_per_source=20, target_optic_name=lens_name, 
            grating_search_bounds=(0.0, 25.4), acceptance_angle_range=(70, 110), 
            grating_period=10.0, beam_energy=0.99
        )
    init_state = _save_ray_state(tracer_init)
    n_total = len(init_state['x'])
    
    lens.y_center = original_y
    manager._calculate_surface_geometry(lens)

    def objective(y):
        lens.y_center = y
        manager._calculate_surface_geometry(lens)
        
        tracer = RayTracer(manager)
        _inject_fixed_rays(tracer, init_state, n_total)
            
        vx, vy, captured_mask = _trace_through_lens(tracer, lens_idx)
        return _collimation_metric(vx, vy, tracer._source_id, captured_mask)

    bounds = (center_y - search_window, center_y + search_window)
    res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-4})
    best_y = res.x
    min_div = res.fun
    
    print(f"✅ Exact Collimation Position: y = {best_y:.4f} mm (Divergence Score: {min_div:.6f})")

    if show_plot: 
        plot_window = 2.5 
        test_ys = np.linspace(best_y - plot_window, best_y + plot_window, steps)
        grid_divs = [objective(y) for y in test_ys]
        _plot_results(test_ys, grid_divs, best_y, f"{lens_name} Collimation", f"{lens_name} y_center (mm)", "Pure Beam Divergence Score")
    
    lens.y_center = best_y
    manager._calculate_surface_geometry(lens) 
    return best_y

def optimize_telecentric_spacing(manager, lens_name="Lens 2", guess_y=None, search_window=50.0, steps=20, show_plot=True):
    print(f"--- Optimizing {lens_name} for Telecentric Spacing ---")
    lens2 = next(el for el in manager.elements if el.name == lens_name)
    lens1 = next((el for el in manager.elements if el.name == "Lens 1"), None)
    
    if guess_y is not None:
        center_y = guess_y
    elif lens1 is not None:
        f_est1 = _estimate_focal_length(lens1)
        f_est2 = _estimate_focal_length(lens2)
        center_y = lens1.y_center + f_est1 + f_est2
    else:
        center_y = lens2.y_center

    tracer_init = RayTracer(manager)
    with redirect_stdout(io.StringIO()):
        tracer_init.generate_smart_spr_source(
            n_sources=3, rays_per_source=20, target_optic_name=lens_name, 
            grating_search_bounds=(0.0, 25.4), acceptance_angle_range=(70, 110), 
            grating_period=10.0, beam_energy=0.99
        )
    init_state = _save_ray_state(tracer_init)
    n_total = len(init_state['x'])

    def objective(y):
        lens2.y_center = y
        manager._calculate_surface_geometry(lens2)
        tracer = RayTracer(manager)
        _inject_fixed_rays(tracer, init_state, n_total)
        
        for t in np.arange(0, 3500, 50.0): tracer.run_time_step(t, 50.0)
        snap = tracer.snapshots[-1]
        
        if len(snap['ids']) > 5:
            src_ids = tracer._source_id[snap['ids']]
            mean_vxs = [np.mean(snap['vx'][src_ids == src]) for src in np.unique(src_ids)]
            return np.sqrt(np.std(mean_vxs)**2 + np.mean(mean_vxs)**2)
        return 1e6

    bounds = (center_y - search_window, center_y + search_window)
    res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-4})
    best_y = res.x
    min_error = res.fun
            
    print(f"✅ Exact Telecentric Position: y = {best_y:.4f} mm (Error: {min_error:.6f})")
    
    if show_plot: 
        plot_window = 10.0 
        test_ys = np.linspace(best_y - plot_window, best_y + plot_window, steps)
        grid_errors = [objective(y) for y in test_ys]
        _plot_results(test_ys, grid_errors, best_y, f"{lens_name} Telecentric Spacing", f"{lens_name} y_center (mm)", "Telecentric Error")
    
    lens2.y_center = best_y
    manager._calculate_surface_geometry(lens2) 
    return best_y

def plot_field_curvature(manager, lens2_name="Lens 2", search_window=50.0):
    print("--- Mapping Focal Surface (Field Curvature) ---")
    l2 = next(el for el in manager.elements if el.name == lens2_name)
    approx_focus_y = l2.y_center + _estimate_focal_length(l2)
    
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
    
    focal_points_x, focal_points_y = [], []
    unique_sources = np.unique(src_ids)
    
    for src in unique_sources:
        mask = (src_ids == src)
        src_x0, src_y0 = x0[mask], y0[mask]
        src_vx, src_vy = vx[mask], vy[mask]
        
        # 🚀 FIX: Use a continuous solver to find the exact sub-micron focal plane for EACH source
        def spot_size_obj(plane_y):
            projected_x = src_x0 + src_vx * ((plane_y - src_y0) / src_vy)
            return np.std(projected_x)
            
        bounds = (approx_focus_y - search_window, approx_focus_y + search_window)
        res = minimize_scalar(spot_size_obj, bounds=bounds, method='bounded', options={'xatol': 1e-5})
        
        best_y = res.x
        min_spot = res.fun
        best_x = np.mean(src_x0 + src_vx * ((best_y - src_y0) / src_vy))
                
        focal_points_x.append(best_x)
        focal_points_y.append(best_y)
        print(f"Source {src}: Focus at X={best_x:.2f}, Y={best_y:.3f} (Spot RMS={min_spot:.4f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(focal_points_x, focal_points_y, 'o-', color='crimson', markersize=8, linewidth=2, label='Focal Surface')
    
    if len(focal_points_x) > 2:
        z = np.polyfit(focal_points_x, focal_points_y, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(focal_points_x), max(focal_points_x), 100)
        ax.plot(x_smooth, p(x_smooth), 'r--', alpha=0.5, label='Petzval Arc Fit')

    ax.axhline(np.mean(focal_points_y), color='k', linestyle='--', alpha=0.3, label='Average Focal Plane')
    ax.set_title("Field Curvature: Focus Location per Source")
    ax.set_xlabel("Transverse Position X (mm)")
    ax.set_ylabel("Longitudinal Focal Position Y (mm)")
    ax.grid(True)
    ax.legend()
    ax.invert_yaxis() 
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    display(Image(data=buf.getvalue()))
    
    return focal_points_x, focal_points_y

def optimize_compressor_gratings(manager, g1_name="Grating 1", g2_name="Grating 2", show_plot=False):
    g1 = next(el for el in manager.elements if el.name == g1_name)
    g2 = next(el for el in manager.elements if el.name == g2_name)
    
    # Extract baseline
    init_g1_y = g1.y_center
    init_g2_x = g2.x_center
    init_g2_y = g2.y_center
    init_g2_angle = g2.orientation_angle

    from code_vi.ray_trace import RayTracer
    import numpy as np
    import io
    from contextlib import redirect_stdout
    from scipy.optimize import minimize

    # Setup Ray Tracer once
    tracer_init = RayTracer(manager)
    with redirect_stdout(io.StringIO()):
        tracer_init.generate_smart_spr_source(
            n_sources=5, rays_per_source=20, target_optic_name="Lens 2", 
            grating_search_bounds=(0, 25.4), acceptance_angle_range=(70, 110), 
            grating_period=10.0, beam_energy=0.99
        )
    # Handle module reloading safety
    if hasattr(manager, '_save_ray_state'):
        init_state = manager._save_ray_state(tracer_init)
    else:
        init_state = __import__('code_vi.optimization', fromlist=['_save_ray_state'])._save_ray_state(tracer_init)
    
    n_total = len(init_state['x'])

    print(f"--- AUTO-DETECTING BEAM REFERENCE FRAME ---")
    # Temporarily banish Grating 2 so we can measure the free-flying diffracted beam
    g2.x_center = 1e6
    manager._calculate_surface_geometry(g2)

    path_tracer = RayTracer(manager)
    __import__('code_vi.optimization', fromlist=['_inject_fixed_rays'])._inject_fixed_rays(path_tracer, init_state, n_total)
    for t in np.arange(0, 5000, 50.0): path_tracer.run_time_step(t, 50.0)

    # Measure the exact centroid velocity of the rainbow leaving G1
    path_snap = path_tracer.snapshots[-1]
    beam_vx = np.mean(path_snap['vx'])
    beam_vy = np.mean(path_snap['vy'])
    beam_angle_rad = np.arctan2(beam_vy, beam_vx)
    
    print(f"   Diffracted Beam Angle: {np.degrees(beam_angle_rad):.2f}°")

    # Restore Grating 2 and calculate the initial user separation distance
    g2.x_center = init_g2_x
    manager._calculate_surface_geometry(g2)
    
    dx_init = init_g2_x - g1.x_center
    dy_init = init_g2_y - init_g1_y
    init_separation = np.sqrt(dx_init**2 + dy_init**2)

    print(f"\n--- INITIALIZING NESTED STEPPER OPTIMIZATION ---")
    print(f" Starting G1 Y      : {init_g1_y:.3f} mm")
    print(f" Starting Separation: {init_separation:.3f} mm (Distance along beam)")
    print(f" Starting G2 Angle  : {init_g2_angle:.3f}°\n")

    # ==========================================
    # INNER LOOP: 2D Optimizer (Separation & Angle)
    # ==========================================
    def optimize_g2_for_given_g1(current_g1_y, guess_separation, guess_g2_angle):
        g1.y_center = current_g1_y
        manager._calculate_surface_geometry(g1)

        def objective_g2(params):
            separation, dg2_angle = params
            
            # Penalize if it tries to put the grating too close, too far, or spin it wildly
            if not (10.0 <= separation <= 800.0) or not (-15.0 <= dg2_angle <= 15.0):
                return 1e12

            # THE MAGIC: Lock Grating 2 onto the diagonal beam vector
            g2.x_center = g1.x_center + separation * np.cos(beam_angle_rad)
            g2.y_center = g1.y_center + separation * np.sin(beam_angle_rad)
            g2.orientation_angle = guess_g2_angle + dg2_angle
            manager._calculate_surface_geometry(g2)

            tracer = RayTracer(manager)
            __import__('code_vi.optimization', fromlist=['_inject_fixed_rays'])._inject_fixed_rays(tracer, init_state, n_total)
            for t in np.arange(0, 5000, 50.0): tracer.run_time_step(t, 50.0)

            snap = tracer.snapshots[-1]
            if len(snap['ids']) < (n_total * 0.5): return 1e10

            vx, vy = snap['vx'], snap['vy']
            ray_angles = np.arctan2(vy, vx)
            theta = np.mean(ray_angles)
            spatial_spread = np.std(ray_angles)

            x, y = snap['x'], snap['y']
            dx, dy = x - np.mean(x), y - np.mean(y)
            z_prime = dx * np.cos(theta) + dy * np.sin(theta)
            dt_fs = -(z_prime / 0.29979) * 1000.0

            wls = tracer._wavelength[snap['ids']]
            try:
                chirp_slope, _ = np.polyfit(wls, dt_fs, 1)
            except:
                return 1e9

            return (spatial_spread * 1e7) + abs(chirp_slope)

        # Hunt for the perfect distance and tilt
        res = minimize(objective_g2, [guess_separation, 0.0], method='Nelder-Mead',
                       bounds=((10.0, 800.0), (-15.0, 15.0)),
                       options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-2})

        best_separation = res.x[0]
        best_g2_angle = guess_g2_angle + res.x[1]
        
        # Calculate the final absolute coordinates
        best_g2_x = g1.x_center + best_separation * np.cos(beam_angle_rad)
        best_g2_y = g1.y_center + best_separation * np.sin(beam_angle_rad)
        
        return best_separation, best_g2_angle, best_g2_x, best_g2_y, objective_g2(res.x)

    # ==========================================
    # OUTER LOOP: The Grating 1 Stepper Algorithm
    # ==========================================
    print(f">>> STEP 0: Evaluating Baseline Guess")
    current_g1_y = init_g1_y
    best_sep, best_g2_angle, best_g2_x, best_g2_y, best_score = optimize_g2_for_given_g1(current_g1_y, init_separation, init_g2_angle)
    print(f"    Baseline Score: {best_score:.2f}\n")

    step_size = 5.0  
    direction = 1.0  

    print(f">>> STEP 1: Testing Forward Direction (+{step_size} mm)")
    test_g1_y = current_g1_y + (step_size * direction)
    test_sep, test_g2_angle, test_g2_x, test_g2_y, test_score = optimize_g2_for_given_g1(test_g1_y, best_sep, best_g2_angle)

    if test_score > best_score:
        print(f"    Score worsened ({test_score:.2f} > {best_score:.2f}). Reversing direction!\n")
        direction = -1.0
        test_g1_y = current_g1_y + (step_size * direction)
        print(f">>> STEP 1 (Reversed): Testing Backward Direction (-{step_size} mm)")
        test_sep, test_g2_angle, test_g2_x, test_g2_y, test_score = optimize_g2_for_given_g1(test_g1_y, best_sep, best_g2_angle)
    else:
        print(f"    Correct direction found! Score improved to {test_score:.2f}\n")

    step_count = 2
    while test_score < best_score and step_count <= 15:
        current_g1_y = test_g1_y
        best_sep = test_sep
        best_g2_angle = test_g2_angle
        best_g2_x = test_g2_x
        best_g2_y = test_g2_y
        best_score = test_score

        test_g1_y = current_g1_y + (step_size * direction)
        print(f">>> STEP {step_count}: Marching to G1 Y = {test_g1_y:.2f} mm")
        test_sep, test_g2_angle, test_g2_x, test_g2_y, test_score = optimize_g2_for_given_g1(test_g1_y, best_sep, best_g2_angle)
        
        if test_score < best_score:
            print(f"    Improvement! Score dropped to {test_score:.2f}\n")
        else:
            print(f"    Score worsened ({test_score:.2f} > {best_score:.2f}). Bottom of the valley found!\n")
            
        step_count += 1

    print(f"✅ STEPPER FINISHED. LOCKING BENCH:")
    print(f"   Final G1 Y     : {current_g1_y:.3f} mm")
    print(f"   Final G2 X,Y   : ({best_g2_x:.3f}, {best_g2_y:.3f}) mm")
    print(f"   Final G2 Angle : {best_g2_angle:.3f}°")
    print(f"   Separation Dist: {best_sep:.3f} mm")
    print(f"   Final Score    : {best_score:.3f}")

    g1.y_center = current_g1_y
    manager._calculate_surface_geometry(g1)
    g2.x_center = best_g2_x
    g2.y_center = best_g2_y
    g2.orientation_angle = best_g2_angle
    manager._calculate_surface_geometry(g2) 
    
    return current_g1_y, best_g2_x, best_g2_y, best_g2_angle, best_score

def optimize_aspheric_profile(manager, lens_name="Lens 2", target_surface=1, show_plot=True):
    print(f"--- Optimizing Aspheric Profile for {lens_name} (Surface {target_surface}) ---")
    lens = next(el for el in manager.elements if el.name == lens_name)
    lens_idx = next(i for i, el in enumerate(manager.elements) if el.name == lens_name)

    if target_surface == 1 and not lens.coeffs1: lens.coeffs1 = [0.0]
    if target_surface == 2 and not lens.coeffs2: lens.coeffs2 = [0.0]

    optimize_R = "1" not in lens_name
    R_init = lens.R1 if target_surface == 1 else lens.R2

    k_bounds = (-2.0, 2.0)       
    A4_bounds = (-5e-5, 5e-5)    

    tracer_init = RayTracer(manager)
    with redirect_stdout(io.StringIO()):
        tracer_init.generate_smart_spr_source(
            n_sources=5, rays_per_source=20, target_optic_name=lens_name, 
            grating_search_bounds=(0.0, 25.4), acceptance_angle_range=(70, 110), 
            grating_period=10.0, beam_energy=0.99
        )
    init_state = _save_ray_state(tracer_init)
    n_total = len(init_state['x'])

    def objective(params):
        if optimize_R:
            R_val, k_val, a4_val = params[0], params[1], params[2]
        else:
            R_val = R_init  
            k_val, a4_val = params[0], params[1]

        if target_surface == 1:
            lens.R1, lens.k1, lens.coeffs1[0] = R_val, k_val, a4_val
        else:
            lens.R2, lens.k2, lens.coeffs2[0] = R_val, k_val, a4_val

        manager._calculate_surface_geometry(lens)

        tracer = RayTracer(manager)
        _inject_fixed_rays(tracer, init_state, n_total)

        if "1" in lens_name:
            vx, vy, captured_mask = _trace_through_lens(tracer, lens_idx)
            return _collimation_metric(vx, vy, tracer._source_id, captured_mask) * 1000.0
            
        else:
            for t in np.arange(0, 4500, 50.0): tracer.run_time_step(t, 50.0)

            snap = tracer.snapshots[-1]
            if len(snap['ids']) < 5: 
                penalty = (k_val**2) + ((a4_val * 1e4)**2)
                if optimize_R: penalty += ((R_val - R_init)**2 / 1e4)
                return 1e6 + penalty

            x0, y0 = snap['x'], snap['y']
            vx, vy = snap['vx'], snap['vy']
            src_ids = tracer._source_id[snap['ids']]
            approx_focus_y = lens.y_center + 250.0
            test_planes = np.linspace(approx_focus_y - 100.0, approx_focus_y + 100.0, 100)

            focal_ys, spot_sizes = [], []
            for src in np.unique(src_ids):
                mask = (src_ids == src)
                src_x0, src_y0, src_vx, src_vy = x0[mask], y0[mask], vx[mask], vy[mask]
                
                min_spot, best_y = np.inf, approx_focus_y
                for plane_y in test_planes:
                    projected_x = src_x0 + src_vx * ((plane_y - src_y0) / src_vy)
                    spot = np.std(projected_x)
                    if spot < min_spot: min_spot, best_y = spot, plane_y
                        
                focal_ys.append(best_y)
                spot_sizes.append(min_spot)

            field_curvature = np.ptp(focal_ys)
            avg_spot = np.mean(spot_sizes)

            return field_curvature + (avg_spot * 10.0)

    k_init = lens.k1 if target_surface == 1 else lens.k2
    a4_init = lens.coeffs1[0] if target_surface == 1 else lens.coeffs2[0]

    if optimize_R:
        if R_init > 0: R_bounds_val = (25.0, 2000.0)
        else: R_bounds_val = (-2000.0, -25.0)
        init_guess = [R_init, k_init, a4_init]
        bnds = (R_bounds_val, k_bounds, A4_bounds)
    else:
        init_guess = [k_init, a4_init]
        bnds = (k_bounds, A4_bounds)

    val1 = objective(init_guess)
    res = minimize(objective, init_guess, method='Nelder-Mead', bounds=bnds, 
                   options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-4})
    
    print(f"   Optimizer result: {res.fun:.6f} (started at {val1:.6f})")

    if optimize_R: R_opt, k_opt, a4_opt = res.x
    else: R_opt, k_opt, a4_opt = R_init, res.x[0], res.x[1]

    r_edge = lens.diameter / 2.0
    edge_departure = abs(a4_opt) * r_edge**4

    if target_surface == 1:
        lens.R1, lens.k1, lens.coeffs1[0] = R_opt, k_opt, a4_opt
        print(f"✅ Aspheric Profile -> R1: {R_opt:.2f} mm, k1: {k_opt:.4f}, A4: {a4_opt:.4e}")
    else:
        lens.R2, lens.k2, lens.coeffs2[0] = R_opt, k_opt, a4_opt
        print(f"✅ Aspheric Profile -> R2: {R_opt:.2f} mm, k2: {k_opt:.4f}, A4: {a4_opt:.4e}")
    
    print(f"   Edge Departure: {edge_departure:.3f} mm ({edge_departure*1000:.1f} µm) at r={r_edge:.1f} mm")

    manager._calculate_surface_geometry(lens)

    if show_plot and "2" in lens_name:
        plot_field_curvature(manager, lens2_name=lens_name)

    return res.x