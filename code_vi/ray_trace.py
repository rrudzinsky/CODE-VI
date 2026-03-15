import numpy as np
import pandas as pd
from .materials import MaterialLib

class RayTracer:
    def __init__(self, manager, mode="geometric"):
        self.manager = manager
        self.mode = mode  # "geometric" or "wave"
        self.c_mm_ps = 0.299792458
        self._reset_arrays()

    def _reset_arrays(self):
        """Clears all simulation data to prepare for a new run."""
        self._x = np.array([])
        self._y = np.array([])
        self._vx = np.array([])
        self._vy = np.array([])
        self._t_total = np.array([])
        self._wavelength = np.array([])
        self._active = np.array([], dtype=bool)
        self._source_id = np.array([], dtype=int)
        
        self._current_optic_idx = np.array([], dtype=int)
        self._current_surf_in_optic = np.array([], dtype=int)
        self._current_n = np.array([])
        
        # --- WAVE OPTICS ARRAYS ---
        if self.mode == "wave":
            self._opl = np.array([])
            self._phase = np.array([])
            
        self.rays = pd.DataFrame()
        self.history = {}
        self.snapshots = []

    # ------------------------------------------------------------------
    #   SMART SOURCE GENERATOR (Adaptive Phase-Space Sampling)
    # ------------------------------------------------------------------

    def generate_smart_spr_source(self, n_sources=5, rays_per_source=20, 
                                  target_optic_name="Lens 2", 
                                  grating_search_bounds=(0.0, 100.0),
                                  acceptance_angle_range=(70.0, 110.0),
                                  grating_period=10.0,
                                  beam_energy=0.99):
        """
        Adaptive Generator:
        1. Scans grating to find useful spatial extent (First/Last valid point).
        2. Places 'n_sources' within that extent.
        3. For each source, uses bisection to find EXACT angular limits that hit the target.
        4. Generates rays strictly within those limits.
        """
        print(f"--- Smart Source Generation (Target: {target_optic_name}) ---")

        # 1. Locate Target Index
        target_idx = -1
        for i, el in enumerate(self.manager.elements):
            if el.name == target_optic_name:
                target_idx = i
                break
        if target_idx == -1: 
            raise ValueError(f"Target Optic '{target_optic_name}' not found in OpticsManager.")

        # 2. Find Spatial Extent (Coarse Scan)
        print("   Step 1: Finding valid grating length...")
        
        valid_x_min, valid_x_max = self._find_spatial_bounds(
            target_idx, grating_search_bounds, acceptance_angle_range,
            grating_period, beam_energy
        )
        
        if valid_x_min is None:
            print("   Error: No part of the grating can see the target.")
            return

        extent = valid_x_max - valid_x_min
        print(f"   -> Valid Grating Region: {valid_x_min:.2f}mm to {valid_x_max:.2f}mm (Extent: {extent:.2f}mm)")

        # 3. Calculate Source Positions
        source_positions = np.linspace(valid_x_min, valid_x_max, n_sources)
        
        optimized_sources = []

        print(f"   Step 2: Optimizing angles for {n_sources} sources...")
        source_y = 0.0 # Assuming grating is on axis
        
        # 4. Angular Optimization Loop
        for i, sx in enumerate(source_positions):
            theta_min, theta_max = self._find_angular_limits_bisection(
                sx, source_y, target_idx, acceptance_angle_range,
                grating_period, beam_energy
            )
            
            optimized_sources.append({
                'x': sx, 
                'min_ang': theta_min, 
                'max_ang': theta_max
            })

        # 5. Clear Arrays and Generate REAL Rays
        self._reset_arrays()
        
        start_idx = 0
        electron_speed = self.c_mm_ps * beam_energy

        for i, src in enumerate(optimized_sources):
            sx = src['x']
            theta_min = src['min_ang']
            theta_max = src['max_ang']
            
            ang_span = theta_max - theta_min
            
            if ang_span < 1e-6:
                angles = np.array([(theta_min + theta_max)/2])
                n_rays_here = 1
            else:
                angles = np.linspace(theta_min, theta_max, rays_per_source)
                n_rays_here = rays_per_source
            
            print(f"     Src {i+1}: X={sx:.1f}mm | Angles=[{theta_min:.1f}°, {theta_max:.1f}°] ({n_rays_here} rays)")

            # Physics Calculations
            t_delay = sx / electron_speed
            rads = np.deg2rad(angles)
            
            wls = grating_period * (1.0/beam_energy - np.cos(rads))
            
            # Append Data
            self._x = np.concatenate([self._x, np.full(n_rays_here, sx)])
            self._y = np.concatenate([self._y, np.full(n_rays_here, source_y)])
            self._vx = np.concatenate([self._vx, np.cos(rads)])
            self._vy = np.concatenate([self._vy, np.sin(rads)])
            self._t_total = np.concatenate([self._t_total, np.full(n_rays_here, t_delay)])
            
            self._wavelength = np.concatenate([self._wavelength, wls])
            self._active = np.concatenate([self._active, np.zeros(n_rays_here, dtype=bool)])
            self._source_id = np.concatenate([self._source_id, np.full(n_rays_here, i, dtype=int)])
            
            self._current_optic_idx = np.concatenate([self._current_optic_idx, np.zeros(n_rays_here, dtype=int)])
            self._current_surf_in_optic = np.concatenate([self._current_surf_in_optic, np.zeros(n_rays_here, dtype=int)])
            self._current_n = np.concatenate([self._current_n, np.ones(n_rays_here)])

            # --- WAVE OPTICS INIT ---
            if self.mode == "wave":
                self._opl = np.concatenate([self._opl, np.zeros(n_rays_here)])
                # Initial phase is set by the electron arrival time (t_delay * frequency)
                # But for relative pulse compression, setting to 0.0 is often sufficient as a baseline.
                self._phase = np.concatenate([self._phase, np.zeros(n_rays_here)])

            for r in range(n_rays_here):
                self.history[start_idx + r] = [(sx, source_y)]
            
            start_idx += n_rays_here

        print(f"   Done. Generated {len(self._x)} optimized rays.")
        self._sync_to_dataframe()

    # ------------------------------------------------------------------
    #   OPTIMIZATION HELPERS (Bisection Logic)
    # ------------------------------------------------------------------

    def _find_spatial_bounds(self, target_idx, search_bounds, ang_range, grating_period, beam_energy):
        probes = np.linspace(search_bounds[0], search_bounds[1], 20)
        valid_indices = []
        
        for x in probes:
            if self._check_single_point_viability(x, 0.0, target_idx, ang_range, grating_period, beam_energy):
                valid_indices.append(x)
        
        if not valid_indices: return None, None
        return min(valid_indices), max(valid_indices)

    def _find_angular_limits_bisection(self, x, y, target_idx, ang_range, grating_period, beam_energy):
        seeds = np.linspace(ang_range[0], ang_range[1], 40) 
        results = [self._fast_trace_check(x, y, ang, target_idx, grating_period, beam_energy) for ang in seeds]
        
        if not any(results):
            mid = (ang_range[0]+ang_range[1])/2
            return mid, mid

        first_valid_idx = results.index(True)
        last_valid_idx = len(results) - 1 - results[::-1].index(True)
        
        if first_valid_idx == 0:
            theta_min = seeds[0] 
        else:
            low = seeds[first_valid_idx - 1]
            high = seeds[first_valid_idx]
            for _ in range(10):
                mid = (low + high) / 2
                if self._fast_trace_check(x, y, mid, target_idx, grating_period, beam_energy):
                    high = mid 
                else:
                    low = mid 
            theta_min = high

        if last_valid_idx == len(seeds) - 1:
            theta_max = seeds[-1] 
        else:
            low = seeds[last_valid_idx]
            high = seeds[last_valid_idx + 1]
            for _ in range(10):
                mid = (low + high) / 2
                if self._fast_trace_check(x, y, mid, target_idx, grating_period, beam_energy):
                    low = mid 
                else:
                    high = mid 
            theta_max = low
            
        return theta_min, theta_max
    
    def _check_single_point_viability(self, x, y, target_idx, ang_range, grating_period, beam_energy):
        test_angles = np.linspace(ang_range[0], ang_range[1], 7)
        for ang in test_angles:
            if self._fast_trace_check(x, y, ang, target_idx, grating_period, beam_energy):
                return True
        return False

    def _fast_trace_check(self, x, y, angle_deg, target_idx, grating_period, beam_energy):
        rads = np.deg2rad(angle_deg)
        wl_mm = grating_period * (1.0/beam_energy - np.cos(rads))
        
        self._x = np.array([x])
        self._y = np.array([y])
        self._vx = np.array([np.cos(rads)])
        self._vy = np.array([np.sin(rads)])
        
        self._t_total = np.zeros(1)
        self._wavelength = np.array([wl_mm]) 
        self._active = np.zeros(1, dtype=bool) 
        self._source_id = np.zeros(1, dtype=int)
        self._current_optic_idx = np.zeros(1, dtype=int)
        self._current_surf_in_optic = np.zeros(1, dtype=int)
        self._current_n = np.ones(1)

        if self.mode == "wave":
            self._opl = np.zeros(1)
            self._phase = np.zeros(1)

        self.history = {0: [(x, y)]} 
        
        max_dist = 4000.0 
        dt = 50.0 
        sim_time = 0.0
        
        for _ in range(int(max_dist/dt)):
            self.run_time_step(sim_time, dt) 
            sim_time += dt 
            
            if self._current_optic_idx[0] > target_idx:
                break
            if self._current_optic_idx[0] >= len(self.manager.elements):
                break
                
        if self._current_optic_idx[0] <= target_idx:
            return False

        points = self.history[0]
        point_idx = 1 
        
        for opt_idx in range(target_idx + 1):
            el = self.manager.elements[opt_idx]
            ap_radius = el.clear_aperture / 2.0
            
            ang_rad = np.deg2rad(el.orientation_angle)
            tx, ty = -np.sin(ang_rad), np.cos(ang_rad)
            
            num_surfs = len(el.surface_names)
            for _ in range(num_surfs):
                if point_idx < len(points):
                    px, py = points[point_idx]
                    
                    dx = px - el.x_center
                    dy = py - el.y_center
                    radial_dist = abs(dx * tx + dy * ty)
                    
                    if radial_dist > ap_radius:
                        return False 
                    
                    point_idx += 1
                    
        return True

    # ------------------------------------------------------------------
    #   CORE PHYSICS ENGINE (Standard + Wave Toggle)
    # ------------------------------------------------------------------

    def run_time_step(self, t_current, dt):
        t_end_of_step = t_current + dt

        activation_condition = (self._t_total < t_end_of_step) & (~self._active)
        self._active[activation_condition] = True
        
        if not np.any(self._active): return
        
        active_indices = np.where(self._active)[0]
        remaining_dt = np.zeros(len(self._x))
        
        start_times = self._t_total[active_indices]
        effective_start_times = np.maximum(start_times, t_current)
        remaining_dt[active_indices] = t_end_of_step - effective_start_times
        remaining_dt[active_indices] = np.maximum(0.0, remaining_dt[active_indices])
        
        iteration = 0
        while iteration < 20: 
            moving_indices = np.where(remaining_dt > 1e-6)[0]
            if len(moving_indices) == 0: break
            
            speeds = self.c_mm_ps / self._current_n[moving_indices]
            hits, dist_hits, normals, surface_names = self._check_intersections_with_skeleton(moving_indices)
            
            max_dist_for_step = remaining_dt[moving_indices] * speeds + 1e-7
            valid_hit_mask = hits & (dist_hits <= max_dist_for_step)
            
            # --- Handle HITS ---
            if np.any(valid_hit_mask):
                local_hit_idx = np.where(valid_hit_mask)[0] 
                global_hit_idx = moving_indices[local_hit_idx]
                
                h_dist = dist_hits[local_hit_idx]
                self._x[global_hit_idx] += self._vx[global_hit_idx] * h_dist
                self._y[global_hit_idx] += self._vy[global_hit_idx] * h_dist
                
                # WAVE OPTICS UPDATE: Accumulate OPL and Phase for hits
                if self.mode == "wave":
                    opl_step = h_dist * self._current_n[global_hit_idx]
                    self._opl[global_hit_idx] += opl_step
                    # phase = 2pi * OPL / lambda (converting lambda microns to mm)
                    self._phase[global_hit_idx] += (2.0 * np.pi * opl_step) / (self._wavelength[global_hit_idx] / 1000.0)

                t_used = h_dist / speeds[local_hit_idx]
                self._t_total[global_hit_idx] += t_used
                remaining_dt[global_hit_idx] -= t_used
                
                for i, g_idx in enumerate(global_hit_idx):
                    self.history[g_idx].append((self._x[g_idx], self._y[g_idx]))
                    
                    loc = local_hit_idx[i]
                    nx, ny = normals[0][loc], normals[1][loc]
                    s_name = surface_names[loc]
                    
                    self._apply_physics_numpy(g_idx, (nx, ny), s_name)

            # --- Handle FLYING ---
            fly_mask = ~valid_hit_mask
            if np.any(fly_mask):
                local_fly_idx = np.where(fly_mask)[0]
                global_fly_idx = moving_indices[local_fly_idx]
                
                dist = speeds[local_fly_idx] * remaining_dt[global_fly_idx]
                self._x[global_fly_idx] += self._vx[global_fly_idx] * dist
                self._y[global_fly_idx] += self._vy[global_fly_idx] * dist

                # WAVE OPTICS UPDATE: Accumulate OPL and Phase for flyers
                if self.mode == "wave":
                    opl_step = dist * self._current_n[global_fly_idx]
                    self._opl[global_fly_idx] += opl_step
                    self._phase[global_fly_idx] += (2.0 * np.pi * opl_step) / (self._wavelength[global_fly_idx] / 1000.0)
                
                self._t_total[global_fly_idx] += remaining_dt[global_fly_idx]
                remaining_dt[global_fly_idx] = 0 
            
            iteration += 1

        if np.any(self._active):
            snapshot = {
                't': t_end_of_step, 
                'x': self._x[self._active].copy(), 
                'y': self._y[self._active].copy(),
                'vx': self._vx[self._active].copy(),
                'vy': self._vy[self._active].copy(),
                'ids': np.where(self._active)[0]
            }
            if self.mode == "wave":
                snapshot['opl'] = self._opl[self._active].copy()
                snapshot['phase'] = self._phase[self._active].copy()
                
            self.snapshots.append(snapshot)

    def _check_intersections_with_skeleton(self, active_indices):
        n_rays = len(active_indices)
        best_dist = np.full(n_rays, np.inf)
        best_nx, best_ny = np.zeros(n_rays), np.zeros(n_rays)
        best_surf_name = [None] * n_rays
        has_hit = np.zeros(n_rays, dtype=bool)

        for i, ray_idx in enumerate(active_indices):
            optic_idx = self._current_optic_idx[ray_idx]
            if optic_idx >= len(self.manager.elements): continue
            
            optic = self.manager.elements[optic_idx]
            
            if optic.optic_type == "Prism":
                surfaces_to_check = range(len(optic.surface_names))
            else:
                surfaces_to_check = [self._current_surf_in_optic[ray_idx]]

            for surf_idx in surfaces_to_check:
                s_name = optic.surface_names[surf_idx]
                if s_name not in optic.coords: continue

                skel_x = optic.coords[s_name]["skeleton_x"]
                skel_y = optic.coords[s_name]["skeleton_y"]
                
                s1x, s1y = skel_x[:-1], skel_y[:-1]
                s2x, s2y = skel_x[1:], skel_y[1:]
                ox, oy = self._x[ray_idx], self._y[ray_idx]
                dx, dy = self._vx[ray_idx], self._vy[ray_idx]
                
                vx1, vy1 = ox - s1x, oy - s1y
                vx2, vy2 = s2x - s1x, s2y - s1y
                vx3, vy3 = -dy, dx 
                
                denom = vx2 * vx3 + vy2 * vy3
                t_ray = (vx2 * vy1 - vy2 * vx1) / (denom + 1e-12)
                t_seg = (vx1 * vx3 + vy1 * vy3) / (denom + 1e-12)
                
                hits = (t_seg >= 0) & (t_seg <= 1) & (t_ray > 1e-4)
                
                if np.any(hits):
                    current_min_dist = np.min(t_ray[hits])
                    
                    if current_min_dist < best_dist[i]:
                        best_dist[i] = current_min_dist
                        has_hit[i] = True
                        best_surf_name[i] = s_name

                        hit_x, hit_y = ox + dx * current_min_dist, oy + dy * current_min_dist
                        lx, ly = hit_x - optic.x_center, hit_y - optic.y_center
                        ang_rad = np.deg2rad(optic.orientation_angle)
                        
                        local_h = -lx * np.sin(ang_rad) + ly * np.cos(ang_rad)
                        _, slope = self._get_analytical_slope(optic, local_h, s_name)
                        
                        if optic.optic_type == "Prism":
                            local_nz = 1.0; local_nh = -slope 
                        else:
                            local_nz = -1.0 if s_name == "Front" else 1.0
                            local_nh = slope if s_name == "Front" else -slope
                        
                        mag = np.sqrt(local_nz**2 + local_nh**2)
                        local_nz /= mag; local_nh /= mag
                        
                        nx_glob = local_nz * np.cos(ang_rad) - local_nh * np.sin(ang_rad)
                        ny_glob = local_nz * np.sin(ang_rad) + local_nh * np.cos(ang_rad)
                        
                        best_nx[i], best_ny[i] = nx_glob, ny_glob

        return has_hit, best_dist, (best_nx, best_ny), best_surf_name

    def _get_analytical_slope(self, el, y, surf_name):
        if el.optic_type == "Prism": return 0.0, 0.0 

        R = el.R1 if surf_name in ["Front", "Center", "Interface"] else el.R2
        k = el.k1 if surf_name in ["Front", "Center", "Interface"] else el.k2
        coeffs = el.coeffs1 if surf_name in ["Front", "Center", "Interface"] else el.coeffs2
        
        r_eff = y + el.off_axis_distance
        if np.isinf(R) or R == 0: return 0.0, 0.0
        c = 1.0/R
        r2 = r_eff**2
        denom = np.sqrt(max(0, 1 - (1+k)*c**2*r2))
        slope = (c * r_eff) / (denom + 1e-12)
        
        if coeffs:
            for i, coeff in enumerate(coeffs):
                p = 4 + 2*i
                slope += coeff * p * (r_eff**(p-1))
        return 0.0, slope

    def _apply_physics_numpy(self, idx, normal, surf_name):
        el = self.manager.elements[self._current_optic_idx[idx]]
        nx, ny = normal
        vx, vy = self._vx[idx], self._vy[idx]
        
        # --- MIRRORS ---
        if el.optic_type == "Mirror":
            dot = vx*nx + vy*ny
            self._vx[idx] -= 2*dot*nx
            self._vy[idx] -= 2*dot*ny
            self._current_optic_idx[idx] += 1
            self._current_surf_in_optic[idx] = 0

        # --- GRATINGS ---
        elif el.optic_type == "Grating":
            dot = vx*nx + vy*ny
            
            if dot > 0:
                nx, ny = -nx, -ny
                dot = -dot
                
            tx, ty = -ny, nx
            t_in = vx*tx + vy*ty
            
            G = getattr(el, 'groove_density', 0.0)
            m = getattr(el, 'diffraction_order', 1)
            wl_mm = self._wavelength[idx] / 1000.0 
            
            t_out = t_in + (m * wl_mm * G)

            # WAVE OPTICS UPDATE: Instantaneous Grating Phase Jump
            if self.mode == "wave":
                # Find lateral position along the grating surface
                lx = self._x[idx] - el.x_center
                ly = self._y[idx] - el.y_center
                ang_rad = np.deg2rad(el.orientation_angle)
                y_local = -lx * np.sin(ang_rad) + ly * np.cos(ang_rad)
                
                # Apply phase jump based on grating equation
                phase_jump = 2.0 * np.pi * m * G * y_local
                self._phase[idx] += phase_jump
            
            if abs(t_out) > 1.0:
                self._active[idx] = False 
            else:
                n_out = np.sqrt(1.0 - t_out**2)
                self._vx[idx] = t_out*tx + n_out*nx
                self._vy[idx] = t_out*ty + n_out*ny
                
            self._current_optic_idx[idx] += 1
            self._current_surf_in_optic[idx] = 0
            
        # --- LENSES & PRISMS ---
        elif el.optic_type in ["Lens", "Prism"]:
            n1 = self._current_n[idx]
            wl_microns = self._wavelength[idx]
            
            if n1 < 1.01:
                n2 = MaterialLib.get_index(el.material, wl_microns)
            else:
                n2 = 1.0
            
            mu = n1 / n2
            dot = vx * nx + vy * ny
            
            if dot > 0:
                nx, ny = -nx, -ny
                dot = -dot
            
            cos_i = -dot
            sin2_t2 = mu**2 * (1.0 - cos_i**2)
            
            if sin2_t2 > 1.0: 
                self._vx[idx] = vx - 2 * dot * nx
                self._vy[idx] = vy - 2 * dot * ny
            else:
                factor = mu * cos_i - np.sqrt(max(0, 1.0 - sin2_t2))
                self._vx[idx] = mu*vx + factor*nx
                self._vy[idx] = mu*vy + factor*ny
                self._current_n[idx] = n2
            
            if el.optic_type == "Lens":
                self._current_surf_in_optic[idx] += 1
                if self._current_surf_in_optic[idx] >= len(el.surface_names):
                    self._current_optic_idx[idx] += 1
                    self._current_surf_in_optic[idx] = 0
            
            elif el.optic_type == "Prism":
                if n2 < 1.01:
                    self._current_optic_idx[idx] += 1
                    self._current_surf_in_optic[idx] = 0
                else:
                    self._current_surf_in_optic[idx] = 0
        
        elif el.optic_type in ["Detector", "Source"]:
            self._current_optic_idx[idx] = 999 

    def _sync_to_dataframe(self):
        data_dict = {
            'x': self._x, 'y': self._y, 'vx': self._vx, 'vy': self._vy,
            'source_id': self._source_id, 'active': self._active,
            'wavelength': self._wavelength, 't_total': self._t_total
        }
        
        # Pass OPL and Phase to the final DataFrame for analysis
        if self.mode == "wave":
            data_dict['opl'] = self._opl
            data_dict['phase'] = self._phase
            
        self.rays = pd.DataFrame(data_dict)