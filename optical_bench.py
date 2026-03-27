import time
import numpy as np
import pickle
import argparse
from code_vi.system import OpticsManager
from code_vi.elements import OpticalElement
from code_vi.ray_trace import RayTracer
from code_vi import optimization 

def build_system():
    """Phase 0: System Initialization"""
    manager = OpticsManager()

    lens1 = OpticalElement(
        name="Lens 1", optic_type="Lens",
        x_center=12.7, y_center=250.405,
        orientation_angle=90.0,         
        clear_aperture=22.86, diameter=25.4, 
        center_thickness=2.4,           
        R1=np.inf, R2=-350.67,            
        k2=0, coeffs2=[], material="ZnSe"
    )

    lens2 = OpticalElement(
        name="Lens 2", optic_type="Lens",
        x_center=12.7, y_center=750.0,
        orientation_angle=90.0,          
        clear_aperture=22.86, diameter=25.4, 
        center_thickness=2.4,
        R1=350.67, R2=np.inf,            
        k1=0, coeffs1=[], material="ZnSe"
    )

    focal_plane_y = 750 + 185.567
    focal_plane_offset = 0

    grating1 = OpticalElement(
        name="Grating 1", optic_type="Grating",
        x_center=12.7, y_center=focal_plane_y + focal_plane_offset, 
        orientation_angle=-91.433,   
        clear_aperture=45, diameter=50.8,
        groove_density=50.0, diffraction_order=1, material="Gold"
    )

    grating2 = OpticalElement(
        name="Grating 2", optic_type="Grating",
        x_center=60.0, y_center=900.0, 
        orientation_angle=125.0,   
        clear_aperture=70.0, diameter=76.0,
        groove_density=50.0, diffraction_order=1, material="Gold" 
    )

    manager.add_element(lens1)
    manager.add_element(lens2)
    manager.add_element(grating1)
    manager.add_element(grating2)
    
    print(f"System loaded with: {[opt.name for opt in manager.elements]}\n")
    return manager

def run_lens_optimization(manager):
    """Phases 1-3: Lens & Focus Optimization"""
    print("===============================================")
    print(" PHASE 1 & 2: LENS OPTIMIZATION")
    print("===============================================")
    
    optimization.optimize_lens_collimation(manager, lens_name="Lens 1", show_plot=False)
    optimization.optimize_aspheric_profile(manager, lens_name="Lens 1", target_surface=2, show_plot=False)
    opt_L1_y = optimization.optimize_lens_collimation(manager, lens_name="Lens 1", show_plot=False)

    optimization.optimize_telecentric_spacing(manager, lens_name="Lens 2", show_plot=False)
    optimization.optimize_aspheric_profile(manager, lens_name="Lens 2", target_surface=1, show_plot=False)
    opt_L2_y = optimization.optimize_telecentric_spacing(manager, lens_name="Lens 2", show_plot=False)

    focal_x, focal_y = optimization.plot_field_curvature(manager, lens2_name="Lens 2")
    opt_focus_y = np.mean(focal_y)
    
    print(f"✅ SYSTEM LOCKED.")
    print(f"Lens 1 Final Position: {opt_L1_y:.3f} mm")
    print(f"Lens 2 Final Position: {opt_L2_y:.3f} mm")
    print(f"Focus Final Position:  {opt_focus_y:.3f} mm (avg of per-source foci)\n")
    return manager

def run_compressor_optimization(manager):
    """Phase 4: Auto-Stepping Vector Compressor Optimization"""
    print("=======================================================")
    print(" PHASE 4: AUTO-STEPPING OPTIMIZER (VECTOR LOCKED)")
    print("=======================================================")
    
    start_time = time.time()
    
    g1 = next(el for el in manager.elements if el.name == "Grating 1")
    g2 = next(el for el in manager.elements if el.name == "Grating 2")

    print(f"Anchoring search to current bench layout:")
    print(f"  G1 Y: {g1.y_center:.2f} | G2 X: {g2.x_center:.2f} | G2 Angle: {g2.orientation_angle:.2f}°\n")

    opt_g1_y, opt_g2_x, opt_g2_y, opt_g2_angle, final_score = optimization.optimize_compressor_gratings(
        manager, 
        g1_name="Grating 1", 
        g2_name="Grating 2"
    )

    print(f"\nOptimization completed in {(time.time() - start_time):.1f} seconds.\n")
    return manager

def run_wave_sim(manager):
    """Phase 5: High-Resolution Wave Simulation"""
    print("--- Running High-Resolution Wave Simulation ---")
    tracer = RayTracer(manager, mode="wave")
    
    tracer.generate_smart_spr_source(
        n_sources=11,               
        rays_per_source=20,         
        target_optic_name="Lens 2", 
        grating_search_bounds=(0, 25.4), 
        acceptance_angle_range=(70, 110), 
        grating_period=10.0,
        beam_energy=0.99
    )   
    
    # Updated to 4000 to match your new bounds
    for t in np.arange(0, 4000, 50.0):
        tracer.run_time_step(t, 50.0)
        
    tracer._sync_to_dataframe()
    return tracer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPR Optical Bench Simulation Engine")
    parser.add_argument('--opt_lenses', action='store_true', help="Run the heavy lens collimation/focuser optimization loops")
    parser.add_argument('--opt_compressor', action='store_true', help="Run the grating auto-stepping optimization")
    parser.add_argument('--wave', action='store_true', help="Run the wave propagation tracer")
    args = parser.parse_args()

    # 1. Build Base Geometry
    system_manager = build_system()

    # 2. Execute requested modules
    if args.opt_lenses:
        system_manager = run_lens_optimization(system_manager)
        
    if args.opt_compressor:
        system_manager = run_compressor_optimization(system_manager)
    
    # 3. Handle Tracer State
    system_tracer = RayTracer(system_manager, mode="wave") # Default empty tracer state
    if args.wave:
        system_tracer = run_wave_sim(system_manager)

    # 4. Final Data Export
    print("Saving system state to system_state.pkl...")
    with open('system_state.pkl', 'wb') as f:
        pickle.dump((system_manager, system_tracer), f)
    print("✅ State saved successfully. You can now refresh the Jupyter Viewer.")