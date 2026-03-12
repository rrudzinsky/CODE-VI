# %%
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from code_vi.system import OpticsManager
from code_vi.elements import OpticalElement
from code_vi.ray_trace import RayTracer
from code_vi.visualization import Draw
from code_vi import optimization 

# 1. Initialize
manager = OpticsManager()

lens1 = OpticalElement(
    name="Lens 1", optic_type="Lens",
    x_center=12.7, y_center=250.694,
    orientation_angle=90.0,         
    clear_aperture=22.86, diameter=25.4, 
    center_thickness=2.4,           
    R1=np.inf, R2=-350.67,            
    k2=0, coeffs2=[], material="ZnSe"
)

lens2 = OpticalElement(
    name="Lens 2", optic_type="Lens",
    x_center=12.7, y_center=750.389,   # Initial Guess 
    orientation_angle=90.0,          
    clear_aperture=22.86, diameter=25.4, 
    center_thickness=2.4,           
    R1=350.67, R2=np.inf,            
    k1=0, coeffs1=[], material="ZnSe"
)

focal_plane_y = 1008.45               # Initial Guess
focal_plane_offset = 0.0

grating1 = OpticalElement(
    name="Grating 1", optic_type="Grating",
    x_center=12.7, y_center=focal_plane_y + focal_plane_offset, 
    orientation_angle=-90.0,   
    clear_aperture=45, diameter=50.8,
    groove_density=75.0, diffraction_order=-1, material="Gold"
)

grating2 = OpticalElement(
    name="Grating 2", optic_type="Grating",
    x_center=-10.0, y_center=(focal_plane_y - 25.0) + focal_plane_offset, 
    orientation_angle=0.0,   
    clear_aperture=45, diameter=50.8,
    groove_density=25.0, diffraction_order=1, material="Gold" 
)

manager.add_element(lens1)
manager.add_element(lens2)
# manager.add_element(grating1)
# manager.add_element(grating2)

print(f"System loaded with: {[opt.name for opt in manager.elements]}\n")

# %%
import numpy as np
import importlib
from code_vi import optimization 
importlib.reload(optimization)

print("===============================================")
print(" PHASE 1: LENS 1 (COLLIMATOR) OPTIMIZATION")
print("===============================================")
print("Step 1a: Initial Lens 1 Collimation...")
optimization.optimize_lens_collimation(manager, lens_name="Lens 1", show_plot=False)

print("\nStep 1b: Optimizing Lens 1 Aspheric Profile...")
optimization.optimize_aspheric_profile(manager, lens_name="Lens 1", target_surface=2, show_plot=False)

print("\nStep 1c: Final Lens 1 Collimation...")
opt_L1_y = optimization.optimize_lens_collimation(manager, lens_name="Lens 1", show_plot=True)

print("\n===============================================")
print(" PHASE 2: LENS 2 (FOCUSER) OPTIMIZATION")
print("===============================================")
print("Step 2a: Initial Lens 2 Telecentric Spacing...")
optimization.optimize_telecentric_spacing(manager, lens_name="Lens 2", show_plot=False)

print("\nStep 2b: Optimizing Lens 2 Aspheric Profile...")
optimization.optimize_aspheric_profile(manager, lens_name="Lens 2", target_surface=1, show_plot=False)

print("\nStep 2c: Final Lens 2 Telecentric Spacing...")
opt_L2_y = optimization.optimize_telecentric_spacing(manager, lens_name="Lens 2", show_plot=True)

print("\n===============================================")
print(" PHASE 3: FINAL FOCUS & SYSTEM LOCK")
print("===============================================")
print("Mapping Field Curvature...")
focal_x, focal_y = optimization.plot_field_curvature(manager, lens2_name="Lens 2")
opt_focus_y = np.mean(focal_y)

print(f"\n✅ SYSTEM LOCKED.")
print(f"Lens 1 Final Position: {opt_L1_y:.3f} mm")
print(f"Lens 2 Final Position: {opt_L2_y:.3f} mm")
print(f"Focus Final Position:  {opt_focus_y:.3f} mm (avg of per-source foci)")

# %%
%matplotlib widget
tracer = RayTracer(manager)

tracer.generate_smart_spr_source(
    n_sources=5,               
    rays_per_source=30,         
    target_optic_name="Lens 2", 
    grating_search_bounds=(0, 25.4), 
    acceptance_angle_range=(70, 110), 
    grating_period=10.0,
    beam_energy=0.99
)    

print("Running High-Resolution Final Simulation...")
for t in np.arange(0, 5500, 50.0):
    tracer.run_time_step(t, 50.0)

tracer._sync_to_dataframe()

Draw.interactive_session(
    manager, 
    tracer, 
    show_curvature=False, 
    show_skeleton=True,       
    draw_beam_arrow=True,
    show_intersection_points=False
)

# %%
