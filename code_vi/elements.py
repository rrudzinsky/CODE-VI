from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
import numpy as np

@dataclass
class OpticalElement:
    """
    Defines the physical and geometrical properties of a single optical element.
    
    Attributes:
        name (str): Unique identifier for the element.
        optic_type (str): "Lens", "Mirror", "Grating", "Prism", "Detector", or "Source".
        
        geometry:
            diameter (float): Mechanical diameter [mm].
            clear_aperture (float): Optically active diameter [mm].
            center_thickness (float): Thickness at the mechanical axis [mm].
            
        position:
            x_center, y_center (float): Global coordinates of the element center [mm].
            orientation_angle (float): Rotation in degrees (0 = Normal points +X).
            
        physics:
            material (str): Material name for refractive index lookup.
            R1 (float): Front radius of curvature [mm] (Infinity = Flat).
            R2 (float): Back radius of curvature [mm].
            k1, k2 (float): Conic constants (0=Sphere, -1=Parabola).
            coeffs1, coeffs2 (list): Polynomial aspheric coefficients [A4, A6, ...].
            off_axis_distance (float): Internal shift for OAPs [mm].
            
        OAP_specifics:
            RFL (float): Reflected Focal Length [mm]. Defines OAP geometry if set.
            reflection_angle (float): Total bend angle [deg].
            handedness (str): "Right" or "Left".
    """
    name: str
    optic_type: Literal["Lens", "Mirror", "Grating", "Prism", "Detector", "Source"]
    
    # --- GEOMETRY ---
    center_thickness: float = 5.0
    diameter: float = 25.4
    clear_aperture: float = 24.0
    
    # --- POSITION & ORIENTATION ---
    x_center: float = 0.0
    y_center: float = 0.0
    orientation_angle: float = 0.0
    
    material: str = "N-BK7"
        
    # --- CATALOG SPECS (High-level inputs for OAPs) ---
    RFL: Optional[float] = None
    reflection_angle: float = 90.0
    handedness: Literal["Right", "Left"] = "Right"

    # --- PHYSICAL SPECS (Detailed surface definitions) ---
    R1: float = np.inf
    R2: float = np.inf
    k1: float = 0.0
    k2: float = 0.0
    
    # Aspheric Coefficients: [A4, A6, A8, ...]
    # Sag Equation: z = z_conic + A4*r^4 + A6*r^6 ...
    coeffs1: List[float] = field(default_factory=list) 
    coeffs2: List[float] = field(default_factory=list)
    
    # Internal usage for off-axis optics
    off_axis_distance: float = 0.0
    
    # --- PRISM SPECIFICS ---
    side_length: float = 0.0
    
    # --- INTERNAL DATA (Calculated by system.py) ---
    coords: Dict = field(default_factory=dict)
    surface_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validates inputs and auto-configures complex optics (like OAPs) upon creation."""
        
        # 1. OAP AUTO-CONFIGURATION
        # If RFL is defined, we treat this as an Off-Axis Parabola and overwrite physical specs.
        if self.optic_type == "Mirror" and self.RFL is not None:
            # Check if user accidentally tried to set manual physics params too
            conflicts = []
            if self.R1 != np.inf: conflicts.append(f"R1={self.R1}")
            if self.k1 != 0.0:    conflicts.append(f"k1={self.k1}")
            if self.coeffs1:      conflicts.append(f"coeffs1={self.coeffs1}")
            
            if conflicts:
                print(f"Warning '{self.name}': OAP Spec (RFL={self.RFL}) detected, but conflicting manual params found: [{', '.join(conflicts)}].")
                print(f" -> IGNORING manual params. Overwriting with calculated OAP geometry.")
            
            self._configure_as_oap()

        # 2. DATA NORMALIZATION
        self.orientation_angle = self.orientation_angle % 360.0

        # 3. SAFETY CHECKS
        if self.diameter <= 0:
            raise ValueError(f"Error '{self.name}': Diameter must be positive.")
            
        if self.center_thickness <= 0 and self.optic_type in ["Lens", "Prism"]:
            print(f"Warning '{self.name}': Thickness is {self.center_thickness} mm. Physical elements usually have >0 thickness.")

        # Hemisphere Check (Only relevant for standard spherical surfaces)
        if self.k1 == 0 and self.k2 == 0 and self.off_axis_distance == 0 and not self.coeffs1 and not self.coeffs2:
            for val, label in [(self.R1, "R1"), (self.R2, "R2")]:
                if not np.isinf(val) and val != 0:
                    if abs(val) < (self.diameter / 2):
                        print(f"CRITICAL Warning '{self.name}': Radius {label}={val} is smaller than radius of aperture ({self.diameter/2}). Physical hemisphere violation.")

        if self.clear_aperture > self.diameter:
            print(f"Warning '{self.name}': Clear Aperture ({self.clear_aperture}) > Diameter ({self.diameter}). Physically impossible.")

    def _configure_as_oap(self):
        """Internal logic to convert RFL/Angle into Radius/Conic/Shift."""
        if self.RFL <= 0:
             raise ValueError(f"Error '{self.name}': RFL must be positive.")
             
        theta_rad = np.deg2rad(self.reflection_angle)
        
        # Calculate Parent Focal Length (PFL) from Reflected Focal Length
        PFL = self.RFL * (np.cos(theta_rad / 2.0) ** 2)
        
        # Set physical parameters for a Parabola
        self.R1 = 2 * PFL
        self.k1 = -1.0
        self.coeffs1 = [] # Pure conic, no higher order terms
        
        # Calculate the off-axis shift required to achieve the reflection angle
        shift = 2 * PFL * np.tan(theta_rad / 2.0)
        
        if self.handedness == "Left":
            self.off_axis_distance = -shift
        else:
            self.off_axis_distance = shift

    # --- Helper Methods for Alignment ---
    def point_axis_toward(self, target_optic):
        """Rotates this element to face its center toward another element."""
        dx = target_optic.x_center - self.x_center
        dy = target_optic.y_center - self.y_center
        self.orientation_angle = np.degrees(np.arctan2(dy, dx)) % 360.0

    def align_for_reflection(self, source_optic, target_optic):
        """Aligns this mirror to reflect light from Source to Target (Angle Bisection)."""
        v_sx = source_optic.x_center - self.x_center
        v_sy = source_optic.y_center - self.y_center
        v_tx = target_optic.x_center - self.x_center
        v_ty = target_optic.y_center - self.y_center
        
        ang_s = np.arctan2(v_sy, v_sx)
        ang_t = np.arctan2(v_ty, v_tx)
        
        # Use vector bisector approach
        u_s = np.array([np.cos(ang_s), np.sin(ang_s)])
        u_t = np.array([np.cos(ang_t), np.sin(ang_t)])
        bisector = u_s + u_t
        
        self.orientation_angle = np.degrees(np.arctan2(bisector[1], bisector[0])) % 360.0