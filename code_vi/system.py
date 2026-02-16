from .elements import OpticalElement
import numpy as np

class OpticsManager:
    """
    Manages the optical system. Responsible for calculating the geometry
    (surface coordinates) of all elements based on their parameters.
    """
    def __init__(self, thin_lens_approx: bool = False, ambient_material: str = "Air"):
        self.thin_lens_approx = thin_lens_approx
        self.ambient_material = ambient_material
        self.elements = []
        
    def add_element(self, element: OpticalElement):
        """Calculates geometry and adds the element to the system."""
        self._calculate_surface_geometry(element)
        self.elements.append(element)
        
    def _calculate_surface_geometry(self, opt):
        """Dispatcher: Decides which geometry generation method to use."""
        theta = np.deg2rad(opt.orientation_angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # 1. LENS
        if opt.optic_type == "Lens":
            if self.thin_lens_approx:
                self._calc_complex_surface(opt, "Center", np.inf, 0.0, [], 0.0, cos_t, sin_t)
                opt.surface_names = ["Center"]
            else:
                # Sequence: Front then Back
                self._calc_complex_surface(opt, "Front", opt.R1, opt.k1, opt.coeffs1, 0.0, cos_t, sin_t)
                self._calc_complex_surface(opt, "Back", opt.R2, opt.k2, opt.coeffs2, opt.center_thickness, cos_t, sin_t)
                opt.surface_names = ["Front", "Back"]

        # 2. MIRROR / GRATING
        elif opt.optic_type in ["Mirror", "Grating"]:
            # Active surface is only the Front for the RayTracer's sequential logic
            self._calc_complex_surface(
                opt, "Front", opt.R1, opt.k1, opt.coeffs1, 0.0, cos_t, sin_t, off_axis=opt.off_axis_distance
            )
            # We still calculate "Back" for drawing purposes, but RayTracer won't look for it
            self._calc_flat_surface(opt, "Back", -opt.center_thickness, cos_t, sin_t)
            opt.surface_names = ["Front"]

        # 3. PRISM
        elif opt.optic_type == "Prism":
            self._calc_prism_detailed(opt, cos_t, sin_t)
            opt.surface_names = ["FaceA", "FaceB", "FaceC"]

        # 4. FLAT ELEMENTS
        elif opt.optic_type in ["Source", "Detector"]:
            self._calc_flat_surface(opt, "Front", 0.0, cos_t, sin_t)
            opt.surface_names = ["Front"]

    def _calc_complex_surface(self, opt, name, R, k, coeffs, z_local_offset, cos_t, sin_t, off_axis=0.0):
        """Calculates global (x,y) coordinates and aperture bounds for any asphere."""
        # Create Grid
        h_vals = np.linspace(-opt.diameter/2, opt.diameter/2, 61)
        r_eff = h_vals + off_axis
        
        # Calculate Sag
        if np.isinf(R) or R == 0:
            z_surf = np.zeros_like(h_vals)
            center_sag = 0
        else:
            c = 1.0/R
            r2 = r_eff**2
            disc = 1 - (1 + k) * c**2 * r2
            z_surf = (c * r2) / (1 + np.sqrt(np.maximum(0, disc)))
            
            if coeffs:
                for i, coeff in enumerate(coeffs):
                    z_surf += coeff * (r_eff ** (4 + 2*i))

            # Normalize so mechanical center (h=0) is at local Z=0
            r_c = off_axis
            disc_c = 1 - (1 + k) * c**2 * r_c**2
            center_sag = (c * r_c**2) / (1 + np.sqrt(np.maximum(0, disc_c)))
            if coeffs:
                for i, coeff in enumerate(coeffs):
                    center_sag += coeff * (r_c ** (4 + 2*i))
            z_surf -= center_sag 

        z_surf += z_local_offset
        
        # Transform to Global
        x_rot = opt.x_center + z_surf * cos_t - h_vals * sin_t
        y_rot = opt.y_center + z_surf * sin_t + h_vals * cos_t
        
        # Aperture Extents
        def get_g_point(h_val):
            r_val = h_val + off_axis
            if np.isinf(R) or R==0: z = 0.0
            else:
                c = 1.0/R
                z = (c*r_val**2)/(1+np.sqrt(max(0, 1-(1+k)*c**2*r_val**2)))
                if coeffs:
                    for i, cf in enumerate(coeffs): z += cf * (r_val ** (4 + 2*i))
                z -= center_sag
            z += z_local_offset
            return (opt.x_center + z * cos_t - h_val * sin_t, 
                    opt.y_center + z * sin_t + h_val * cos_t)

        ap_top = get_g_point(opt.clear_aperture/2)
        ap_bot = get_g_point(-opt.clear_aperture/2)

        opt.coords[name] = {
            "skeleton_x": x_rot, 
            "skeleton_y": y_rot,
            "vertex_x": x_rot[len(x_rot)//2], 
            "vertex_y": y_rot[len(y_rot)//2],
            "aperture_x": np.array([ap_top[0], ap_bot[0]]), 
            "aperture_y": np.array([ap_top[1], ap_bot[1]])
        }

    def _calc_flat_surface(self, opt, name, z_local_offset, cos_t, sin_t):
        self._calc_complex_surface(opt, name, np.inf, 0.0, [], z_local_offset, cos_t, sin_t)

    def _calc_prism_detailed(self, opt, cos_t, sin_t):
        # 1. Calculate Vertices for Visualization (The Missing Part)
        sl = opt.side_length
        # Distance from center to vertex for equilateral triangle
        R_vertex = sl / np.sqrt(3) 
        # Standard vertices are at 90, 210, 330 degrees relative to orientation
        angles_vertex = np.deg2rad(np.array([90, 210, 330]) + opt.orientation_angle)
        
        vx_list = opt.x_center + R_vertex * np.cos(angles_vertex)
        vy_list = opt.y_center + R_vertex * np.sin(angles_vertex)
        
        # Store them so visualization.py can find them
        opt.coords["Vertices"] = {"x": vx_list, "y": vy_list}

        # 2. Calculate Faces for Ray Tracing (Your existing logic)
        h = sl * np.sqrt(3) / 2
        d_face = h / 3
        angles_face = np.deg2rad(np.array([30, 150, 270]) + opt.orientation_angle)
        face_names = ["FaceA", "FaceB", "FaceC"]
        
        for i, ang in enumerate(angles_face):
            vx = opt.x_center + d_face * np.cos(ang)
            vy = opt.y_center + d_face * np.sin(ang)
            tx, ty = -np.sin(ang), np.cos(ang)
            x1, y1 = vx + (sl/2)*tx, vy + (sl/2)*ty
            x2, y2 = vx - (sl/2)*tx, vy - (sl/2)*ty
            opt.coords[face_names[i]] = {
                "vertex_x": vx, "vertex_y": vy,
                "skeleton_x": np.array([x1, x2]), "skeleton_y": np.array([y1, y2]),
                "aperture_x": np.array([x1, x2]), "aperture_y": np.array([y1, y2])
            }