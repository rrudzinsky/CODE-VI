import numpy as np

class MaterialLib:
    """
    Central repository for Material Properties (Color & Refractive Index).
    Includes Sellmeier dispersion equations for high-precision optics.
    """
    
    @staticmethod
    def get_color(material_name):
        """Returns a valid Matplotlib color string/hex for visualization."""
        mat = material_name.lower() if material_name else ""
        
        # --- Reflective Optic Custom Colors (Mirrors/Coatings) ---
        if "gold" in mat:      return "gold"
        if "aluminum" in mat:  return "silver"
        if "silver" in mat:    return "silver"
        
        # --- Transmissive Optic Custom Colors (Lenses/Substrates) ---
        if "znse" in mat:      return "orange"
        if "germanium" in mat: return "darkgrey"
        if "silicon" in mat:   return "darkslategray"
        if "bk7" in mat:       return "skyblue"
        if "fused" in mat:     return "lightcyan"
        if "air" in mat:       return "white"
        
        # Fallback
        return "lightgray"

    @staticmethod
    def get_index(material_name, wavelength_um):
        """
        Returns Refractive Index (n) at a specific wavelength [microns].
        Required for Ray Tracing.
        """
        if not material_name: return 1.0
        mat = material_name.lower()
        
        # 1. Background / Simple
        if mat == "vacuum": return 1.0
        if mat == "air":    return 1.00027
        
        # 2. Custom Equations (Sellmeier)
        if "znse" in mat:
            return MaterialLib.n_ZnSe(wavelength_um)
            
        # 3. Standard Catalog (Placeholder fixed values for now)
        # TODO: Add Sellmeier coefficients for N-BK7 and Fused Silica if needed.
        if "bk7" in mat: return 1.5168
        if "fused" in mat: return 1.458
        
        return 1.0 # Default if unknown

    @staticmethod
    def n_ZnSe(lam): 
        """
        ZnSe Refractive Index Formula (Sellmeier-like).
        Source: User provided constants.
        Input: Wavelength in MICRONS (um).
        """
        # Constants
        first_term = 1 - 0.689818 
        
        lam_sq = lam ** 2.0
        
        second_term = (4.855169 * lam_sq) / (lam_sq - 0.056359)
        third_term  = (0.673922 * lam_sq) / (lam_sq - 0.056336)
        fourth_term = (2.481890 * lam_sq) / (lam_sq - 2222.114)
        
        n_squared = first_term + second_term + third_term + fourth_term
        
        # Safety check for sqrt to avoid domain errors
        if n_squared < 0: return 1.0
        return np.sqrt(n_squared)