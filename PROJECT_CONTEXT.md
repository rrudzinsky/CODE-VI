# Project Overview: SPR Compressor Architecture

## The Goal
The objective of this codebase is to simulate the generation of Smith-Purcell Radiation (SPR) from a metallic-coated grating as a Laser Wakefield Accelerator Electron Bunch passes by it, and subsequently compress the resulting broad (~86 ps), diverging pulse using a series of optical elements to a transform-limited duration (sub-ps). 

## SPR Pulse Characteristics:
Unlike a standard ultrafast laser pulse, Smith-Purcell Radiation (SPR) is not born as a clean, collimated beam. It is generated as a divergent fan of radiation directly from the interaction of an electron beam and a source grating. This creates several immediate difficulties:
    - Angular Dispersion & Spatial Chirp: Different wavelengths are emitted at different angles according to the Smith-Purcell relation. This means the "colors" of the pulse are spatially separated from the moment of birth, creating a massive spatial chirp as the beam propagates.
    - Pulse Front Tilt (PFT): Because of this angular distribution and the fact that the front edge of the grating emits first relative to the back edge of the grating as the electron bunch passes over it, the pulse envelope is not perpendicular to the direction of propagation. This Pulse Front Tilt must be corrected to achieve any meaningful temporal compression.
    - Divergence: The radiation is not a single "ray" but a distribution. It would be nice in the future to come up with a tunable source (or high-bandwidth of collection). For now though the radiation needs to be capturable with a 1st optic of some sort. The SPR grating acts as its own spectrometer so to say, creating a fan. 

## Design Constraints: 
We cannot simply use a standard "Martinez" or "Treacy" compressor configuration for this problem due to the following physical constraints:

    - Textbook Limitations: Standard compressors assume an incoming collimated beam that already has a temporal chirp. In our case, the "source" is the grating itself. The compressor must handle a divergent, angularly dispersed field, which is inherently produced with a negative chirp which violates the "parallel-input" assumption of standard compressor math.

    - Geometric Realizability (The Normal Trap): We cannot simply place a second grating parallel to the SPR source. If the second grating has a matching groove_density and is oriented parallel to the spr source grating, the resultant diffracted radiation would be normal to the face of the grating. This is physically unrealizable, as the light cannot be collected or steered without being blocked by the grating substrate itself.

    - Simplicity: Using lenses instead of OAPs if possible is better. The design wavelength currently targeted is 10 um (30 THz). The chromatic aberrations over a wide wavelength range should be relatively flat such that lenses are an easy choice for collecting/collimating light when needed. 

    - Telescope Configuration: To successfully compress the negatively chirped SPR pulse (where high-frequency "blue" leads low-frequency "red"), I would like for starters to incorporate initially a telescopic lens configuration. This is included for three critical physical reasons:
    
        - GDD Sign Inversion (Positive Dispersion): A standard grating pair (Treacy) always produces negative Group Delay Dispersion (GDD). By placing a telescope (two lenses in an afocal arrangement) between the gratings, we effectively create a "negative" optical path length. This flips the sign of the dispersion to positive GDD, which is required to delay the leading blue components and allow the trailing red components to catch up.

        - Imaging and Spatial Recombination: The telescope is configured as a 4f imaging system. This images the angularly dispersed fan of radiation from the first grating directly onto the second grating. This ensures that all frequency components are properly "mapped" to the second grating, facilitating the elimination of spatial chirp and lateral displacement that would otherwise prevent the rays from recombining into a single, clean pulse.

        - Magnification Control: The two-lens configuration allows us to adjust the magnification ($M$) of the system. By changing the ratio of the focal lengths ($f_2 / f_1$), we can magnify the angular dispersion. This provides a "lever" to scale the total amount of dispersion achieved without physically moving the gratings to unrealizable distances in the lab.Beam Collimation (Afocal Stability): Unlike a single-lens Martinez setup, a two-lens telescope is afocal. It ensures that if the incoming beam is collimated (after Lens 1), the beam exiting the telescope remains collimated before hitting the second grating. This prevents unwanted quadratic phase shifts (focusing/divergence) from being introduced into the pulse's phase profile, which would complicate the temporal compression.

    - Lens Aberrations: While lenses provide a way to handle SPR collection easily they will also inherently produce spherical aberrations (if they have spherical surfaces). In order to correct for this we must be careful not just about the focal lengths of said lenses but also if they are aspherical or not. Ideally, we should correct the field of curvature of the focus after a telescopic configuration such that all the rays focus to the same line/plane when incident on another optic (for instance a 2nd grating).

    - The Solution Path: We must use the simulation engine to find a composite compressor layout involving gratings or prisms, with lenses, mirrors, etc. that provide the necessary Group Delay Dispersion (GDD) to compress it spatially and longitudinally (with no residual pulse-front tilt). 


## The Codebase (`code_vi`)
The simulation relies on a custom Python library (`code_vi`) that handles exact ray tracing, transfer matrices, and optical path length calculations. The main operational script is `optical_bench.py`, which initializes the `OpticsManager` and `RayTracer`.