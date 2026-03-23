"""
config.py — Simulation Configuration
=====================================
Single source of truth for all mission parameters and perturbation toggles.
Edit values here (or pass a custom SimConfig to main.run()) to configure the
simulation without touching any other file.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np



@dataclass
class SimConfig:
    # ==================================================================================================
    # ORBITAL PARAMETERS
    # ==================================================================================================
    a_geo: float = 45_000_000.0                         # Semi-major axis [m] (LEO for fast debugging)
    e_geo: float = 0.8                                  # Eccentricity
    base_i_deg: float = 0.0                             # Base inclination [deg]
    base_raan_deg: float = 0.0                          # RAAN [deg]
    base_omega_deg: float = 0.0                         # Argument of periapsis [deg]
    start_eccentric_anomaly_deg: float = 0.0             # Starting E [deg] (for jump-starting sim)
    time_init_string: str = "2024 APRIL 10 00:00:00.0"   # SPICE epoch SN2024aggi

    # ==================================================================================================
    # FORMATION DESIGN
    # ==================================================================================================
    target_focal_length: float = 5000.0   # [m]  desired cross-track separation at peak

    # Detector orbital element offsets relative to Aperture
    # (Both share a, e, RAAN, ω.  Only inclination and initial true anomaly differ.)
    use_focal_designator: bool  = True    # If True, auto-compute det_delta_i_deg from target_focal_length
    det_delta_i_deg: float      = 0.012  # [deg] inclination offset (overridden when use_focal_designator=True)
    det_lag_f_deg:   float      = 0.005    # [deg] true anomaly lag: f_det = f_app - det_lag_f_deg

    # The star_vector is computed automatically from orbital_plane_normal(i, RAAN)
    # in run() — it is the aperture orbit normal, which the aperture body +Z tracks.
    star_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])


    # star_vector is set automatically by orbital_plane_normal(i, RAAN) in run()

    # ==================================================================================================
    # MISSION TIMING
    # ==================================================================================================
    ### -important
    period_multiple: float = 0.15               # THIS ONE Fraction of orbital period to simulate
    ### -important
    cal_window_sec: float = 600.0               # Calibration phase duration [s]
    obs_window_sec: float = 600.0              # Observation phase duration [s]
    target_eccentric_anomaly_deg: float = 90.0  # Eccentric anomaly for peak concentration [deg]
    time_step_sec: float = 0.1                  # Simulation step [s]

    # ==================================================================================================
    # CONTROL GAINS
    # ==================================================================================================
    calibration_kp: float = 5.0
    calibration_kd: float = 48.0
    observation_kp: float = 5.0         # Same as cal — no gain drop at obs boundary
    observation_kd: float = 48.0
    ki_fraction: float = 0.2             # Integral gain = ki_fraction * kp
    integral_limit: float = 500.0        # Anti-windup clamp per axis [N·s]

    # ==================================================================================================
    # SPACECRAFT PROPERTIES — APERTURE (Chief)
    # ==================================================================================================
    app_shape: str = "hexagonal"
    mirror_ftf: float = 5               # [m]
    segment_gap: float = 1.0            # [cm]
    app_side: float = 5.0               # [m]
    app_height: float = 1.0             # [m]
    app_thrust: float = 0.0             # [N] (unused — aperture has no thrusters)
    app_srp_model: str = "plate"        # "cannonball" or "plate"
    app_diameter: float = 1000.0        # [m] effective aperture diameter (if plate)
    app_mass_density: float = 200.0     # [kg/m^3]
    rings: int = 1
    gap: float = 1.0                    # [cm]
    flat_to_flat: float = 5.0           # [m]


    # ==================================================================================================
    # SPACECRAFT PROPERTIES — DETECTOR (Deputy)
    # ==================================================================================================
    det_mass: float = 100.0
    det_shape: str = "hexagonal"
    det_side: float = 1                      # [m]
    det_height: float = 0.4                  # [m]
    det_thrust: float = 20.0                 # [N]
    det_num_pixels: int = 2048               # [pixels]
    det_pixel_size: float = 2.5e-6           # [m]
    det_size = det_num_pixels*det_pixel_size # detector total size
    wideview: float = 0.5                    # [m]
    read_every: int = 100                    # [steps]
    

    # ==================================================================================================
    # DETECTOR CONTROL ENABLE
    # ==================================================================================================
    detector_control_on: bool = True    # Master switch for translation control

    # ==================================================================================================
    # MIRROR CONTROLS
    # ==================================================================================================
    mirror_control_on: bool = True    # Master switch for mirror control
    mirror_mass: float = 1.0          # [kg]
    mirror_I = np.diag([5/72, 5/72, 5/36]) * mirror_mass * mirror_ftf**2
    mirror_Q = np.diag([1, 1, 1, 0.1, 0.1, 0.1])    # LQR state cost matrix
    mirror_R = np.diag([1, 1, 1])                   # LQR control cost matrix
    mirror_J_theta: float = 1.25                    # mirror THETA stiffness
    mirror_J_phi: float = 1.25                      # mirror PHI stiffness
    mirror_S: float = 0.5                           # mirror PISTON stiffness
    mirror_dampting_theta: float = 0.1              # mirror THETA damping
    mirror_dampting_phi: float = 0.1                # mirror PHI damping
    mirror_dampting_S: float = 0.1                  # mirror PISTON damping
    actuator_rotor_inertia: float = 1e-6            # actuator rotor inertia
    mirror_rotor_speed: float = 1e-2                # mirror rotor speed
    initial_random_actuation: float = 1e-2          # initial random actuation for mirrors [urad]
    
    # ==================================================================================================
    # PERTURBATION FLAGS
    # ==================================================================================================
    # Each flag independently enables/disables a physical perturbation.
    # Set all to False for a pure two-body simulation.

    enable_rw_jitter: bool = True
    """Enable high-frequency structural micro-vibrations from the spinning reaction wheels."""

    rw_initial_omega_radps: float = 104.7  # ~1000 RPM
    """
    Constant momentum bias applied to wheels at t=0. 
    Required because physically simulated Jitter scales with Omega^2! 
    Because Aperture is perfectly 0-locked, its wheels would never spin up without this bias.
    """

    enable_metrology_noise: bool = True
    """Enable measurement noise (white noise & random walk) on the relative navigation sensors."""

    metrology_noise_std: float = 0.0005
    """1-sigma standard deviation for metrology position noise [m]. Controller sees fuzzed relative positions."""

    # COMMON REACTION WHEEL PROFILES (simIncludeRW.py):
    # ------------------------------------------------------------------
    # "BCT_RWP015"   | Max Mom: 0.015 Nms | Cubesat Scale, Micro-Jitter
    # "Sinclair_rx1" | Max Mom: 1.0 Nms   | SmallSat Scale
    # "Honeywell_HR12"| Max Mom: 12-25 Nms| Mid-size Sat (Medium Jitter)
    # "Honeywell_HR16"| Max Mom: 50-100 Nms| Flagship Sat (High Jitter)

    enable_j2: bool = True
    """Analytical J2 via ExtForceTorque. Safe with all other perturbations."""

    enable_srp: bool = True
    """Native RadiationPressure module (cannonball, SPICE-driven).
    Aperture: Cr=1.2, A≈100 m²; Detector: Cr=1.4, A≈1.66 m².
    NOTE: incompatible with useSphericalHarmonicsGravityModel (Basilisk bug)."""

    enable_third_body: bool = True
    """Sun + Moon third-body gravity via SPICE ephemeris.
    Required for enable_srp=True (provides sun position message)."""

    enable_css_noise: bool = True
    """6-element CSS array + WLS sun-direction estimator on each spacecraft.
    Introduces realistic sun-tracking angular noise into attitude estimation."""

    # CSS sensor parameters (only used when enable_css_noise=True)
    css_noise_std: float = 0.001    # 1σ Gaussian noise per sensor [cosine units ~0.057°]
    css_bias: float = 0.0002        # Constant bias per sensor [cosine units ~0.011°]

    # Physical Simulation Limits (Systems Engineering Constraints)
    metrology_resolution_m: float = 0.00001 # [m] resolution - controller cannot see less than this
    thruster_mib_n: float = 0.001          # [N] minimum force increment
    #control_deadband_m: float = 0.00005     # [m] idle zone for controller mechanism
    control_deadband_m: float = 0.0
    
    # these two must be EITHER, not OR - Hys would choke PVM
    use_hysteresis: bool = False            # [bool] enable Schmitt trigger to prevent MIB chatter
    """ designed to stop thrusters from switiching - waits for huge error before jumping to next MIB"""
    use_pvm: bool = True                   # [bool] enable Delta-Sigma PWM (error diffusion)
    """ designed to keep average force precise to 1/1000th of an MIB
        witching as much as possible to keep average
        HANDLES GRAV DIFF AND SRP BETTER"""

    # ==================================================================================================
    # SRP REFLECTIVITY (used even when enable_srp=True via native module)
    # ==================================================================================================
    app_cr: float = 1.2             # Aperture reflectivity coefficient
    det_cr: float = 1.4             # Detector reflectivity coefficient

    # ==================================================================================================
    # RANDOM SEED
    # ==================================================================================================
    random_seed: int = 1

    # ==================================================================================================
    # PARALLEL RUN FLAGS
    # ==================================================================================================
    save_data: bool = True           # Master switch to save h5 data and sim config
    disable_progress: bool = False   # Set True in monte_carlo workers to suppress tqdm bars
    mirror_plotting: bool = True     # Run mirror_plotting.run() after the sim (requires mirror_control_on)

    # ==================================================================================================
    # OUTPUT DIRECTORY
    # ==================================================================================================
    results_base: str = f"./DBG_2_results_pvm_{use_pvm}_hysteresis_{use_hysteresis}/ring_vs_focal_sweep"