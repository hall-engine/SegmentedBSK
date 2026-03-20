"""
gravity.py — Gravity, SRP, and Perturbation Setup
===================================================
Provides three setup functions called from main.py:

  setup_gravity()  — gravity factory, Earth, optional Sun/Moon, SPICE
  setup_j2()       — optional analytical J2 via ExtForceTorque
  setup_srp()      — optional native RadiationPressure effectors

All functions are gated by the SimConfig perturbation flags so the caller
does not need any conditional logic.
"""

import numpy as np
from Basilisk.utilities import simIncludeGravBody
from Basilisk.simulation import extForceTorque, radiationPressure
from Basilisk.architecture import messaging

from config import SimConfig

# ─── Physical constants ───────────────────────────────────────────────────────
J2_CONST = 1.08262668e-3    # EGM2008
R_EARTH  = 6_378_136.3      # [m]

SRP_P1AU  = 4.56e-6         # Solar pressure at 1 AU [N/m²]
SRP_AU    = 1.496e11        # 1 AU in metres


# =============================================================================
# PUBLIC API
# =============================================================================

def setup_gravity(sim, task_name: str, cfg: SimConfig):
    """
    Create the gravity factory, Earth (central body), optional Sun + Moon
    (third-body), and the SPICE interface.

    Returns
    -------
    grav_factory : simIncludeGravBody.gravBodyFactory
    earth        : the Earth gravity body
    mu           : Earth gravitational parameter [m³/s²]
    sun_msg      : SPICE sun position message (or None if third_body disabled)
    """
    grav_factory = simIncludeGravBody.gravBodyFactory()

    earth = grav_factory.createEarth()
    earth.isCentralBody = True

    # NOTE: useSphericalHarmonicsGravityModel crashes with RadiationPressure +
    # 2 spacecraft (Basilisk linkInStates bug). J2 is handled analytically.
    if cfg.enable_j2:
        print("[Gravity] J2 will be applied analytically via ExtForceTorque "
              f"(J2={J2_CONST:.4e}, R_E={R_EARTH:.1f} m).")
    else:
        print("[Gravity] J2 DISABLED.")

    if cfg.enable_third_body:
        # Creation order determines SPICE message indices:
        #   planetStateOutMsgs[0] = Earth
        #   planetStateOutMsgs[1] = Sun
        #   planetStateOutMsgs[2] = Moon
        grav_factory.createSun()
        grav_factory.createMoon()
        print("[Gravity] Sun + Moon third-body gravity ENABLED (SPICE).")
    else:
        print("[Gravity] Third-body gravity DISABLED (Sun + Moon omitted).")

    grav_factory.createSpiceInterface(time=cfg.time_init_string, epochInMsg=True)
    grav_factory.spiceObject.zeroBase = "Earth"
    sim.AddModelToTask(task_name, grav_factory.spiceObject)

    # Sun message — used by SRP and CSS; None when third-body is off
    if cfg.enable_third_body:
        sun_msg = grav_factory.spiceObject.planetStateOutMsgs[1]
    else:
        sun_msg = None

    mu = earth.mu
    return grav_factory, earth, mu, sun_msg


def setup_j2(sim, task_name: str, cfg: SimConfig, app_sc, det_sc):
    """
    Attach analytical J2 ExtForceTorque effectors to both spacecraft.

    When cfg.enable_j2 is False, returns (None, None, None) — the caller
    should treat a None compute_fn as "no J2 force".

    Returns
    -------
    j2_msg_app   : CmdForceInertialMsg for Aperture (or None)
    j2_msg_det   : CmdForceInertialMsg for Detector (or None)
    compute_fn   : callable(r_vec, mass) → force [N] in ECI (or None)
    """
    if not cfg.enable_j2:
        return None, None, None

    def _compute_j2(r_vec, mass, mu):
        """
        Manually calculate the J2 disturbance force vector.
        We avoid using simIncludeGravBody's useSphericalHarmonicsGravityModel
        as it currently triggers a memory bug when dealing with multiple spacecraft
        and radiationPressure in Basilisk. Instead, standard continuous equations 
        are used here on each tick of main.py.
        """
        x, y, z = r_vec
        r2 = x*x + y*y + z*z
        r  = np.sqrt(r2)
        
        # Calculate scaling coefficients 
        factor = 1.5 * J2_CONST * mu * R_EARTH**2 / r**5
        fz2_r2 = 5.0 * z * z / r2
        
        # Build 3D acceleration vector
        ax = factor * x * (fz2_r2 - 1.0)
        ay = factor * y * (fz2_r2 - 1.0)
        az = factor * z * (fz2_r2 - 3.0)
        
        return mass * np.array([ax, ay, az])

    # Aperture J2
    j2_eff_app = extForceTorque.ExtForceTorque()
    j2_eff_app.ModelTag = "Aperture_J2"
    app_sc.addDynamicEffector(j2_eff_app)
    sim.AddModelToTask(task_name, j2_eff_app)
    j2_msg_app = messaging.CmdForceInertialMsg()
    j2_eff_app.cmdForceInertialInMsg.subscribeTo(j2_msg_app)

    # Detector J2
    j2_eff_det = extForceTorque.ExtForceTorque()
    j2_eff_det.ModelTag = "Detector_J2"
    det_sc.addDynamicEffector(j2_eff_det)
    sim.AddModelToTask(task_name, j2_eff_det)
    j2_msg_det = messaging.CmdForceInertialMsg()
    j2_eff_det.cmdForceInertialInMsg.subscribeTo(j2_msg_det)

    return j2_msg_app, j2_msg_det, _compute_j2


def setup_srp(sim, task_name: str, cfg: SimConfig, app_sc, det_sc, sun_msg):
    """
    Configure SRP for both spacecraft.
    Aperture uses an analytical plate model; Detector uses analytical cannonball.

    Returns
    -------
    srp_app_fn : callable(r_sc, r_sun, sigma_BN, mass) → force [N] in ECI
    srp_det_fn : callable(r_sc, r_sun, sigma_BN, mass) → force [N] in ECI
    """
    if not cfg.enable_srp or sun_msg is None:
        if not cfg.enable_srp:
            print("[SRP] DISABLED.")
        elif sun_msg is None:
            print("[SRP] Skipped — requires enable_third_body=True for sun position.")
        return None, None

    # Constants
    app_area = float(np.pi * (cfg.app_diameter / 2.0)**2)  # circular plate [m²]
    det_area = float(2.0 * np.sqrt(3.0) * cfg.det_side ** 2)  # hex prism cross section [m²]

    def _compute_srp_plate(r_sc, r_sun_n, sigma_BN, mass, area, cr):
        """Analytical flat plate SRP model (Double-Sided)."""
        # 1. Sun vector in inertial frame
        s_n = r_sun_n - r_sc
        d_sun = np.linalg.norm(s_n)
        s_hat_n = s_n / d_sun

        # 2. Body normal in inertial frame (Body +Z is the plate normal)
        s2 = np.dot(sigma_BN, sigma_BN)
        s_cross = np.array([[0, -sigma_BN[2], sigma_BN[1]],
                           [sigma_BN[2], 0, -sigma_BN[0]],
                           [-sigma_BN[1], sigma_BN[0], 0]])
        BN = np.eye(3) + 8*s_cross @ s_cross / (1+s2)**2 + 4*(1-s2)*s_cross / (1+s2)**2
        NB = BN.T
        n_hat_n = NB @ np.array([0, 0, 1])

        # 3. Double-sided logic
        cos_theta = np.dot(s_hat_n, n_hat_n)
        
        # Determine which side is illuminated and flip normal/cos if necessary
        if cos_theta < 0:
            effective_n_hat = -n_hat_n
            effective_cos = -cos_theta
        else:
            effective_n_hat = n_hat_n
            effective_cos = cos_theta

        if effective_cos <= 1e-9:
            return np.zeros(3)

        p_sun = SRP_P1AU * (SRP_AU / d_sun)**2
        
        # 4. Force calculation (simple specular + absorption)
        # F = -P * A * cos(theta) * [(2-Cr)*s_hat + 2*(Cr-1)*cos(theta)*n_hat]
        spec = cr - 1.0
        force_n = -p_sun * area * effective_cos * ((2.0 - cr) * s_hat_n + 2.0 * spec * effective_cos * effective_n_hat)
        return force_n

    def _compute_srp_cannonball(r_sc, r_sun_n, sigma_BN, mass, area, cr):
        """Analytical cannonball SRP model."""
        s_n = r_sun_n - r_sc
        d_sun = np.linalg.norm(s_n)
        s_hat_n = s_n / d_sun
        p_sun = SRP_P1AU * (SRP_AU / d_sun)**2
        return -p_sun * area * cr * s_hat_n

    # Define the per-SC wrappers
    def srp_app_fn(r_sc, r_sun, sigma_BN, mass):
        if cfg.app_srp_model == "plate":
            return _compute_srp_plate(r_sc, r_sun, sigma_BN, mass, app_area, cfg.app_cr)
        return _compute_srp_cannonball(r_sc, r_sun, sigma_BN, mass, app_area, cfg.app_cr)

    def srp_det_fn(r_sc, r_sun, sigma_BN, mass):
        # Detector currently stays cannonball
        return _compute_srp_cannonball(r_sc, r_sun, sigma_BN, mass, det_area, cfg.det_cr)

    print(f"[SRP] ENABLED (Analytical) — "
          f"Aperture: {cfg.app_srp_model} (A={app_area:.1f} m² Cr={cfg.app_cr}) | "
          f"Detector: cannonball (A={det_area:.2f} m² Cr={cfg.det_cr})")

    return srp_app_fn, srp_det_fn
