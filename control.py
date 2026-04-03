"""
control.py — Mission Phase Logic & PID Translation Controller
==============================================================
MissionController encapsulates:
  - Orbital timing (peak eccentric anomaly calculation)
  - Mission-phase determination (Drifting / Calibration / Observation)
  - PID + feed-forward translation force computation
  - Delta-V accounting
  - True sun-body-frame computation (for sun tracker deviation logging)
"""

import numpy as np
from config import SimConfig
from Basilisk.utilities import RigidBodyKinematics


class MissionController:
    """
    Manages the mission phase timeline and PID translation control for
    the Detector (deputy) spacecraft.

    Parameters
    ----------
    cfg     : SimConfig
    mu      : float — Earth gravitational parameter [m³/s²]
    j2_fn   : callable(r_vec, mass, mu) → force [N], or None if J2 disabled
    """

    def __init__(self, cfg: SimConfig, mu: float, j2_fn=None, aperture_frame_dcm=None):
        self.cfg    = cfg
        self.mu     = mu
        self.j2_fn  = j2_fn
        # Frozen perifocal DCM (ECI → aperture frame).  All PID + quantization
        # operates in this frame so that metrology and MIB rounding are
        # independent of RAAN / orbital orientation in ECI.
        self.aperture_frame_dcm = aperture_frame_dcm if aperture_frame_dcm is not None else np.eye(3)

        # Pre-compute orbital timing
        self.period = 2.0 * np.pi * np.sqrt(cfg.a**3 / mu)   # [s]
        n           = np.sqrt(mu / cfg.a**3)                   # mean motion
        E_peak      = np.radians(cfg.target_eccentric_anomaly_deg)
        M_peak      = E_peak - cfg.eccentricity * np.sin(E_peak)
        t_peak_periapsis = M_peak / n                              # [s from periapsis]

        # The simulation clock starts at E = start_eccentric_anomaly_deg, NOT at periapsis.
        # We must express t_peak relative to the simulation start time, so that the
        # compute_phase() comparison (t_now_sec vs self.t_peak) fires at the correct
        # real simulation time.  Without this correction the calibration/observation
        # windows fire too early by exactly the periapsis-to-start transit time.
        E_start  = np.radians(cfg.start_eccentric_anomaly_deg)
        M_start  = E_start - cfg.eccentricity * np.sin(E_start)
        t_start  = M_start / n                                     # [s from periapsis to sim t=0]

        # Time from sim start to peak, wrapped into [0, period) so it is always positive.
        self.t_peak = (t_peak_periapsis - t_start) % self.period   # [s from sim start]

        # Star unit vector (constant, inertial)
        sv = np.array(cfg.star_vector, dtype=float)
        self.star_hat = sv / np.linalg.norm(sv)

        # Control state
        self._integral         = np.zeros(3)
        self.prev_force_n      = np.zeros(3)
        self.force_accumulator = np.zeros(3) # For Delta-Sigma PWM

    # ─── Phase Logic ──────────────────────────────────────────────────────────

    def compute_phase(self, t_now_sec: float):
        """
        Determine the current mission phase from simulation time.

        Parameters
        ----------
        t_now_sec : current simulation time [s] (absolute, not mod period)

        Returns
        -------
        is_engaged : bool
        mode_str   : 'Drifting' | 'Calibration' | 'Observation'
        kp, kd     : current PD gains (0 if drifting)
        """
        cfg = self.cfg
        dt  = (t_now_sec % self.period) - self.t_peak

        # Wrap to (−period/2, +period/2)
        if dt >  self.period / 2.0: dt -= self.period
        if dt < -self.period / 2.0: dt += self.period

        half_obs = cfg.obs_window_sec / 2.0

        if abs(dt) <= half_obs:
            return True, "Fine Observation", cfg.observation_kp, cfg.observation_kd

        if -(half_obs + cfg.cal_window_sec) < dt < -half_obs:
            # Switch to observation gains at obs_gain_switch_fraction through the cal window.
            # cal window runs from dt = -(half_obs + cal_window_sec) → dt = -half_obs.
            # The switch point (in dt) is the tail of calibration:
            #   dt_switch = -half_obs - cal_window_sec * (1 - fraction)
            # Beyond that point (i.e. dt > dt_switch) the obs gains are already active.
            switch_dt = -half_obs - cfg.cal_window_sec * (1.0 - cfg.obs_gain_switch_fraction)
            if dt >= switch_dt:
                return True, "Pre-Observation", cfg.observation_kp, cfg.observation_kd
            return True, "Calibration", cfg.calibration_kp, cfg.calibration_kd

        return False, "Drifting", 0.0, 0.0

    # ─── Force Computation ────────────────────────────────────────────────────

    def compute_force(self,
                      r_c: np.ndarray, v_c: np.ndarray,
                      r_d: np.ndarray, v_d: np.ndarray,
                      kp: float, kd: float,
                      j2_app: np.ndarray,
                      focal_length: float) -> np.ndarray:
        """
        Compute the total inertial translation force on the Detector.

        Includes:
          - Differential gravity feed-forward (J2 if enabled)
          - Proportional-Derivative control
          - Integral term with anti-windup

        Parameters
        ----------
        r_c, v_c : chief position/velocity [m, m/s] in ECI
        r_d, v_d : deputy position/velocity [m, m/s] in ECI
        kp, kd   : current PD gains
        j2_app   : J2 force on chief [N] (or zeros array if J2 disabled)
        focal_length : target separation [m]

        Returns
        -------
        force_n : np.ndarray [3] — commanded force in ECI [N]
        """
        cfg = self.cfg
        mu  = self.mu
        dcm = self.aperture_frame_dcm      # ECI → aperture frame

        r_target = r_c + self.star_hat * focal_length
        v_target = v_c.copy()

        # ── Feed-forward: differential gravity (computed in ECI, then rotated) ──
        r_c_norm = np.linalg.norm(r_c)
        r_t_norm = np.linalg.norm(r_target)
        g_chief  = -mu * r_c / r_c_norm**3 + (j2_app / cfg.app_mass if self.j2_fn else 0.0)
        if self.j2_fn:
            g_target = -mu * r_target / r_t_norm**3 + self.j2_fn(r_target, 1.0, mu)
        else:
            g_target = -mu * r_target / r_t_norm**3
        delta_g  = g_target - g_chief
        force_ff_ap = dcm @ (cfg.det_mass * delta_g)   # rotate to aperture frame

        # ── PID (all in aperture frame) ──────────────────────────────────────
        pos_err_raw = dcm @ (r_target - r_d)           # aperture frame
        vel_err     = dcm @ (v_target - v_d)           # aperture frame

        # 1. Metrology Resolution (quantize per-aperture-axis)
        res = cfg.metrology_resolution_m
        pos_err = np.round(pos_err_raw / res) * res

        # 2. Control Deadband (magnitude — frame-invariant)
        if np.linalg.norm(pos_err) < cfg.control_deadband_m:
            pos_err = np.zeros(3)

        self._integral += pos_err * cfg.ff_control_dt
        self._integral  = np.clip(self._integral,
                                   -cfg.integral_limit, cfg.integral_limit)

        ki       = cfg.ki_fraction * kp
        force_ap_raw = force_ff_ap + kp * pos_err + ki * self._integral + kd * vel_err

        # 3. Thruster Quantization (per-aperture-axis: Delta-Sigma PWM / Hysteresis / Simple Rounding)
        mib = cfg.thruster_mib_n
        
        if cfg.use_pvm:
            # PWM / Error Diffusion: maintain sub-MIB precision via duty-cycling
            self.force_accumulator += force_ap_raw
            force_ap = np.round(self.force_accumulator / mib) * mib
            self.force_accumulator -= force_ap
        elif cfg.use_hysteresis:
            # Schmitt Trigger / Hysteresis fallback:
            slack = 0.2 * mib 
            diff  = force_ap_raw - self.prev_force_n
            force_ap = self.prev_force_n.copy()
            for axis in range(3):
                if abs(diff[axis]) > (0.5 * mib + slack):
                    force_ap[axis] = np.round(force_ap_raw[axis] / mib) * mib
        else:
            # Simple Rounding
            force_ap = np.round(force_ap_raw / mib) * mib
        
        self.prev_force_n = force_ap.copy()
        # Rotate back to ECI for Basilisk's inertial force message
        return dcm.T @ force_ap

    def reset_integral(self):
        """Reset the integral accumulator (called when leaving engaged phase)."""
        self._integral = np.zeros(3)
        self.prev_force_n = np.zeros(3)
        self.force_accumulator = np.zeros(3) # Also clear PWM history

    def target_position(self, r_c: np.ndarray, focal_length: float) -> np.ndarray:
        """Return the desired deputy position in ECI."""
        return r_c + self.star_hat * focal_length

    # ─── Delta-V Accounting ───────────────────────────────────────────────────

    def dv_increment(self, force_n: np.ndarray) -> tuple:
        """
        Compute scalar and per-axis Delta-V increment for this time step.

        Returns
        -------
        dv_scalar : float [m/s]
        dv_xyz    : np.ndarray [3] [m/s], absolute per-axis contribution
        """
        accel_mag = np.linalg.norm(force_n) / self.cfg.det_mass
        accel_xyz = np.abs(force_n)          / self.cfg.det_mass
        dt        = self.cfg.ff_control_dt
        return accel_mag * dt, accel_xyz * dt

    # ─── Sun Tracker Utilities ────────────────────────────────────────────────

    @staticmethod
    def true_sun_body(sc_state, sun_pos_eci: np.ndarray) -> np.ndarray:
        """
        Compute the true sun unit vector in body frame.

        Parameters
        ----------
        sc_state    : BSK scStateOutMsg payload
        sun_pos_eci : Sun position in ECI [m] (from SPICE)

        Returns
        -------
        [3] unit vector in body frame, or zeros if degenerate.
        """
        r_sc     = np.array(sc_state.r_CN_N)
        sigma_bn = np.array(sc_state.sigma_BN)
        dcm_bn   = RigidBodyKinematics.MRP2C(sigma_bn)
        sun_vec  = sun_pos_eci - r_sc
        norm     = np.linalg.norm(sun_vec)
        if norm < 1.0:
            return np.zeros(3)
        return dcm_bn @ (sun_vec / norm)

    # ─── Timing Info ─────────────────────────────────────────────────────────

    def print_timing(self):
        """Print mission timing summary to stdout."""
        print("=" * 80)
        print(f">> Mission Timing")
        print("=" * 80)
        print(f"    Orbital Period   : {self.period / 3600.0:.2f} hours")
        print(f"    Start Anomaly    : E={self.cfg.start_eccentric_anomaly_deg:.1f}°")
        print(f"    Base Parameters  : a={self.cfg.a:.1e} m, e={self.cfg.eccentricity:.1f}, i={self.cfg.base_i_deg:.1f}°, Ω={self.cfg.base_raan_deg:.1f}°, ω={self.cfg.base_omega_deg:.1f}°")
        print(f"    Peak (E={self.cfg.target_eccentric_anomaly_deg:.1f}°)      : {self.t_peak / 60.0:.1f} min from sim start  "
              f"(start anomaly E={self.cfg.start_eccentric_anomaly_deg:.1f}°)")
        print(f"    Cal window       : {self.cfg.cal_window_sec:.0f} s "
              f"(starts at sim t={(self.t_peak - self.cfg.obs_window_sec/2 - self.cfg.cal_window_sec)/60:.1f} min)")
        print(f"    Obs window       : {self.cfg.obs_window_sec:.0f} s "
              f"(±{self.cfg.obs_window_sec/2:.0f} s of peak)")
        print("=" * 80)

