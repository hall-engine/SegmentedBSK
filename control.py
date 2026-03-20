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

    def __init__(self, cfg: SimConfig, mu: float, j2_fn=None):
        self.cfg    = cfg
        self.mu     = mu
        self.j2_fn  = j2_fn

        # Pre-compute orbital timing
        self.period = 2.0 * np.pi * np.sqrt(cfg.a_geo**3 / mu)   # [s]
        n           = np.sqrt(mu / cfg.a_geo**3)                   # mean motion
        E_peak      = np.radians(cfg.target_eccentric_anomaly_deg)
        M_peak      = E_peak - cfg.e_geo * np.sin(E_peak)
        self.t_peak = M_peak / n                                   # [s from periapsis]

        # Star unit vector (constant, inertial)
        sv = np.array(cfg.star_vector, dtype=float)
        self.star_hat = sv / np.linalg.norm(sv)

        # Control state
        self._integral = np.zeros(3)

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

        r_target = r_c + self.star_hat * focal_length
        v_target = v_c.copy()

        # ── Feed-forward: differential gravity (J2 only; native SRP is internal) ──
        r_c_norm = np.linalg.norm(r_c)
        r_t_norm = np.linalg.norm(r_target)
        g_chief  = -mu * r_c / r_c_norm**3 + (j2_app / cfg.app_mass if self.j2_fn else 0.0)
        if self.j2_fn:
            g_target = -mu * r_target / r_t_norm**3 + self.j2_fn(r_target, 1.0, mu)
        else:
            g_target = -mu * r_target / r_t_norm**3
        delta_g  = g_target - g_chief
        force_ff = cfg.det_mass * delta_g

        # ── PID ──────────────────────────────────────────────────────────────
        pos_err_raw = r_target - r_d
        vel_err     = v_target - v_d

        # 1. Metrology Resolution (Quantize input error)
        res = cfg.metrology_resolution_m
        pos_err = np.round(pos_err_raw / res) * res

        # 2. Control Deadband (Idle if error is too small)
        if np.linalg.norm(pos_err) < cfg.control_deadband_m:
            pos_err = np.zeros(3)

        self._integral += pos_err * cfg.time_step_sec
        self._integral  = np.clip(self._integral,
                                   -cfg.integral_limit, cfg.integral_limit)

        ki       = cfg.ki_fraction * kp
        force_n_raw = force_ff + kp * pos_err + ki * self._integral + kd * vel_err

        # 3. Thruster Quantization (Minimum Impulse Bit equivalent force)
        mib = cfg.thruster_mib_n
        force_n = np.round(force_n_raw / mib) * mib

        return force_n

    def reset_integral(self):
        """Reset the integral accumulator (called when leaving engaged phase)."""
        self._integral = np.zeros(3)

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
        dt        = self.cfg.time_step_sec
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
        print(f"    Base Parameters  : i={self.cfg.base_i_deg:.1f}°, Ω={self.cfg.base_raan_deg:.1f}°, ω={self.cfg.base_omega_deg:.1f}°")
        print(f"    Peak (E={self.cfg.target_eccentric_anomaly_deg:.1f}°)      : {self.t_peak / 60.0:.1f} min from periapsis")
        print(f"    Cal window       : {self.cfg.cal_window_sec:.0f} s "
              f"(starts {(self.t_peak - self.cfg.obs_window_sec/2 - self.cfg.cal_window_sec)/60:.1f} min)")
        print(f"    Obs window       : {self.cfg.obs_window_sec:.0f} s "
              f"(±{self.cfg.obs_window_sec/2:.0f} s of peak)")
        print("=" * 80)

