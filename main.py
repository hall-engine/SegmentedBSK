"""
main.py — Formation Simulation Entry Point (Native BSK SysModel)
================================================================
Run this file directly:

    cd SegmentedBSK/fastersimulation
    python main.py

Or call run() programmatically with a custom config:

    from config import SimConfig
    import main
    cfg = SimConfig(enable_j2=False, enable_srp=False)
    main.run(cfg, show_plots=True)

Perturbation flags (all in SimConfig):
    enable_j2          : Analytical J2 via ExtForceTorque
    enable_srp         : Native RadiationPressure (cannonball, SPICE sun)
    enable_third_body  : Sun + Moon third-body gravity (SPICE)
    enable_css_noise   : CSS/WLS navigation noise on both spacecraft

Performance notes:
    This version replaces the Python `while` loop with a native Basilisk
    SysModel (CustomFlightSoftwareContext).  Basilisk now drives the clock
    entirely in C++ and calls UpdateState() once per tick, eliminating
    ~14,000 Python/C++ start-stop transitions per run.
"""

import numpy as np
from tqdm import tqdm
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros, orbitalMotion, SimulationBaseClass, RigidBodyKinematics
from Basilisk.simulation import spacecraft as bsk_spacecraft, extForceTorque
import matplotlib.pyplot as plt
import os

from config import SimConfig
from gravity import setup_gravity, setup_j2, setup_srp
from spacecraft import SpacecraftManager
from formation import FormationManager
from control import MissionController
import plotting
import segmented_optics
from utilities import create_sim_dir, save_sim_config, save_states_h5
import mirror_plotting

print()
print()


# =============================================================================
# NATIVE BSK FLIGHT SOFTWARE MODULE
# =============================================================================

class CustomFlightSoftwareContext(sysModel.SysModel):
    """
    Native Basilisk SysModel that encapsulates all Python-side flight software.

    Basilisk calls UpdateState() on every simulation tick.  This eliminates
    the ~14,000 Python↔C++ engine start-stop transitions that the old while
    loop caused.

    Parameters
    ----------
    cfg                : SimConfig
    controller         : MissionController
    sim                : SimBaseClass  (needed for CurrentNanos timestamp)
    app_sc, det_sc     : Spacecraft objects
    app_cmd_force_msg,
    det_cmd_force_msg  : CmdForceInertialMsg handles for writing forces
    j2_msg_app,
    j2_msg_det         : CmdForceInertialMsg handles for J2
    j2_fn              : callable or None — analytical J2 function
    srp_app_fn,
    srp_det_fn         : callable or None — analytical SRP functions
    mu                 : gravitational parameter
    focal_length       : formation focal length [m]
    app_mgr, det_mgr   : SpacecraftManager instances
    grav_factory       : gravity factory (for SPICE sun/moon msgs)
    sun_msg            : sun body message (or None)
    has_css            : bool — whether CSS noise is active
    hexdm, states,
    mirror_controllers : optics objects (or None if mirror_control_on=False)
    stop_time_nano     : int — total sim stop time in nanoseconds
    step_nano          : int — time-step in nanoseconds
    """

    def __init__(self,
                 cfg, controller, sim,
                 app_sc, det_sc,
                 app_cmd_force_msg, det_cmd_force_msg,
                 j2_msg_app, j2_msg_det, j2_fn,
                 srp_app_fn, srp_det_fn,
                 mu, focal_length,
                 app_mgr, det_mgr,
                 grav_factory, sun_msg, has_css,
                 hexdm, states, mirror_controllers,
                 stop_time_nano, step_nano):
        super().__init__()

        # ── Book-keeping references ────────────────────────────────────────
        self.cfg            = cfg
        self.controller     = controller
        self.sim            = sim
        self.app_sc         = app_sc
        self.det_sc         = det_sc
        self.app_cmd_force_msg = app_cmd_force_msg
        self.det_cmd_force_msg = det_cmd_force_msg
        self.j2_msg_app     = j2_msg_app
        self.j2_msg_det     = j2_msg_det
        self.j2_fn          = j2_fn
        self.srp_app_fn     = srp_app_fn
        self.srp_det_fn     = srp_det_fn
        self.mu             = mu
        self.focal_length   = focal_length
        self.app_mgr        = app_mgr
        self.det_mgr        = det_mgr
        self.grav_factory   = grav_factory
        self.sun_msg        = sun_msg
        self.has_css        = has_css
        self.hexdm          = hexdm
        self.states         = states
        self.mirror_controllers = mirror_controllers
        self.stop_time_nano = stop_time_nano
        self.step_nano      = step_nano

        # ── Scalar helpers ─────────────────────────────────────────────────
        self.cumul_dv      = 0.0
        self.cumul_dv_xyz  = np.zeros(3)
        self.zero3         = np.zeros(3)
        self.force_n_cmd  = self.zero3.copy()

        # ── Progress bar ───────────────────────────────────────────────────
        self.pbar = tqdm(
            total=float(stop_time_nano * macros.NANO2SEC),
            unit="s", desc="Simulating",
            disable=cfg.disable_progress,
        )

        # ── Data accumulation lists (plain Python lists — SWIG-safe) ──────
        # NOTE: r1/r2 positions and RW speeds are logged by BSK recorders (app_log,
        # det_log, rw_log) and read from there at plot-time.  Only derived / mission-
        # level quantities that have no native BSK recorder are kept here.
        self.r_rel_B_list       = []
        self.engaged_list       = []
        self.phase_list         = []
        self.dv_list            = []
        self.dv_xyz_list        = []
        self.pos_err_list       = []
        self.css_sun_app_list   = []
        self.css_sun_det_list   = []
        self.true_sun_app_list  = []
        self.true_sun_det_list  = []
        self.srp_app_log_list   = []
        self.srp_det_log_list   = []
        self.sun_pos_list       = []
        self.moon_pos_list      = []
        self.mirror_time_list     = []   # timestamps [s] for engaged-phase ticks only
        self.mirror_r_rel_B_list  = []   # aperture-frame r_rel at engaged ticks (matches mirror_time_list)
        
        # ── Control Timers ─────────────────────────────────────────────────
        self.last_ff_time  = -1.0
        self.last_mir_time = -1.0

    def __del__(self):
        # Hand ownership back to SWIG so it doesn't warn about a missing
        # destructor for this Python-subclassed C++ SysModel at process exit.
        self.thisown = False

    # ── Internal helpers ───────────────────────────────────────────────────

    def _read_sun_pos(self):
        if self.sun_msg is not None:
            return np.array(
                self.grav_factory.spiceObject.planetStateOutMsgs[1].read().PositionVector
            )
        return self.zero3.copy()

    def _read_moon_pos(self):
        if self.cfg.enable_third_body:
            return np.array(
                self.grav_factory.spiceObject.planetStateOutMsgs[2].read().PositionVector
            )
        return self.zero3.copy()

    def _read_wls(self, mgr):
        if self.has_css:
            return np.array(mgr.css_wls.navStateOutMsg.read().vehSunPntBdy)
        return self.zero3.copy()

    # ── Basilisk entry-point ───────────────────────────────────────────────

    def UpdateState(self, CurrentSimNanos):
        """Called by Basilisk once per simulation tick."""
        cfg        = self.cfg
        controller = self.controller

        # ── Read current SC states ─────────────────────────────────────────
        app_state = self.app_sc.scStateOutMsg.read()
        det_state = self.det_sc.scStateOutMsg.read()
        r_c = np.array(app_state.r_CN_N);  v_c = np.array(app_state.v_CN_N)
        r_d = np.array(det_state.r_CN_N);  v_d = np.array(det_state.v_CN_N)
        dist = np.linalg.norm(r_d - r_c)

        # ── Mission phase ──────────────────────────────────────────────────
        t_now_sec = CurrentSimNanos * macros.NANO2SEC
        is_engaged, mode, kp, kd = controller.compute_phase(t_now_sec)
        self.pbar.set_description(f"Dist={dist:.3f}m | {mode}")

        # ── Mirror / optics (only during engaged phases) ───────────────────
        if is_engaged and cfg.mirror_control_on:
            if (t_now_sec - self.last_mir_time) >= (cfg.mirror_control_dt - 1e-9):
                self.last_mir_time = t_now_sec
                self.states = segmented_optics.gauge_where_pointing(
                    self.hexdm, self.states, app_state, det_state, cfg,
                    aperture_frame_dcm=self.aperture_frame_dcm
                )
                for s in self.states:
                    self.states[s] = self.mirror_controllers[0].lqr_control_full(
                        self.states[s], cfg.mirror_control_dt
                    )
                    self.states[s].store_histories()
                self.mirror_time_list.append(t_now_sec)
                self.mirror_r_rel_B_list.append(self.aperture_frame_dcm @ (r_d - r_c))

        # ── J2 forces (analytical) ─────────────────────────────────────────
        j2_app_vec = self.zero3.copy()
        j2_det_vec = self.zero3.copy()
        if self.j2_fn is not None:
            j2_app_vec = self.j2_fn(r_c, cfg.app_mass, self.mu)
            j2_det_vec = self.j2_fn(r_d, cfg.det_mass, self.mu)

            _p = messaging.CmdForceInertialMsgPayload()
            _p.forceRequestInertial = j2_app_vec.tolist()
            self.j2_msg_app.write(_p, CurrentSimNanos)

            _p = messaging.CmdForceInertialMsgPayload()
            _p.forceRequestInertial = j2_det_vec.tolist()
            self.j2_msg_det.write(_p, CurrentSimNanos)

        # ── SRP forces (analytical) ────────────────────────────────────────
        srp_app_vec = self.zero3.copy()
        srp_det_vec = self.zero3.copy()
        if self.srp_app_fn is not None:
            sun_n     = self._read_sun_pos()
            sigma_app = np.array(app_state.sigma_BN)
            sigma_det = np.array(det_state.sigma_BN)
            srp_app_vec = self.srp_app_fn(r_c, sun_n, sigma_app, cfg.app_mass)
            srp_det_vec = self.srp_det_fn(r_d, sun_n, sigma_det, cfg.det_mass)

        self.srp_app_log_list.append(srp_app_vec)
        self.srp_det_log_list.append(srp_det_vec)

        # ── Translation control ────────────────────────────────────────────
        if is_engaged and cfg.detector_control_on:
            if (t_now_sec - self.last_ff_time) >= (cfg.ff_control_dt - 1e-9):
                self.last_ff_time = t_now_sec
                self.force_n_cmd = controller.compute_force(
                    r_c, v_c, r_d, v_d, kp, kd,
                    j2_app_vec + srp_app_vec, self.focal_length
                )
            
            force_n = self.force_n_cmd
            # Scalar dv and progress logging
            dv_s, _         = controller.dv_increment(force_n)
            force_ap        = self.aperture_frame_dcm @ force_n   # project to aperture frame
            dv_xyz          = force_ap / cfg.det_mass * cfg.time_step_sec  # use simulation step for accumulation scale
            self.cumul_dv     += dv_s * (cfg.time_step_sec / cfg.ff_control_dt) # Rescale for sub-stepping
            self.cumul_dv_xyz += np.abs(dv_xyz) 
        else:
            force_n = self.zero3.copy()
            self.force_n_cmd = self.zero3.copy()
            self.last_ff_time = -1.0 # Reset timers when leaving engaged
            controller.reset_integral()


        # Write force to Aperture (J2 + SRP)
        _f_app = messaging.CmdForceInertialMsgPayload()
        _f_app.forceRequestInertial = (j2_app_vec + srp_app_vec).tolist()
        self.app_cmd_force_msg.write(_f_app, CurrentSimNanos)

        # Write force to Detector (J2 + SRP + Control)
        _f_det = messaging.CmdForceInertialMsgPayload()
        _f_det.forceRequestInertial = (j2_det_vec + srp_det_vec + force_n).tolist()
        self.det_cmd_force_msg.write(_f_det, CurrentSimNanos)

        # ── Collect data ───────────────────────────────────────────────────
        # r1, r2, and rw_speeds are captured by BSK recorders (app_log / det_log / rw_log).
        # Only derived/mission quantities not available from recorders are stored here.
        sigma_BN = np.array(app_state.sigma_BN)
        # r_rel in APERTURE FRAME: fixed ECI frame (z=star_vec, y=v_perigee, x=y×z).
        # We use the pre-computed frozen DCM rather than the instantaneous body frame
        # so that the coordinates are independent of RAAN/ω/Ω and attitude residuals.
        self.r_rel_B_list.append(self.aperture_frame_dcm @ (r_d - r_c))

        self.engaged_list.append(is_engaged)
        self.phase_list.append(mode)
        self.dv_list.append(self.cumul_dv)
        self.dv_xyz_list.append(self.cumul_dv_xyz.copy())
        _tgt = controller.target_position(r_c, self.focal_length)
        self.pos_err_list.append(
            self.aperture_frame_dcm @ (r_d - _tgt)   # aperture-frame: negative = below target
        )

        self.css_sun_app_list.append(self._read_wls(self.app_mgr))
        self.css_sun_det_list.append(self._read_wls(self.det_mgr))
        sun_pos  = self._read_sun_pos()
        moon_pos = self._read_moon_pos()
        self.sun_pos_list.append(sun_pos)
        self.moon_pos_list.append(moon_pos)
        self.true_sun_app_list.append(
            MissionController.true_sun_body(self.app_sc.scStateOutMsg.read(), sun_pos)
        )
        self.true_sun_det_list.append(
            MissionController.true_sun_body(self.det_sc.scStateOutMsg.read(), sun_pos)
        )

        self.pbar.update(float(self.step_nano * macros.NANO2SEC))


# =============================================================================
# FORMATION DESIGN HELPERS
# =============================================================================

def orbital_plane_normal(base_i_deg: float, base_raan_deg: float) -> list:
    """
    Unit normal to the orbital plane (angular momentum direction ĥ) in ECI.

    For inclination i and RAAN Ω:
        ĥ = [ +sin(i)·sin(Ω),  -sin(i)·cos(Ω),  cos(i) ]

    ω has no effect on the plane orientation and is intentionally ignored.
    """
    i = np.radians(base_i_deg)
    O = np.radians(base_raan_deg)
    h = np.array([
         np.sin(i) * np.sin(O),
        -np.sin(i) * np.cos(O),
         np.cos(i)
    ])
    return h.tolist()

def calculate_optimal_detector_state(cfg: SimConfig, mu: float, oe_app_start: orbitalMotion.ClassicElements):
    """
    Compute detector orbital elements using the user-specified formation geometry.

    Formation geometry
    ------------------
    Both spacecraft share:  a,  e,  RAAN (Ω),  argument of periapsis (ω)

    Aperture:   i = base_i_deg,          f = f_app  (from start_eccentric_anomaly_deg)
    Detector:   i = base_i_deg + Δi,     f = f_app − det_lag_f_deg  (starts behind)

    where:
      Δi           = cfg.det_delta_i_deg   [deg → rad]  — inclination offset
      det_lag_f_deg = cfg.det_lag_f_deg                 — how far behind in true anomaly

    Star vector
    -----------
    star_vector = orbital_plane_normal(base_i_deg, base_raan_deg)
    = orbit normal of the APERTURE's plane.
    (Set automatically in run() before this function is called.)
    """
    delta_i  = np.radians(cfg.det_delta_i_deg)
    delta_O  = np.radians(getattr(cfg, '_det_delta_O_deg', 0.0))   # set by USE_FOCAL_DESIGNATOR solve
    lag_f    = np.radians(cfg.det_lag_f_deg)

    oe_det        = orbitalMotion.ClassicElements()
    oe_det.a      = oe_app_start.a
    oe_det.e      = oe_app_start.e
    oe_det.Omega  = oe_app_start.Omega + delta_O        # RAAN offset (zeroes lateral component)
    delta_w  = np.radians(getattr(cfg, '_det_delta_omega_deg', 0.0))
    oe_det.omega  = oe_app_start.omega + delta_w     # adjusted to keep perigee direction fixed
    oe_det.i      = oe_app_start.i + delta_i            # inclination offset
    oe_det.f      = oe_app_start.f - lag_f              # along-track lag


    # ── Print geometry summary ──────────────────────────────────────────────

    r_d0, v_d0 = orbitalMotion.elem2rv(mu, oe_det)
    r_a0, v_a0 = orbitalMotion.elem2rv(mu, oe_app_start)
    sep0 = np.linalg.norm(np.array(r_d0) - np.array(r_a0))
    sv   = np.array(cfg.star_vector)
    z0   = np.dot(np.array(r_d0) - np.array(r_a0), sv)
    print(f">> Formation geometry:")
    print(f"     Δi = {cfg.det_delta_i_deg:.4f} deg | lag_f = {cfg.det_lag_f_deg:.2f} deg")
    print(f"     star_vector = {[f'{v:.4f}' for v in cfg.star_vector]}")
    print(f"     t=0 separation = {sep0:.1f} m  |  z(star_vec) = {z0:.1f} m")

    return oe_det










# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run(cfg: SimConfig = None, show_plots: bool = True, **kwargs):
    """
    Execute the full formation simulation.

    Parameters
    ----------
    cfg          : SimConfig instance (uses defaults if None)
    show_plots   : generate and save all plots when True
    **kwargs     : Overwrites for SimConfig attributes (e.g., app_diameter=50.0,
                   mirror_plotting=False to skip mirror animation)
    """
    if cfg is None:
        cfg = SimConfig()

    # Resolve results_base relative to this script, not the CWD.
    # This ensures the output path is the same regardless of which
    # directory main.py is launched from.
    _here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(cfg.results_base):
        cfg.results_base = os.path.join(_here, cfg.results_base.lstrip('./'))

    # Apply parametric overwrites
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            print(f"  [Overwrite] {key} = {value}")
        else:
            print(f"  [Message] No SimConfig overwrites detected - using congig.py")

    # Seed all stochastic components using cfg.random_seed (set AFTER kwarg overwrites
    # so that random_seed can be overridden from the Monte Carlo grid / SLURM array).
    import random as _random
    np.random.seed(cfg.random_seed)
    _random.seed(cfg.random_seed)
    print(f"  [RNG] Seeded with random_seed = {cfg.random_seed}")

    # create correct semi-major axis
    cfg.a_geo = cfg.r_geo * (1 - cfg.e_geo)

    # Derive star_vector from the orbital plane normal (must come AFTER kwarg overwrites
    # so that any base_i_deg / base_raan_deg overrides are already applied).
    cfg.star_vector = orbital_plane_normal(cfg.base_i_deg, cfg.base_raan_deg)
    
    print(f'>> Star vector (ĥ) = {[f"{v:.4f}" for v in cfg.star_vector]}'
          f'   (i={cfg.base_i_deg}°, Ω={cfg.base_raan_deg}°)')


    # =========================================================================
    # 0. Formation Geometry Setup
    # =========================================================================
    # focal_length is the controller's target cross-track separation [m].
    focal_length = float(cfg.target_focal_length)
    cfg.focal_length = focal_length

    # If USE_FOCAL_DESIGNATOR is set, auto-compute det_delta_i_deg so that the
    # z-separation in the aperture frame equals exactly target_focal_length at E_peak.
    #
    # Formula (first-order, aperture frame z = orbit normal):
    #   δz_peak = r_peak · Δi · sin(u_peak) · cos(2·i)
    #   ⟹  Δi = focal_length / (r_peak · sin(u_peak) · cos(2·i))
    #
    # where u_peak = ω + f(E_peak).  This is purely from the inclination offset;
    # the 1° along-track lag (det_lag_f_deg) contributes only in-plane separation.
    if cfg.use_focal_designator:
        _e  = cfg.e_geo
        _i  = np.radians(cfg.base_i_deg)
        _O  = np.radians(cfg.base_raan_deg)
        _w  = np.radians(cfg.base_omega_deg)
        _si, _ci = np.sin(_i), np.cos(_i)
        _sO, _cO = np.sin(_O), np.cos(_O)
        _sw, _cw = np.sin(_w), np.cos(_w)

        # E_peak and f_peak — purely ellipse-defined, independent of ω
        _E_pk = np.radians(cfg.target_eccentric_anomaly_deg)
        _r_pk = cfg.a_geo * (1.0 - _e * np.cos(_E_pk))
        _f_pk = 2.0 * np.arctan(np.sqrt((1+_e)/(1-_e)) * np.tan(_E_pk/2))

        # ── Perigee-axis tilt ────────────────────────────────────────────────
        # Tilt the detector orbit by ε around the perigee direction (e_hat),
        # which is the x-axis of the frozen aperture frame.
        # This gives:  z_sep(E) = r(E) · ε · sin(f(E))
        # — a function of E and e only, completely ω-invariant.
        # At E=0 (perigee): f=0, sin(f)=0 → z_sep=0 always. ✓
        # Maximum at E where r·sin(f) peaks (E=90° for e=0.8). ✓
        _eps = focal_length / (_r_pk * np.sin(_f_pk))   # tilt angle [rad]

        # Orbit normal h_hat and perigee direction e_hat in ECI
        _h_hat = np.array([_si*_sO, -_si*_cO, _ci])
        _e_hat = np.array([_cO*_cw - _sO*_ci*_sw,
                           _sO*_cw + _cO*_ci*_sw,
                           _si*_sw])

        # Velocity at perigee direction: q_hat = h_hat × e_hat
        _q_hat = np.cross(_h_hat, _e_hat)
        _q_hat /= np.linalg.norm(_q_hat)

        # ── Exact extraction of detector orbital elements ────────────────────
        # Compute h_new = h_hat rotated by ε around e_hat (perigee direction).
        # Then extract i_d, Ω_d, ω_d exactly — works for all i including i=0.
        _h_new = _h_hat - _eps * _q_hat
        _h_new /= np.linalg.norm(_h_new)

        # i_d = arccos(h_new_z)
        _i_det = float(np.arccos(np.clip(_h_new[2], -1.0, 1.0)))

        # Ω_d = arctan2(h_new_x, -h_new_y)  [standard formula]
        _O_det = float(np.arctan2(_h_new[0], -_h_new[1])) % (2*np.pi)

        # ω_d: angle from new ascending node to perigee (e_hat) in detector plane
        _n_det = np.array([np.cos(_O_det), np.sin(_O_det), 0.0])
        _p_det = np.cross(_h_new, _n_det); _p_det /= np.linalg.norm(_p_det)
        _w_det = float(np.arctan2(np.dot(_e_hat, _p_det), np.dot(_e_hat, _n_det)))

        def _wrap(x):
            return (x + np.pi) % (2*np.pi) - np.pi

        _di_rad  = _wrap(_i_det - _i)
        _dO_rad  = _wrap(_O_det - _O)
        _dw_rad  = _wrap(_w_det - _w)

        cfg.det_delta_i_deg      = float(np.degrees(_di_rad))
        cfg._det_delta_O_deg     = float(np.degrees(_dO_rad))
        cfg._det_delta_omega_deg = float(np.degrees(_dw_rad))

        print(f">> Focal designator (perigee-tilt):  E_peak={cfg.target_eccentric_anomaly_deg:.1f}°  "
              f"ε={np.degrees(_eps)*1e3:.4f} mdeg  "
              f"δi={cfg.det_delta_i_deg*1e3:.4f} mdeg  "
              f"δΩ={cfg._det_delta_O_deg*1e3:.4f} mdeg  "
              f"δω={cfg._det_delta_omega_deg*1e3:.4f} mdeg")
    else:
        cfg._det_delta_O_deg    = 0.0
        cfg._det_delta_omega_deg = 0.0


    # Initialise optics system mapping
    if cfg.mirror_control_on:
        hexdm, states, mirror_controllers, cfg = segmented_optics.initialise_aperture_segments(cfg)
        
    # mass calcs using properly mapped aperture diameter
    cfg.app_mass = ((cfg.app_diameter/2)**2)*np.pi*cfg.app_height*cfg.app_mass_density
    cfg.app_estimate_area = ((cfg.app_diameter/2)**2)*np.pi
    cfg.time_step_sec = min(cfg.ff_control_dt, cfg.mirror_control_dt)

    print("=" * 80)
    print(">> Optical Reef — Formation Simulation")
    print("=" * 80)
    print(f">> Perturbations:")
    print(f"    J2           : {'ON' if cfg.enable_j2==True else 'OFF'}")
    print(f"    SRP          : {'ON' if cfg.enable_srp==True else 'OFF'}")
    print(f"    Third-body   : {'ON' if cfg.enable_third_body==True else 'OFF'}")
    print(f"    CSS noise    : {'ON' if cfg.enable_css_noise==True else 'OFF'}")
    print("=" * 80)
    print(">> Simulation Date and Time:")
    print(f"                             {cfg.time_init_string}")
    print("=" * 80)
    print(">> Segmented Mirror Properties")
    print(f"    Segment gap:             {cfg.segment_gap} cm")
    print(f"    Segment flat-to-flat:    {cfg.flat_to_flat} m")
    print(f"    Number of rings:         {cfg.rings}")
    print(f"    Number of segments:      {len(states)}")
    print(f"    Focal length:            {cfg.target_focal_length} m")
    print("=" * 80)
    print(">> Spacecraft Properties")
    print(f"    Aperture mass:           {cfg.app_mass} kg")
    print(f"    Aperture diameter:       {cfg.app_diameter} m")
    print(f"    Aperture estimate area:  {cfg.app_estimate_area} m²")
    print(f"    Detector mass:           {cfg.det_mass} kg")
    print(f"    Detector side:           {cfg.det_side} m")
    print(f"    FF control:              {cfg.ff_control_dt} s")
    print(f"    Mirror control:          {cfg.mirror_control_dt} s")
    print(f"    Simulation time step:    {cfg.time_step_sec} s")
    print("=" * 80)


    # =========================================================================
    # 1. Simulation Container
    # =========================================================================
    sim        = SimulationBaseClass.SimBaseClass()
    dyn_proc   = sim.CreateNewProcess("dynProcess", 2)
    task_name  = "dynTask"
    dyn_proc.addTask(sim.CreateNewTask(task_name, macros.sec2nano(cfg.time_step_sec)))

    # =========================================================================
    # 2. Gravity, SPICE, SRP
    # =========================================================================
    grav_factory, earth, mu, sun_msg = setup_gravity(sim, task_name, cfg)

    # J2 effectors are created at Step 4 after spacecraft objects exist.

    # =========================================================================
    # 3. Spacecraft Objects
    # =========================================================================
    # Add both SCs to the task BEFORE gravFactory.addBodiesTo() — required by BSK.
    app_sc = bsk_spacecraft.Spacecraft()
    app_sc.ModelTag = "Aperture"
    sim.AddModelToTask(task_name, app_sc)

    det_sc = bsk_spacecraft.Spacecraft()
    det_sc.ModelTag = "Detector"
    sim.AddModelToTask(task_name, det_sc)

    grav_factory.addBodiesTo(app_sc)
    grav_factory.addBodiesTo(det_sc)

    # =========================================================================
    # 4. J2 & SRP (now that SC objects exist)
    # =========================================================================
    j2_msg_app, j2_msg_det, j2_fn = setup_j2(sim, task_name, cfg, app_sc, det_sc)
    srp_app_fn, srp_det_fn               = setup_srp(sim, task_name, cfg, app_sc, det_sc, sun_msg)

    srp_props = None
    if srp_app_fn is not None:
        srp_props = {
            "aperture": {"Cr": cfg.app_cr, "area": float(np.pi * (cfg.app_diameter / 2.0)**2)},
            "detector": {"Cr": cfg.det_cr, "area": float(2.0 * np.sqrt(3.0) * cfg.det_side ** 2)},
        }

    # =========================================================================
    # 5. Detector Translation Control (ExtForceTorque in inertial frame)
    # =========================================================================
    app_ext_force = extForceTorque.ExtForceTorque()
    app_ext_force.ModelTag = "Aperture_ExtForce"
    app_sc.addDynamicEffector(app_ext_force)
    sim.AddModelToTask(task_name, app_ext_force)

    app_cmd_force_msg = messaging.CmdForceInertialMsg()
    app_ext_force.cmdForceInertialInMsg.subscribeTo(app_cmd_force_msg)

    det_ext_force = extForceTorque.ExtForceTorque()
    det_ext_force.ModelTag = "Detector_ExtForce"
    det_sc.addDynamicEffector(det_ext_force)
    sim.AddModelToTask(task_name, det_ext_force)

    det_cmd_force_msg = messaging.CmdForceInertialMsg()
    det_ext_force.cmdForceInertialInMsg.subscribeTo(det_cmd_force_msg)

    # =========================================================================
    # 6. Spacecraft Managers (hardware + FSW, minus attitude — done in formation)
    # =========================================================================
    app_mgr = SpacecraftManager(sim, app_sc, task_name)
    app_mgr.set_geometric_properties(cfg.app_mass, cfg.app_shape,
                                      cfg.app_side, cfg.app_height)
    app_mgr.add_rw_cluster(enable_jitter=cfg.enable_rw_jitter, initial_omega=cfg.rw_initial_omega_radps)
    app_mgr.add_simple_nav(enable_noise=cfg.enable_metrology_noise, noise_std=cfg.metrology_noise_std)
    # NOTE: attitude control for Aperture is set up inside FormationManager

    det_mgr = SpacecraftManager(sim, det_sc, task_name)
    det_mgr.set_geometric_properties(cfg.det_mass, cfg.det_shape,
                                      cfg.det_side, cfg.det_height)
    det_mgr.add_rw_cluster(enable_jitter=cfg.enable_rw_jitter, initial_omega=cfg.rw_initial_omega_radps)
    # NOTE: Translation is commanded via ExtForceTorque (idealized impulsive control).
    # ThrusterDynamicEffector is intentionally not registered — thrusters are unused.
    det_mgr.add_simple_nav(enable_noise=cfg.enable_metrology_noise, noise_std=cfg.metrology_noise_std)


    # Optional CSS navigation noise
    if cfg.enable_css_noise and sun_msg is not None:
        app_mgr.add_css_array(sun_msg, noise_std=cfg.css_noise_std, bias=cfg.css_bias)
        det_mgr.add_css_array(sun_msg, noise_std=cfg.css_noise_std, bias=cfg.css_bias)
    elif cfg.enable_css_noise and sun_msg is None:
        print("[CSS] Skipped — requires enable_third_body=True for sun message.")

    # Keep message references alive
    _msgs = app_mgr.get_messages() + det_mgr.get_messages()

    # =========================================================================
    # 7. Formation Attitude Setup
    # =========================================================================
    form_mgr = FormationManager(sim, task_name)
    form_mgr.setup_star_alignment(app_sc, app_mgr, det_sc, det_mgr,
                                   cfg.star_vector, focal_length)

    # =========================================================================
    # 8. Initial Orbital Conditions
    # =========================================================================
    oe_app        = orbitalMotion.ClassicElements()
    oe_app.a      = cfg.a_geo
    oe_app.e      = cfg.e_geo
    oe_app.i      = np.radians(cfg.base_i_deg)
    oe_app.Omega  = np.radians(cfg.base_raan_deg)
    oe_app.omega  = np.radians(cfg.base_omega_deg)


    # Convert starting Eccentric Anomaly to True Anomaly
    E0            = np.radians(cfg.start_eccentric_anomaly_deg)
    f0            = 2.0 * np.arctan(np.sqrt((1.0 + cfg.e_geo) / (1.0 - cfg.e_geo)) * np.tan(E0 / 2.0))
    oe_app.f      = f0
    r1, v1        = orbitalMotion.elem2rv(mu, oe_app)


    # ── Override star_vector with the true orbit normal from initial r×v ──────
    # This is the definitive source: no formula sign ambiguity, guaranteed correct
    # for any RAAN / ω / i.  The cfg.orbital_plane_normal() formula is now correct
    # (sign fix), but we override here as a belt-and-suspenders check so the
    # aperture frame, attitude controller, and translation controller are all
    # guaranteed to use exactly the same ĥ.
    _h_vec = np.cross(np.array(r1, dtype=float), np.array(v1, dtype=float))
    _h_hat = _h_vec / np.linalg.norm(_h_vec)
    _sv_formula = np.array(cfg.star_vector, dtype=float)
    _sv_formula /= np.linalg.norm(_sv_formula)
    _angle_diff_deg = float(np.degrees(np.arccos(np.clip(np.dot(_h_hat, _sv_formula), -1, 1))))
    cfg.star_vector = _h_hat.tolist()
    print(f">> star_vector (r1×v1): {_h_hat.round(4)}  [formula diff={_angle_diff_deg:.4f}°]")


    oe_det = calculate_optimal_detector_state(cfg, mu, oe_app)
    r2, v2 = orbitalMotion.elem2rv(mu, oe_det)

    app_sc.hub.r_CN_NInit = r1;  app_sc.hub.v_CN_NInit = v1
    det_sc.hub.r_CN_NInit = r2;  det_sc.hub.v_CN_NInit = v2

    # ── Initial attitude: start already pointing at star (skip coarse-slew transient) ──
    # Compute the MRP that rotates body +Z to align with star_vector (ECI).
    # This replicates the same math as SpacecraftManager.add_inertial_pointing_control().
    def _star_to_mrp(target_vec):
        """Return the MRP sigma_RN for body +Z → target_vec."""
        s_n = np.array(target_vec, dtype=float)
        s_n /= np.linalg.norm(s_n)
        z_b = np.array([0., 0., 1.])
        if np.allclose(s_n, z_b):
            return [0., 0., 0.]
        if np.allclose(s_n, -z_b):
            return [1., 0., 0.]
        v  = np.cross(z_b, s_n)
        s  = np.linalg.norm(v)
        c  = np.dot(z_b, s_n)
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R  = np.eye(3) + vx + vx @ vx * ((1.0-c)/s**2)
        d  = 1.0 + R[0,0] + R[1,1] + R[2,2]
        if d > 1e-6:
            z    = np.sqrt(d)
            sigma = np.array([R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0]]) / (z*(z+2))
        else:
            sigma = np.array([1., 0., 0.])
        return sigma.tolist()

    # Aperture: point +Z at the star vector
    sigma_app = _star_to_mrp(cfg.star_vector)
    app_sc.hub.sigma_BNInit = [[sigma_app[0]], [sigma_app[1]], [sigma_app[2]]]
    app_sc.hub.omega_BN_BInit = [[0.], [0.], [0.]]   # zero initial angular rate

    # Detector: point +Z toward the Aperture (LOS direction, same as spacecraftPointing reference)
    r1_np = np.array(r1); r2_np = np.array(r2)
    los_vec = r1_np - r2_np          # from Detector toward Aperture
    sigma_det = _star_to_mrp(los_vec)
    det_sc.hub.sigma_BNInit = [[sigma_det[0]], [sigma_det[1]], [sigma_det[2]]]
    det_sc.hub.omega_BN_BInit = [[0.], [0.], [0.]]   # zero initial angular rate

    print(f">> Aperture σ_BN = {[f'{x:.6f}' for x in sigma_app]}")
    print(f">> Detector σ_BN = {[f'{x:.6f}' for x in sigma_det]}")

    # ── Aperture Frame: fixed ECI frame defined at t=0 ─────────────────────
    # z = star_vector (orbit normal, fixed in ECI)
    # y = aperture velocity direction at perigee (t=0)
    # x = y × z  (radially outward at perigee — perifocal frame)
    # This frame is INVARIANT to RAAN/ω/Ω: same a/e/i always give the same
    # separation coordinates regardless of other elements.
    z_hat = np.array(cfg.star_vector, dtype=float); z_hat /= np.linalg.norm(z_hat)
    y_hat = np.array(v1, dtype=float);               y_hat /= np.linalg.norm(y_hat)
    x_hat = np.cross(y_hat, z_hat);                 x_hat /= np.linalg.norm(x_hat)
    # Re-orthogonalise y in case v1 had a small z component
    y_hat = np.cross(z_hat, x_hat);                 y_hat /= np.linalg.norm(y_hat)
    aperture_frame_dcm = np.vstack((x_hat, y_hat, z_hat))   # (3×3), rows are basis vectors
    print(f">> Aperture frame (ECI, frozen at t=0):")
    print(f"     x = {x_hat.round(4)}  (radial at perigee)")
    print(f"     y = {y_hat.round(4)}  (velocity at perigee)")
    print(f"     z = {z_hat.round(4)}  (orbit normal / star_vector)")


    # =========================================================================
    # 9. Logging
    # =========================================================================
    app_log   = app_sc.scStateOutMsg.recorder()
    det_log   = det_sc.scStateOutMsg.recorder()
    rw_log    = det_mgr.rw_effector.rwSpeedOutMsg.recorder()
    sim.AddModelToTask(task_name, app_log)
    sim.AddModelToTask(task_name, det_log)
    sim.AddModelToTask(task_name, rw_log)

    # =========================================================================
    # 10. Mission Controller
    # =========================================================================
    controller = MissionController(cfg, mu, j2_fn)
    controller.star_hat = np.array(cfg.star_vector, dtype=float)  # ensure uses corrected ĥ
    controller.print_timing()

    stop_time = macros.min2nano(
        controller.period * cfg.period_multiple / 60.0
    )
    step_nano = macros.sec2nano(cfg.time_step_sec)

    # ── CSS / helper flags ─────────────────────────────────────────────────
    has_css = cfg.enable_css_noise and sun_msg is not None and app_mgr.has_css()

    # ── Seed t=0 data before handing off to the native module ────────────
    _zero3   = np.zeros(3)

    def _seed_sun_pos():
        if sun_msg is not None:
            return np.array(
                grav_factory.spiceObject.planetStateOutMsgs[1].read().PositionVector
            )
        return _zero3.copy()

    def _seed_moon_pos():
        if cfg.enable_third_body:
            return np.array(
                grav_factory.spiceObject.planetStateOutMsgs[2].read().PositionVector
            )
        return _zero3.copy()

    def _seed_wls():
        # CSS WLS C-message is not yet written before InitializeSimulation;
        # reading it would segfault.  Use zeros as a safe t=0 placeholder.
        return _zero3.copy()

    sun0        = _seed_sun_pos()
    initial_r_target = controller.target_position(r1, focal_length)

    seed_r1          = np.array(app_sc.scStateOutMsg.read().r_CN_N)
    seed_r2          = np.array(det_sc.scStateOutMsg.read().r_CN_N)
    _app_state_seed  = app_sc.scStateOutMsg.read()
    seed_r_rel_B     = aperture_frame_dcm @ (seed_r2 - seed_r1)

    seed_sun_app     = MissionController.true_sun_body(_app_state_seed, sun0)
    seed_sun_det     = MissionController.true_sun_body(det_sc.scStateOutMsg.read(), sun0)

    # =========================================================================
    # 12. Native BSK Flight-Software Module
    # =========================================================================
    custom_fsw = CustomFlightSoftwareContext(
        cfg=cfg,
        controller=controller,
        sim=sim,
        app_sc=app_sc,
        det_sc=det_sc,
        app_cmd_force_msg=app_cmd_force_msg,
        det_cmd_force_msg=det_cmd_force_msg,
        j2_msg_app=j2_msg_app,
        j2_msg_det=j2_msg_det,
        j2_fn=j2_fn,
        srp_app_fn=srp_app_fn,
        srp_det_fn=srp_det_fn,
        mu=mu,
        focal_length=focal_length,
        app_mgr=app_mgr,
        det_mgr=det_mgr,
        grav_factory=grav_factory,
        sun_msg=sun_msg,
        has_css=has_css,
        hexdm=hexdm if cfg.mirror_control_on else None,
        states=states if cfg.mirror_control_on else None,
        mirror_controllers=mirror_controllers if cfg.mirror_control_on else None,
        stop_time_nano=stop_time,
        step_nano=step_nano,
    )
    custom_fsw.ModelTag = "custom_fsw"
    custom_fsw.aperture_frame_dcm = aperture_frame_dcm  # frozen ECI frame for r_rel decomposition

    # Seed the t=0 data entries into the FSW module's lists before adding to task.
    # Note: r1/r2/rw_speeds are NOT seeded here — they come from BSK recorders which
    # start at t=0 naturally after InitializeSimulation(), so no trim is needed.
    custom_fsw.r_rel_B_list.append(seed_r_rel_B)
    custom_fsw.engaged_list.append(False)
    custom_fsw.phase_list.append("Drifting")
    custom_fsw.dv_list.append(0.0)
    custom_fsw.dv_xyz_list.append(_zero3.copy())
    custom_fsw.pos_err_list.append(aperture_frame_dcm @ (seed_r2 - initial_r_target))
    custom_fsw.css_sun_app_list.append(_seed_wls())
    custom_fsw.css_sun_det_list.append(_seed_wls())
    custom_fsw.true_sun_app_list.append(seed_sun_app)
    custom_fsw.true_sun_det_list.append(seed_sun_det)
    custom_fsw.srp_app_log_list.append(_zero3.copy())
    custom_fsw.srp_det_log_list.append(_zero3.copy())
    custom_fsw.sun_pos_list.append(sun0)
    custom_fsw.moon_pos_list.append(_seed_moon_pos())

    # Hook into the simulation task (runs every tick)
    sim.AddModelToTask(task_name, custom_fsw)

    # =========================================================================
    # 13. Initialize & Run
    # =========================================================================
    print(">> Initializing simulation...")
    sim.InitializeSimulation()
    sim.ConfigureStopTime(stop_time)
    sim.ExecuteSimulation()

    custom_fsw.pbar.close()
    print(">> Simulation complete.")

    # =========================================================================
    # 14. Data Saving & Plots
    # =========================================================================
    if show_plots or getattr(cfg, 'save_data', True):
        fsw = custom_fsw  # alias for readability

        # The FSW lists are seeded with one t=0 entry before the sim runs,
        # giving N+1 items vs the recorder's N ticks.  Trim to recorder length.
        # r1_n / r2_n / rw_speeds now come directly from BSK recorders (no trim needed).
        n_rec = len(app_log.times())
        def _trim(lst): return lst[-n_rec:] if len(lst) > n_rec else lst

        extra = {
            # Spacecraft State & Trajectory — sourced from BSK recorders
            "r1_n":         app_log.r_BN_N,
            "r2_n":         det_log.r_BN_N,

            # Mission & Control Status
            "engaged":      _trim(fsw.engaged_list),
            "phase":        _trim(fsw.phase_list),
            "pos_err":      np.array(_trim(fsw.pos_err_list)),

            # Actuators & Effort
            "dv":           np.array(_trim(fsw.dv_list)),
            "dv_xyz":       np.array(_trim(fsw.dv_xyz_list)),
            "rw_speeds":    rw_log.wheelSpeeds,  # sourced from BSK recorder

            # Environment & Sensors
            "star_vector":        cfg.star_vector,
            "aperture_frame_dcm": aperture_frame_dcm,  # frozen ECI frame: rows = x,y,z axes
            "cfg":                cfg,                  # full config for orbital geometry diagram
            "sun_pos_n":          np.array(_trim(fsw.sun_pos_list)),
            "moon_pos_n":         np.array(_trim(fsw.moon_pos_list)),

            # Solar Radiation Pressure & Sun Sensors
            "css_sun_app":  np.array(_trim(fsw.css_sun_app_list)),
            "css_sun_det":  np.array(_trim(fsw.css_sun_det_list)),
            "true_sun_app": np.array(_trim(fsw.true_sun_app_list)),
            "true_sun_det": np.array(_trim(fsw.true_sun_det_list)),
            "srp_app_vec":  np.array(_trim(fsw.srp_app_log_list)),
            "srp_det_vec":  np.array(_trim(fsw.srp_det_log_list)),
        }
        out_dir = create_sim_dir(cfg, results_base=cfg.results_base)
        cfg.out_dir = out_dir

        if getattr(cfg, 'save_data', True):
            save_sim_config(cfg, out_dir)
            t_arr = app_log.times() * macros.NANO2SEC   # full-sim timestamps [s]
            if cfg.mirror_control_on:
                _full_time    = t_arr                                  # full-sim timestamps [s]
                _mirror_time  = np.array(fsw.mirror_time_list)        # engaged-phase timestamps [s]
                _rel_pos_B    = np.array(fsw.mirror_r_rel_B_list)     # engaged-phase body-frame r_rel
                _phase        = list(_trim(fsw.phase_list))            # full-sim phase labels

                # Compute attitude conversions (Full Cadence)
                from Basilisk.utilities import RigidBodyKinematics as rbk
                sigma_SN = rbk.C2MRP(aperture_frame_dcm)
                
                # Vectorised conversions
                _sigma_app_star = np.array([rbk.subMRP(s, sigma_SN) for s in app_log.sigma_BN])
                _sigma_det_star = np.array([rbk.subMRP(s, sigma_SN) for s in det_log.sigma_BN])
                _rel_sigma_B    = np.array([rbk.subMRP(sd, sa) for sd, sa in zip(det_log.sigma_BN, app_log.sigma_BN)])

                save_states_h5(custom_fsw.states, out_dir,
                               time_arr=_full_time,
                               mirror_time_arr=_mirror_time,
                               cfg=cfg,
                               r_app_eci=app_log.r_BN_N,
                               r_det_eci=det_log.r_BN_N,
                               phase_arr=_phase,
                               rel_pos_B_arr=_rel_pos_B,
                               sigma_app_star=_sigma_app_star,
                               sigma_det_star=_sigma_det_star,
                               rel_sigma_B_arr=_rel_sigma_B,
                               dv_arr=np.array(_trim(fsw.dv_list)),
                               dv_xyz_arr=np.array(_trim(fsw.dv_xyz_list)))


        if show_plots:
            t_arr = app_log.times() * macros.NANO2SEC   # full-sim timestamps [s]
            plotting.run_all(app_log, det_log,
                             t_arr,
                             extra_data=extra,
                             out_dir=out_dir)

    # Mirror animation — gated on both config flags
    if cfg.mirror_plotting and cfg.mirror_control_on:
        import mirror_plotting
        mirror_plotting.run(cfg, read_every=cfg.read_every, opd_vmax=0.005, debug=False)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Uses defaults from config.py. Edit that file or pass overrides here.
    # cfg = SimConfig()
    # multiple = 0.15
    # cal_window_sec = 1200
    # overwriting script
    # for app_diam_val in range(70, 1000, 10):
    # for app_diam_val in [1000]:
        # for obs_window_val in [300]:
        # for obs_window_val in range(15, 270, 30):
            # run(cfg, show_plots=True, app_diameter=app_diam_val, obs_window_sec=obs_window_val, period_multiple=multiple, cal_window_sec=cal_window_sec)
            # plt.close('all')
    
    # RUN SIMULATION
    cfg = SimConfig()
    run(cfg,
        read_every          = 1000,     # mirror plotting frame interval
        ff_control_dt       = 0.01,
        mirror_control_dt   = 0.01,
        start_eccentric_anomaly_deg = 55.0,
        r_geo               = 20_000_000.0,
        show_plots          = True,   # save all plots after each sim
        save_data           = True,    # keep h5 and config saved
        mirror_plotting     = True,   # run mirror animation (slow — keep False for sweeps)
        disable_progress    = False,    # suppress tqdm in workers
        )
    plt.close('all')


    ##### table of useful constants
    # =================================
    # focal legnth | rings | opd_vmax
    # ---------------------------------
    # 5000 m       | 1     | 0.005
    # 5000 m       | 2     | 0.010
    # 5000 m       | 3     | 0.025
    # 5000 m       | 4     | 0.050
    # 5000 m       | 5     | 0.060
    # 5000 m       | 6     | 0.072
