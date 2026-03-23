"""
spacecraft.py — SpacecraftManager
===================================
Manages the hardware components of a single spacecraft:
  - Mass / inertia properties
  - Reaction wheel cluster
  - Navigation (SimpleNav)
  - Attitude control (inertial pointing via MRP feedback + RW motor torque)
  - Thruster set (6-DOF RCS)
  - CSS array + WLS sun-direction estimator (optional)

Usage
-----
    mgr = SpacecraftManager(sim, sc, task_name)
    mgr.set_geometric_properties(mass=..., shape=..., side=..., height=...)
    mgr.add_rw_cluster()
    mgr.add_thrusters(thrust=...)
    mgr.add_simple_nav()
    # Optionally:
    mgr.add_css_array(sun_msg, noise_std=..., bias=...)
"""

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import (
    attTrackingError, inertial3D, spacecraftPointing,
    mrpFeedback, rwMotorTorque, cssWlsEst,
)
from Basilisk.simulation import (
    reactionWheelStateEffector, simpleNav,
    thrusterDynamicEffector, coarseSunSensor,
)
from Basilisk.utilities import simIncludeRW, simIncludeThruster, unitTestSupport


# =============================================================================
# CSS AGGREGATOR — Native SysModel that collects CSS sensor readings
# =============================================================================

class CSSAggregator(sysModel.SysModel):
    """
    Thin Basilisk SysModel that reads from each CoarseSunSensor's output
    and writes the aggregated CSSArraySensorMsg each tick.

    In the old while-loop approach, SpacecraftManager.read_css_vals() did this
    from Python before each sim.ExecuteSimulation() call. Now that the sim runs
    natively, this model is placed on the task right before the cssWlsEst module.
    """
    def __init__(self, css_list, css_array_msg):
        super().__init__()
        self._css_list      = css_list
        self._css_array_msg = css_array_msg

    def UpdateState(self, CurrentSimNanos):
        payload = messaging.CSSArraySensorMsgPayload()
        for i, css in enumerate(self._css_list):
            payload.CosValue[i] = css.cssDataOutMsg.read().OutputData
        self._css_array_msg.write(payload, CurrentSimNanos)

    def __del__(self):
        # Hand ownership back to SWIG so it doesn't warn about a missing
        # destructor for this Python-subclassed C++ SysModel at process exit.
        self.thisown = False


class SpacecraftManager:
    """Builds and wires a spacecraft's onboard hardware and FSW chain."""

    def __init__(self, sim, sc, task_name: str):
        self.sim = sim
        self.sc = sc
        self.task_name = task_name
        self._messages = []     # keeps Python references alive (prevents GC)

    # ─── Mass / Inertia ───────────────────────────────────────────────────────

    def set_geometric_properties(self, mass: float, shape: str,
                                  side: float, height: float):
        """
        Set mass and derive inertia tensor from geometry.

        Parameters
        ----------
        shape : 'square' | 'hexagonal'
        side  : side length (or vertex radius for hex) [m]
        height: body height [m]
        """
        self.mass = mass
        if shape == "square":
            Izz      = (1.0 / 6.0)  * mass * side**2
            Ixx = Iyy = (1.0 / 12.0) * mass * (side**2 + height**2)
        elif shape == "hexagonal":
            Izz      = (5.0 / 12.0) * mass * side**2
            Ixx = Iyy = (5.0 / 24.0) * mass * side**2 + (1.0 / 12.0) * mass * height**2
        else:
            raise ValueError(f"Unknown shape '{shape}'. Use 'square' or 'hexagonal'.")

        inertia = [Ixx, 0., 0., 0., Ixx, 0., 0., 0., Izz]
        self.sc.hub.mHub = mass
        self.sc.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(inertia)

        payload = messaging.VehicleConfigMsgPayload()
        payload.ISCPntB_B = inertia
        self.vcMsg = messaging.VehicleConfigMsg().write(payload)
        self._messages.append(self.vcMsg)

    # ─── Reaction Wheels ──────────────────────────────────────────────────────

    def add_rw_cluster(self, num_wheels: int = 3, max_momentum: float = 50.0, enable_jitter: bool = False, initial_omega: float = 0.0):
        """Add an orthogonal 3-wheel Honeywell HR16 RW cluster."""
        factory = simIncludeRW.rwFactory()
        dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        rw_model = messaging.JitterSimple if enable_jitter else messaging.BalancedWheels
        for i in range(min(num_wheels, 3)):
            if enable_jitter:
                # Bias wheels to initial spin (rad/s) so Jitter forces actually activate!
                factory.create("Honeywell_HR16", dirs[i],
                               maxMomentum=max_momentum,
                               RWModel=rw_model,
                               Omega=initial_omega)
            else:
                factory.create("Honeywell_HR16", dirs[i],
                               maxMomentum=max_momentum,
                               RWModel=rw_model)

        self.rw_effector = reactionWheelStateEffector.ReactionWheelStateEffector()
        self.rw_effector.ModelTag = f"{self.sc.ModelTag}_RW"
        factory.addToSpacecraft(f"{self.sc.ModelTag}_RWs", self.rw_effector, self.sc)
        self.sim.AddModelToTask(self.task_name, self.rw_effector)

        self.rw_cfg_msg = factory.getConfigMessage()
        self._messages.append(self.rw_cfg_msg)
        return self.rw_cfg_msg

    # ─── Navigation ───────────────────────────────────────────────────────────

    def add_simple_nav(self, enable_noise: bool = False, noise_std: float = 0.0005):
        """
        Add SimpleNav translator/attitude navigation module.
        
        The simpleNav module in Basilisk is a perfectly precise 'true internal' 
        navigator. It acts as an idealized perfect sensor that bypasses full 
        state estimation filters (like kalman filters) and directly passes 
        the true simulation state to the flight software.
        """
        self.nav = simpleNav.SimpleNav()
        self.nav.ModelTag = f"{self.sc.ModelTag}_Nav"
        
        if enable_noise:
            # Add Gaussian white noise to the translation outputs (metrology position noise)
            # SimpleNav state vector is 18x1. Position is indices [0, 1, 2].
            PMatrix = [[0.0]*18 for _ in range(18)]
            PMatrix[0][0] = PMatrix[1][1] = PMatrix[2][2] = noise_std**2
            self.nav.PMatrix = PMatrix
            
            walkBounds = [0.0]*18
            walkBounds[0] = walkBounds[1] = walkBounds[2] = 10.0 * noise_std
            self.nav.walkBounds = walkBounds
            
        # Subscribe to spacecraft output message to receive true r_CN_N and v_CN_N states
        self.nav.scStateInMsg.subscribeTo(self.sc.scStateOutMsg)
        
        # Add simpleNav object to primary dynamic task
        self.sim.AddModelToTask(self.task_name, self.nav)
        return self.nav

    # ─── Attitude Control ─────────────────────────────────────────────────────

    def add_inertial_pointing_control(self, target_vector, K=3.5, Ki=-1.0, P=30.0):
        """
        Point body +Z at a fixed inertial direction (star vector).
        Sets up: inertial3D → attTrackingError → mrpFeedback → rwMotorTorque.
        """
        s_n  = np.array(target_vector, dtype=float)
        s_n /= np.linalg.norm(s_n)
        z_b  = np.array([0., 0., 1.])

        if np.allclose(s_n, z_b):
            sigma_RN = [0., 0., 0.]
        elif np.allclose(s_n, -z_b):
            sigma_RN = [1., 0., 0.]
        else:
            v  = np.cross(z_b, s_n)
            s  = np.linalg.norm(v)
            c  = np.dot(z_b, s_n)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R  = np.eye(3) + vx + vx @ vx * ((1.0 - c) / s**2)
            d  = 1.0 + R[0, 0] + R[1, 1] + R[2, 2]
            if d > 1e-6:
                zeta = np.sqrt(d)
                sigma = np.array([R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0]]) / (zeta*(zeta+2))
            else:
                sigma = np.array([1., 0., 0.])
            sigma_RN = sigma.tolist()

        guid = inertial3D.inertial3D()
        guid.ModelTag   = f"{self.sc.ModelTag}_InertialPoint"
        guid.sigma_R0N  = sigma_RN
        self.sim.AddModelToTask(self.task_name, guid)

        self._add_tracking_error(guid.attRefOutMsg, K, Ki, P)

    def add_los_pointing_control(self, chief_nav_trans_msg, K=3.5, Ki=-1.0, P=30.0):
        """Point body +Z toward the chief spacecraft (line-of-sight)."""
        los = spacecraftPointing.spacecraftPointing()
        los.ModelTag = f"{self.sc.ModelTag}_LOSPoint"
        los.chiefPositionInMsg.subscribeTo(chief_nav_trans_msg)
        los.deputyPositionInMsg.subscribeTo(self.nav.transOutMsg)
        # Add a 1e-6 epsilon to X to permanently prevent mathematical Gimbal Lock
        # Divide-by-Zero singularity when attempting to pitch 180-degrees perfectly
        # along the Z-axis (which happens exclusively during i=0.0 equatorial orbits!)
        los.alignmentVector_B = [1e-6, 0., 1.]
        self.sim.AddModelToTask(self.task_name, los)

        self._add_tracking_error(los.attReferenceOutMsg, K, Ki, P)

    def _add_tracking_error(self, att_ref_msg, K, Ki, P):
        """Wire attTrackingError → mrpFeedback → rwMotorTorque."""
        err = attTrackingError.attTrackingError()
        err.ModelTag = f"{self.sc.ModelTag}_AttErr"
        err.attNavInMsg.subscribeTo(self.nav.attOutMsg)
        err.attRefInMsg.subscribeTo(att_ref_msg)
        self.sim.AddModelToTask(self.task_name, err)

        ctrl = mrpFeedback.mrpFeedback()
        ctrl.ModelTag = f"{self.sc.ModelTag}_MRPCtrl"
        ctrl.guidInMsg.subscribeTo(err.attGuidOutMsg)
        ctrl.vehConfigInMsg.subscribeTo(self.vcMsg)
        ctrl.rwParamsInMsg.subscribeTo(self.rw_cfg_msg)
        ctrl.rwSpeedsInMsg.subscribeTo(self.rw_effector.rwSpeedOutMsg)
        ctrl.K, ctrl.Ki, ctrl.P = K, Ki, P
        self.sim.AddModelToTask(self.task_name, ctrl)

        motor = rwMotorTorque.rwMotorTorque()
        motor.ModelTag = f"{self.sc.ModelTag}_RWMotor"
        motor.vehControlInMsg.subscribeTo(ctrl.cmdTorqueOutMsg)
        motor.rwParamsInMsg.subscribeTo(self.rw_cfg_msg)
        motor.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self.rw_effector.rwMotorCmdInMsg.subscribeTo(motor.rwMotorTorqueOutMsg)
        self.sim.AddModelToTask(self.task_name, motor)

    # ─── Thrusters ────────────────────────────────────────────────────────────

    def add_thrusters(self, thrust: float = 1.0):
        """
        Add a 6-thruster ±X/Y/Z RCS set (MOOG Monarc-1 model).
        Used for translation via ExtForceTorque in the main loop.
        """
        def _flatten3(x):
            return [float(x[k][0]) if hasattr(x[k], "__len__") else float(x[k])
                    for k in range(3)]

        locations  = [[.5,0,0],[-.5,0,0],[0,.5,0],[0,-.5,0],[0,0,.5],[0,0,-.5]]
        directions = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]

        factory = simIncludeThruster.thrusterFactory()
        for loc, direc in zip(locations, directions):
            factory.create("MOOG_Monarc_1", loc, direc, MaxThrust=float(thrust))

        eff = thrusterDynamicEffector.ThrusterDynamicEffector()
        eff.ModelTag = f"{self.sc.ModelTag}_Thrusters"
        factory.addToSpacecraft(f"{self.sc.ModelTag}_ThrCluster", eff, self.sc)
        self.sim.AddModelToTask(self.task_name, eff)

        cfg = messaging.THRArrayConfigMsgPayload()
        cfg.numThrusters = len(factory.thrusterList)
        for i, thr in enumerate(factory.thrusterList.values()):
            th = messaging.THRConfigMsgPayload()
            th.maxThrust      = float(thr.MaxThrust)
            th.rThrust_B      = _flatten3(thr.thrLoc_B)
            th.tHatThrust_B   = _flatten3(thr.thrDir_B)
            messaging.ThrustConfigArray_setitem(cfg.thrusters, i, th)

        self.thr_cfg_msg = messaging.THRArrayConfigMsg().write(cfg)
        self._messages.append(self.thr_cfg_msg)
        print(f"[{self.sc.ModelTag}] {cfg.numThrusters} thrusters provisioned "
              f"(MaxThrust={thrust} N each).")

    # ─── CSS / Navigation Noise ───────────────────────────────────────────────

    def add_css_array(self, sun_msg, noise_std: float = 0.001, bias: float = 0.0):
        """
        Add a 6-element Coarse Sun Sensor array + WLS sun-direction estimator.

        Produces a noisy body-frame sun direction each step.
        Call ``read_css_vals()`` BEFORE every ``sim.ExecuteSimulation()`` call.

        Parameters
        ----------
        sun_msg    : SPICE sun ephemeris message
        noise_std  : 1σ Gaussian noise per sensor [cosine units, ~0.001 ≈ 0.057°]
        bias       : Constant offset per sensor [cosine units]
        """
        nhat_list = [
            [1,0,0],[-1,0,0],
            [0,1,0],[0,-1,0],
            [0,0,1],[0,0,-1],
        ]
        self._css_list = []
        for i, nhat in enumerate(nhat_list):
            css = coarseSunSensor.CoarseSunSensor()
            css.ModelTag      = f"{self.sc.ModelTag}_CSS_{i}"
            css.nHat_B        = nhat
            css.fov           = np.pi / 2.0
            css.senNoiseStd   = noise_std
            css.senBias       = bias
            css.scaleFactor   = 1.0
            css.maxOutput     = 1.0
            css.minOutput     = 0.0
            css.stateInMsg.subscribeTo(self.sc.scStateOutMsg)
            css.sunInMsg.subscribeTo(sun_msg)
            self.sim.AddModelToTask(self.task_name, css)
            self._css_list.append(css)

        self._css_array_msg = messaging.CSSArraySensorMsg()

        cfg_payload = messaging.CSSConfigMsgPayload()
        cfg_payload.nCSS = len(nhat_list)
        for i, nhat in enumerate(nhat_list):
            unit = messaging.CSSUnitConfigMsgPayload()
            unit.nHat_B = nhat
            unit.CBias  = bias
            cfg_payload.cssVals[i] = unit
        self._css_cfg_msg = messaging.CSSConfigMsg().write(cfg_payload)
        self._messages.append(self._css_cfg_msg)

        self.css_wls = cssWlsEst.cssWlsEst()
        self.css_wls.ModelTag = f"{self.sc.ModelTag}_CSSWls"
        self.css_wls.cssDataInMsg.subscribeTo(self._css_array_msg)
        self.css_wls.cssConfigInMsg.subscribeTo(self._css_cfg_msg)
        self.css_wls.sensorUseThresh = 0.1

        # CSSAggregator runs BEFORE cssWlsEst each tick, populating _css_array_msg
        self._css_aggregator = CSSAggregator(self._css_list, self._css_array_msg)
        self._css_aggregator.ModelTag = f"{self.sc.ModelTag}_CSSAgg"
        self.sim.AddModelToTask(self.task_name, self._css_aggregator)
        self.sim.AddModelToTask(self.task_name, self.css_wls)

        print(f"[CSS] {self.sc.ModelTag}: 6-element array "
              f"(noise={noise_std:.4f}, bias={bias:.4f}).")

    def has_css(self) -> bool:
        """True if a CSS array has been added to this spacecraft."""
        return hasattr(self, "_css_list")

    # ─── Utilities ────────────────────────────────────────────────────────────

    def get_messages(self):
        """Return all BSK message objects (prevents Python garbage collection)."""
        return self._messages
