import numpy as np
import scipy.linalg
from config import SimConfig

class segmentLQR:
    def __init__(self, cfg: SimConfig):
            """
            Computes the LQR gain for the full state-space model.
            
            Input:
            -params from the simulation
            
            Outputs:
            Sets instance variables:
                - self.K_full: the computed LQR gain matrix.
                - self.S: the solution to the Riccati equation.
                - self.E: the eigenvalues of the closed-loop system.
            """
            self.cfg = cfg
            # initialise every parameters to compute the LQR gain
            self.mass = cfg.mirror_mass
            # Mirror mass
            self.I = cfg.mirror_I
            # Mirror moment of inertia.
            # self.L = cfg.mirror_L (Unused)
            # Mirror half-length.
            #self.A_full = params.A_angles
            # State-space A matrix.
            #self.B_full = params.B_angles
            # State-space B matrix.
            self.Q = cfg.mirror_Q
            # LQR state cost matrix.
            self.R = cfg.mirror_R
            # LQR control cost matrix.
            self.J_theta = cfg.mirror_J_theta
            # Torsional stiffness (theta).
            self.J_phi = cfg.mirror_J_phi
            # Torsional stiffness (phi).
            self.S = cfg.mirror_S
            # Piston surface area.
            self.d_theta = cfg.mirror_dampting_theta
            # Damping (theta).
            self.d_phi = cfg.mirror_dampting_phi
            # Damping (phi).
            self.d_z = cfg.mirror_dampting_S
            # Damping (vertical motion).
            # Define the matrix that are representing the miror on the piston, X_dot = A@X +Bu, X = state vector = [tip, tilt, piston, omega_tip, omega_tilt, piston_velocity], u = command =[torque_tip, torque_tilt, pressure in the piston]
            self.A = np.array([
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, -self.d_theta / self.J_theta, 0, 0],
                                    [0, 0, 0, 0, -self.d_phi / self.J_phi, 0],
                                    [0, 0, 0, 0, 0, -self.d_z / self.mass]
                                ])
            self.B = np.array([
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [1 / self.J_theta, 0, 0],
                                    [0, 1 / self.J_phi, 0],
                                    [0, 0, self.S / self.mass]
                                ])
            # compute the LQR gain using scipy
            self.S = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            self.K_full = np.linalg.inv(self.R) @ self.B.T @ self.S
            self.E = np.linalg.eigvals(self.A - self.B @ self.K_full)


    def lqr_control_full(self, state, dt):
        """
        Computes the LQR control input.

        State vector:
        X = [tip, tilt, piston, tip_dot, tilt_dot, piston_dot]
        """
        debug = False
        # ensure numpy arrays
        current_mirror_state = np.asarray(state.mirror_actuation, dtype=float)
        desired_mirror_state = np.asarray(state.desired_mirror_actuation, dtype=float)
        # safety check
        if current_mirror_state.size != 6:
            raise ValueError("mirror_actuation must contain 6 elements")
        if desired_mirror_state.size != 6:
            raise ValueError("desired_mirror_actuation must contain 6 elements")
        # compute error
        error = current_mirror_state - desired_mirror_state
        # LQR control law
        command = -self.K_full @ error

        # ── 1. Force / torque saturation ───────────────────────────────────────
        # Hard physical limit: actuator cannot produce more than max force/torque.
        max_tau = getattr(self.cfg, 'mirror_max_torque_nm',      0.5)
        max_fz  = getattr(self.cfg, 'mirror_max_piston_force_n', 100.0)
        command[0] = np.clip(command[0], -max_tau, max_tau)   # tip torque
        command[1] = np.clip(command[1], -max_tau, max_tau)   # tilt torque
        command[2] = np.clip(command[2], -max_fz,  max_fz)   # piston force

        # system dynamics
        dx = (self.A @ current_mirror_state) + (self.B @ command)
        # Euler integration
        new_mirror_state = current_mirror_state + dx * dt

        # ── 2. Slew rate limiting ──────────────────────────────────────────────
        # Clamp velocities to actuator bandwidth limits.
        max_tt_rate = getattr(self.cfg, 'mirror_max_tiptilt_rate_radps', 50e-3)
        max_pz_rate = getattr(self.cfg, 'mirror_max_piston_rate_mps',    5e-3)
        new_mirror_state[3] = np.clip(new_mirror_state[3], -max_tt_rate, max_tt_rate)  # tip rate
        new_mirror_state[4] = np.clip(new_mirror_state[4], -max_tt_rate, max_tt_rate)  # tilt rate
        new_mirror_state[5] = np.clip(new_mirror_state[5], -max_pz_rate, max_pz_rate)  # piston rate

        # ── 3. Stroke limiting ─────────────────────────────────────────────────
        # Clamp displacement to physical actuator travel.
        max_tt_stroke = getattr(self.cfg, 'mirror_max_tiptilt_stroke_rad', 1e-3)
        max_pz_stroke = getattr(self.cfg, 'mirror_max_piston_stroke_m',   100e-6)
        new_mirror_state[0] = np.clip(new_mirror_state[0], -max_tt_stroke, max_tt_stroke)  # tip
        new_mirror_state[1] = np.clip(new_mirror_state[1], -max_tt_stroke, max_tt_stroke)  # tilt
        new_mirror_state[2] = np.clip(new_mirror_state[2], -max_pz_stroke, max_pz_stroke)  # piston

        if debug:
            print(state.number)
            print(f'error = {error}')
            print(f'command = {command}')
            print(f'dx = {dx}')
            print(f'new mirror state = {new_mirror_state}')
            print()

        # ── Actuator quantization (Option A + C fine metrology) ────────────────
        # Mirrors the thruster MIB logic in control.py: the actuator has a
        # finite step size set by DAC resolution.  Commands smaller than the
        # LSB are rounded to zero, preventing the LQR from chasing noise below
        # the mechanical floor.
        if getattr(self.cfg, 'enable_ao_metrology_noise', False):
            res_piston  = getattr(self.cfg, 'mirror_actuator_resolution_piston_m',    1e-9)
            res_tiptilt = getattr(self.cfg, 'mirror_actuator_resolution_tiptilt_rad', 1e-9)
            new_mirror_state[0] = np.round(new_mirror_state[0] / res_tiptilt) * res_tiptilt  # tip
            new_mirror_state[1] = np.round(new_mirror_state[1] / res_tiptilt) * res_tiptilt  # tilt
            new_mirror_state[2] = np.round(new_mirror_state[2] / res_piston)  * res_piston   # piston

        # update state
        state.mirror_actuation = new_mirror_state
        # -------------------------------------------------------
        # Actuator rotor dynamics (tip / tilt)
        # -------------------------------------------------------
        Irot = getattr(self.cfg, 'actuator_rotor_inertia', 0.0)
        if Irot is not None and np.any(np.asarray(Irot) != 0.0):
            if not hasattr(state, 'mirror_rotor_speed'):
                state.mirror_rotor_speed = np.array([0.0, 0.0])
            if np.isscalar(Irot):
                Irot = np.array([Irot, Irot])
            tau_tip = command[0]
            tau_tilt = command[1]
            alpha_tip = tau_tip / Irot[0]
            alpha_tilt = tau_tilt / Irot[1]
            state.mirror_rotor_speed[0] += alpha_tip * dt
            state.mirror_rotor_speed[1] += alpha_tilt * dt
        return state