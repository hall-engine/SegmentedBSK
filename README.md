# SegmentedBSK Simulator Data Fields

This document describes the data fields collected and plotted by the simulator. The simulation results are stored and available in the `extra` dictionary passed to the plotting routines.

## Spacecraft State & Trajectory
* **`r1_n`**: Position vector of Aperture spacecraft in the inertial N-frame (`np.array` of 3D vectors over time).
* **`r2_n`**: Position vector of Detector spacecraft in the inertial N-frame (`np.array` of 3D vectors over time).

## Mission & Control Status
* **`engaged`**: Boolean flag indicating when the active formation control (e.g., observation mode) is engaged versus drifting/calibrating.
* **`phase`**: The current mission phase name (e.g., "Drifting", "Observation", etc.) over time.
* **`pos_err`**: The relative position error vector. Measures how far the Detector is from its target formation position relative to the Aperture and the star vector.

## Actuators & Effort (Fuel/Power)
* **`dv`**: Total accumulated Delta-V (change in velocity) expended by the thrusters over time. Useful for fuel budget analysis.
* **`dv_xyz`**: The individual X, Y, and Z components of the accumulated Delta-V, showing which axes are using the most fuel.
* **`rw_speeds`**: The angular velocities of the Reaction Wheels over time, used to monitor momentum buildup and attitude control effort.

## Environment & Sensors
* **`star_vector`**: The fixed inertial unit vector pointing toward the target star being observed.
* **`sun_pos_n`**: The position vector of the Sun in the inertial N-frame over time.
* **`moon_pos_n`**: The position vector of the Moon in the inertial N-frame over time (used for third-body gravity perturbations).

## Solar Radiation Pressure (SRP) & Sun Sensors
* **`css_sun_app`**: The sun direction vector as measured/estimated by the Coarse Sun Sensors (CSS) algorithms on the Aperture spacecraft.
* **`css_sun_det`**: The sun direction vector as measured/estimated by the Coarse Sun Sensors (CSS) algorithms on the Detector spacecraft.
* **`true_sun_app`**: The ground-truth (actual) sun direction relative to the Aperture spacecraft, used to calculate how noisy/inaccurate the CSS readings are.
* **`true_sun_det`**: The ground-truth (actual) sun direction relative to the Detector spacecraft.
* **`srp_app_vec`**: The Solar Radiation Pressure force/acceleration perturbation vector acting on the Aperture spacecraft.
* **`srp_det_vec`**: The Solar Radiation Pressure force/acceleration perturbation vector acting on the Detector spacecraft.
