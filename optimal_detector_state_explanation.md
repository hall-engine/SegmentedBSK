# The Precise Orbit Inserter (calculate_optimal_detector_state)

## The Goal
The objective is to find the **exact initial orbital elements** ($a, e, i, \Omega, \omega, f$) for the Detector spacecraft at the start of the simulation so that, when the observation window occurs (e.g., at $E=90^\circ$ or $E=0^\circ$), the Detector naturally drifts into a "hover" state exactly `5000` meters away from the Aperture, perfectly along its optical axis, with zero relative velocity.

## Why delta_i Failed
Previously, the simulation inserted the Detector by taking the Aperture's orbit and simply adding a slight inclination shift ($\Delta i$). 
While this created a perfect stationary point when the Argument of Periapsis ($\omega$) was $0^\circ$, pushing $\omega$ to any non-zero value (like $10^\circ$) broke the geometric alignment. The natural maximum separation would occur at the wrong time, missing your observation window. Because the natural orbit was trying to drift away during the observation phase, the PID controller had to fight the orbital mechanics. This caused the massive Delta-V spikes you noticed.

## Step-by-Step Breakdown of the New Algorithm

### 1. Fast-Forward the Aperture to the Observation Window
Instead of guessing how the orbits will behave at the start, the function first computes the target end-state. The code takes the Aperture's initial orbital elements and mathematically fast-forwards them to compute exactly what its Cartesian state (ECI Position **`r_c_peak`**, Velocity **`v_c_peak`**) will be at your `target_eccentric_anomaly_deg`.

### 2. Define the Perfect Detector State at that Exact Moment
Now that we know exactly where the Aperture will be during the peak of the observation, we construct the *perfect* physical state for the Detector:
*   **The Line of Sight (`n_hat`)**: The function calculates the normal vector to the orbital plane by taking the cross product of the Aperture's position and velocity ($\vec{r} \times \vec{v}$). 
*   **Position (`r_d_peak`)**: The strict target position for the Detector is defined as the Aperture's position *plus* the focal length (`5000m`) extending exactly along that orbital normal vector.
*   **Velocity (`v_d_peak`)**: The Detector is assigned the exact same inertial velocity as the Aperture. By perfectly matching the velocity at this moment, we guarantee the relative drift (velocity) between the two spacecraft will be virtually zero at the peak, making it a perfect "stationary point."

### 3. Reverse-Engineer the Detector's Orbit
At this point, we have a hypothetical Cartesian state (`r_d_peak` and `v_d_peak`) that perfectly describes the Detector hovering 5000m away from the Aperture.
The function passes this Cartesian state into Basilisk's math engine `orbitalMotion.rv2elem()`. This converts those physical coordinates back into formal Keplerian orbital elements. This new element set automatically contains the precise tiny tweaks to Inclination, RAAN, and Argument of Periapsis required to make the orbit natively cross that exact point in space without thrusting!

### 4. "Rewind" the Orbit Back to the Simulation Start
We now have the perfect orbital elements for the Detector, but its true anomaly represents where it is during the *peak observation segment*, not at the start of the simulation. We must rewind it so it can be initialized properly by the engine.
1. The code calculates the time difference ($\Delta t$) between the simulation start anomaly and the exact target anomaly using the Aperture's mean motion.
2. It uses the Detector's newly computed mean motion to calculate how much Mean Anomaly the Detector would traverse over that exact $\Delta t$.
3. It subtracts that traversed anomaly from the peak anomaly, giving us the Mean Anomaly the Detector *must have been at* when the simulation started.
4. Finally, it uses a Newton-Raphson numerical solver to convert that starting Mean Anomaly back into a starting True Anomaly ($f_{d, \text{start}}$).

### 5. Return the Initial State
The function assigns this rewound True Anomaly to our newly generated Detector elements and hands them back to Basilisk. 

### The Result
By starting with the physical target constraints and mathematically tricking the propagator into solving for the orbital elements backwards, the spacecraft will organically follow the laws of physics and naturally slide into the optimal geometry exactly when you want it to. The controller no longer has to use heavy thrusters to hold the spacecraft in place, effectively mitigating extreme Delta-V spikes completely.
