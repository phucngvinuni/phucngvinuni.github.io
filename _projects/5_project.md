---
layout: page
title: Quadcopter Design, FEA Simulation & Teensy 4.1 Flight Controller
description: "ELEC3030 Intelligent Physical Systems at VinUniversity. Custom 400Hz C++ flight controller firmware, Madgwick IMU sensor fusion, ANSYS structural FEA, and Carbon Fiber CNC frame optimization."
img: assets/img/projects/drone/drone_hardware_assembly.jpg
importance: 5
category: work
related_publications: false
---

A complete autonomous aerial vehicle engineering project developed for **ELEC3030: Intelligent Physical Systems** at VinUniversity, spanning mechanical CAD/FEA simulation, propulsion modeling, and custom real-time flight controller firmware.

Unlike commercial off-the-shelf flight controllers (such as Betaflight), this project built an entire Quad-X flight management system from bare metal in **C++ on a Teensy 4.1 (ARM Cortex-M7 @ 600MHz)**, featuring 400Hz PID attitude stabilization, Madgwick sensor fusion, and structural Finite Element Analysis (FEA) comparing 3D-printed polymers with CNC-milled Carbon Fiber.

<div class="project-meta-box p-3 mb-4 rounded" style="background-color: var(--global-card-bg-color, #f8f9fa); border-left: 4px solid var(--global-theme-color, #0076df);">
  <div class="row text-center text-md-left">
    <div class="col-6 col-md-3 mb-2">
      <strong>Course:</strong><br>ELEC3030 IPS, VinUniversity
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Flight Controller:</strong><br>Teensy 4.1 (Cortex-M7 @ 600MHz)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Control Loop:</strong><br>400Hz PID (422Hz sustained)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Frame & FEA:</strong><br>ANSYS Static & Drop Test (CF vs. PLA)
    </div>
  </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_hardware_assembly.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 1: Fully assembled Quad-X prototype on the experimental test bench &mdash; Featuring the custom reinforced chassis, Teensy 4.1 flight computer, MPU6050 6-DOF IMU, 30A ESCs, and RS2205 brushless motors." %}
    </div>
</div>

---

## 1. Frame Design Evolution & Finite Element Analysis (FEA)

The structural chassis underwent multiple iterative design phases to balance mechanical rigidity, component payload volume, and impact shock tolerance.

### Failure Analysis of Initial Prototypes

Early 3D-printed iterations revealed critical mechanical failure modes:
- **Iteration 1:** The interior bay was too compact (< 50 mm clearance) to house the power distribution board and ESCs cleanly.
- **Iteration 2:** While providing sufficient space, thin structural arm walls (4 mm) and low infill density (20%) fractured during motor thrust oscillations and drop impacts.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_previous_design.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 2: Initial 3D frame iterations and failure modes under structural loading." %}
    </div>
</div>

### ANSYS Static Loading & Drop Test FEA Simulation

To systematically evaluate material resilience before fabrication, the team conducted comprehensive **ANSYS Finite Element Analysis (FEA)** comparing Poly-Lactic Acid (PLA) polymer against woven Carbon Fiber composite:

1. **Drop Test Simulation:** Evaluated dynamic shock wave propagation from an inverted 1.5 m impact. The integration of landing legs was shown to distribute impact reaction forces across the perimeter, shielding sensitive electronics (Teensy MCU, IMU) from peak deceleration shock.
2. **ANSYS Static Loading Simulation:** Under full motor thrust (4 × 800 g), Carbon Fiber exhibited maximum Von-Mises stress of only 10.03 MPa with negligible elastic strain (0.00033), whereas PLA experienced 42.17 MPa stress. While Carbon Fiber provided superior stiffness-to-weight ratio, high-infill PLA (60%, 8 mm thickness) proved mechanically sufficient for rapid prototyping.
3. **SolidWorks Topographic Optimization:** Performed material removal algorithms to identify low-stress regions, carving aerodynamic weight-reduction cutouts that cut frame mass by 28%.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_fea_simulation.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 3: ANSYS FEA simulation &mdash; Drop test impact stress distribution (Top), Static loading Von-Mises stress comparison (Middle), and SolidWorks topographic mass optimization (Bottom)." %}
    </div>
</div>

### Final CAD Geometry & Dimensions

The final design utilizes a **178.73 mm diagonal wheelbase** Quad-X configuration, manufactured via a hybrid combination of CNC-milled 3 mm Carbon Fiber base plates and reinforced 8 mm, 60%-infill 3D-printed motor arms:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_frame_cad.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 4: Final frame design &mdash; 3D printed arms with 60% infill and CNC milling toolpath simulation for 3mm carbon fiber cutting." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_multiview.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 5: Multi-view CAD orthographic and trimetric projections of the assembled airframe." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_dimensions.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 6: Dimensional layout &mdash; 178.73mm diagonal motor-to-motor wheelbase." %}
    </div>
</div>

---

## 2. Propulsion & Power Electronics Integration

The electrical propulsion subsystem was engineered to deliver a thrust-to-weight ratio > 2.2:1:

- **Motors:** Four **RS2205 2300kV** Brushless DC (BLDC) outrunner motors capable of delivering up to 1,024 g maximum thrust each on 5045 bullnose propellers.
- **Speed Controllers:** Four **30A Electronic Speed Controllers (ESCs)** running BLHeli firmware, receiving low-latency PWM drive signals from the Teensy flight controller.
- **Power Management:** A 3S 11.1 V LiPo battery delivers high discharge current (up to 75C). High-power ground loops and motor back-EMF spikes are decoupled using a low-ESR electrolytic capacitor bank and dedicated LC filtering for the flight computer.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_propulsion.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 7: Propulsion hardware &mdash; RS2205 2300kV BLDC motors and 30A high-frequency ESCs." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_electronics_wiring.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 8: Electrical schematic & wiring topology &mdash; Showing Teensy 4.1 flight computer, MPU6050 IMU, power distribution, and RC receiver channels." %}
    </div>
</div>

---

## 3. Real-Time Flight Controller Firmware & Sensor Fusion

The flight control firmware was developed from scratch in **C++** on the **Teensy 4.1**, taking advantage of the NXP i.MXRT1062 ARM Cortex-M7 running at 600MHz:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_software_architecture.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 9: High-level software architecture &mdash; Non-blocking task loop coordinating RC capture, sensor fusion, PID stabilization, and motor mixing." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_setup_flowchart.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 10: Startup initialization sequence &mdash; Gyroscope bias calibration, ESC arming protocol, and RC interrupt watchdog." %}
    </div>
</div>

### Madgwick Sensor Fusion Algorithm

Raw data from the onboard **MPU6050 6-DOF IMU** presents significant challenges:
- **Gyroscopes** offer high responsiveness but suffer from unbounded integration drift over time.
- **Accelerometers** provide a reliable gravity reference in the long term but are easily corrupted by high-frequency motor vibrations and linear accelerations.

To achieve robust attitude estimation, we implemented the **Madgwick Orientation Filter**:
- Uses quaternion gradient descent to compute the direction of the gravity field from accelerometer readings, directly correcting the orientation computed from integrated angular rates.
- Achieves equivalent or superior accuracy to an Extended Kalman Filter (EKF) with vastly reduced computational overhead ($pprox 10\,\mu\text{s}$ per iteration on Cortex-M7).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_madgwick_filter.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 11: Madgwick sensor fusion mathematical pipeline &mdash; Correcting gyro quaternion drift via accelerometer gradient descent." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_filter_testing.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 12: Dynamic roll angle tracking validation &mdash; Comparing raw noisy accelerometer data, drifting gyro integration, and the stable Madgwick estimate." %}
    </div>
</div>

---

## 4. 400Hz PID Attitude Stabilization & Telemetry

Attitude control is achieved through three independent **Proportional-Integral-Derivative (PID)** control loops operating on the Roll, Pitch, and Yaw axes at **400Hz**:


$$
u(t) = K_p \, e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d \, \frac{de(t)}{dt}
$$


- **Proportional (Kp):** Provides instantaneous restoring torque proportional to orientation error.
- **Integral (Ki):** Eliminates steady-state attitude offsets caused by minor battery mass asymmetry or aerodynamic drag.
- **Derivative (Kd):** Dampens high-speed rotational oscillations and prevents angular overshoot.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_pid_control.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 13: PID control topology &mdash; Mapping pilot setpoints and IMU states into differential motor throttle adjustments." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_pid_code.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 14: Core C++ implementation snippet of the PID error computation and anti-windup clamping." %}
    </div>
</div>

### Real-Time Loop Timing

Benchmarking confirmed a sustained loop execution frequency of **422Hz** (cycle time T_loop = 2.37 ms), well exceeding the 400Hz target to guarantee deterministic motor update deadlines with zero jitter:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_loop_timing.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 15: High-speed real-time telemetry streaming at 422Hz &mdash; Demonstrating microsecond-level timing determinism across IMU sampling and motor updates." %}
    </div>
</div>

---

## 5. Experimental Validation on Test Jig

Before free-flight testing, the drone was constrained to an instrumented multi-axis test stand to tune PID gains (Kp, Ki, Kd) and evaluate step-response settling times:

- **Roll & Yaw Step Responses:** Tested disturbance rejection by applying sudden rotational impulses. The controller returned to level within < 180 ms with zero steady-state oscillation.
- **Pitch Dynamic Tracking:** Demonstrated tight setpoint tracking across rapid pilot input transitions from -20° to +20°.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_roll_yaw_test.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 16: Experimental test jig &mdash; Real-time Roll and Yaw axis dynamic step response verification." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/drone/drone_pitch_test.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 17: Pitch axis disturbance rejection and angular settling verification on the test stand." %}
    </div>
</div>

---

## 6. Team & Project Information

- **Course:** ELEC3030 &mdash; Intelligent Physical Systems, VinUniversity
- **Institution:** College of Engineering & Computer Science, VinUniversity
- **Team Members:**
  - **Nguyen Hong Phuc** (Embedded C++ Firmware Architecture, Madgwick Sensor Fusion, 400Hz PID Control Loop)
  - **Le The Doan** (Chassis CAD Design, ANSYS Structural FEA Simulation, SolidWorks Optimization)
  - **Dinh Nguyen Phuong** (Power Electronics, Propulsion Hardware Testing)
  - **Vo Viet Duc** (Mechanical Assembly & Wiring)
