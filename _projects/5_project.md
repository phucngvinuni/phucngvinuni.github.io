---
layout: page
title: Teensy 4.1 Flight Controller Firmware
description: C++, Embedded Systems, Control Theory — Jan. 2026
img:
importance: 5
category: work
---

Custom flight controller firmware developed on **Teensy 4.1** for a Quad-X drone, implementing a full 400Hz PID stabilization loop.

**Key Features:**
- **400Hz PID control loop** for Quad-X stabilization
- **Madgwick sensor fusion** for precise attitude estimation (MPU6050 IMU)
- Real-time RC signal noise filtering
- Safety arming logic and optimized Serial plotting at 500k baud for PID tuning and diagnostics

**Technologies:** C++, Teensy 4.1, MPU6050, PID Control, Madgwick Filter, Embedded Systems
