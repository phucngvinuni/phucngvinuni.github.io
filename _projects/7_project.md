---
layout: page
title: Privacy-Preserving Thermal Fall Detection on Edge AI
description: "Research internship at HKU AIoT Lab (The University of Hong Kong). Real-time edge hardening on NVIDIA Jetson Orin (6.7 → 20.8 FPS, latency < 0.3s), 91,800-frame multimodal dataset, and 10× false-positive rate reduction (40.7% to 1.5%) via passive infrared array sensing."
img: assets/img/projects/fall-detection/thermal_fall_cover.jpg
importance: 1
category: research
related_publications: false
---

A research project conducted during my research internship at the **HKU AIoT Lab (The University of Hong Kong)**, focusing on deploying, optimizing, and hardening **TaFall** &mdash; a balance-informed fall detection system powered by a low-cost, ultra-low-resolution passive infrared (thermal) sensor &mdash; for autonomous real-time operation on resource-constrained edge hardware.

While the core TaFall theoretical framework was published by the laboratory (*Li, Zhang, Zhu, Jiang & Wu, arXiv:2604.09693*), my internship tackled the critical engineering and algorithmic gap between an academic proof-of-concept and a trustworthy edge-deployed production system: **rebuilding the real-time inference loop on NVIDIA Jetson Orin, collecting and auto-annotating a 91,800-frame multimodal thermal dataset, and isolating a preprocessing bug to achieve a 10&times; reduction in empty-room false alarms.**

<div class="project-meta-box p-3 mb-4 rounded" style="background-color: var(--global-card-bg-color, #f8f9fa); border-left: 4px solid var(--global-theme-color, #0076df);">
  <div class="row text-center text-md-left">
    <div class="col-6 col-md-3 mb-2">
      <strong>Host Laboratory:</strong><br>HKU AIoT Lab (HKU)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Thermal Sensor:</strong><br>Meridian MI48 (80&times;62 @ 30 FPS, ~$18)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Edge Platform:</strong><br>NVIDIA Jetson Orin (FP16 PyTorch)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Edge Throughput:</strong><br>20.8 FPS (Display Latency &lt; 0.28s)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Curated Dataset:</strong><br>45 Sessions, 91,800 Thermal Frames
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Empty-Room FPR:</strong><br>40.7% &rarr; 1.5% (10&times; reduction)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Lab Mentors:</strong><br>Prof. Wu, Dr. Xie Zhang, Chengxiao Li
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Deliverables:</strong><br>Edge Pipeline, C++ / Python Driver, Web UI
    </div>
  </div>
</div>

---

## Live System Demonstration

Below is a live screen capture demonstrating the end-to-end edge pipeline running on the **NVIDIA Jetson Orin**. The top panel displays the real-time multi-modal evaluation stream (synchronized Intel RealSense RGB vs. Meridian 80&times;62 thermal array), while the bottom panel showcases the deployed browser dashboard streaming at **20.8 FPS** with independent real-time confidence scores for human presence detection and balance-loss classification:

<div class="row mt-3 mb-4">
    <div class="col-12 text-center">
        <div class="embed-responsive shadow-lg rounded" style="max-width: 960px; margin: 0 auto; background: #000;">
            <video width="100%" height="auto" controls poster="{{ 'assets/img/projects/fall-detection/demo_poster.jpg' | relative_url }}" preload="metadata" style="border-radius: 8px;">
                <source src="{{ 'assets/video/thermal_fall_detection_demo.mp4' | relative_url }}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <p class="text-muted mt-2 small">
            <em>Demonstration: Real-time multi-person tracking and fall detection on NVIDIA Jetson Orin (Meridian MI48 thermal array, 20.8 FPS live streaming).</em>
        </p>
    </div>
</div>

---

## 1. The Challenge: Private Fall Monitoring & The 2% FPR Trap

Falls represent the leading cause of fatal and non-fatal injuries among older adults:
- **Epidemiology:** Between **28% and 35%** of individuals over age 65 fall at least once each year, accounting for 10–15% of all emergency department admissions.
- **The Psychological Toll:** Between 15% and 55% of older adults subsequently restrict their daily activities out of fear of falling, initiating a cycle of physical decline.
- **The Privacy Dilemma:** The majority of severe falls occur in **bedrooms and bathrooms** &mdash; exactly the private living spaces where optical RGB cameras are completely unacceptable and legally impermissible.

| Sensing Modality | Fall Accuracy | Privacy Preservation | Operates in Darkness | User Compliance | Hardware Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wearables (Pendant/Watch)** | Moderate | High | Yes | **Very Poor (forgotten/uncharged)** | Low (~$50–$150) |
| **Optical Cameras (RGB-D)** | Very High | **Unacceptable** | No (fails in dark) | High (passive) | Medium (~$100–$300) |
| **mmWave FMCW Radar** | Moderate | High | Yes | High (passive) | High (~$200–$500) |
| **Passive Infrared Array (MI48)** | **High (TaFall)** | **Guaranteed (80&times;62 px)**| **Yes (Total Darkness)**| **High (100% Passive)** | **Ultra-Low (~$18 BOM)** |

### The 2% False-Positive Rate Trap
In academic literature, a 98% specificity (a **2% False Positive Rate**) is frequently hailed as state-of-the-art. However, when deployed in a real-world home or hospital room:
- A monitoring system evaluating 1-minute temporal windows runs **1,440 inferences per day**.
- A 2% false-positive rate translates into **~29 false emergency alarms every single day**.
- Within 48 hours, caregivers experience complete alarm fatigue and switch the device off entirely.

> *"A fall detection system that cries wolf 30 times a day in an empty room is worse than no system at all. My internship was dedicated to driving that false-positive number down to zero on empty rooms without sacrificing fall sensitivity."*

---

## 2. The Baseline: TaFall & Balance-Informed Pose Dynamics

Prior thermal fall systems relied on crude velocity thresholds, height drops, or bounding-box bounding aspect ratios, generating pervasive false alarms whenever an occupant bent down to tie shoes, sat rapidly on a couch, or dropped a blanket.

The HKU AIoT Lab established **TaFall** (*arXiv:2604.09693*), founded on the World Health Organization (WHO) definition: **a fall is fundamentally a transient, unrecoverable loss of balance**, not simply a vertical descent.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/fig1_tafall.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 1: Core TaFall Paradigm (Li et al., 2026) &mdash; Top: Synchronized RGB ground truth. Middle: Ultra-low-resolution 80&times;62 thermal map. Bottom: Biomechanical balance state (Green: stable balance; Red: unrecoverable loss of balance; Amber: impact shock). The blue arrow indicates the Center of Mass (CoM) projection onto the floor; the green ellipse denotes the Base of Support (BoS). Once CoM exits BoS, a fall is biomechanically inevitable." %}
    </div>
</div>

### System Architecture (Li et al., 2026)
TaFall employs a dual-branch neural architecture:
1. **Branch (a) &mdash; Balance-Informed Fall Detection:** An appearance-motion fusion module extracts 2.5D human pose skeletons from consecutive thermal frames, which are fed into a balance-aware pose network to calculate the dynamic relationship between the Center of Mass (CoM) and the Base of Support (BoS).
2. **Branch (b) &mdash; Out-of-Vocabulary Enhancement:** Pre-trained on extensive public motion-capture (MoCap) repositories projected into diverse 2.5D synthetic camera angles to reject unseen everyday actions (squatting, stretching, yoga).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/fig3_tafall.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 2: TaFall System Architecture (Li et al., 2026, Fig. 3) &mdash; Five-stage pipeline: Sensor Acquisition &rarr; CenterNet Person Detector &rarr; SimpleIoU Tracker &rarr; Skeleton Pose Estimator &rarr; Balance-Aware Fall Classifier." %}
    </div>
</div>

---

## 3. Workstream 1: Real-Time Edge Deployment on NVIDIA Jetson Orin

The original research prototype achieved only **6.7 FPS** in Python with an unacceptable end-to-end display latency of **~3.0 seconds**, creating severe UI lag where a fall was reported long after the subject had already recovered or left the field of view.

I completely re-architected the edge runtime across multi-threaded sensor acquisition, sliding window management, and convolutional inference:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/inference_loop_architecture.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 3: Rebuilt Multi-Threaded Edge Inference Architecture &mdash; Decoupling MI48 sensor acquisition (30 FPS thread) from the CenterNet detector, SimpleIoU tracker, Skeleton Streamer, and sequence fall classifier." %}
    </div>
</div>

### Four Core Runtime Optimizations

| Optimization | Original Prototype | Edge-Hardened Implementation | Physical Rationale & Safety |
| :--- | :--- | :--- | :--- |
| **Sliding Window Size** | 40 frames (~1.6s) | **20 frames (~0.8s)** | Biomechanically verified: fall impact occurs within &le; 0.6s. 20 frames captures the complete balance transition while halving latency. |
| **Inference Stride** | Evaluated every frame ($S=1$) | **Evaluated every 4th frame ($S=4$)** | At 25–30 FPS, $S=4$ yields ~6.25 full classifications/sec, providing sub-160ms decision updates. |
| **CNN Feature Cache** | None (20 encodes/tick) | **Circular Modulo Cache (4 encodes/tick)** | Exploits sliding temporal overlap: 16 of 20 frames already encoded. **Eliminates 80% of convolutional operations.** |
| **Execution Precision** | Full FP32 | **Mixed-Precision FP16 (TensorRT/PyTorch)** | GPU tensor core acceleration with loss scaling; zero degradation in pose joint coordinates. |

### The Modulo Circular Feature Cache
Because the 20-frame sliding window advances with a stride of 4, frames $4 \dots 19$ have already been processed in the preceding tick. To eliminate redundant forward passes through the deep backbone:
- We index a circular buffer using `frame_idx % 20`.
- Because consecutive window indices are strictly within 20 frames of each other, mathematical collisions are strictly impossible ($\Delta < 20 \implies i \not\equiv j \pmod{20}$).
- The oldest 4 frames leaving the window are automatically overwritten by the newest 4 frames arriving, with **zero dynamic memory reallocation or garbage collection stalls**.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/cache_modulo_diagram.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 4: Modulo-Based Circular Feature Cache Architecture &mdash; Cold start encodes 20 frames; all subsequent ticks achieve an 80% cache hit rate (16 cached, 4 new), dropping CNN encoding calls from 20 to 4 per tick on the Jetson Orin." %}
    </div>
</div>

### Overcoming Real Hardware Traps (Socket Buffering & Thermal Throttling)
During bench testing on the NVIDIA Jetson Orin, two critical hardware bottlenecks were diagnosed and resolved:
1. **The MJPEG TCP Buffer Lag:** When streaming live video over an SSH tunnel (`ssh -L`), frames backed up in the TCP socket buffer. Although throughput showed 20 FPS, the video lagged by over 2.0 seconds. Refactoring to direct LAN WebSocket/MJPEG streaming collapsed latency to **0.28 seconds**.
2. **Thermal Throttling Collapse:** After 5–7 minutes of sustained inference, GPU clock frequencies throttled down, dropping throughput from 21 FPS to 7 FPS. By locking the power profile using `nvpmodel -m 0` (MAXN mode) and pinning GPU/CPU clock governors via `jetson_clocks`, sustained performance remained completely flat over hours of testing.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/jetson_latency_breakdown.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 5: Measured Runtime Acceleration on NVIDIA Jetson Orin &mdash; Sustained frame rate boosted from 6.7 to 20.8 FPS; latency reduced from 3.0s to under 0.3s." %}
    </div>
</div>

---

## 4. Workstream 2: 45-Session, 91,800-Frame Multimodal Dataset

The initial Phase 1 dataset suffered from two severe distribution gaps: almost no empty-room recordings and only one multi-person recording. 

To overcome this, I designed and conducted **Phase 2 Data Collection**, expanding the corpus to **45 structured recording sessions totaling 91,800 thermal frames (~51 minutes of continuous telemetry)**:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/dataset_overview.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 6: Comprehensive Dataset Distribution &mdash; (A) Frame breakdown by occupant count (including 21,608 empty-room frames); (B) Session catalog across 6 activity categories; (C) Pixel temperature distributions highlighting the critical tail above 30°C." %}
    </div>
</div>

### Autonomous Cross-Modal Labelling (Zero Manual Annotation)
Manually labelling 91,800 noisy 80&times;62 thermal frames with bounding boxes and 17-joint skeletons was infeasible. I engineered an automated cross-modal transfer pipeline utilizing an Intel RealSense RGB camera co-mounted with the MI48 sensor:

1. **High-Confidence RGB Pose Extraction:** Run `YOLOv8n-Pose` (confidence threshold = 0.55) on the timestamp-aligned RGB frame.
2. **Depth-Calibrated Perspective Projection:** Project bounding boxes from $480 \times 640$ RGB coordinates into $62 \times 80$ thermal space using calibrated Field-of-View (FOV) transformation matrices.
3. **Automated Thermal Gating Filter (My Core Contribution):** Due to minor optical parallax between sensors, projected boxes occasionally fell onto cold walls or furniture. I implemented a physical temperature verification gate: **reject any projected box whose peak internal pixel temperature is $< 24^\circ\text{C}$**. Because living humans always exceed ambient room temperatures, this eliminated false projection artifacts without human intervention.
4. **Explicit Negative Session Injection:** For the 9 empty-room sessions, empty bounding-box lists were injected into training, explicitly teaching the model what an unoccupied room looks like.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/labeled_samples.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 7: Autonomous Multi-Modal Annotation Pipeline &mdash; Synchronized RealSense RGB detection projected into thermal coordinates, verified via the 24°C temperature gate." %}
    </div>
</div>

---

## 5. Workstream 3: Detector Fine-Tuning & 10&times; False-Alarm Reduction

The initial detector (`v1`) exhibited an unacceptable **40.7% False Positive Rate on empty rooms**. Three initial rounds of retraining with additional data brought this down to only **34.3%** &mdash; an insignificant improvement.

Investigating the raw thermal histogram revealed the root cause: **a silent preprocessing bug**.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/temp_distribution.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 8: Pixel Temperature Density Analysis &mdash; Blue: Ambient background (20–21°C). Orange: Human body pixels (averaging 23.4°C with a long tail extending beyond 35°C). The original code clipped input temperatures to [20, 30]°C, completely truncating warm torso centers and cool extremities." %}
    </div>
</div>

### The 1-Line Preprocessing Fix
The original codebase clipped input thermal values to $[20^\circ\text{C}, 30^\circ\text{C}]$. Consequently:
- Warm human torsos ($> 30^\circ\text{C}$) were clipped to 30°C.
- Cooler extremities and thin clothing ($< 20^\circ\text{C}$) were clipped to 20°C (indistinguishable from the room background).
- The neural network was trained on artificially flattened, truncated contrast.

By expanding the normalization clip range from $[20, 30]^\circ\text{C}$ to **$[15, 37]^\circ\text{C}$**, input contrast was restored:
- **Empty-room false positive rate instantly plummeted from 34.3% to 3.6% &mdash; a 10&times; reduction from a single line of preprocessing code.**
- Subsequent unfreezing of the entire convolutional backbone across the full 45 sessions (`v4_full`) brought empty-room FPR down to **1.5%**.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/eval_all_models.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 9: Quantitative Benchmark Across All 6 Detector Versions at IoU@0.5 &mdash; (A) F1 score by scenario; (B) Precision vs. Recall (precision climbs from 0.52 to 0.65 while recall remains rock-solid at ~0.53); (C) Dramatic collapse of empty-room False Positive Rate from 40.7% down to 1.5%." %}
    </div>
</div>

### Qualitative Verification: Baseline vs. Deployed Model

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/before_after_grid.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 10: Qualitative Benchmark on Scene 1 &mdash; Row 1: Completely empty room (Original: 1.05 false boxes/frame at 0.76 confidence; Tuned v4_aug: 0.02 boxes/frame). Row 2: Laptop thermal exhaust plume (Original: false detection on vent at 0.76; Tuned: 0 false detections). Row 3: Standing person (Original: phantom split detection; Tuned: single clean detection with confidence boosted from 0.82 to 0.93)." %}
    </div>
</div>

---

## 6. Edge Hardening: Sensor Calibration & Multi-Person Dynamics

### Compensating Systematic Sensor Domain Shift
When transferring models from the laboratory data collection rig (sensor unit `COM14`) to the standalone Jetson Orin deployment unit (`COM7`), we observed an unexpected degradation in detection stability.

Empirical calibration revealed that unit `COM7` read systematically **$1.07^\circ\text{C}$ warmer** across 99.7% of all spatial pixels on identical scenes:
- The standard deviation of the difference was only $0.40^\circ\text{C}$, confirming a **systematic analog ADC offset rather than random sensor noise or optical distortion**.
- Applying a software compensation bias ($\Delta T = -1.07^\circ\text{C}$) restored model detection thresholds without requiring retraining.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/sensor_domain_shift.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 11: Systematic Thermal Offset Calibration &mdash; Inter-unit calibration comparing data-collection sensor COM14 vs. Jetson deployment sensor COM7, demonstrating a uniform 1.07°C baseline shift." %}
    </div>
</div>

### Geometric Filtering for Non-Human Heat Sources
To safeguard against edge cases without inflating neural network parameter count:
1. **Laptop & Appliance Vent Rejection:** Thermal vents emit localized hotspots (~33°C) that resemble human torsos at $80 \times 62$ resolution. We implemented an instantaneous connected-component geometric filter: reject any hot cluster whose area is $< 120$ pixels with an aspect ratio (Height / Width) $< 1.3$. An upright human body is tall and narrow; a thermal fan exhaust is squat and square.
2. **Multi-Person Heatmap Ceiling Analysis:** Quantitative evaluation revealed that 2-person F1 ($0.388$) and 3-person F1 ($0.230$) remained lower than single-person F1 ($0.749$). We proved that this is an **inherent architectural resolution limit**: at $80 \times 62$ input resolution, the detector downsamples to a $16 \times 20$ output heatmap. When three individuals stand in close proximity, their Gaussian heat signatures blend into a single continuous blob prior to Non-Maximum Suppression (NMS).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/fall-detection/threshold_analysis.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 12: Heatmap Peak & NMS Threshold Analysis &mdash; Demonstrating dual-peak detection thresholds (primary 0.36, secondary 0.18) for multi-occupant scenarios." %}
    </div>
</div>

---

## 7. Project Summary & Technical Report

### Key Quantified Outcomes
- **Edge Acceleration:** Throughput elevated from **6.7 FPS to 20.8 FPS** on NVIDIA Jetson Orin; display latency reduced by **86%** (from 2.0s to 0.28s).
- **Thermal Dataset:** Created, verified, and cataloged **45 sessions (91,800 frames)** with automated cross-modal ground truth.
- **Reliability Hardening:** Empty-room false-positive rate reduced from **40.7% to 1.5%** via preprocessing range correction ($15\text{--}37^\circ\text{C}$) and full backbone unfreezing.
- **Inter-Unit Calibration:** Formulated and deployed a **$1.07^\circ\text{C}$ systematic calibration bias** to overcome hardware unit variance.

### Research Team & Acknowledgements
- **Author & Edge System Engineer:** **Nguyen Hong Phuc** (VinUniversity)
- **Host Institution:** **HKU AIoT Lab**, Department of Computer Science, The University of Hong Kong
- **Base Algorithm (TaFall):** Chengxiao Li, Dr. Xie Zhang, Y. Zhu, S. Jiang, and Prof. Chenshu Wu (*arXiv:2604.09693*)
- **Special Gratitude:** I would like to express my sincere gratitude to **Prof. Chenshu Wu**, **Dr. Xie Zhang**, and **Chengxiao Li** for their mentorship, technical guidance, and invaluable support throughout my research internship at HKU AIoT Lab.

<div class="text-center mt-4 mb-4">
    <a href="{{ 'assets/pdf/Thermal_Fall_Detection_Report.pdf' | relative_url }}" class="btn btn-outline-primary btn-lg" target="_blank" rel="noopener noreferrer">
        <i class="fas fa-file-pdf mr-2"></i> Download Full Internship Technical Report (PDF, 21 Slides)
    </a>
</div>
