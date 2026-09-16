---
layout: page
title: Smart Marine Aquaculture – Nha Trang
description: "Field-deployable embedded systems, underwater imaging, bioacoustics, and 5-parameter water-quality sensing infrastructure."
img: assets/img/projects/smart-aquaculture/smart-marine-aquaculture.jpg
importance: 1
category: research
related_publications: true
github: https://github.com/korobn0608/aquasense
github_stars: korobn0608/aquasense
---

An intelligent, field-deployable multi-sensory aquaculture monitoring and edge AI infrastructure deployed across operational breeding raceways, outdoor circular tanks, and offshore floating sea cages in **Nha Trang, Vietnam**.

Developed through an interdisciplinary collaboration between the **[ICCL Lab](https://icclabo.github.io/icc/)**, **VinUniversity Smart Green Transformation Center ([GREEN-X](https://vinuni.edu.vn/))**, the **Vietnam National University of Agriculture (VNUA)**, and the **Research Institute for Aquaculture (RIA 1 & RIA 3)**. Funded by a **$3,500 Student Research Grant** awarded to lead research on smart aquaculture IoT and multimodal AI systems at VinUniversity.

<div class="project-meta-box p-3 mb-4 rounded" style="background-color: var(--global-card-bg-color, #f8f9fa); border-left: 4px solid var(--global-theme-color, #0076df);">
  <div class="row text-center text-md-left">
    <div class="col-6 col-md-3 mb-2">
      <strong>Timeline:</strong><br>Feb. 2025 – Present
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Sensing Modalities:</strong><br>CV + Bioacoustics + 5-Parameter Chemistry
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Field Deployments:</strong><br>4 Phases (VNUA &rarr; RIA 1 &rarr; Nha Phu &rarr; RIA 3)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Core Stack:</strong><br>PyTorch, Edge AI, Hydrophone DSP, LTE IoT
    </div>
  </div>
</div>

> **Project Links & Documentation:**
> - **GitHub Repository:** [AquaSense (IoT Node, Multimodal AI & Web App)](https://github.com/korobn0608/aquasense)
> - **Field Deployment Log:** [Smart Marine Aquaculture Case Study (by Minh-Hoang Pham)](https://hoang-pm-7604119.github.io/myself/projects/smart-marine-aquaculture/)

---

## 1. The Field Challenge

Smart marine aquaculture in coastal waters near Nha Trang presents harsh environmental constraints that disrupt traditional manual monitoring:
- **High Water Turbidity & Variable Lighting:** Suspended organic solids, plankton blooms, and ambient light attenuation drastically degrade submerged vision.
- **Rapid Marine Biofouling:** Algae, barnacles, and bacterial biofilms rapidly coat optical viewports and probe membranes, leading to severe sensor drift within days.
- **Saline Corrosion & Tropical Swells:** Open-sea floating cages face tidal currents, saltwater spray, mechanical wave shocks, and intense tropical solar heat.
- **Offshore Power & Connectivity Limits:** Operating kilometers from the mainland requires complete electrical self-sufficiency and resilient long-range telemetry.

Sustainable offshore farming demands continuous, automated insight into water chemistry, underwater acoustics, and fish behavior without requiring hazardous and infrequent manual diving.

---

## 2. Embedded System Architecture: 3 Sensing Modalities & Integrated AI

To provide continuous, end-to-end monitoring for offshore aquaculture operations, the research team engineered an integrated embedded hardware platform combining three primary sensing modalities coupled with edge AI algorithms:

1. **Submerged Computer Vision (Underwater Optical Sensing):**  
   Waterproof subsea camera rigs capture high-resolution imagery and video streams under challenging underwater illumination. Embedded vision models continuously observe fish biomass, schooling density, swimming kinematics, and automatically evaluate optical viewport biofouling severity in real time.
2. **Subsea Bioacoustics (Underwater Audio Sensing):**  
   Subsea hydrophone arrays capture ambient soundscapes and high-frequency acoustic dynamics. Bioacoustic signal processing models analyze fish chewing, pellet collision, and swimming acoustic signatures to quantify feeding intensity and appetite in real time.
3. **Five Key Physicochemical Water-Quality Parameters:**  
   An industrial-grade sensor manifold continuously monitors 5 critical physicochemical indicators:
   - **Dissolved Oxygen (DO)**
   - **pH**
   - **Water Temperature**
   - **Salinity**
   - **Turbidity**  
   Safeguarding against sudden hypoxia events, thermal shocks, or salinity shifts.
4. **Applied Edge Intelligence (Edge AI & DSP):**  
   On-site edge computing units (PyTorch, YOLO) and signal processing algorithms analyze video and acoustic streams locally, converting raw high-bandwidth sensor feeds into actionable operational insights, feeding schedule optimizations, and early stress/disease warnings.
5. **Task-Oriented Semantic Communication & Telemetry:**  
   Extracts task-relevant semantic representations to compress transmission payloads over bandwidth-constrained marine IoT links, streaming telemetry via encrypted MQTT pipelines to edge and mainland servers backed by PostgreSQL time-series storage, MinIO media buckets, and live operational dashboards.

---

## 3. Evolution of Field Deployments: From Laboratory to Open Ocean

The system underwent an iterative four-phase engineering trajectory, validating hardware resilience, fluidics, and telemetry across diverse real-world aquaculture environments.

### Phase 1: Laboratory Benchmarking & Multiplexed Fluidics at VNUA (Hanoi)

Development began at the Vietnam National University of Agriculture (VNUA), where controlled multi-tank environments allowed precise calibration of optical sensors and water quality instrumentation.

To overcome the high cost of duplicating industrial probes across multiple tanks, the team engineered a custom fluidic manifold box. Using automated solenoid valve cycling, a single high-precision sensing chamber could cyclically sample distinct water tanks in sequence, with automated freshwater flush cycles to prevent cross-contamination.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/smart-aquaculture/stage1-vnua-lab-testing.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 1: Multi-tank experimental test bench at VNUA during overnight calibration of automated water circulation and optical tracking rigs." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/smart-aquaculture/stage1-vnua-sensor-manifold.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 2: Solenoid valve manifold assembly multiplexing fluid intake lines from distinct aquaculture tanks into a single sensor chamber." %}
    </div>
</div>

---

### Phase 2: Pilot Raceway Hardening at RIA 1 (Hai Phong)

Moving beyond benchtop testing, the prototype was deployed at the Research Institute for Aquaculture No. 1 (RIA 1) in Hai Phong. This phase tested the system inside operational concrete raceways and indoor breeding pools under high-humidity, saline aerosol conditions.

The team validated continuous underwater camera telemetry, watertight cable pass-throughs, and real-time data streaming to the central GREEN-X cloud dashboard, isolating and resolving ground-loop electrical noise caused by high-power industrial water aerators.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/smart-aquaculture/stage2-ria1-indoor-aquaculture.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 3: High-density indoor aquaculture raceway at RIA 1 Hai Phong instrumented with submerged water-quality probes and an overhead camera rig." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/smart-aquaculture/stage2-ria1-field-team.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 4: Research team conducting real-time data ingestion checks and telemetry validation on the VinUniversity GREEN-X monitoring station." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/smart-aquaculture/stage2-ria1-circuit-inspection.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 5: Technical inspection of power management circuitry, signal conditioning boards, and IP-rated marine enclosure cable glands." %}
    </div>
</div>

---

### Phase 3: Offshore Floating Sea-Cage Deployment at Nha Phu Bay (Nha Trang)

The ultimate test of marine resilience took place in the open waters of Nha Phu Bay, Khanh Hoa province. Here, commercial fish cages float kilometers offshore, subjected to tidal currents, heavy wave swells, salt spray, and tropical sunlight.

To ensure total self-sufficiency, the team engineered a dual-solar-powered station equipped with high-capacity lithium iron phosphate (LiFePO4) battery buffering and high-gain 4G/LTE cellular communications. The installation operated autonomously on the floating wooden platform, continuously beaming environmental parameters to mainland servers.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/smart-aquaculture/stage3-nhaphu-floating-cages.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 6: Autonomous dual-solar telemetry station mounted atop offshore floating sea cages in Nha Phu Bay, Nha Trang." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/smart-aquaculture/stage3-nhaphu-team-deployment.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 7: Field engineering team on the floating raft platform in Nha Phu Bay celebrating successful offshore commissioning." %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <video controls playsinline preload="metadata" poster="{{ '/assets/img/projects/smart-aquaculture/setup-nha-phu-bay-poster.jpg' | relative_url }}" class="img-fluid rounded z-depth-1" style="width: 100%;">
            <source src="https://hoang-pm-7604119.github.io/myself/videos/projects/smart-aquaculture/setup-nha-phu-bay.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p class="caption text-center mt-2"><strong>Video 1: System Setup at Nha Phu Bay</strong> &mdash; Aerial drone survey of the autonomous solar-powered telemetry station deployed on offshore floating aquaculture cages in Nha Phu Bay.</p>
    </div>
</div>

---

### Phase 4: Circular Outdoor Tanks & Bioacoustics at RIA 3 (Nha Trang)

In the latest operational stage, the infrastructure was installed at the Research Institute for Aquaculture No. 3 (RIA 3) in Nha Trang across large outdoor circular aquaculture pools.

This installation unified environmental telemetry with underwater bioacoustic monitoring. An industrial embedded processing unit and dedicated multi-channel audio interface were integrated into the outdoor weatherproof enclosure, enabling real-time hydrophone signal capture to study fish feeding sounds and swimming dynamics in correlation with water quality fluctuations.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/smart-aquaculture/stage4-ria3-system-assembly.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 8: Assembly of the outdoor telemetry enclosure, integrating 4G LTE communications, hydrophone audio interface, and embedded processor." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/smart-aquaculture/stage4-ria3-tank-overview.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 9: Elevated view of RIA 3 outdoor circular aquaculture tanks under continuous surveillance by the installed solar AIoT telemetry station." %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <video controls playsinline preload="metadata" poster="{{ '/assets/img/projects/smart-aquaculture/setup-ria3-nha-trang-poster.jpg' | relative_url }}" class="img-fluid rounded z-depth-1" style="width: 100%;">
            <source src="https://hoang-pm-7604119.github.io/myself/videos/projects/smart-aquaculture/setup-ria3-nha-trang.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p class="caption text-center mt-2"><strong>Video 2: System Setup at RIA 3 Nha Trang</strong> &mdash; Outdoor telemetry enclosure, power distribution, and monitoring cameras mounted under the aquaculture tank canopy at RIA 3.</p>
    </div>
</div>

---

## 4. Technical Contributions & Scientific Impact

- **Field-Tested Marine AIoT:** Demonstrated sustained, autonomous operation across indoor raceways, land-based circular tanks, and offshore floating sea cages under severe coastal conditions.
- **Biofouling-Aware Vision:** Engineered camera viewport cleanliness assessment algorithms, forming the empirical foundation of the published *CleanCam* benchmark dataset for underwater optical surveillance.
- **Multimodal Environmental & Acoustic Sensing:** Fused 5-parameter physicochemical time-series with submerged bioacoustic telemetry to enable closed-loop, data-driven feeding schedule optimization.
- **Semantic Data Transmission:** Designed task-oriented semantic communication to compress and transmit critical feature representations over bandwidth-constrained IoT links.

---

## 5. Project Leadership & Collaborating Institutions

### Principal Investigators (PI)
- **Dr. Van-Dinh Nguyen** (PI, VinUniversity)
- **Dr. Do Danh Cuong** (PI, VinUniversity)
- **Dinh Van Dung** (Co-PI)

### Engineering, Telemetry & Research Team
- **Minh-Hoang Pham** (Technical Lead & Research Assistant)
- **Phan Tuan Khoi** (Research Assistant)
- **Nguyen Thanh Trung** (Research Assistant)
- **Nguyen Hong Phuc** (Research Assistant & Student Grant Lead)
- **Trinh Cong Son** (Research Assistant)
- **Nguyen Xuan Quyen** (Research Assistant)

### Partner Institutions
- **Smart Green Transformation Center ([GREEN-X](https://vinuni.edu.vn/)), VinUniversity**
- **Vietnam National University of Agriculture ([VNUA](https://vnua.edu.vn/))**
- **Research Institute for Aquaculture No. 1 & No. 3 ([RIA 1](https://ria1.org/) & [RIA 3](https://ria3.vn/))**
