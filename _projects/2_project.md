---
layout: page
title: WinCart – Edge AI Smart Shopping Assistant & Indoor Navigation
description: "Top 20 Finalist at IoT Challenge 2025 (FPT Software & Silicon Labs). Real-time BLE indoor positioning with Triplet Metric Learning + KNN, quantized on-device LLM on Raspberry Pi, and A* pathfinding."
img: assets/img/projects/wincart/wincart_ui_map.jpg
importance: 3
category: work
related_publications: false
---

An autonomous, AI-powered smart shopping cart navigation and conversational assistance system developed as a **Top 20 Finalist in the IoT Challenge 2025**, organized by **FPT Software** and **Silicon Labs**.

WinCart transforms traditional supermarket shopping into a hands-free, intelligent experience by combining sub-meter BLE indoor positioning, on-cart edge computing with quantized Large Language Models (LLM), and real-time pathfinding across complex multi-aisle retail environments.

<div class="project-meta-box p-3 mb-4 rounded" style="background-color: var(--global-card-bg-color, #f8f9fa); border-left: 4px solid var(--global-theme-color, #0076df);">
  <div class="row text-center text-md-left">
    <div class="col-6 col-md-3 mb-2">
      <strong>Competition:</strong><br>IoT Challenge 2025 (Top 20)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Hardware Stack:</strong><br>Silicon Labs EFR32 + Raspberry Pi
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Positioning AI:</strong><br>Triplet Network + KNN (Sub-meter)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Conversational AI:</strong><br>Qwen2.5-0.5B via llama.cpp
    </div>
  </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_ui_map.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 1: WinCart interactive touch-screen interface &mdash; Showing real-time cart positioning, multi-item shopping list, and shortest-path A* route visualization on the supermarket floor map." %}
    </div>
</div>

---

## 1. The Retail Problem & Market Opportunity

Modern hypermarkets present significant customer friction:
- **Shopper Navigation Fatigue:** Customers waste an estimated 20–30% of their in-store shopping time searching for items across dense multi-aisle layouts.
- **GPS-Denied Indoor Environments:** Satellite GPS signals cannot penetrate commercial building structures, necessitating dedicated indoor positioning systems (IPS).
- **Existing Cart Complexities:** Solutions like Amazon Dash Cart rely on heavy, expensive multi-camera arrays and weight sensors (\$5,000+ per cart), making wide deployment economically unviable for Southeast Asian retail markets.

WinCart addresses this with an ultra-cost-effective architecture: low-cost Bluetooth Low Energy (BLE) infrastructure coupled with on-cart edge intelligence.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_user_flow.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 2: End-to-end customer journey &mdash; From voice search / list upload, optimal route generation, real-time aisle tracking, to frictionless cashierless checkout." %}
    </div>
</div>

---

## 2. System Architecture & Hardware Stack

The WinCart architecture is partitioned into three cooperative layers:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_system_architecture.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 3: High-level system architecture &mdash; Central cart computer, BLE positioning subsystem, and cloud backend." %}
    </div>
</div>

### Hardware Subsystems

1. **Central Cart Unit (Raspberry Pi):**  
   Acts as the cart's primary computing hub, executing the touchscreen GUI, 3-stage voice processing pipeline, local conversational LLM, and real-time A* pathfinding.
2. **Positioning Subsystem (Silicon Labs EFR32):**  
   A dedicated Silicon Labs EFR32 Wireless Gecko microcontroller mounted on the cart operates as a high-speed central BLE scanner, capturing periodic advertising packets from fixed aisle beacons.
3. **Fixed Shelf Beacons:**  
   Battery-efficient Silicon Labs BLE beacons placed along supermarket shelving emit synchronized periodic advertising packets containing beacon IDs and transmission power metadata.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_hardware_overview.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 4: Hardware modules &mdash; Silicon Labs EFR32 development board and Raspberry Pi central computing platform." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_ble_packet.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 5: Periodic advertising packet structure used for synchronized beacon telemetry." %}
    </div>
</div>

---

## 3. Sub-Meter Indoor Positioning via Triplet Metric Learning

Traditional RSSI trilateration fails catastrophically in indoor retail environments due to:
- **Severe Multipath Reflections:** Metal shelving and refrigeration units create complex constructive/destructive RF interference.
- **Human Body Shadowing:** Dynamic crowds absorb and scatter 2.4 GHz signals, causing non-linear RSSI fluctuations up to 15 dB.

### The Triplet Network + KNN Architecture

Rather than relying on noisy geometric distance formulas, WinCart employs **Deep Metric Learning**:

1. **Offline Training Phase:**  
   A deep neural network (Triplet Network) is trained on multi-beacon RSSI fingerprint vectors using **Triplet Margin Loss**:
   
$$
\mathcal{L}(a, p, n) = \max\Big(0, \,\mathcal{D}\big(f(a), f(p)\big) - \mathcal{D}\big(f(a), f(n)\big) + \alpha\Big)
$$

   Where a is an anchor fingerprint, p is a positive sample from the same spatial cell, n is a negative sample from a distant cell, and α is the enforcement margin.
2. **Embedding Space:**  
   The network learns to project noisy high-dimensional RSSI vectors into an invariant low-dimensional embedding space where spatial proximity is preserved regardless of RF multipath distortions.
3. **Online Inference (KNN):**  
   During runtime, incoming RSSI scans from the Silicon Labs receiver are projected into the embedding space, and a K-Nearest Neighbors (KNN) regressor estimates the cart's precise coordinates with **sub-meter accuracy**.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_triplet_knn_ml.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 6: Machine Learning positioning engine &mdash; Triplet Network offline feature representation learning and online KNN coordinate classification." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_central_device.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 7: Central positioning receiver execution flow and periodic scanning state machine on the Silicon Labs controller." %}
    </div>
</div>

---

## 4. 3-Stage Voice Pipeline & On-Cart Quantized LLM

To enable natural hands-free interaction, WinCart incorporates a fully local edge voice assistant pipeline:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_voice_pipeline.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 8: 3-Stage Voice Pipeline &mdash; Stage 1 Wake Word Detection, Stage 2 Silero Voice Activity Detection (VAD), and Stage 3 Speech-to-Text (Whisper STT)." %}
    </div>
</div>

1. **Stage 1 &mdash; Wake Word Detection:** Listens continuously with minimal CPU overhead for the activation phrase *"Hey WinCart"*.
2. **Stage 2 &mdash; Voice Activity Detection (Silero VAD):** Trims background supermarket noise and segments speech boundaries with millisecond precision.
3. **Stage 3 &mdash; Speech-to-Text (Whisper STT):** Transcribes Vietnamese and English queries into structured text commands.

### On-Device Conversational LLM (Qwen2.5 via llama.cpp)

Rather than paying recurring cloud API fees or suffering network drops in underground grocery stores, WinCart runs a localized **Qwen2.5-0.5B** model quantized in GGUF format via `llama.cpp` directly on the Raspberry Pi:
- **Zero Cloud Latency:** Instant response to complex conversational questions (e.g., *"Where can I find organic gluten-free pasta on sale today?"*).
- **Hybrid Intent Parser:** A hybrid Regex + LLM pipeline resolves exact item names to database product IDs and passes them to the A* navigation planner.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_local_llm.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 9: Local edge chatbot architecture integrating structured catalog queries with conversational recommendations." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_qwen_llamacpp.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 10: Quantized Qwen2.5 deployment on edge hardware via llama.cpp." %}
    </div>
</div>

---

## 5. Supermarket Map Editor & Fleet Scalability

To support rapid rollout across diverse retail branches, the team engineered a visual **Supermarket Config Editor**:
- Store operators can draw walls, aisles, and checkout zones directly in a web interface.
- Automatically compiles store layouts into graph navigation meshes for A* pathfinding.
- Supports Over-The-Air (OTA) firmware deployment for dynamic beacon calibration.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_map_editor.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 11: Supermarket Config Editor &mdash; Visual block customization and aisle coordinate grid mapping." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_rpi_call_graph.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 12: MainAppController call graph detailing the multithreaded application architecture on the Raspberry Pi." %}
    </div>
</div>

---

## 6. Competitive Advantage & Impact

| Feature | Computer Vision Carts (Amazon Dash) | Retail Mobile Apps | **WinCart** |
| :--- | :--- | :--- | :--- |
| **Unit Hardware Cost** | Extremely High (\$5,000+) | Low (BYOD) | **Low (~\$150/cart)** |
| **Positioning Accuracy** | Visual Odometry (Drift-prone) | Cellular/Wi-Fi (3–5m error) | **BLE Triplet Metric Learning (< 1m)** |
| **User Experience** | Heavy, restricted cart design | Small phone screen, battery drain | **Dedicated cart touchscreen + Voice AI** |
| **Edge AI Assistance** | Barcode/Vision checkout only | Cloud chatbot (network dependent) | **Local LLM (Zero cloud latency)** |
| **Shelf Infrastructure** | None | None | **Low-cost battery-operated BLE beacons** |

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/wincart/wincart_competitive_edge.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 13: Competitive matrix highlighting WinCart's strategic advantages in deployment cost, latency, and positioning accuracy." %}
    </div>
</div>

---

## 7. Project Credits & Recognition

- **Competition:** IoT Challenge 2025 (Top 20 Finalist)
- **Organizers:** FPT Software & Silicon Labs
- **Core Contributors:** WinCart Engineering Team (VinUniversity)
  - **Nguyen Hong Phuc** (BLE Positioning Architecture, Triplet Network Metric Learning, Edge LLM / Software Integration)
