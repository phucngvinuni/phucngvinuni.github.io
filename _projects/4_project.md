---
layout: page
title: Deep JSCC for Visible Light Communication
description: "Hardware-in-the-loop Deep JSCC transmitting semantic features via analog light intensity using a custom 8-bit R-2R DAC."
img: assets/img/projects/vlc/vlc_hardware_setup.jpg
importance: 4
category: work
related_publications: true
github: https://github.com/phucngvinuni/DAC-JSCC
github_stars: phucngvinuni/DAC-JSCC
---

A hardware-in-the-loop **Deep Joint Source-Channel Coding (Deep JSCC)** Visible Light Communication (VLC) system developed for **ELEC4010: Introduction to Microelectronics** at VinUniversity.

Unlike traditional digital communication that transmits raw binary bitstreams (0s and 1s) over discrete modulation schemes, this system compresses high-dimensional images into semantic latent representations ($k = 16$) using a Deep Convolutional Autoencoder and transmits them directly as **analog light intensities** via a custom-built 8-bit R-2R resistor ladder DAC, op-amp buffer, and BJT emitter-follower driver.

<div class="project-meta-box p-3 mb-4 rounded" style="background-color: var(--global-card-bg-color, #f8f9fa); border-left: 4px solid var(--global-theme-color, #0076df);">
  <div class="row text-center text-md-left">
    <div class="col-6 col-md-3 mb-2">
      <strong>Course:</strong><br>ELEC4010 Microelectronics
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Hardware Architecture:</strong><br>8-bit R-2R DAC + LM358 + 2N2222
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Compression Ratio:</strong><br>98% (~49&times; payload reduction)
    </div>
    <div class="col-6 col-md-3 mb-2">
      <strong>Transmission Speedup:</strong><br>~50&times; faster than UART
    </div>
  </div>
</div>

> **Project Links & Documentation:**
> - **GitHub Repository:** [phucngvinuni/DAC-JSCC](https://github.com/phucngvinuni/DAC-JSCC)
> - **Download Project Report:** [Full 18-Page Technical Report (PDF)]({{ '/assets/pdf/DAC_JSCC_Project_Report.pdf' | relative_url }})
> - **Team:** Nguyen Hong Phuc (Lead & Design) & Vo Viet Duc (Hardware Assembly)

---

## 🎥 Live Demonstration: End-to-End Optical Transmission

The video below demonstrates the complete end-to-end hardware-in-the-loop system in action. A transmitter PC encodes handwritten digits into 16 analog symbols, sends them through the 8-bit DAC and LED driver over free-space light to a TEMT6000 photodetector, and the receiver PC reconstructs the digit in real time:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <video controls playsinline preload="metadata" poster="{{ '/assets/img/projects/vlc/vlc_video_poster.jpg' | relative_url }}" class="img-fluid rounded z-depth-1" style="width: 100%;">
            <source src="{{ '/assets/video/vlc_deepjscc_demo.mp4' | relative_url }}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p class="caption text-center mt-2"><strong>Video 1: Hardware-in-the-Loop VLC Demo</strong> &mdash; Real-time transmission of handwritten digits from the transmitter PC to the receiver PC via analog light intensity modulation.</p>
    </div>
</div>

---

## 1. Motivation: Analog Deep JSCC vs. Brittle Digital Transmission

Traditional Visible Light Communication (VLC / Li-Fi) systems transmit digitized bits using binary modulation techniques such as On-Off Keying (OOK) or Pulse Amplitude Modulation (PAM). For image transmission, raw pixels are digitized into hundreds or thousands of bytes:
- Transmitting a single $28 \times 28$ grayscale image (e.g., MNIST) requires **784 bytes (6,272 bits)**.
- In low Signal-to-Noise Ratio (SNR) or turbulent optical conditions, a single corrupted bit can cause catastrophic bit errors or desynchronization (*the "cliff effect"*).

**The Deep JSCC Paradigm:**  
Rather than separating source compression and channel coding, our system uses a **Deep Convolutional Autoencoder** to jointly compress the source image and encode it into continuous semantic features ($k = 16$). Each feature is mapped directly to a continuous analog light intensity. When channel noise occurs, the reconstructed image experiences smooth, graceful degradation rather than total structural failure.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_system_architecture.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 1: End-to-End System Architecture &mdash; The transmitter neural encoder compresses the 784-pixel input image into a 16-dimensional latent vector, which is converted to analog light intensities by the 8-bit DAC and transmitted over the optical channel to the receiver neural decoder." %}
    </div>
</div>

### Benchmarking: Analog JSCC vs. Traditional Digital UART

| Metric | Traditional Digital Transmission (UART) | Proposed Analog Deep JSCC |
| :--- | :--- | :--- |
| **Data Representation** | Raw Pixels (Digital Bits) | Latent Semantic Features (Analog Voltages) |
| **Payload Size** | 784 bytes | **16 analog symbols** |
| **Channel Symbols** | 6,272 bits (1 byte/pixel) | **16 physical optical pulses** |
| **Noise Resilience** | Brittle (single bit-error corrupts pixel) | **Robust (graceful degradation)** |
| **Transmission Speed** | $1\times$ (Baseline) | **~50&times; faster** |
| **Bandwidth Savings** | Baseline ($0\%$) | **~98% compression** |

---

## 2. Microelectronics Circuit Design & Hardware Implementation

The hardware layer converts parallel 8-bit digital words from an Arduino into precise, continuous analog optical levels across three tightly coupled circuit stages:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_block_diagram.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 2: System-level block diagram showing the 8-bit R-2R ladder, op-amp buffer, BJT LED driver, and the optical channel." %}
    </div>
</div>

### Circuit Topology & Stage Breakdown

1. **8-bit R-2R Ladder Network:**  
   Constructed from precision $1\%$ metal film resistors ($R = 1\,\text{k}\Omega, 2R = 2\,\text{k}\Omega$). Driven by 8 digital GPIO pins ($0\text{--}5\,\text{V}$), the ladder produces 256 discrete analog voltage steps:
   $$\displaystyle V_{\text{DAC}} = V_{\text{ref}} \sum_{i=0}^{7} \frac{b_i}{2^{8-i}} = 5\,\text{V} \times \frac{D}{255}$$
   with an ideal step size (LSB) of $V_{\text{LSB}} = \frac{5\,\text{V}}{256} \approx 19.53\,\text{mV}$.
2. **LM358 Op-Amp Voltage Follower (Buffer):**  
   The raw output impedance of an R-2R ladder equals $R = 1\,\text{k}\Omega$. Connecting a load directly causes substantial voltage sag. Feeding $V_{\text{DAC}}$ into an LM358 op-amp configured as a unity-gain buffer ($V_{\text{out}} = V_{\text{in}}$) provides near-infinite input impedance ($> 1\,\text{M}\Omega$) to prevent loading, and low output impedance to drive the subsequent stage.
3. **2N2222 BJT Emitter-Follower LED Driver:**  
   Standard op-amps cannot supply high continuous drive currents. The buffered analog voltage drives the base of a 2N2222 NPN BJT in emitter-follower configuration. With current gain $\beta \approx 100\text{--}300$, the emitter supplies proportional current through a high-brightness Blue LED.
4. **Isolated Optical Channel & TEMT6000 Receiver:**  
   The transmitter LED and TEMT6000 phototransistor receiver are aligned within a light-shielded cylindrical optical tube to eliminate ambient room lighting interference. The sensor's analog output is sampled by the receiver's ADC.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_4bit_r2r_schematic.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 3: Conceptual 4-bit R-2R ladder DAC schematic with op-amp buffer." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_8bit_dac_schematic.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 4: Complete schematic of the 8-bit R-2R DAC with LM358 buffer and 2N2222 LED driver." %}
    </div>
</div>

### Simulation & Experimental Hardware Setup

Prior to breadboard fabrication, the ladder and buffer stages were simulated in PSpice to verify step uniformity and monotonicity:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_pspice_simulation.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 5: PSpice transient simulation results showing a 4-bit binary counting sequence producing a discrete, linear staircase output." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_hardware_setup.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 6: Experimental benchtop setup featuring the transmitter workstation, breadboard R-2R DAC and LED driver, optical isolation tube, and receiver workstation." %}
    </div>
</div>

---

## 3. Physical Channel Characterization & Differentiable Learning

Real optical hardware deviates substantially from textbook linear models. Accurately characterizing physical non-linearities and embedding them into the deep learning pipeline was essential for reliable communication.

### Physical Non-Linearity & Noise Analysis

To model the physical link, we collected an empirical dataset of **5,000 samples** (`final_merged_dataset.csv`) sweeping DAC input values from $0$ to $255$ and recording photodetector ADC responses:

1. **Non-Linear Transfer Characteristic:**  
   The Blue LED requires a threshold voltage ($V_{\text{th}} \approx 2.7\,\text{V}$) before conducting significant current. Below $D \approx 120$, the LED emits negligible light (dead zone). At high DAC values ($D > 180$), current saturation and phototransistor non-linearity cause response flattening.
2. **Heteroscedastic Hardware Noise:**  
   Analysis revealed that noise is **heteroscedastic** &mdash; the standard deviation $\sigma_{\text{noise}}$ is not constant but scales with optical intensity (shot noise and optical fluctuations dominate at higher illumination).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_transfer_function.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 7: Empirical transfer function of the optical channel (DAC input vs. Photodetector ADC output), showing the turn-on dead zone, active linear region, and saturation." %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_noise_analysis.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 8: Hardware noise analysis &mdash; Residual error distribution (Left) and heteroscedastic noise standard deviation scaling with signal amplitude (Right)." %}
    </div>
</div>

### Dynamic Waveform Validation

Oscilloscope measurements across multiple dynamic input trials confirmed high signal stability (consistent period $\approx 4.9\,\text{ms}$), discrete quantization steps, and monotonicity across full sweeps:

<div class="row mt-3">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_waveform_trial1.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 9a: Trial 1 &mdash; Quantization steps visibility." %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_waveform_trial2.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 9b: Trial 2 &mdash; Signal stability verification." %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_waveform_trial3.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 9c: Trial 3 &mdash; High-frequency consistency." %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_waveform_trial4.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 9d: Trial 4 &mdash; Strict monotonicity ramp check." %}
    </div>
</div>

### Channel Linearization & Differentiable Approximation

To enable gradient-based backpropagation through the physical channel:
1. **Channel Linearization (`createlinear.py`):**  
   We isolated the monotonic, high-sensitivity operating regime ($D \in [140, 170]$ corresponding to $V \in [2.74\,\text{V}, 3.33\,\text{V}]$) and constructed an optimal mapping table (`good_dac_map.json`).
2. **Differentiable Channel Layer (`real_channel.py` / `trainlinear.py`):**  
   During training, the PyTorch autoencoder uses a differentiable hardware-in-the-loop simulation layer that applies the empirical transfer function and heteroscedastic noise distribution. This allows the neural network to learn feature encodings that are inherently robust to optical non-linearities and physical noise.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_dac_distribution.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 10: Histogram distribution of DAC values generated by the neural encoder across the test set, demonstrating optimal utilization of the linear operating window." %}
    </div>
</div>

---

## 4. Experimental Results & Performance Evaluation

The end-to-end system was evaluated on the MNIST dataset using a latent dimension of $k = 16$.

### Semantic Reconstruction Quality

Despite compressing 784 image pixels into just 16 analog light pulses (**98% compression**), the receiver neural decoder faithfully reconstructed digit semantics, stroke continuity, and morphology under real physical channel noise:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_mnist_reconstruction.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 11: Experimental transmission results &mdash; Original MNIST digits (Top row) versus Reconstructed digits received through the physical analog VLC channel (Bottom row)." %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/vlc/vlc_reconstruction_results.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 12: High-resolution side-by-side reconstruction validation across multiple digit classes." %}
    </div>
</div>

### Power Consumption Analysis

To measure current consumption through the R-2R network without breaking the circuit, the **Voltage Drop method** was applied across individual branches:
- Total current consumption of the R-2R ladder remained minimal at approximately **$1.69\,\text{mA}$** during peak operation.
- The emitter follower efficiently sourced current directly from the $5\,\text{V}$ rail, preventing any loading or thermal drift on the precision ladder resistors.

---

## 5. Bill of Materials & Technical Specifications

| Component | Function / Specification | Quantity |
| :--- | :--- | :--- |
| **Arduino Uno / R4** | Microcontroller for 8-bit parallel GPIO control & 10-bit ADC sampling | 2 |
| **Precision Resistors (1kΩ, 1%)** | R-2R Ladder Series Resistors ($R$) | 9 |
| **Precision Resistors (2kΩ, 1%)** | R-2R Ladder Shunt Resistors ($2R$) | 8 |
| **LM358 Operational Amplifier** | Dual General-Purpose Op-Amp configured as Unity-Gain Buffer | 1 |
| **2N2222 NPN BJT** | Emitter-Follower Current Buffer for LED Drive | 1 |
| **High-Brightness Blue LED** | Optical Transmitter ($\lambda \approx 460\text{--}470\,\text{nm}$) | 1 |
| **TEMT6000 Sensor** | Silicon NPN Phototransistor Ambient Light Sensor | 1 |
| **Optical Enclosure** | Cylindrical light-shielded optical isolation chamber | 1 |

---

## 6. Credits & Project Information

- **Course:** ELEC4010 &mdash; Introduction to Microelectronics, Fall 2025
- **Institution:** College of Engineering & Computer Science, VinUniversity
- **Authors:**
  - **Nguyen Hong Phuc** (System Architecture, Circuit Design, PSpice Simulation, Deep JSCC Autoencoder, Hardware-in-the-Loop Integration)
  - **Vo Viet Duc** (Hardware Assembly)
- **Supervising Faculty:** VinUniversity Microelectronics Faculty
- **Source Code & Report:** [GitHub Repository](https://github.com/phucngvinuni/DAC-JSCC) &middot; [Download Report (PDF)]({{ '/assets/pdf/DAC_JSCC_Project_Report.pdf' | relative_url }})
