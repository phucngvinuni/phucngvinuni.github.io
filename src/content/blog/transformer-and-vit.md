---
title: "From Words to Pixels: A Deep Dive into Transformers and Vision Transformers"
description: "A comprehensive technical guide to the Transformer architecture (Attention Is All You Need) and Vision Transformer (ViT), covering scaled dot-product attention, multi-head attention, positional encodings, patch embeddings, and how a single architecture unifies NLP and computer vision."
pubDate: "Feb 21 2026"
heroImage: "/transformer-vit-thumb.svg"
badge: "Deep Learning"
tags: ["Transformers", "ViT", "Deep Learning", "Computer Vision", "Attention Mechanism", "NLP"]
---

> **Papers covered:**
> - *Attention Is All You Need* — Vaswani et al., NeurIPS 2017 ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))
> - *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* — Dosovitskiy et al., ICLR 2021 ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929))

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: Why Attention?](#2-background-why-attention)
3. [The Transformer Architecture](#3-the-transformer-architecture)
   - [3.1 Scaled Dot-Product Attention](#31-scaled-dot-product-attention)
   - [3.2 Multi-Head Attention](#32-multi-head-attention)
   - [3.3 Positional Encoding](#33-positional-encoding)
   - [3.4 Position-wise Feed-Forward Networks](#34-position-wise-feed-forward-networks)
   - [3.5 Encoder Stack](#35-encoder-stack)
   - [3.6 Decoder Stack](#36-decoder-stack)
   - [3.7 Full Architecture](#37-full-architecture)
   - [3.8 Training Details](#38-training-details)
   - [3.9 Results](#39-results)
4. [Vision Transformer (ViT)](#4-vision-transformer-vit)
   - [4.1 The Core Idea: Images as Sequences](#41-the-core-idea-images-as-sequences)
   - [4.2 Patch Embedding](#42-patch-embedding)
   - [4.3 Class Token and Positional Embeddings](#43-class-token-and-positional-embeddings)
   - [4.4 Transformer Encoder for Images](#44-transformer-encoder-for-images)
   - [4.5 The Complete Forward Pass](#45-the-complete-forward-pass)
   - [4.6 Model Variants](#46-model-variants)
   - [4.7 Pre-training and Fine-tuning](#47-pre-training-and-fine-tuning)
   - [4.8 Key Findings and Ablations](#48-key-findings-and-ablations)
   - [4.9 Results](#49-results)
5. [Transformer vs. ViT: Key Differences](#5-transformer-vs-vit-key-differences)
6. [Complexity and Efficiency Analysis](#6-complexity-and-efficiency-analysis)
7. [Inductive Bias: The Central Tension](#7-inductive-bias-the-central-tension)
8. [What Attention Heads Actually Learn](#8-what-attention-heads-actually-learn)
9. [Conclusion](#9-conclusion)
10. [Quick Reference: Key Equations](#10-quick-reference-key-equations)

---

## 1. Introduction

For decades, deep learning practitioners faced a fundamental split: **recurrent networks** (RNNs, LSTMs) for sequences and **convolutional networks** (CNNs) for images. Each architecture imposed strong structural assumptions to make learning tractable — recurrence for temporal ordering, local convolutions for spatial locality.

Two landmark papers shattered these assumptions:

1. **"Attention Is All You Need" (2017):** Replaced recurrence entirely with attention mechanisms, creating the **Transformer** — a sequence-to-sequence model that parallelizes completely across tokens and achieves state-of-the-art machine translation.

2. **"An Image is Worth 16×16 Words" (2021):** Applied the *same unmodified Transformer encoder* to images by treating image patches as tokens, creating the **Vision Transformer (ViT)** — which, given sufficient data, matches or exceeds the best CNNs.

Together, these papers established the Transformer as a universal architecture across modalities. This blog provides an exhaustive technical tour of both.

---

## 2. Background: Why Attention?

### The Problem with RNNs

Before the Transformer, sequence modeling relied on RNNs and LSTMs:

```
 x₁     x₂     x₃     x₄
  │      │      │      │
  ▼      ▼      ▼      ▼
[h₁]──▶[h₂]──▶[h₃]──▶[h₄]──▶ output
```

**Fundamental issues:**
- **Sequential computation:** Step `t` depends on step `t-1`, making parallelization impossible during training.
- **Long-range dependencies:** Information from `x₁` must pass through every intermediate hidden state to reach `h₄`. The gradient signal degrades over long paths (vanishing/exploding gradients).
- **Maximum path length:** `O(n)` between any two positions in the sequence.

Attention mechanisms were originally added *on top of* RNNs as a workaround — attending over all encoder hidden states when generating each decoder output. The Transformer's insight was radical: **remove the RNN entirely and use only attention**.

### Why Attention Works

The key intuition: for any pair of positions in a sequence, attention computes their interaction in **O(1) sequential operations** with a **maximum path length of O(1)**. Compare this to the O(n) path length in RNNs.

```
Self-Attention: Any token can attend to any other directly

   "The  animal  didn't  cross  the  street  because  it   was  tired"
     │     │       │       │     │     │        │       │     │    │
     └─────┴───────┴───────┴─────┴─────┴────────┴───┐   │     │    │
                                                     └───┘<────┘    │
                             (single hop: "it" → "animal")         │
```

---

## 3. The Transformer Architecture

The Transformer follows an **encoder-decoder** structure, designed initially for sequence-to-sequence tasks like machine translation. Let us build it up from first principles.

### 3.1 Scaled Dot-Product Attention

This is the atomic building block of every Transformer.

**Inputs:**
- **Q** (Queries): What we are looking for — matrix of shape `(n_q, d_k)`
- **K** (Keys): What each position offers — matrix of shape `(n_k, d_k)`
- **V** (Values): The actual content to retrieve — matrix of shape `(n_k, d_v)`

**The mechanism:**

```
               softmax
                  │
           ┌──────┴──────┐
           │  Q·Kᵀ/√d_k  │   ← similarity scores, scaled
           └──────┬──────┘
                  │
     Q ──────────►├◄──────── K (transposed)
                  │
                  └──────►× V ──────► Output
```

**Formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

**Step-by-step breakdown:**

| Step | Operation | Shape | Purpose |
|------|-----------|-------|---------|
| 1 | `Q · Kᵀ` | `(n_q, n_k)` | Raw similarity scores |
| 2 | `÷ √d_k` | `(n_q, n_k)` | Scale to prevent vanishing gradients |
| 3 | `softmax(·)` | `(n_q, n_k)` | Normalize scores into a probability distribution |
| 4 | `(·) · V` | `(n_q, d_v)` | Weighted sum of values |

**Why scale by `√d_k`?**

If Q and K are random vectors with zero mean and unit variance, their dot product `q·k = Σᵢ qᵢkᵢ` has variance `d_k`. For large `d_k`, this pushes the softmax into saturated regions where gradients are nearly zero:

```
Softmax output (no scaling, d_k=512):   [≈0, ≈0, ≈0, ..., ≈1]   ← almost one-hot
Softmax output (scaled by 1/√d_k):     [0.1, 0.2, 0.15, ..., 0.1]  ← soft distribution
```

Dividing by `√d_k` brings the variance back to 1.0, keeping the softmax in its informative, gradient-rich region.

### 3.2 Multi-Head Attention

Rather than performing one attention function with full `d_model`-dimensional keys/queries/values, the paper proposes running `h` attention functions **in parallel** on different linear projections of the inputs.

```
                         Input (Q, K, V)
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
     [Head 1]            [Head 2]     ...    [Head h]
   Q·W₁Q, K·W₁K        Q·W₂Q, K·W₂K      Q·WₕQ, K·WₕK
   Attn(·,·,V·W₁V)     Attn(·,·,V·W₂V)  Attn(·,·,V·WₕV)
          │                   │                   │
          └───────────────────┼───────────────────┘
                         Concat(·)
                              │
                           × Wᴼ
                              │
                           Output
```

**Formula:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

**Projection matrices (learned parameters):**

| Matrix | Shape | Purpose |
|--------|-------|---------|
| `Wᵢᴼ` | `d_model × d_k` | Project input down to query subspace for head i |
| `Wᵢᴷ` | `d_model × d_k` | Project input down to key subspace for head i |
| `Wᵢᵛ` | `d_model × d_v` | Project input down to value subspace for head i |
| `Wᴼ` | `h·d_v × d_model` | Project concatenated heads back to model dim |

**Default hyperparameters:**
- `h = 8` heads
- `d_k = d_v = d_model / h = 512 / 8 = 64`

This keeps the total computation roughly equal to single-head attention with full dimensionality.

**Why multiple heads?** A single attention head must produce one weighted average over all positions. With multiple heads, each head can specialize: one head may focus on syntactic dependencies, another on coreference, another on proximity. The heads operate in different subspaces, then pool their information through `Wᴼ`.

### 3.3 Positional Encoding

Self-attention is **permutation-equivariant** — it treats the input as a set, not a sequence. Swapping two tokens produces the same output, just swapped. To make the model position-aware, positional information is injected by **adding** a fixed encoding to each token embedding before the first layer.

The paper uses **sinusoidal positional encodings:**

$$PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

Where `pos` is the position index and `i` is the dimension index.

**Visualization of the encoding matrix:**

```
Dimension →
      0       1       2       3     ...   511
   ┌─────────────────────────────────────────┐
 0 │ sin(0)  cos(0)  sin(0)  cos(0) ...     │   ← Position 0
 1 │ sin(1)  cos(1)  sin(ε)  cos(ε) ...     │   ← Position 1
 2 │ sin(2)  cos(2)  ...                    │   ← Position 2
 . │  ...                                   │
 . │                    High-freq → Low-freq│
```

- Small `i` → high-frequency sinusoids (rapid oscillation across positions)
- Large `i` → low-frequency sinusoids (slow, almost constant change)

**Key mathematical property:** For any fixed offset `k`, `PE(pos+k)` is a **linear function** of `PE(pos)`. This means the model can learn to attend by relative position using linear operations.

The paper tested learned positional embeddings too and found nearly identical results (25.7 vs 25.8 BLEU), but chose sinusoidal because it may generalize to longer sequences than seen during training.

### 3.4 Position-wise Feed-Forward Networks

After each attention sub-layer, a **two-layer fully connected network** is applied *independently and identically to each position*:

$$\text{FFN}(x) = \max(0,\; x W_1 + b_1) W_2 + b_2$$

```
Position i:   x_i → [Linear d_model→d_ff] → [ReLU] → [Linear d_ff→d_model] → output_i
Position j:   x_j → [Linear d_model→d_ff] → [ReLU] → [Linear d_ff→d_model] → output_j
(same weights W₁, W₂ shared across positions within a layer, different across layers)
```

**Hyperparameters:**
- Inner dimension: `d_ff = 2048` (4× the model dimension)
- This is equivalent to two 1×1 convolutions applied over the position dimension

The FFN provides **position-local transformation** — a complement to attention's global, cross-position mixing.

### 3.5 Encoder Stack

Each encoder layer wraps its sub-layers with **residual connections** and **layer normalization**:

```
Input x
   │
   ├──────────────────────────────┐
   │                              │
   ▼                              │
Multi-Head Self-Attention         │
   │                              │
   └──────────────────────────────► Add → LayerNorm → x'
                                          │
                              ┌───────────┤
                              │           │
                              ▼           │
                     Position-wise FFN   │
                              │           │
                              └───────────► Add → LayerNorm → output
```

**Formula for each sub-layer:**

$$\text{Output} = \text{LayerNorm}\bigl(x + \text{Sublayer}(x)\bigr)$$

The encoder stack has `N = 6` such layers. Each has the same structure but different learned weights. All layers output `d_model = 512` dimensional representations.

**Dimensions throughout:**

```
Input tokens → [Embedding] → R^(seq_len × 512)
                             ↓
                        + Positional Encoding (same shape)
                             ↓
               [Encoder Layer 1]   (512 → 512 per position)
                             ↓
               [Encoder Layer 2]
                             ↓
                           ...
                             ↓
               [Encoder Layer 6]   → encoder output: R^(seq_len × 512)
```

### 3.6 Decoder Stack

The decoder is more complex. Each of the `N = 6` decoder layers has **three sub-layers**:

```
Target input (shifted right)
   │
[Embedding + Positional Encoding]
   │
┌──┴──────────────────────────────────────────────────────────┐
│                    DECODER LAYER                            │
│                                                             │
│  (1) MASKED Multi-Head Self-Attention                       │
│      [Causal mask: position i can only see positions ≤ i]   │
│      Add & LayerNorm                                        │
│             │                                               │
│  (2) Cross-Attention (Encoder-Decoder Attention)            │
│      Q = from decoder step (1)                              │
│      K, V = from encoder output (memory)                    │
│      Add & LayerNorm                                        │
│             │                                               │
│  (3) Position-wise Feed-Forward Network                     │
│      Add & LayerNorm                                        │
└───────────────────────────── × N ──────────────────────────┘
   │
[Linear projection: d_model → vocab_size]
[Softmax]
   │
Output probabilities
```

**Masking in the decoder:** Future positions are masked by setting their pre-softmax attention scores to `-∞`, so `softmax(-∞) = 0`. This enforces **auto-regressive generation**: when generating token at position `t`, only tokens `0..t-1` are visible.

**The three attention modes in one Transformer:**

```
┌──────────────────────────────────────────────────────────┐
│ Location              │ Q from     │ K, V from           │
├───────────────────────┼────────────┼─────────────────────┤
│ Encoder self-attn     │ Encoder    │ Same encoder layer   │
│ Decoder self-attn     │ Decoder    │ Same decoder layer   │
│   (masked, causal)    │            │   (masked)           │
│ Encoder-decoder attn  │ Decoder    │ Encoder output       │
└───────────────────────┴────────────┴─────────────────────┘
```

### 3.7 Full Architecture

```
SOURCE SENTENCE                        TARGET SENTENCE (shifted right)
    │                                          │
[Embedding × √d_model]               [Embedding × √d_model]
    │                                          │
[Positional Encoding]                 [Positional Encoding]
    │                                          │
    ▼                                          ▼
╔══════════════════╗               ╔══════════════════════════╗
║   ENCODER        ║               ║   DECODER                ║
║ ┌──────────────┐ ║               ║ ┌──────────────────────┐ ║
║ │ Multi-Head   │ ║               ║ │ Masked Multi-Head    │ ║
║ │ Self-Attn    │ ║               ║ │ Self-Attention       │ ║
║ │ Add & Norm   │ ║               ║ │ Add & Norm           │ ║
║ └──────────────┘ ║               ║ └──────────────────────┘ ║
║ ┌──────────────┐ ║               ║ ┌──────────────────────┐ ║
║ │ Feed-Forward │ ║               ║ │ Cross-Attention      │◄══╗
║ │ Add & Norm   │ ║               ║ │ (K, V from Encoder)  │  ║
║ └──────────────┘ ║               ║ │ Add & Norm           │  ║
║      × N=6       ║               ║ └──────────────────────┘  ║
╚══════════════════╝               ║ ┌──────────────────────┐  ║
         ║                         ║ │ Feed-Forward         │  ║
         ║─────── memory ──────────╝ │ Add & Norm           │  ║
                                   ║ └──────────────────────┘  ║
                                   ║      × N=6                 ║
                                   ╚═══════════════════════════╝
                                             │
                                      [Linear Layer]
                                      [Softmax]
                                             │
                                   Output Probabilities
```

### 3.8 Training Details

| Hyperparameter | Base Model | Big Model |
|----------------|-----------|-----------|
| Layers N | 6 | 6 |
| d_model | 512 | 1024 |
| d_ff | 2048 | 4096 |
| Heads h | 8 | 16 |
| d_k = d_v | 64 | 64 |
| Dropout | 0.1 | 0.3 |
| Label Smoothing ε | 0.1 | 0.1 |
| Parameters | 65M | 213M |
| Training Steps | 100K | 300K |
| Training Time | ~12 hours | ~3.5 days |
| Hardware | 8× P100 GPUs | 8× P100 GPUs |

**Optimizer:** Adam with warmup learning rate schedule:

$$\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)$$

```
Learning Rate

  ▲
  │         ╱╲
  │        ╱  ╲
  │       ╱    ╲
  │      ╱      ╲___
  │     ╱            ╲___
  │    ╱                  ╲___
  │───╱                        ╲___
  └──────────────────────────────────▶ Steps
         ↑
    warmup_steps = 4000
    (linear increase, then ∝ step^{-0.5} decrease)
```

**Three regularization strategies:**
1. **Residual Dropout** (`p = 0.1`): Applied after each sub-layer output, and after embedding + positional encoding sums.
2. **Label Smoothing** (`ε = 0.1`): Softens the one-hot target distribution, hurting perplexity but improving BLEU.
3. **Checkpoint Averaging**: Last 5 (base) or 20 (big) checkpoints averaged at inference.

**Inference:** Beam search with beam size 4, length penalty `α = 0.6`.

### 3.9 Results

**Machine Translation — WMT 2014 (Table 2 from paper):**

```
EN-DE Translation (BLEU score):

  28.4 │████████████████████████████████████████ Transformer Big  ★ SOTA
  27.3 │████████████████████████████████████████ Transformer Base
  26.36│█████████████████████████████████████   ConvS2S Ensemble
  26.30│█████████████████████████████████████   GNMT+RL Ensemble
  26.03│████████████████████████████████████    MoE
  25.16│███████████████████████████████████     ConvS2S
  24.6 │██████████████████████████████████      GNMT+RL
  ─────┴─────────────────────────────────────────────────────
        BLEU
```

```
EN-FR Translation (BLEU score):

  41.8 │███████████████████████████████████████████ Transformer Big  ★ SOTA
  41.29│██████████████████████████████████████████  ConvS2S Ensemble
  41.16│█████████████████████████████████████████   GNMT+RL Ensemble
  40.56│████████████████████████████████████████    MoE
  40.46│████████████████████████████████████████    ConvS2S
  ─────┴────────────────────────────────────────────────────────────
Transformer (Big): 41.8 BLEU, less than 1/4 training cost of ConvS2S Ensemble
```

**Compute comparison (EN-DE, FLOPs):**

| Model | FLOPs | BLEU |
|-------|-------|------|
| Transformer (base) | 3.3×10¹⁸ | 27.3 |
| Transformer (big) | 2.3×10¹⁹ | 28.4 |
| ConvS2S Ensemble | 7.7×10¹⁹ | 26.36 |
| GNMT+RL Ensemble | 1.8×10²⁰ | 26.30 |

Transformer achieves better quality at **dramatically lower cost**.

---

## 4. Vision Transformer (ViT)

### 4.1 The Core Idea: Images as Sequences

The central question ViT answers: *Can we feed images to a Transformer encoder with minimal modification?*

The challenge is that images are 2D grids of pixels — a 224×224 image has 50,176 pixels. Self-attention scales quadratically with sequence length (`O(N²)`), making attention over individual pixels computationally prohibitive.

**The solution: treat non-overlapping image patches as tokens.**

```
Original Image (224×224×3)          Patch Grid (14×14 patches of 16×16 pixels)

┌────────────────────────┐          ┌──┬──┬──┬──┬──┬──┬──┬──┐
│                        │          │p1│p2│p3│p4│p5│p6│p7│p8│
│       A cat            │   ═══►   ├──┼──┼──┼──┼──┼──┼──┼──┤
│                        │          │  │  │  │  │  │  │  │  │
│                        │          ├──┼──┼──┼──┼──┼──┼──┼──┤
└────────────────────────┘          │  │  │  │  │  │  │  │  │
 H=224, W=224, C=3                  └──┴──┴──┴──┴──┴──┴──┴──┘
                                    14×14 = 196 patches
                                    Each patch: 16×16×3 = 768 values
```

With `P=16` and `H=W=224`: `N = (224/16)² = 14² = 196` patches. Each flattened patch has dimension `P²·C = 16²·3 = 768`. This is a manageable sequence of 196 tokens.

### 4.2 Patch Embedding

Each flattened patch is linearly projected to the model's hidden dimension `D`:

```
Patch xₚⁱ ∈ R^(P²·C)   →   [Linear E ∈ R^(P²·C × D)]   →   embedding ∈ R^D

Example (ViT-Base):
   Flattened patch: R^768 → Linear (768×768) → Token: R^768
```

**Formally:** For image `x ∈ R^(H×W×C)`:

$$\mathbf{x}_p = \bigl[\mathbf{x}^1_p; \mathbf{x}^2_p; \ldots; \mathbf{x}^N_p\bigr] \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

The patch embedding matrix `E ∈ R^(P²·C × D)` projects each patch to dimension `D`.

**Equivalence to convolution:** This linear projection is equivalent to a 2D convolution with kernel size `P×P`, stride `P`, and output channels `D`. In practice, ViT often implements patch embedding as a strided convolution.

### 4.3 Class Token and Positional Embeddings

**Class Token** — borrowed directly from BERT:

A learnable embedding vector `x_class ∈ R^D` is **prepended** to the sequence of `N` patch embeddings, giving a sequence of length `N+1`. After passing through all `L` Transformer layers, the output at position 0 (the class token's position) carries the global image representation.

```
Before encoder:   [CLS | patch₁ | patch₂ | ... | patchₙ]
                     ↑                              ↑
                  learnable                    projected patches
                  token (not                  from image
                  from image)

After L layers:   [CLS_output | patch₁_out | ... | patchₙ_out]
                       ↓
                  Classification head
```

**Positional Embeddings:**

Since Transformers are permutation-equivariant, spatial ordering must be injected. ViT adds **1D learnable positional embeddings** `E_pos ∈ R^((N+1)×D)` to the full token sequence (including the class token):

```
Input sequence z₀:

  E_pos[0]  E_pos[1]  E_pos[2]    ...   E_pos[N]
     +         +         +                 +
  x_class  patch₁E  patch₂E     ...   patchₙE

     =        =         =                 =
   z₀[0]    z₀[1]    z₀[2]      ...    z₀[N]
```

**Why not 2D positional embeddings?** Ablation studies show essentially no difference between 1D, 2D, and relative positional encodings at the patch scale. At 14×14 patches, the spatial resolution is small enough that even simple 1D embeddings learn the 2D structure automatically (the learned embeddings exhibit clear row/column structure).

```
Cosine similarity of learned position embeddings (14×14 grid):

Patch at row 3, col 5 has highest similarity to:
  ┌──────────────────────────────┐
  │ . . . . . . . . . . . . . . │
  │ . . . . . . . . . . . . . . │
  │ . . . . . . . . . . . . . . │
  │ . . . . . [●] . . . . . . . │  ← shows cross-shaped
  │ . . . . . . . . . . . . . . │    similarity pattern
  │ . . . . . . . . . . . . . . │    (same row + same col)
  └──────────────────────────────┘
```

### 4.4 Transformer Encoder for Images

ViT uses only the **encoder** part of the original Transformer — no decoder is needed for classification.

Each of the `L` encoder layers follows:

```
z_{ℓ-1}
   │
   ├──────────────────────────────────┐
   │                                  │
   ▼                                  │
[LayerNorm]                           │  (Pre-Norm: LN before sublayer)
   │                                  │
[Multi-Head Self-Attention]           │
   │                                  │
   └──────────────────────────────────► + (Residual)
                                          │
                                     z'_ℓ │
                                          ├──────────────────────────────┐
                                          │                              │
                                          ▼                              │
                                     [LayerNorm]                         │
                                          │                              │
                                     [MLP: Linear → GELU → Linear]      │
                                          │                              │
                                          └──────────────────────────────► + (Residual)
                                                                            │
                                                                          z_ℓ
```

**Key difference from original Transformer:** ViT uses **pre-norm** (LayerNorm before the sublayer), while the original Transformer uses **post-norm** (LayerNorm after residual addition). Pre-norm improves training stability for large models.

**MLP Block:** Each encoder layer's MLP consists of two linear transformations with GELU activation:

$$\text{MLP}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

With hidden dimension `= 4D` (e.g., for ViT-Base with D=768: MLP hidden dim = 3072).

### 4.5 The Complete Forward Pass

The full ViT forward pass in four equations:

$$\mathbf{z}_0 = \bigl[\mathbf{x}_\text{class};\; \mathbf{x}_p^1 \mathbf{E};\; \mathbf{x}_p^2 \mathbf{E};\; \ldots;\; \mathbf{x}_p^N \mathbf{E}\bigr] + \mathbf{E}_\text{pos} \tag{1}$$

$$\mathbf{z}'_\ell = \text{MSA}\!\bigl(\text{LN}(\mathbf{z}_{\ell-1})\bigr) + \mathbf{z}_{\ell-1}, \quad \ell = 1 \ldots L \tag{2}$$

$$\mathbf{z}_\ell = \text{MLP}\!\bigl(\text{LN}(\mathbf{z}'_\ell)\bigr) + \mathbf{z}'_\ell, \quad \ell = 1 \ldots L \tag{3}$$

$$\mathbf{y} = \text{LN}\!\bigl(\mathbf{z}_L^0\bigr) \tag{4}$$

**Complete ViT architecture diagram:**

```
Image x ∈ R^(H×W×C)
   │
   ▼ Split into N patches
[Patch₁, Patch₂, ..., PatchN]  each ∈ R^(P²·C)
   │
   ▼ Linear projection E
[Patch₁E, Patch₂E, ..., PatchNE]  each ∈ R^D
   │
   ▼ Prepend class token + add positional embeddings
z₀ = [x_class | Patch₁E | ... | PatchNE] + E_pos  ∈ R^((N+1)×D)
   │
   │        ┌─────────────────────────────────┐
   ▼        │      TRANSFORMER ENCODER        │
z'₁ ← MSA(LN(z₀)) + z₀                       │
z₁  ← MLP(LN(z'₁)) + z'₁                      │
   │                                           │
z'₂ ← MSA(LN(z₁)) + z₁        × L layers     │
z₂  ← MLP(LN(z'₂)) + z'₂                      │
   │        │                                  │
  ...       │                                  │
   │        └─────────────────────────────────┘
   ▼
z_L ∈ R^((N+1)×D)
   │
   ▼ Take class token output (position 0)
y = LN(z_L[0]) ∈ R^D
   │
   ▼ Classification head (pre-training: MLP with tanh; fine-tuning: linear)
logits ∈ R^K   (K = number of classes)
```

### 4.6 Model Variants

ViT comes in three sizes, following BERT's naming convention:

| Model | Layers L | Hidden dim D | MLP dim | Attention Heads | Parameters |
|-------|----------|--------------|---------|-----------------|------------|
| ViT-Base (B) | 12 | 768 | 3,072 | 12 | **86M** |
| ViT-Large (L) | 24 | 1,024 | 4,096 | 16 | **307M** |
| ViT-Huge (H) | 32 | 1,280 | 5,120 | 16 | **632M** |

**Naming convention:** `ViT-L/16` = Large model with 16×16 patches.
Smaller `P` → longer sequence N → more expensive but more fine-grained.

**Sequence length comparison:**

| Patch size P | Sequence length N (224×224 image) |
|---|---|
| 32×32 | `(224/32)² = 49` tokens |
| 16×16 | `(224/16)² = 196` tokens |
| 14×14 | `(224/14)² = 256` tokens |

### 4.7 Pre-training and Fine-tuning

#### Pre-training

ViT is pre-trained on large datasets using **image classification** with supervised labels (or masked patch prediction for self-supervised). Three datasets are used:

| Dataset | Classes | Images | Notes |
|---------|---------|--------|-------|
| ImageNet | 1,000 | 1.3M | "Small scale" |
| ImageNet-21k | 21,000 | 14M | "Medium scale" |
| JFT-300M | 18,291 | 303M | "Large scale" |

**Pre-training settings:**

| Setting | Value |
|---------|-------|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Batch size | 4,096 |
| Weight decay | 0.1 |
| LR schedule | Linear warmup (10k steps) + linear decay |
| Resolution | 224×224 |

#### Fine-tuning

```
Pre-trained ViT
   │
   Remove MLP head (tanh, hidden layer)
   │
   Attach new linear head: D × K  (zero-initialized)
   │
   Fine-tune on downstream dataset
   │
   Optional: Higher resolution (384×384)
              → 2D-interpolate positional embeddings to new grid
              → Keep patch size fixed → longer sequence
```

Higher resolution consistently improves accuracy. At 384×384 with patch size 16, the sequence grows from 196 to 576 tokens.

**Fine-tuning optimizer:** SGD with momentum 0.9, cosine learning rate decay, gradient clipping at global norm 1.

**Why zero-initialize the new head?** It ensures the model starts as an identity function for the pre-trained representations, causing no initialization shock.

### 4.8 Key Findings and Ablations

#### Finding 1: Data Scale is Critical

```
ImageNet top-1 accuracy (5-shot proxy):

                      Small data (ImageNet 1.3M)
                  ┌─────────────────────────────────┐
BiT ResNet        │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ← CNN wins
ViT-B             │░░░░░░░░░░░░░░░░░░░░░          │ ← Transformer lags
                  └─────────────────────────────────┘

                      Medium data (ImageNet-21k, 14M)
                  ┌─────────────────────────────────┐
BiT ResNet        │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
ViT-L             │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ ← Approaching parity
                  └─────────────────────────────────┘

                      Large data (JFT-300M)
                  ┌─────────────────────────────────┐
BiT ResNet        │░░░░░░░░░░░░░░░░░░░░░░░░░░░     │
ViT-H             │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ← Transformer wins!
                  └─────────────────────────────────┘
```

**Interpretation:** CNNs have built-in inductive biases (locality, translation equivariance) that help with small data. Transformers must learn these from data. At 300M images, data compensates entirely, and Transformers outperform CNNs.

#### Finding 2: Depth Matters Most

Ablation on ViT's architectural dimensions (5-shot ImageNet accuracy):

```
Dimension         Improvement when doubled
Depth (layers)    ████████████████████ (largest)
Patch size        ████████████ (second, reducing P)
MLP width         ████ (small)
Model width D     ██ (smallest)
```

**Conclusion:** Invest compute primarily in more layers, then in finer patch granularity.

#### Finding 3: Positional Embedding Type Does Not Matter (Much)

| Type | 5-shot ImageNet | Notes |
|------|----------------|-------|
| No position | 0.6138 | Large gap |
| 1D learned (default) | 0.6421 | — |
| 2D learned | 0.6400 | ≈ same |
| Relative | 0.6403 | ≈ same |

The large gap between no position vs. any position encoding confirms that ordering is important. The negligible differences between encoding types indicate that at 14×14 spatial resolution, spatial structure is easy to learn regardless of encoding.

#### Finding 4: Class Token ≈ Global Average Pooling

Both approaches reach the same accuracy when trained with their respective optimal learning rates. The class token approach uses `lr = 8×10⁻⁴`; GAP uses `lr = 3×10⁻⁴`. Initial experiments showing worse GAP performance were due to learning rate mismatch, not architectural inferiority.

#### Finding 5: Self-Supervised ViT Shows Promise

**Masked Patch Prediction** (analogous to BERT's masked token prediction):
- Corrupt 50% of patch embeddings (replace with `[mask]` token, random patch, or keep)
- Predict the **3-bit mean color** (512 classes) of each corrupted patch
- Result: ViT-B/16 → **79.9% top-1 on ImageNet** (vs. ~79.5% for training from scratch, vs. ~84% for supervised pre-training)

Still lagging supervised pre-training by ~4%, but demonstrates the self-supervised potential. Later works like MAE (Masked Autoencoders) and DINO built on this foundation.

### 4.9 Results

**State-of-the-Art Comparison (ImageNet top-1 accuracy):**

```
Model                              | ImageNet | Compute (TPUv3·days)
─────────────────────────────────────────────────────────────────────
Noisy Student (EffNet-L2, extra data) | 88.4%  | 12,300
BiT-L (ResNet 152×4, JFT)           | 87.54%  |  9,900
ViT-H/14 (JFT)           ★          | 88.55%  |  2,500   ← 4× less compute
ViT-L/16 (JFT)                      | 87.76%  |    680
ViT-L/16 (ImageNet-21k)             | 85.30%  |    230   ← only 21k data
```

**Transfer Learning Benchmark (fine-tuned at 384px):**

| Pre-train | Model | CIFAR-10 | CIFAR-100 | Oxford Pets | Oxford Flowers |
|-----------|-------|----------|-----------|-------------|----------------|
| ImageNet | ViT-B/16 | 98.13% | 87.13% | 93.81% | 89.49% |
| ImageNet-21k | ViT-L/16 | 99.16% | 93.44% | 94.73% | 99.61% |
| JFT-300M | ViT-H/14 | 99.50% | 94.55% | 97.56% | 99.68% |

**VTAB-1k (19 diverse tasks):**

| Model | Natural | Specialized | Structured | Mean |
|-------|---------|-------------|------------|------|
| ViT-H/14 (JFT) | ~88% | ~87% | ~63% | **77.63%** |
| BiT-L | ~87% | ~85% | ~62% | 76.29% |
| ViT-L/16 (ImageNet-21k) | ~83% | ~85% | ~55% | 72.72% |

---

## 5. Transformer vs. ViT: Key Differences

| Aspect | Transformer (NLP) | Vision Transformer (ViT) |
|--------|------------------|--------------------------|
| **Input** | Token embeddings (discrete vocab) | Patch embeddings (continuous image patches) |
| **Input dim** | Vocabulary size → d_model | P²·C → D |
| **Sequence component** | Encoder + Decoder | Encoder only |
| **Position encoding** | Fixed sinusoidal | Learned 1D (prepended CLS token) |
| **Output token** | All positions (language model) / decoder | Position 0 (CLS token) |
| **Classification head** | Not applicable (generation) | Linear D→K (fine-tuning) |
| **Norm placement** | Post-norm (after residual) | Pre-norm (before sublayer) |
| **Activation in FFN** | ReLU | GELU |
| **Pre-training task** | Language modeling / translation | Image classification (or masked patches) |
| **Key challenge** | Long sequences, grammar | Inductive bias, data requirement |
| **Scaling law** | More data + params = better | Requires 14M–300M images to compete with CNNs |

---

## 6. Complexity and Efficiency Analysis

### Self-Attention Complexity

For sequence of length `N` and dimension `d`:

| Layer Type | Complexity / Layer | Sequential Ops | Max Path Length |
|---|---|---|---|
| **Self-Attention** | `O(N² · d)` | `O(1)` | **O(1)** |
| Recurrent (RNN/LSTM) | `O(N · d²)` | `O(N)` | `O(N)` |
| Convolutional (kernel k) | `O(k · N · d²)` | `O(1)` | `O(logₖ N)` |
| Self-Attn (restricted r) | `O(r · N · d)` | `O(1)` | `O(N/r)` |

**Self-attention beats recurrent layers when `N < d`** — almost universal with modern tokenization. However, at `N = 196` patches vs. pixel-level `N = 50176`, ViT's `O(N²)` is manageable.

### ViT vs. CNN Compute

```
Compute to reach 85% ImageNet transfer accuracy (avg. 5 datasets):

ViT-L/16   ▓▓▓▓▓               (≈0.4k TPUv3-days)
ResNet-L    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (≈1.6k TPUv3-days)

ViT requires ~2–4× LESS compute to match ResNet performance
```

### Memory Complexity

- Self-attention attention matrix: `O(N²)` memory per layer per head
- For ViT with N=196: `196² = 38,416` values per head — very manageable
- For pixel-level attention N=50K: `50K² = 2.5B` values — infeasible without approximation

---

## 7. Inductive Bias: The Central Tension

The most conceptually important finding from ViT:

**CNNs** bake in three inductive biases that hold for natural images:
1. **Locality:** Features are local. A pixel's meaning depends on its neighbors.
2. **Translation equivariance:** A dog in the top-left looks the same as a dog in the bottom-right.
3. **2D spatial hierarchy:** Low-level edges→textures→objects, spatially nested.

**Transformers** bake in essentially none of these:
- Self-attention is **fully global** — any token can attend to any other
- The MLP is **locally applied** (per position) but uses no 2D structure
- The only manual 2D structure in ViT: (1) the initial patch extraction, (2) 2D-interpolated positional embeddings at fine-tuning

```
CNN                          ViT
 ┌───────────────────┐        ┌──────────────────────┐
 │     Pixel x       │        │     Patch token x_i   │
 │  ┌─┬─┐           │        │                       │
 │  │k│e│           │        │   Attends to ALL       │
 │  └─┴─┘ kernel    │        │   other patches        │
 │   local receptive│        │   from layer 1         │
 │   field          │        │                        │
 └───────────────────┘        └──────────────────────┘
   Strong locality bias          No locality bias
   ✓ Works with 1K images        ✗ Needs 14M–300M images
   ✗ Less flexible at scale      ✓ More powerful at scale
```

**The trade-off:** CNNs use domain knowledge as a shortcut; Transformers learn everything from data. When data is plentiful, learned representations beat hand-crafted biases.

---

## 8. What Attention Heads Actually Learn

### In the Original Transformer (NLP)

Visualization studies in the paper reveal specialized behaviors:

**Head 1 (Layer 5):** Resolves long-distance syntactic dependencies
```
"The  Law  will  not  make  men  free;  it  is  men  who  have  got  to make law free."
   └──────────────────────────────────────────────────────────────────────┘
   "its" → "Law" (coreference, 11 tokens apart)
```

**Head 2 (Layer 5):** Anaphora resolution — "it" attends sharply to "Law" and "application"

**Head 3 (different layers):** Phrase-level grouping, position-local attention, global discourse attention

Different heads clearly divide the representational labor, justifying the multi-head design.

### In ViT (Vision)

Three key findings from attention analysis:

**1. Global receptive field from layer 1:**
```
Layer 1 attention distance:

  Some heads:   ●●●●●● (very local, few pixels)
  Other heads:  ●●●●●●●●●●●●●●●●●●●●●●●●● (global, 200+ pixels)

Mean attention distance grows from layer 1 to layer L:

Layer 1:  ■■■■■■                 (short mean distance)
Layer 6:  ■■■■■■■■■■■■           (medium)
Layer 12: ■■■■■■■■■■■■■■■■■■■■  (long, global)
```

Unlike CNNs where receptive field grows slowly layer by layer, ViT's attention heads attend globally from the very first layer.

**2. Learned position embeddings encode 2D structure:**
The similarity structure of the 1D learned embeddings naturally organizes into rows and columns — the model discovers 2D structure without explicit 2D inductive bias.

**3. Patch embeddings resemble Gabor filters:**
The principal components of the learned patch embedding matrix `E` look like oriented edge detectors and color filters — similar to the first-layer features of CNNs, but learned from raw RGB patches rather than hand-coded.

---

## 9. Conclusion

### The Transformer's Legacy

"Attention Is All You Need" dismantled the sequential processing paradigm. By removing recurrence and using only attention mechanisms, the Transformer achieved:
- **O(1)** maximum path length between any two positions (vs. O(n) for RNNs)
- Complete parallelizability during training
- 28.4 BLEU on EN-DE — surpassing all prior models including ensembles

The architecture's impact extended far beyond machine translation. GPT, BERT, T5, and virtually every modern large language model is a Transformer variant.

### ViT's Legacy

"An Image is Worth 16×16 Words" established that:
- **No CNN components are necessary** for world-class image recognition, given sufficient data
- The Transformer, applied to image patches, can be a **universal image representation learner**
- **Large-scale pre-training** compensates for the lack of inductive biases
- ViT-H/14 achieves 88.55% on ImageNet at **4× less compute** than BiT-L ResNets

ViT spawned an entire ecosystem: DeiT (efficient training), Swin Transformer (hierarchical), BEiT (self-supervised), MAE (masked autoencoders), CLIP (vision-language), DINO, SAM, and more.

### The Unifying Principle

Both breakthroughs share a core philosophy:

> **Attention mechanisms, applied to sequences of discrete tokens, are sufficient to learn powerful representations across modalities — given enough data and compute.**

NLP tokens or image patches, the key abstraction is the same: split the input into a sequence, project each element to a common space, and let attention mechanisms discover the relationships that matter for the task at hand.

---

## 10. Quick Reference: Key Equations

### Transformer

| Name | Formula |
|------|---------|
| Scaled Dot-Product Attention | $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ |
| Multi-Head Attention | $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$ |
| Head projection | $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ |
| Position-wise FFN | $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$ |
| Sub-layer residual | $\text{LayerNorm}(x + \text{Sublayer}(x))$ |
| Sinusoidal PE (even) | $PE_{(\text{pos},2i)} = \sin\!\left(\text{pos}/10000^{2i/d_{\text{model}}}\right)$ |
| Sinusoidal PE (odd) | $PE_{(\text{pos},2i+1)} = \cos\!\left(\text{pos}/10000^{2i/d_{\text{model}}}\right)$ |
| Warmup LR | $\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$ |

### Vision Transformer

| Name | Formula |
|------|---------|
| Input sequence | $\mathbf{z}_0 = [\mathbf{x}_\text{cls};\; \mathbf{x}_p^1\mathbf{E};\;\ldots;\; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_\text{pos}$ |
| MSA block | $\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$ |
| MLP block | $\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$ |
| Output | $\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$ |
| Patch count | $N = HW/P^2$ |
| QKV projection | $[q, k, v] = \mathbf{z}\, U_{qkv},\quad U_{qkv} \in \mathbb{R}^{D \times 3D_h}$ |
| Attention weights | $A = \text{softmax}(qk^\top / \sqrt{D_h})$ |
| Single-head SA | $\text{SA}(\mathbf{z}) = Av$ |
| Multi-head SA | $\text{MSA}(\mathbf{z}) = [\text{SA}_1(\mathbf{z});\ldots;\text{SA}_k(\mathbf{z})] U_\text{msa}$ |

### Key Hyperparameter Summary

| Param | Transformer (base) | ViT-Base | ViT-Large | ViT-Huge |
|-------|-------------------|----------|-----------|----------|
| Layers | 6 | 12 | 24 | 32 |
| d_model / D | 512 | 768 | 1024 | 1280 |
| d_ff / MLP | 2048 | 3072 | 4096 | 5120 |
| Heads h | 8 | 12 | 16 | 16 |
| d_k = d_v | 64 | 64 | 64 | 80 |
| Parameters | 65M | 86M | 307M | 632M |

---

*This blog is based on the original papers:*
- *Vaswani et al., "Attention Is All You Need", NeurIPS 2017, arXiv:1706.03762*
- *Dosovitskiy et al., "An Image is Worth 16×16 Words", ICLR 2021, arXiv:2010.11929*
