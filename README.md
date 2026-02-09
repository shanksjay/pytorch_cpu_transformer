# PyTorch CPU Transformer Profiling (Apple Silicon)

## Overview

This repository is a **hands-on exploration of PyTorch CPU execution on Apple Silicon**, with a specific focus on **understanding how individual PyTorch operations map to low‑level kernels, BLAS calls, and optimized attention implementations**.

The core goal is **not model training or accuracy**, but **execution tracing and performance analysis**:

- Which PyTorch ops dominate runtime on CPU?
- When does PyTorch dispatch to **Accelerate / BLAS**?
- When is **Scaled Dot‑Product Attention (SDPA)** fused into FlashAttention‑style kernels?
- How do architectural choices (RoPE, RMSNorm, SwiGLU) affect operator mix?
- What does `torch.compile()` actually change for a GEMM‑heavy Transformer?

The repo intentionally keeps the model small and readable so you can reason about **what runs where**.

---

## What’s in this repo

### `transformer_llama.py`
A **minimal LLaMA‑style decoder‑only Transformer**, implemented for **profiling clarity** rather than completeness.

Key architectural choices:
- Decoder‑only Transformer (LLaMA‑style)
- Rotary Position Embeddings (RoPE)
- RMSNorm (pre‑norm)
- SwiGLU MLP
- Grouped‑query compatible attention
- Causal self‑attention
- PyTorch SDPA fast‑path (CPU FlashAttention when available)

The script supports:
- CPU vs MPS execution (Apple Silicon)
- PyTorch profiler integration
- Optional `torch.compile()`

### `main.py`
A lightweight entry point / helper script used during experimentation.

---

## Primary intent of this repository

This repo exists to help answer questions like:

> *“Why is my Transformer fast or slow on CPU?”*

More concretely:

- Trace execution **from Python → ATen → C++ kernels → BLAS / SDPA**
- Observe how **attention masking choices** affect kernel selection
- Verify when PyTorch uses:
  - `aten::mm` / `aten::addmm` (BLAS / Accelerate)
  - `aten::_scaled_dot_product_flash_attention_for_cpu`
- Compare **explicit attention vs SDPA**
- Measure how much time is actually spent outside GEMMs

The repo is deliberately friendly to:
- `torch.profiler`
- `lldb` / symbol breakpoints
- assembly‑level inspection of PyTorch operators

---

## Requirements

- macOS on Apple Silicon (M1 / M2 / M3)
- Python 3.10+
- PyTorch ≥ 2.1.0 (CPU SDPA FlashAttention support)

---

## Environment setup (recommended)

On Apple Silicon, explicitly setting CPU threads often improves consistency:

```bash
export OMP_NUM_THREADS=8
export TORCH_NUM_THREADS=8
```

Optional (helps GEMM selection):

```python
torch.set_float32_matmul_precision("high")
```

---

## Running CPU profiling

### Basic CPU profile

```bash
python transformer_llama.py --device cpu --profile
```

This runs a single forward pass under `torch.profiler` and prints a breakdown of **ATen operators**.

### Example output (truncated)

```
aten::mm                                   ~70%
aten::_scaled_dot_product_flash_attention  ~18%
aten::mul / aten::add / aten::silu          small remainder
```

This tells you immediately:
- GEMMs dominate runtime (expected)
- Attention is fused into the SDPA FlashAttention CPU kernel
- Elementwise ops are a small fraction of total time

---

## CPU vs MPS

You can switch devices explicitly:

```bash
# CPU (Accelerate / BLAS)
python transformer_llama.py --device cpu --profile

# MPS (Metal)
python transformer_llama.py --device mps --profile
```

Note:
- **CPU** uses Apple Accelerate (BLAS) + SDPA CPU kernels
- **MPS** uses Metal kernels (no BLAS involvement)

---

## torch.compile() experimentation

You can enable PyTorch compilation via:

```bash
python transformer_llama.py --device cpu --compile --profile
```

This repo intentionally demonstrates that:

- `torch.compile()` **does not speed up GEMMs**
- For GEMM‑ and SDPA‑dominated workloads, compile may:
  - provide small gains
  - provide no change
  - or slightly regress due to graph wrapper overhead

This makes the repo useful as a **realistic counter‑example** to “compile always helps”.

---

## Debugging & deep inspection

This codebase is designed to work well with:

### PyTorch Profiler
- `aten::mm`
- `aten::scaled_dot_product_attention`
- SDPA FlashAttention CPU kernels

### LLDB

Example:

```bash
lldb python -- transformer_llama.py --device cpu --profile
```

You can then:
- set breakpoints on SDPA kernels
- inspect `at::Tensor` layouts
- examine strides, shapes, and data pointers
- step through dispatcher and kernel selection logic

This makes the repo suitable for **learning PyTorch’s execution model from Python down to assembly**.

---
