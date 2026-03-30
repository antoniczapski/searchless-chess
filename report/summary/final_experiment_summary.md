# Final Experiment Summary — Searchless Chess

**Author:** Antoni Czapski  
**Date:** March 2026  
**Course:** Big Data in Healthcare (AGH UST)

---

## Objective

Reimplement and extend a neural chess position evaluator that predicts Stockfish evaluations without tree search, following the architecture from [mateuszgrzyb/searchless-chess](https://github.com/mateuszgrzyb/searchless-chess).

---

## Models Explored

### 1. MLP Baseline
- **Architecture:** 2-layer MLP (768 → 256 → 256 → 1), Tanh output
- **Purpose:** Establish a lower bound for position evaluation quality
- **Result:** ~900 ELO — proves the task is learnable but needs spatial structure

### 2. BDH (Iterative Value Refinement)
- **Architecture:** Board-embedding + K thinking steps with synaptic scratchpad
- **Key idea:** Shared-weight iterative refinement inspired by BDH-GPU (Pathway Technology, 2025)
- **Parameters:** 2.5M, 12 thinking steps
- **Result:** 1177 ELO — iterative refinement helps but linear attention limits expressiveness

### 3. ViT-S v2 (Vision Transformer Small)
- **Architecture:** 64-patch ViT with 5 transformer layers, dim=256, 4 heads
- **Key innovation:** Direct reimplementation of the reference architecture
- **Parameters:** 2.6M
- **Result:** 1621 ELO — strong baseline, close to reference with fewer GPU-hours

### 4. ViT-S + Attention Residuals (Best Model)
- **Architecture:** ViT-S with Windowed AttnRes (W=3), SwiGLU FFN, RMSNorm
- **Key innovation:** Depth-wise softmax retrieval replaces fixed residual connections
- **Parameters:** 2.66M
- **Result:** **1810 ELO** (53.0% puzzle accuracy) — only 7 ELO below reference (1817)

---

## Key Findings

1. **Attention Residuals improve learning efficiency.** ViT-S+AttnRes surpassed standard ViT-S by ~190 ELO with similar parameter count, suggesting depth-wise mixing helps the model learn richer representations.

2. **Score perspective is critical.** A bug that inverted ~50% of labels cost ~400 ELO. Data preprocessing validation is essential.

3. **Windowed AttnRes (W=3) is memory-efficient.** Full AttnRes caused OOM on A100 40GB. The windowed variant achieves similar performance with bounded memory.

4. **SwiGLU + RMSNorm are beneficial.** These modern components replaced GELU FFN and LayerNorm respectively, contributing to both training stability and final performance.

5. **The reference result is reproducible.** Achieving 1810 ELO (vs 1817 reference) with substantially less compute (<50h vs >200h) validates the approach.

---

## Training Infrastructure

- **GPU:** NVIDIA A100-SXM4-40GB (Ares HPC cluster, plgrid)
- **Framework:** PyTorch 2.5.1 + PyTorch Lightning
- **Data:** 50M positions from HuggingFace, uint8 encoding, in-memory loading
- **Optimizer:** AdamW with cosine LR schedule + linear warmup
- **Precision:** bf16 mixed precision

---

## Reproducibility

All experiments can be reproduced using the configs in `configs/`:
```bash
uv sync --group train
uv run python scripts/train.py --config configs/train_vit_attnres_a100.yaml
uv run python scripts/evaluate.py --config configs/train_vit_attnres_a100.yaml --checkpoint path/to/best.pt
```

For a quick local test:
```bash
uv run python scripts/train.py --config configs/smoke/vit_smoke.yaml --smoke-test
```
