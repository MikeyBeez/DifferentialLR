# Chunked Linear Attention Research Framework

A research framework for investigating training dynamics of chunked linear attention, focusing on the interaction between within-chunk softmax attention and cross-chunk linear state mechanisms.

## Overview

This framework implements a hybrid attention mechanism:
- **Within each chunk**: Standard softmax attention O(C²) for fine-grained local modeling
- **Across chunks**: Linear state with gamma decay O(L/C) for long-range dependencies

The key research question: How do within-chunk and cross-chunk components interact during training, and can differential learning rates improve convergence?

## Architecture

TinyLlama-style model (~125M parameters):
- hidden_dim: 768
- num_layers: 12
- num_heads: 12
- head_dim: 64
- intermediate_dim: 3072

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Train a single model

```bash
python -m src.train --chunk-size 64 --gamma 0.95 --max-steps 10000
```

### Run chunk size sweep

```bash
python experiments/baseline_sweep.py --mode chunk --max-steps 10000
```

### Run gamma sweep

```bash
python experiments/baseline_sweep.py --mode gamma --max-steps 10000
```

### Run differential LR experiment

```bash
python experiments/differential_lr.py --mode ratio --max-steps 10000
```

### Analyze results

```bash
python experiments/analyze_results.py outputs/chunk_sweep/TIMESTAMP --type chunk_sweep
```

## Project Structure

```
LinearAttention/
├── src/
│   ├── config.py           # Hyperparameter dataclasses
│   ├── chunked_attention.py # Core attention mechanism
│   ├── model.py            # Full transformer
│   ├── data.py             # Dataset loading
│   ├── train.py            # Training loop
│   └── diagnostics.py      # Gradient/magnitude analysis
├── experiments/
│   ├── baseline_sweep.py   # Chunk size & gamma sweeps
│   ├── differential_lr.py  # Differential LR experiments
│   └── analyze_results.py  # Post-hoc analysis
├── configs/
│   └── default.yaml        # Default configuration
└── requirements.txt
```

## Key Hyperparameters

### Chunk Size
Controls the granularity of local vs global attention:
- Smaller chunks (16-32): More cross-chunk interactions, potentially slower
- Larger chunks (256-512): Approaches full attention, less compression

### Gamma (decay factor)
Controls persistence of cross-chunk state:
- Higher gamma (0.99): More long-range memory, but slower adaptation
- Lower gamma (0.9): Faster forgetting, more local focus

### LR Ratio
Differential learning rates between within-chunk and cross-chunk components:
- Ratio < 1: Cross-chunk learns slower (may help stability)
- Ratio > 1: Cross-chunk learns faster (may help long-range)

## Diagnostics

The framework tracks:
- **Gradient norms**: Per-component gradient magnitudes
- **Output magnitudes**: Within-chunk vs cross-chunk contribution sizes
- **Learning speed**: Parameter update magnitudes over time

## Research Outputs

After running experiments, the analysis script generates:
- Perplexity vs chunk_size curves
- Gradient magnitude plots
- Within-chunk vs cross-chunk output magnitude ratios
- Optimal configuration recommendations
