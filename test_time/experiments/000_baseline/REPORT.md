# Experiment 000 — pretrained baseline

## Question

Can the released unconstrained GSDiff model generate a vector floorplan on Northflank?

## Setup

Original reference: `scripts/test_main.py`; node model `HeterHouseModel`, edge model `EdgeModel`.
Node checkpoint: `structure-1/model1000000.pt` (`99a6567f2c19bd849a026d1931cc56fa5742229a4ec50769bcff8b1e3314e96b`).
Edge checkpoint: `structure-2/model_stage2_best_061000.pt` (`6e9fc4fc8aef2f39c55371cfab6066658b2eed39ada9bb808be6705039da75d1`).

Seed 42, batch size 1, 1,000 original cosine-DDPM steps, no alignment/merging.
Northflank H100 80 GB; Python 3.10.13, PyTorch 2.1.0, CUDA 11.8.
The upstream requirements list PyTorch 2.0.1; this measured environment uses 2.1.0.
RPLAN images were not required. No physical scale was assigned in this initial generation.

## Measured result

| Quantity | Result |
|---|---:|
| Wall junctions | 33 |
| Predicted edges before bridge removal | 40 |
| Edges after original postprocessing | 39 |
| Extracted rooms | 8 |
| Generation time | 9.87 s |
| Peak allocated GPU memory | 0.435 GiB |

The PNG and vector graph were produced on 2026-10-28 and downloaded to Kraken.
The new adapter was checked against the saved raw prediction: nodes, coordinates and raw edges match exactly,
and its model parameters remain unchanged. This is an inference plumbing check, not a quality benchmark.

![Generated seed-42 floorplan](figures/floorplan.png)

[Measured adapter verification](verification.json).

## Reproduce

```bash
python -m test_time.audit --reference /workspace/results/smoke-20261028T110433-573119Z/raw_prediction.npz
```

Next: compare constraints and guided methods under one explicitly assumed physical scale.
