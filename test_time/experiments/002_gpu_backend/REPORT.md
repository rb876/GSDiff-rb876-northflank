# 002 — GPU guidance implementation audit

## Change

The default `--reward_backend gpu` runs intermediate node decoding, edge inference, room extraction,
constraint searches and candidate selection with PyTorch tensors on the model's device.
Use `--reward_backend cpu` to opt into reference CPU guidance. All six scored constraints and auxiliary C034 remain enabled;
C033/C035 still require a shared bed witness. No thresholds or scenario assumptions were relaxed.

SVDD retains compact proposal scores and selection traces, avoiding complete surrogate-graph exports.
Best-of-N can use the same tensor objective. Every final candidate is audited by the reference CPU
checker, whose results determine all reported validity/compliance metrics. Guidance scores are saved
separately. Run metadata distinguishes backends and records GPU/CPU evaluation counts.

## Scope and limitations

The room extractor checks planarity, removes bridges and traverses bounded faces. Verified,
non-overlapping room areas are summed without a polygon-union operation. Invalid structures skip
dimensional searches, receive unknown check statuses and retain the structural penalty.

Geometry uses the configured .25/.05 m grids, up to 20,000 bed positions per room and up to 32
directional circle witnesses. Search misses remain unknown. Bounds can prove some failures;
rectangle and circle witnesses establish passes. These searches differ from the CPU Shapely/polylabel
implementation and can have different unknown rates. Positive-temperature selection uses a PyTorch
random stream rather than the reference NumPy stream. Greedy selection remains the default.

The scenario remains two occupants, 2.5 m ceilings and **assumed** .025 m per drawing pixel.
These are conditional geometric checks, not verified real-world compliance.

## Offline evidence

Replayed the 12 saved pilot candidates on CPU tensors; no checkpoint loading or generation:

| Comparison | Result |
|---|---:|
| Room polygons and labels | 12/12 match |
| Structural validity | 12/12 match |
| All seven check statuses, structurally valid candidates | 10/10 match |
| Tensor passes absent from saved CPU results | 0 |

The two remaining candidates are structurally invalid: tensor checks are unknown because searches
are skipped. Historical CPU evaluations are retained unchanged. This small replay cannot establish
general equivalence or improved generation quality.

Full local comparison: `test_time/outputs/gpu_backend_check/audit.json` (ignored by Git).
Unit tests cover extraction, containment, shared placements, conservative unknowns, tensor selection,
batching, random streams and final CPU auditing: **146 passed, 2 CUDA tests skipped** locally.
CUDA validation and H100 throughput remain unmeasured.
Python dispatch and dynamic packing can still synchronize with the device; no speedup is claimed yet.

## Node commands

Run unit tests and check the saved pilot using CUDA, without generating new designs:

```bash
python -m pytest -q test_time/tests
python -m test_time.audit_gpu --experiment_dir test_time/outputs/pilot_check \
  --device cuda:0 --output test_time/outputs/gpu_backend_check/audit_cuda.json
```

When ready, generate four final designs with K=10, all 999 stochastic guided steps and temperature 0:

```bash
python -m test_time.run --method svdd_pm --trials 4 --seed 1000 \
  --batch_size 4 --inference_batch_size 10
```

The launcher defaults to `REWARD_BACKEND=gpu`; preview without launching with
`bash test_time/launch_eval.sh --dry-run`. Set `REWARD_BACKEND=cpu` to use reference CPU guidance.
Measured runtime calibration is still required before a matched-compute Best-of-N/SVDD comparison.
