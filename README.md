
# EIT 3.0 — ALL-IN-ONE (Per-Channel + Fused INT4 Pack + Strided Controller + Bench)

## Included
- `eit3_kernels.py` — Hybrid attention (legacy) + **per-channel** kernel (`*_pc`)
- `eit3_pack_int4_kernel.py` / `eit3_pack_int4.py` — Fused INT4 pack along N
- `eit3_controller.py` — Controller with **strided sampling** (sample_stride)
- `eit3_engine.py` — EIT3Block wiring controller ↔ attention (per_channel enabled)
- `eit3_toy_decoder.py` — Mini-Llama style demo (per_channel + sample_stride=4)
- `run_phase3_simulation.py` — Streaming simulation
- `run_bench_phase2_vs_phase3.py` — Benchmark Phase2(Torch pack) vs Phase3(Triton pack)

## Quick Start
```bash
# 1) Simulation (per-channel + strided sampling)
python run_phase3_simulation.py

# 2) Toy decoder (stacked blocks)
python eit3_toy_decoder.py

# 3) Benchmarks: Phase 2 vs Phase 3
python run_bench_phase2_vs_phase3.py
```

> Requirements: CUDA GPU, PyTorch, Triton.
