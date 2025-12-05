
import torch
import time
import sys
from eit3_engine import EIT3Block

def run_simulation():
    if not torch.cuda.is_available():
        print("❌ Needs CUDA")
        sys.exit(1)

    device = torch.device("cuda")
    torch.manual_seed(42)

    print("\n🧠 EIT 3.0 Engine: Per-Channel + Strided Sampling Test")
    print("=======================================================")

    D_MODEL = 128
    HEADS = 4
    MAX_STEPS = 12
    SEQ_PER_STEP = 32

    ctrl_cfg = {
        "target_active_ratio": 0.20,
        "recency_keep": 64,
        "cooldown_steps": 2,
        "tail_q": 32,
        "ema_alpha": 0.5,
        "max_freeze_frac": 0.30,
        "min_active_tokens": 64,
        "sample_stride": 4,  # check every 4 tokens
        "sample_cap": None,
        "include_active_in_sample": True,
    }

    model = EIT3Block(D_MODEL, HEADS, controller_config=ctrl_cfg, per_channel=True).to(device).half()

    kv_cache = None
    ctrl_state = None

    total_tokens = 0

    print(f"{'Step':<5} | {'Total Tokens':<12} | {'Active':<8} | {'Frozen':<8} | {'Ratio':<7} | {'Latency(ms)':<12} | {'Status'}")
    print("-" * 90)

    for step in range(MAX_STEPS):
        x = torch.randn(1, SEQ_PER_STEP, D_MODEL, device=device, dtype=torch.float16)

        torch.cuda.synchronize(); t0 = time.time()
        out, kv_cache, ctrl_state = model(x, kv_cache, ctrl_state)
        torch.cuda.synchronize(); dt = (time.time() - t0) * 1000.0

        total_tokens += SEQ_PER_STEP

        mask = ctrl_state.mask
        n_total = mask.numel()
        n_frozen = mask.sum().item()
        n_active = n_total - n_frozen
        ratio = (n_frozen / n_total) * 100.0

        status = "🟢 Init"
        if ratio > 10: status = "🟡 Optimizing"
        if ratio > 70: status = "🔴 HIGH COMPRESSION"

        print(f"{step+1:<5} | {total_tokens:<12} | {n_active:<8} | {n_frozen:<8} | {ratio:6.1f}% | {dt:11.2f} | {status}")

        if torch.isnan(out).any():
            print("❌ NaN Detected. Abort."); break

    print("-" * 90)
    print("\n✅ Simulation Complete.")

if __name__ == "__main__":
    run_simulation()
