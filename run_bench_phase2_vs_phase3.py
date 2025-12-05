
import torch, time, statistics as stats
from eit3_wrapper import EIT3Attention
from eit3_pack_int4 import pack_int4_along_N

@torch.no_grad()
def torch_pack_int4_sim(x, scale, zp):
    scale_val = scale if isinstance(scale, float) else scale.view(1,1,1,-1)
    zp_val = zp if isinstance(zp, float) else zp.view(1,1,1,-1)
    q = torch.clamp(torch.round(x.float() / scale_val) + zp_val, 0, 15).to(torch.uint8)
    q_even = q[:, :, 0::2, :]
    q_odd  = q[:, :, 1::2, :]
    if q_odd.shape[2] != q_even.shape[2]:
        pad = torch.zeros_like(q_even[:, :, :1, :])
        q_odd = torch.cat([q_odd, pad], dim=2)
    return (q_even & 0x0F) | ((q_odd & 0x0F) << 4)

def cuda_timing(fn, iters=40, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        torch.cuda.synchronize()
        start.record(); fn(); end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {
        "mean": sum(times)/len(times),
        "p50": times[len(times)//2],
        "p90": times[int(0.9*len(times))],
        "p99": times[int(0.99*len(times))-1],
        "n": len(times)
    }

def run_case(B=2, H=4, N=2048, D_model=128, heads=4, per_channel=False):
    device="cuda"
    torch.manual_seed(0)
    head_dim = D_model // heads

    q = torch.randn(B, N, D_model, device=device, dtype=torch.float16)
    k = torch.randn(B, N, D_model, device=device, dtype=torch.float16)
    v = torch.randn(B, N, D_model, device=device, dtype=torch.float16)
    mask = torch.zeros(B, N, device=device, dtype=torch.int32); mask[:, 100:] = 1
    scale, zp = 0.05, 8.0

    k4 = k.view(B, N, heads, head_dim).transpose(1, 2).contiguous()

    # Pack-only
    t_phase2 = cuda_timing(lambda: torch_pack_int4_sim(k4, scale, zp))
    t_phase3 = cuda_timing(lambda: pack_int4_along_N(k4, scale, zp))

    # Forward end-to-end
    attn2 = EIT3Attention(D_model, heads, scale=scale, zero_point=zp, per_channel=per_channel).to(device).half()
    attn3 = EIT3Attention(D_model, heads, scale=scale, zero_point=zp, per_channel=per_channel).to(device).half()

    def fwd2():
        import eit3_pack_int4 as mod
        orig = mod.pack_int4_along_N
        mod.pack_int4_along_N = torch_pack_int4_sim
        _ = attn2(q, k, v, mask, q_start=0)
        mod.pack_int4_along_N = orig

    def fwd3():
        _ = attn3(q, k, v, mask, q_start=0)

    f2 = cuda_timing(fwd2)
    f3 = cuda_timing(fwd3)

    toks = B * N
    print(f"=== B={B}, H={heads}, N={N}, D_model={D_model}, per_channel={per_channel} ===")
    print(f"Pack-only  : P2 mean={t_phase2['mean']:.3f} ms | P3 mean={t_phase3['mean']:.3f} ms | speedup {t_phase2['mean']/t_phase3['mean']:.2f}x")
    print(f"Forward    : P2 mean={f2['mean']:.3f} ms | P3 mean={f3['mean']:.3f} ms | speedup {f2['mean']/f3['mean']:.2f}x")
    print(f"Throughput : P2 ~{toks/(f2['mean']/1000):.1f} tok/s | P3 ~{toks/(f3['mean']/1000):.1f} tok/s")
    print("")

if __name__ == "__main__":
    if not torch.cuda.is_available(): 
        print("❌ Needs CUDA"); raise SystemExit
    for N in [1024, 2048, 4096]:
        run_case(B=2, H=4, N=N, D_model=128, heads=4, per_channel=False)
    run_case(B=2, H=4, N=2048, D_model=128, heads=4, per_channel=True)
