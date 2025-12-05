
import torch
import triton
from eit3_pack_int4_kernel import pack_int4_n_axis_kernel

@torch.no_grad()
def pack_int4_along_N(x: torch.Tensor, scale, zero_point):
    assert x.dim() == 4, "expect [B,H,N,D]"
    assert x.dtype in (torch.float16, torch.float32)
    B, H, N, D = x.shape
    out = torch.empty((B, H, (N + 1) // 2, D), device=x.device, dtype=torch.uint8)

    per_channel = isinstance(scale, torch.Tensor)
    if per_channel:
        assert isinstance(zero_point, torch.Tensor)
        assert scale.shape == (D,) and zero_point.shape == (D,)
        scale_t = scale.to(device=x.device, dtype=torch.float32).contiguous()
        zp_t    = zero_point.to(device=x.device, dtype=torch.float32).contiguous()
        stride_sd, stride_zd = scale_t.stride(0), zp_t.stride(0)
        SCALE, ZP = scale_t, zp_t
    else:
        SCALE, ZP = float(scale), float(zero_point)
        stride_sd, stride_zd = 0, 0

    BLOCK_N = 128
    BLOCK_D = 64
    grid = (B * H, triton.cdiv((N + 1) // 2, BLOCK_N), triton.cdiv(D, BLOCK_D))

    pack_int4_n_axis_kernel[grid](
        x, out, SCALE, ZP,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stride_sd, stride_zd,
        B, H, N, D,
        PER_CHANNEL=per_channel, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2
    )
    return out
