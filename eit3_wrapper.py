
import torch
import torch.nn as nn
import triton
from eit3_kernels import eit3_hybrid_attention_kernel, eit3_hybrid_attention_kernel_pc
from eit3_pack_int4 import pack_int4_along_N

class EIT3Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, scale: float = 0.05, zero_point: float = 8.0, per_channel: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = float(scale)
        self.zero_point = float(zero_point)
        self.per_channel = bool(per_channel)

    def _calibrate_per_channel(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale_d = (t.abs().amax(dim=(0,1,2)).float() / 7.0).clamp(min=1e-8)  # [D]
        zp_d = torch.full_like(scale_d, 8.0)
        return scale_d, zp_d

    def forward(self, q, k, v, mask, q_start: int = 1_000_000_000):
        B, Sq, _ = q.shape
        _, Sk, _ = k.shape

        q = q.view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        if self.per_channel:
            sK, zK = self._calibrate_per_channel(k)
            sV, zV = self._calibrate_per_channel(v)
            k_packed = pack_int4_along_N(k, sK, zK)
            v_packed = pack_int4_along_N(v, sV, zV)
        else:
            k_packed = pack_int4_along_N(k, self.scale, self.zero_point)
            v_packed = pack_int4_along_N(v, self.scale, self.zero_point)

        out = torch.empty_like(q)
        B_, Hh, M, D = q.shape
        _, _, N, _ = k.shape

        BLOCK_M = 64
        BLOCK_N = 64
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), B_ * Hh)

        if self.per_channel:
            eit3_hybrid_attention_kernel_pc[grid](
                q, k, k_packed, v, v_packed, mask, out,
                sK, zK, sV, zV,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
                v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
                mask.stride(0), mask.stride(1),
                sK.stride(0), zK.stride(0), sV.stride(0), zV.stride(0),
                B_, Hh, M, N, D,
                int(q_start),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )
        else:
            eit3_hybrid_attention_kernel[grid](
                q, k, k_packed, v, v_packed, mask, out,
                float(self.scale), float(self.zero_point),
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
                v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
                mask.stride(0), mask.stride(1),
                B_, Hh, M, N, D,
                int(q_start),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )

        return out.transpose(1, 2).contiguous().view(B_, M, self.d_model)
