
import triton
import triton.language as tl

@triton.jit
def pack_int4_n_axis_kernel(
    X, OUT,                      # X: [B,H,N,D], OUT: [B,H,(N+1)//2,D]
    SCALE, ZP,                   # scalar or pointer (per-channel D)
    # X strides
    stride_xb, stride_xh, stride_xn, stride_xd,
    # OUT strides
    stride_ob, stride_oh, stride_on, stride_od,
    # per-channel strides
    stride_sd, stride_zd,
    # sizes
    B, Hh, N, D,
    # meta
    PER_CHANNEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_pn = tl.program_id(1)
    pid_d  = tl.program_id(2)

    b = pid_bh // Hh
    h = pid_bh %  Hh

    offs_pn = pid_pn * BLOCK_N + tl.arange(0, BLOCK_N)  # packed index along ceil(N/2)
    offs_d  = pid_d  * BLOCK_D + tl.arange(0, BLOCK_D)

    Np = (N + 1) // 2
    mask_pn = offs_pn < Np
    mask_d  = offs_d  < D

    n_even = offs_pn * 2
    n_odd  = n_even + 1

    mask_even = n_even < N
    mask_odd  = n_odd  < N

    x_base   = X   + b * stride_xb + h * stride_xh
    out_base = OUT + b * stride_ob + h * stride_oh

    x_even = tl.load(
        x_base + (n_even[:, None] * stride_xn + offs_d[None, :] * stride_xd),
        mask=(mask_even[:, None] & mask_d[None, :]),
        other=0
    ).to(tl.float32)

    x_odd = tl.load(
        x_base + (n_odd[:, None] * stride_xn + offs_d[None, :] * stride_xd),
        mask=(mask_odd[:, None] & mask_d[None, :]),
        other=0
    ).to(tl.float32)

    if PER_CHANNEL:
        s = tl.load(SCALE + offs_d * stride_sd, mask=mask_d, other=1.0).to(tl.float32)[None, :]
        z = tl.load(ZP    + offs_d * stride_zd, mask=mask_d, other=8.0).to(tl.float32)[None, :]
    else:
        s = tl.full([1, 1], SCALE, dtype=tl.float32)
        z = tl.full([1, 1], ZP,    dtype=tl.float32)

    q_even = tl.clamp(tl.round(x_even / s) + z, 0, 15).to(tl.uint8)
    q_odd  = tl.clamp(tl.round(x_odd  / s) + z, 0, 15).to(tl.uint8)
    q_odd  = tl.where(mask_odd[:, None], q_odd, tl.zeros_like(q_odd))

    packed = (q_even & 0x0F) | ((q_odd & 0x0F) << 4)

    tl.store(
        out_base + (offs_pn[:, None] * stride_on + offs_d[None, :] * stride_od),
        packed,
        mask=(mask_pn[:, None] & mask_d[None, :])
    )
