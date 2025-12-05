
import triton
import triton.language as tl

@triton.jit
def _quantize_and_pack_kernel(
    in_ptr, out_ptr, n_elements,
    scale, zero_point,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs_out = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_out = offs_out < ((n_elements + 1) // 2)

    offs_in0 = offs_out * 2
    offs_in1 = offs_out * 2 + 1

    x0 = tl.load(in_ptr + offs_in0, mask=offs_in0 < n_elements, other=0)
    x1 = tl.load(in_ptr + offs_in1, mask=offs_in1 < n_elements, other=0)

    q0 = tl.clamp(tl.round(x0 / scale) + zero_point, 0, 15).to(tl.uint8)
    q1 = tl.clamp(tl.round(x1 / scale) + zero_point, 0, 15).to(tl.uint8)

    packed = (q0 & 0x0F) | ((q1 & 0x0F) << 4)
    tl.store(out_ptr + offs_out, packed, mask=mask_out)


@triton.jit
def eit3_hybrid_attention_kernel(
    Q, K_fp16, K_int4, V_fp16, V_int4, EIT_Mask, Out,
    scale_int4, zero_point,

    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_kpb, stride_kph, stride_kpn, stride_kpd,
    stride_vpb, stride_vph, stride_vpn, stride_vpd,
    stride_mb, stride_mn,

    B, Hh, M, N, D,
    q_start,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // Hh
    h = pid_bh % Hh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_base = Q + b * stride_qb + h * stride_qh
    kf_base = K_fp16 + b * stride_kb + h * stride_kh
    vf_base = V_fp16 + b * stride_vb + h * stride_vh
    kp_base = K_int4 + b * stride_kpb + h * stride_kph
    vp_base = V_int4 + b * stride_vpb + h * stride_vph
    m_base  = EIT_Mask + b * stride_mb

    q = tl.load(
        q_base + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd),
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
        other=0
    ).to(tl.float16)

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i  = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i  = tl.zeros((BLOCK_M,), dtype=tl.float32)

    sm_scale = 1.0 / tl.sqrt(tl.float32(D))

    for start_n in range(0, N, BLOCK_N):
        cols = start_n + offs_n

        mask_vals = tl.load(m_base + cols * stride_mn, mask=cols < N, other=1)
        is_frozen = (mask_vals.to(tl.int32) != 0)

        k_active = tl.load(
            kf_base + (cols[:, None] * stride_kn + offs_d[None, :] * stride_kd),
            mask=(cols[:, None] < N) & (offs_d[None, :] < D),
            other=0
        ).to(tl.float16)
        v_active = tl.load(
            vf_base + (cols[:, None] * stride_vn + offs_d[None, :] * stride_vd),
            mask=(cols[:, None] < N) & (offs_d[None, :] < D),
            other=0
        ).to(tl.float16)

        packed_cols = (cols // 2)
        k_packed = tl.load(
            kp_base + (packed_cols[:, None] * stride_kpn + offs_d[None, :] * stride_kpd),
            mask=(cols[:, None] < N) & (offs_d[None, :] < D),
            other=0
        ).to(tl.uint8)
        v_packed = tl.load(
            vp_base + (packed_cols[:, None] * stride_vpn + offs_d[None, :] * stride_vpd),
            mask=(cols[:, None] < N) & (offs_d[None, :] < D),
            other=0
        ).to(tl.uint8)

        is_even = (cols % 2) == 0
        k_nib = tl.where(is_even[:, None], k_packed & 0x0F, (k_packed >> 4) & 0x0F).to(tl.float32)
        v_nib = tl.where(is_even[:, None], v_packed & 0x0F, (v_packed >> 4) & 0x0F).to(tl.float32)

        k_frozen = ((k_nib - zero_point) * scale_int4).to(tl.float16)
        v_frozen = ((v_nib - zero_point) * scale_int4).to(tl.float16)

        k = tl.where(is_frozen[:, None], k_frozen, k_active).to(tl.float16)
        v = tl.where(is_frozen[:, None], v_frozen, v_active).to(tl.float16)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32)
        qk = qk * sm_scale
        qk = tl.where(cols[None, :] < N, qk, -float("inf"))

        abs_q = q_start + offs_m
        causal = cols[None, :] <= abs_q[:, None]
        qk = tl.where(causal, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v).to(tl.float32)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out = (acc / l_i[:, None]).to(tl.float16)
    tl.store(
        Out + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd),
        out,
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < D)
    )


@triton.jit
def eit3_hybrid_attention_kernel_pc(
    Q, K_fp16, K_int4, V_fp16, V_int4, EIT_Mask, Out,
    SCALE_K_PTR, ZP_K_PTR, SCALE_V_PTR, ZP_V_PTR,

    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_kpb, stride_kph, stride_kpn, stride_kpd,
    stride_vpb, stride_vph, stride_vpn, stride_vpd,
    stride_mb, stride_mn,
    stride_skd, stride_zkd, stride_svd, stride_zvd,

    B, Hh, M, N, D,
    q_start,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // Hh
    h = pid_bh % Hh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_base = Q + b * stride_qb + h * stride_qh
    kf_base = K_fp16 + b * stride_kb + h * stride_kh
    vf_base = V_fp16 + b * stride_vb + h * stride_vh
    kp_base = K_int4 + b * stride_kpb + h * stride_kph
    vp_base = V_int4 + b * stride_vpb + h * stride_vph
    m_base  = EIT_Mask + b * stride_mb

    sK = tl.load(SCALE_K_PTR + offs_d * stride_skd, mask=offs_d < D, other=1.0).to(tl.float32)[None, :]
    zK = tl.load(ZP_K_PTR    + offs_d * stride_zkd, mask=offs_d < D, other=8.0).to(tl.float32)[None, :]
    sV = tl.load(SCALE_V_PTR + offs_d * stride_svd, mask=offs_d < D, other=1.0).to(tl.float32)[None, :]
    zV = tl.load(ZP_V_PTR    + offs_d * stride_zvd, mask=offs_d < D, other=8.0).to(tl.float32)[None, :]

    q = tl.load(
        q_base + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd),
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
        other=0
    ).to(tl.float16)

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i  = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i  = tl.zeros((BLOCK_M,), dtype=tl.float32)

    sm_scale = 1.0 / tl.sqrt(tl.float32(D))

    for start_n in range(0, N, BLOCK_N):
        cols = start_n + offs_n

        mask_vals = tl.load(m_base + cols * stride_mn, mask=cols < N, other=1)
        is_frozen = (mask_vals.to(tl.int32) != 0)

        k_active = tl.load(
            kf_base + (cols[:, None] * stride_kn + offs_d[None, :] * stride_kd),
            mask=(cols[:, None] < N) & (offs_d[None, :] < D),
            other=0
        ).to(tl.float16)
        v_active = tl.load(
            vf_base + (cols[:, None] * stride_vn + offs_d[None, :] * stride_vd),
            mask=(cols[:, None] < N) & (offs_d[None, :] < D),
            other=0
        ).to(tl.float16)

        packed_cols = (cols // 2)
        k_packed = tl.load(
            kp_base + (packed_cols[:, None] * stride_kpn + offs_d[None, :] * stride_kpd),
            mask=(cols[:, None] < N) & (offs_d[None, :] < D),
            other=0
        ).to(tl.uint8)
        v_packed = tl.load(
            vp_base + (packed_cols[:, None] * stride_vpn + offs_d[None, :] * stride_vpd),
            mask=(cols[:, None] < N) & (offs_d[None, :] < D),
            other=0
        ).to(tl.uint8)

        is_even = (cols % 2) == 0
        k_nib = tl.where(is_even[:, None], k_packed & 0x0F, (k_packed >> 4) & 0x0F).to(tl.float32)
        v_nib = tl.where(is_even[:, None], v_packed & 0x0F, (v_packed >> 4) & 0x0F).to(tl.float32)

        k_frozen = ((k_nib - zK) * sK).to(tl.float16)
        v_frozen = ((v_nib - zV) * sV).to(tl.float16)

        k = tl.where(is_frozen[:, None], k_frozen, k_active).to(tl.float16)
        v = tl.where(is_frozen[:, None], v_frozen, v_active).to(tl.float16)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32)
        qk = qk * sm_scale
        qk = tl.where(cols[None, :] < N, qk, -float("inf"))

        abs_q = q_start + offs_m
        causal = cols[None, :] <= abs_q[:, None]
        qk = tl.where(causal, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v).to(tl.float32)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out = (acc / l_i[:, None]).to(tl.float16)
    tl.store(
        Out + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd),
        out,
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < D)
    )
