
import torch
import torch.nn as nn
from eit3_wrapper import EIT3Attention
from eit3_controller import EIT3Controller, EIT3ControllerState

class EIT3Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, controller_config: dict | None = None, per_channel: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn = EIT3Attention(d_model, num_heads, per_channel=per_channel)

        if controller_config is None:
            controller_config = {}
        self.controller = EIT3Controller(**controller_config)

    @torch.no_grad()
    def forward(self, x, kv_cache=None, controller_state: EIT3ControllerState | None = None):
        assert x.dtype == torch.float16, "Input x should be FP16"
        B, L_new, D = x.shape
        device = x.device

        q_new = self.q_proj(x).to(dtype=torch.float16)
        k_new = self.k_proj(x).to(dtype=torch.float16)
        v_new = self.v_proj(x).to(dtype=torch.float16)

        past_len = 0 if kv_cache is None else kv_cache[0].shape[1]

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k_new], dim=1)
            v = torch.cat([v_past, v_new], dim=1)
        else:
            k, v = k_new, v_new

        _, N_total, _ = k.shape

        if controller_state is None:
            controller_state = self.controller.initialize(B, N_total, device=device)

        if controller_state.mask.shape[1] < N_total:
            diff = N_total - controller_state.mask.shape[1]
            controller_state.mask = torch.cat([controller_state.mask, torch.zeros(B, diff, device=device, dtype=torch.int32)], dim=1)
            controller_state.ema = torch.cat([controller_state.ema, torch.zeros(B, diff, device=device, dtype=torch.float32)], dim=1)
            controller_state.cooldown = torch.cat([controller_state.cooldown, torch.zeros(B, diff, device=device, dtype=torch.int16)], dim=1)

        controller_state = self.controller.update(q_new, k, controller_state)

        scale, zp = self.controller.maybe_calibrate(k, v)
        self.attn.scale = float(scale)
        self.attn.zero_point = float(zp)

        out_new = self.attn(q_new, k, v, controller_state.mask, q_start=int(past_len))
        out = self.o_proj(out_new).to(dtype=torch.float16)

        return out, (k, v), controller_state
