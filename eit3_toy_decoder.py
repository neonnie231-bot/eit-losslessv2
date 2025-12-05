
import torch
import torch.nn as nn
from eit3_engine import EIT3Block

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        dtype_in = x.dtype
        x_f = x.float()
        norm = x_f.pow(2).mean(-1, keepdim=True)
        x_normed = x_f * torch.rsqrt(norm + self.eps)
        return (self.weight.to(dtype_in) * x_normed.to(dtype_in)).to(dtype_in)

class MLP(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion, bias=False)
        self.fc2 = nn.Linear(d_model * expansion, d_model, bias=False)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class EIT3ToyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, controller_config, per_channel=True):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': RMSNorm(d_model),
                'eit_attn': EIT3Block(d_model, num_heads, controller_config, per_channel=per_channel),
                'norm2': RMSNorm(d_model),
                'mlp': MLP(d_model)
            }))

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None, past_controller_states=None):
        B, L_new = input_ids.shape
        x = self.embed(input_ids).half()

        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        if past_controller_states is None:
            past_controller_states = [None] * self.num_layers

        new_kv_caches = []
        new_ctrl_states = []

        for i, layer in enumerate(self.layers):
            residual = x
            x_norm = layer['norm1'](x)
            attn_out, kv_cache, ctrl_state = layer['eit_attn'](x_norm, past_key_values[i], past_controller_states[i])
            x = residual + attn_out

            new_kv_caches.append(kv_cache)
            new_ctrl_states.append(ctrl_state)

            residual = x
            x_norm = layer['norm2'](x)
            mlp_out = layer['mlp'](x_norm)
            x = residual + mlp_out

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches, new_ctrl_states

if __name__ == "__main__":
    import time, sys
    if not torch.cuda.is_available():
        print("❌ CUDA GPU required for EIT 3.0 Toy Decoder."); sys.exit(1)
    device = torch.device("cuda")

    VOCAB_SIZE = 1000
    D_MODEL = 256
    NUM_HEADS = 4
    NUM_LAYERS = 4

    ctrl_cfg = {
        "target_active_ratio": 0.20,
        "recency_keep": 32,
        "cooldown_steps": 2,
        "max_freeze_frac": 0.30,
        "sample_stride": 4,  # Enable strided sampling
    }

    print("\n🏗️ Building EIT 3.0 Toy Decoder (Per-Channel + Strided Sampling)...")
    model = EIT3ToyDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, ctrl_cfg, per_channel=True).to(device).half()

    batch_size = 1
    seq_per_step = 16
    max_steps = 6

    print(f"🚀 Running Inference Simulation ({NUM_LAYERS} layers deep)...")
    print(f"{'Step':<5} | {'Ctx Len':<8} | {'Latency':<8} | {'Layer 0 Frozen %':<16} | {'Layer 3 Frozen %':<16}")
    print("-" * 70)

    past_kv = None
    past_ctrl = None
    total_tokens = 0

    for step in range(max_steps):
        input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_per_step), device=device)

        torch.cuda.synchronize(); t0 = time.time()
        logits, past_kv, past_ctrl = model(input_ids, past_kv, past_ctrl)
        torch.cuda.synchronize(); dt = (time.time() - t0) * 1000

        total_tokens += seq_per_step

        mask_l0 = past_ctrl[0].mask
        ratio_l0 = (mask_l0.sum() / mask_l0.numel()) * 100

        mask_l3 = past_ctrl[3].mask
        ratio_l3 = (mask_l3.sum() / mask_l3.numel()) * 100

        print(f"{step+1:<5} | {total_tokens:<8} | {dt:6.2f}ms | {ratio_l0:5.1f}%           | {ratio_l3:5.1f}%")
    print("-" * 70)
    print("✅ Completed.")
