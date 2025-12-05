
import math
import torch

class EIT3ControllerState:
    def __init__(self, mask, ema_scores, cooldown, step=0):
        self.mask = mask
        self.ema = ema_scores
        self.cooldown = cooldown
        self.step = step

class EIT3Controller:
    def __init__(
        self,
        target_active_ratio: float = 0.10,
        recency_keep: int = 128,
        cooldown_steps: int = 2,
        tail_q: int = 32,
        ema_alpha: float = 0.80,
        max_freeze_frac: float | None = 0.30,
        min_active_tokens: int = 64,
        sample_stride: int | None = None,
        sample_cap: int | None = None,
        include_active_in_sample: bool = True,
    ):
        assert 0 < target_active_ratio <= 1.0
        assert 0 <= ema_alpha < 1.0
        self.target_active_ratio = float(target_active_ratio)
        self.recency_keep = int(recency_keep)
        self.cooldown_steps = int(cooldown_steps)
        self.tail_q = int(tail_q)
        self.ema_alpha = float(ema_alpha)
        self.max_freeze_frac = max_freeze_frac
        self.min_active_tokens = int(min_active_tokens)
        self.sample_stride = sample_stride
        self.sample_cap = sample_cap
        self.include_active_in_sample = include_active_in_sample

    @torch.no_grad()
    def initialize(self, B: int, N: int, device, initial_keep: int = 100):
        mask = torch.ones(B, N, device=device, dtype=torch.int32)
        keep = max(0, min(N, int(initial_keep)))
        if keep > 0: mask[:, :keep] = 0
        ema = torch.zeros(B, N, device=device, dtype=torch.float32)
        cooldown = torch.zeros(B, N, device=device, dtype=torch.int16)
        return EIT3ControllerState(mask, ema, cooldown, step=0)

    @torch.no_grad()
    def _compute_scores_full(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, L, D = q.shape
        tq = min(self.tail_q, L)
        if tq <= 0:
            return torch.zeros(B, L, device=q.device, dtype=torch.float32)
        q_tail = q[:, -tq:, :].float()
        k_all  = k.float()
        logits = torch.einsum('bqd,bkd->bqk', q_tail, k_all) / math.sqrt(D)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)
        scores = probs.mean(dim=1).to(torch.float32)
        return scores

    @torch.no_grad()
    def _compute_scores_strided(self, q: torch.Tensor, k: torch.Tensor, state: EIT3ControllerState) -> torch.Tensor:
        B, L, D = k.shape
        tq = min(self.tail_q, q.shape[1])
        if tq <= 0:
            return state.ema.clone()

        q_tail = q[:, -tq:, :].float()
        device = q.device
        scores = state.ema.clone()

        for b in range(B):
            idx_list = []

            if self.recency_keep > 0:
                r = min(L, self.recency_keep)
                idx_list.append(torch.arange(L - r, L, device=device))

            if self.include_active_in_sample:
                active_idx = torch.nonzero(state.mask[b] == 0, as_tuple=False).squeeze(-1)
                if active_idx.numel() > 0:
                    idx_list.append(active_idx)

            if self.sample_stride and self.sample_stride > 1:
                stride = self.sample_stride
                upto = max(0, L - (self.recency_keep if self.recency_keep > 0 else 0))
                idx_list.append(torch.arange(0, upto, stride, device=device))

            if not idx_list:
                return self._compute_scores_full(q, k)

            sel = torch.unique(torch.cat(idx_list, dim=0))
            if self.sample_cap is not None and sel.numel() > self.sample_cap:
                sel = sel[:self.sample_cap]

            k_sel = k[b:b+1, sel, :].float()
            logits = torch.einsum('bqd,bkd->bqk', q_tail[b:b+1], k_sel) / math.sqrt(D)
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            sc = probs.mean(dim=1).to(torch.float32).squeeze(0)
            scores[b].scatter_(0, sel, sc)

        return scores

    @torch.no_grad()
    def compute_scores(self, q: torch.Tensor, k: torch.Tensor, state: EIT3ControllerState) -> torch.Tensor:
        if self.sample_stride is None or self.sample_stride <= 1:
            return self._compute_scores_full(q, k)
        return self._compute_scores_strided(q, k, state)

    @torch.no_grad()
    def maybe_calibrate(self, k: torch.Tensor, v: torch.Tensor) -> tuple[float, int]:
        m = torch.maximum(k.abs().amax(), v.abs().amax()).item()
        if m == 0.0: return 1.0, 8
        return float(m / 7.0), 8

    @torch.no_grad()
    def update(self, q: torch.Tensor, k: torch.Tensor, state: EIT3ControllerState) -> EIT3ControllerState:
        B, N, _ = k.shape
        scores = self.compute_scores(q, k, state)

        ema = scores if state.step == 0 else (self.ema_alpha * state.ema + (1.0 - self.ema_alpha) * scores)

        if self.recency_keep > 0:
            keep = min(N, self.recency_keep)
            ema[:, -keep:] = float('inf')

        k_active = max(int(self.target_active_ratio * N), self.min_active_tokens)
        k_active = min(k_active, N)
        _, indices = torch.topk(ema, k_active, dim=1, largest=True)

        new_mask = torch.ones_like(state.mask)
        new_mask.scatter_(1, indices, 0)

        prev = state.mask
        going_frozen = (prev == 0) & (new_mask == 1)
        going_active = (prev == 1) & (new_mask == 0)

        allow_freeze = going_frozen & (state.cooldown >= self.cooldown_steps)
        tentative = going_frozen & ~allow_freeze

        updated = prev.clone()
        updated[allow_freeze] = 1
        updated[going_active] = 0

        cooldown = torch.where(
            tentative, torch.clamp(state.cooldown + 1, max=32767),
            torch.where(updated == 0, torch.zeros_like(state.cooldown), state.cooldown)
        )

        if self.max_freeze_frac is not None:
            cap = int(self.max_freeze_frac * N)
            for b in range(B):
                newly = torch.nonzero((prev[b] == 0) & (updated[b] == 1), as_tuple=False).squeeze(-1)
                if newly.numel() > cap:
                    imp = ema[b, newly]
                    order = torch.argsort(imp)
                    to_keep_active = newly[order[cap:]]
                    updated[b, to_keep_active] = 0

        if self.recency_keep > 0:
            updated[:, -min(N, self.recency_keep):] = 0

        state.mask = updated
        state.ema = ema
        state.cooldown = cooldown
        state.step += 1
        return state
