"""
Project OMEGA: Sovereign Discovery Engine v2.0 (The Ascendant Build)

Neuro-Symbolic AI for Causal Inference. Transitions from pattern-matching
(Symbolic Regression) to true causal physics discovery.

Integrated Fixes in this Build:
  - Global Elite Pool: Cures Generation Amnesia during PPO updates.
  - Zero-Allocation Buffer: Solves Python GC memory choking in IronDome.
  - PPO Sign Reversal: Correctly maximizes positive advantage.
  - Cross-Layer Bloodline: Dynamic constant padding for perfect inheritance.
  - Batched GPU Rollouts: Massively parallelized MCTS simulations.
  - Tensor Boundary Patch: Fixes off-by-one sequence limits (+1 padded sequences).
  - Expected vs Predicted Graphing output at execution end.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sympy as sp
import logging
import matplotlib.pyplot as plt
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.float64)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [OMEGA] | %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. OPERATORS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Op(IntEnum):
    PAD    = 0
    VAR_X1 = 1; VAR_X2 = 2; VAR_X3 = 3; VAR_X4 = 4; VAR_X5 = 5
    CONST  = 6
    ADD = 7; SUB = 8; MUL = 9; DIV = 10
    SIN = 11; COS = 12; EXP = 13; SQ = 14; CUBE = 15


@dataclass
class OmegaConfig:
    action_dim:    int   = 16
    embedding_dim: int   = 64      # Transformer d_model
    generations:   int   = 50      # Tuned for harder challenges
    max_seq_len:   int   = 20      # Expanded to hold deeper operators

    # Search quality
    orthogonality_limit: float = 0.95
    parsimony_penalty:   float = 0.005
    alchemist_steps:     int   = 30
    use_smart_init:      bool  = True
    temperature:         float = 1.2

    # MCTS
    mcts_simulations: int   = 60
    c_puct:           float = 1.5
    rollout_batch:    int   = 128  # Number of simultaneous parallel rollouts

    # Greedy Pruning graft ratio
    graft_alpha: float = 0.15

    # The Shadow (neuro-mutation)
    shadow_patience:  int   = 8
    shadow_noise_std: float = 0.02

    # Causal Perturbation
    causality_threshold: float = 0.5
    fitness_threshold:   float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 2. CAUSAL PERTURBATION MODULE
# ─────────────────────────────────────────────────────────────────────────────

class CausalPerturbationModule(nn.Module):
    def __init__(self, num_nodes: int, epsilon: float = 1e-4):
        super().__init__()
        self.num_nodes = num_nodes
        self.epsilon   = epsilon

    def compute_causal_jacobian(
        self, fn, input_matrix: torch.Tensor
    ) -> List[float]:
        with torch.no_grad():
            baseline = fn(input_matrix)
            n_vars   = input_matrix.shape[1]
            norms    = []
            for i in range(n_vars):
                col_scale = input_matrix[:, i].abs().mean().item()
                eps_i     = max(self.epsilon, self.epsilon * col_scale)
                perturbed = input_matrix.clone()
                perturbed[:, i] = perturbed[:, i] + eps_i
                out_p   = fn(perturbed)
                dydxi   = (out_p - baseline).abs().mean().item() / eps_i
                norms.append(dydxi)
        return norms


# ─────────────────────────────────────────────────────────────────────────────
# 3. MCTS NODE
# ─────────────────────────────────────────────────────────────────────────────

class MCTSNode:
    def __init__(
        self,
        state:  List[int],
        parent: Optional['MCTSNode'] = None,
        prior:  float = 1.0
    ):
        self.state       = list(state)
        self.parent      = parent
        self.children:   Dict[int, 'MCTSNode'] = {}
        self.visits:     int   = 0
        self.value_sum:  float = 0.0
        self.prior:      float = prior
        self.constants:  Optional[torch.Tensor] = None
        self.is_informant: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb(self, c_puct: float = 1.5) -> float:
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value
                + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits))

    def best_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        return max(self.children.values(), key=lambda c: c.ucb(c_puct))

    def is_terminal(self, max_len: int) -> bool:
        return len(self.state) >= max_len

    def backpropagate(self, reward: float) -> None:
        self.visits    += 1
        self.value_sum += reward
        if self.parent:
            self.parent.backpropagate(reward)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PARETO FRONT
# ─────────────────────────────────────────────────────────────────────────────

class ParetoFront:
    def __init__(self, max_size: int = 20, eps: float = 0.01):
        self.front:    List[Dict] = []
        self.max_size: int   = max_size
        self.eps:      float = eps

    def _dominates(self, r2_a, len_a, r2_b, len_b) -> bool:
        return (r2_a >= r2_b - self.eps and len_a <= len_b and
                (r2_a > r2_b + self.eps or len_a < len_b))

    def update(
        self, seq: List[int], constants: torch.Tensor,
        r2: float, causality: float = 1.0
    ) -> None:
        length = sum(1 for x in seq if x != Op.PAD)
        self.front = [
            e for e in self.front
            if not self._dominates(r2, length, e['r2'], e['length'])
        ]
        if not any(self._dominates(e['r2'], e['length'], r2, length)
                   for e in self.front):
            c = constants.clone() if isinstance(constants, torch.Tensor) else constants
            self.front.append(
                {'seq': list(seq), 'constants': c,
                 'r2': r2, 'length': length, 'causality': causality}
            )
        self.front.sort(key=lambda e: -e['r2'])
        self.front = self.front[:self.max_size]

    def best(self) -> Optional[Dict]:
        return self.front[0] if self.front else None


# ─────────────────────────────────────────────────────────────────────────────
# 5. IRON DOME: ZERO-ALLOCATION PHYSICS KERNEL
# ─────────────────────────────────────────────────────────────────────────────

class IronDome:
    EPS = 1e-8
    _buf = [None] * 256  # Zero-allocation static buffer prevents GC choking

    @staticmethod
    def safe_div(a, b):
        return a / (b + IronDome.EPS * torch.sign(b) + IronDome.EPS)

    @staticmethod
    def execute(
        structure:    List[int],
        input_matrix: torch.Tensor,
        constants:    torch.Tensor,
        ones_buffer:  torch.Tensor
    ) -> torch.Tensor:
        buf = IronDome._buf
        ptr = -1
        c_idx = 0
        try:
            for act in structure:
                if 1 <= act <= 5:
                    ptr += 1
                    buf[ptr] = input_matrix[:, act-1:act]
                elif act == Op.CONST:
                    val = (constants[c_idx] if c_idx < len(constants)
                           else torch.tensor(1.0, dtype=torch.float64, device=ones_buffer.device))
                    ptr += 1
                    buf[ptr] = val * ones_buffer
                    c_idx += 1
                elif act == Op.ADD:
                    buf[ptr-1] = buf[ptr-1] + buf[ptr]; ptr -= 1
                elif act == Op.SUB:
                    buf[ptr-1] = buf[ptr-1] - buf[ptr]; ptr -= 1
                elif act == Op.MUL:
                    buf[ptr-1] = buf[ptr-1] * buf[ptr]; ptr -= 1
                elif act == Op.DIV:
                    buf[ptr-1] = IronDome.safe_div(buf[ptr-1], buf[ptr]); ptr -= 1
                elif act == Op.SIN:
                    buf[ptr] = torch.sin(buf[ptr])
                elif act == Op.COS:
                    buf[ptr] = torch.cos(buf[ptr])
                elif act == Op.EXP:
                    buf[ptr] = torch.exp(torch.clamp(buf[ptr], -10, 15))
                elif act == Op.SQ:
                    buf[ptr] = torch.pow(torch.clamp(buf[ptr], -1e3, 1e3), 2)
                elif act == Op.CUBE:
                    buf[ptr] = torch.pow(torch.clamp(buf[ptr], -1e3, 1e3), 3)

            return buf[ptr] if ptr >= 0 else torch.zeros_like(ones_buffer)
        except Exception:
            return torch.zeros_like(ones_buffer) + 1e9

    @staticmethod
    def execute_multi_layer(
        structures:      List[List[int]],
        input_matrix:    torch.Tensor,
        global_constants: torch.Tensor,
        ones_buffer:     torch.Tensor
    ) -> torch.Tensor:
        total = torch.zeros_like(ones_buffer)
        c_off = 0
        for struct in structures:
            nc = struct.count(Op.CONST)
            layer_c = global_constants[c_off: c_off + nc]
            total = total + IronDome.execute(struct, input_matrix, layer_c, ones_buffer)
            c_off += nc
        return total


# ─────────────────────────────────────────────────────────────────────────────
# 6. THE ALCHEMIST: OPTIMIZER + CAUSAL VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class TheAlchemist:
    def __init__(self, cfg: OmegaConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device

    def _r2(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        var = torch.var(target).item() + 1e-9
        mse = torch.mean((pred - target) ** 2).item()
        return max(-1.0, 1.0 - mse / var)

    def optimize(
        self,
        structure:   List[int],
        inputs:      torch.Tensor,
        target:      torch.Tensor,
        ones_buffer: torch.Tensor,
        warm_start:  Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:

        num_c = structure.count(Op.CONST)

        if num_c > 0:
            if warm_start is not None:
                inherited = warm_start.clone().detach()
                if len(inherited) >= num_c:
                    c_init = inherited[:num_c]
                else:
                    mag = torch.mean(torch.abs(target)).item() if self.cfg.use_smart_init else 1.0
                    init_val = mag if mag > 1e-5 else 1.0
                    padding = torch.full((num_c - len(inherited),), init_val, device=self.device, dtype=torch.float64)
                    c_init = torch.cat([inherited, padding])
                c = c_init.to(self.device).requires_grad_(True)

            elif self.cfg.use_smart_init:
                mag      = torch.mean(torch.abs(target)).item()
                init_val = mag if mag > 1e-5 else 1.0
                c = torch.full((num_c,), init_val, requires_grad=True,
                               device=self.device, dtype=torch.float64)
            else:
                c = torch.full((num_c,), 1.0, requires_grad=True,
                               device=self.device, dtype=torch.float64)

            opt = optim.LBFGS([c], lr=1.0, max_iter=self.cfg.alchemist_steps,
                              line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                pred = IronDome.execute(structure, inputs, c, ones_buffer)
                loss = torch.mean((pred - target) ** 2)
                loss.backward()
                return loss

            try:
                opt.step(closure)
            except Exception:
                pass
        else:
            c = torch.tensor([], device=self.device, dtype=torch.float64)

        with torch.no_grad():
            pred = IronDome.execute(structure, inputs, c, ones_buffer)
            r2   = self._r2(pred, target)

        return c.detach(), r2, pred.detach()

    def evaluate_terminal(
        self,
        seq:                 List[int],
        constants:           torch.Tensor,
        inputs:              torch.Tensor,
        target:              torch.Tensor,
        ones_buffer:         torch.Tensor,
        perturbation_engine: CausalPerturbationModule
    ) -> Tuple[float, float]:

        clean = [x for x in seq if 1 <= x < self.cfg.action_dim]
        if not clean:
            return 0.0, 0.0

        with torch.no_grad():
            pred_y  = IronDome.execute(clean, inputs, constants, ones_buffer)
            base_r2 = self._r2(pred_y, target)

        def kernel(inp_mat: torch.Tensor) -> torch.Tensor:
            return IronDome.execute(clean, inp_mat, constants, ones_buffer)

        try:
            norms = perturbation_engine.compute_causal_jacobian(kernel, inputs)
            n_used = sum(1 for j in norms if j > 1e-6)
            if n_used == 0:
                causality = 1.0
            else:
                causal_count = sum(1 for j in norms if j > 1e-4)
                causality = causal_count / n_used
        except Exception:
            causality = 0.1

        return base_r2, causality


# ─────────────────────────────────────────────────────────────────────────────
# 7. THE EXPLORER: TRANSFORMER ACTOR-CRITIC
# ─────────────────────────────────────────────────────────────────────────────

class ExplorerAgent(nn.Module):
    def __init__(self, cfg: OmegaConfig):
        super().__init__()
        d = cfg.embedding_dim
        self.embed   = nn.Embedding(cfg.action_dim, d)
        self.pos_enc = nn.Parameter(torch.zeros(cfg.max_seq_len + 2, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.actor_head  = nn.Linear(d, cfg.action_dim)
        self.critic_head = nn.Linear(d, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        emb  = self.embed(x) + self.pos_enc[:seq_len].unsqueeze(0)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        out    = self.transformer(emb, mask=mask)
        logits = self.actor_head(out)
        values = self.critic_head(out)
        return logits, values

    def mutate(self, noise_std: float = 0.02) -> None:
        with torch.no_grad():
            for p in self.parameters():
                p.add_(torch.randn_like(p) * noise_std)


# ─────────────────────────────────────────────────────────────────────────────
# 8. THE SOVEREIGN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class OmegaIndustrial:
    def __init__(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        cfg:    OmegaConfig,
        device: torch.device
    ):
        self.inputs          = inputs
        self.original_target = target.unsqueeze(1) if target.ndim == 1 else target
        self.cfg             = cfg
        self.device          = device
        self.ones = torch.ones(
            (len(target), 1), dtype=torch.float64, device=device
        )

        self.explorer = ExplorerAgent(cfg).to(device).double()
        self.opt      = optim.Adam(self.explorer.parameters(), lr=0.005)
        self.alchemist = TheAlchemist(cfg, device)

        n_vars = inputs.shape[1] if inputs.ndim > 1 else 1
        self.perturbation_engine = CausalPerturbationModule(num_nodes=n_vars).to(device)

        self.graveyard: Set[tuple] = set()
        self.pareto    = ParetoFront()
        self.discovered_layers: List[Dict] = []

    def check_orthogonality(
        self,
        candidate_pred: torch.Tensor,
        elite_preds:    List[torch.Tensor]
    ) -> bool:
        if not elite_preds:
            return True
        cand = candidate_pred.flatten()
        norm_c = torch.norm(cand)
        if norm_c < 1e-9:
            return False
        for e in elite_preds:
            sim = torch.dot(cand, e) / (norm_c * torch.norm(e) + 1e-9)
            if sim > self.cfg.orthogonality_limit:
                return False
        return True

    @staticmethod
    def _stack_depth_after(tokens: List[int]) -> int:
        depth = 0
        for tok in tokens:
            if 1 <= tok <= 6:
                depth += 1
            elif 7 <= tok <= 10:
                if depth < 2:
                    return -1
                depth -= 1
            elif 11 <= tok <= 15:
                if depth < 1:
                    return -1
        return depth

    def _valid_action_mask(self, stack_depth: int) -> torch.Tensor:
        n_vars = self.inputs.shape[1] if self.inputs.ndim > 1 else 1
        mask = torch.zeros(self.cfg.action_dim, dtype=torch.float64, device=self.device)

        for tok in range(1, min(5, n_vars) + 1):
            mask[tok] = 1.0

        mask[Op.CONST] = 1.0
        if stack_depth >= 1:
            for tok in range(11, 16):   # SIN, COS, EXP, SQ, CUBE — all unary
                mask[tok] = 1.0
        if stack_depth >= 2:
            for tok in range(7, 11):
                mask[tok] = 1.0
        return mask

    def _shadow_mutate(self) -> None:
        self.explorer.mutate(noise_std=self.cfg.shadow_noise_std)

    def _ppo_update(self, elite_batch: List[Tuple[List[int], float]]) -> None:
        if not elite_batch:
            return
        self.opt.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        n_valid = 0
        for seq, reward in elite_batch[:10]:
            if not seq:
                continue
            inp_t = torch.tensor([0] + seq, dtype=torch.long, device=self.device).unsqueeze(0)
            tgt_t = torch.tensor(seq + [0],  dtype=torch.long, device=self.device)
            pad_mask = (tgt_t != Op.PAD).double()

            logits, values = self.explorer(inp_t)
            logits = logits.squeeze(0)
            values = values.squeeze(0).squeeze(-1)

            actor_loss = F.cross_entropy(logits, tgt_t, reduction='none')
            actor_loss = (actor_loss * pad_mask).sum() / (pad_mask.sum() + 1e-9)
            advantage  = reward - values.detach().mean().item()
            critic_loss = F.mse_loss(values, torch.full_like(values, reward))

            total_loss = total_loss + (advantage * actor_loss + 0.5 * critic_loss)
            n_valid += 1

        if n_valid > 0:
            (total_loss / n_valid).backward()
            torch.nn.utils.clip_grad_norm_(self.explorer.parameters(), 1.0)
            self.opt.step()

    def _mcts_simulate(
        self,
        root:           MCTSNode,
        current_target: torch.Tensor,
        elite_preds:    List[torch.Tensor]
    ) -> Optional[Dict]:
        node = root
        while node.children and not node.is_terminal(self.cfg.max_seq_len):
            node = node.best_child(self.cfg.c_puct)

        if not node.is_terminal(self.cfg.max_seq_len) and node.visits > 0:
            with torch.no_grad():
                seq_t  = torch.tensor(
                    [0] + node.state, dtype=torch.long, device=self.device
                ).unsqueeze(0)
                logits, _ = self.explorer(seq_t)
                priors    = torch.softmax(
                    logits[0, -1, :] / self.cfg.temperature, dim=-1
                )
            top_k = min(5, self.cfg.action_dim - 1)
            node_depth = self._stack_depth_after(node.state)
            valid_expand = self._valid_action_mask(max(0, node_depth))
            for tok in torch.topk(priors, top_k).indices.tolist():
                if tok == Op.PAD or tok in node.children or not valid_expand[tok]:
                    continue
                child = MCTSNode(
                    node.state + [tok], parent=node, prior=priors[tok].item()
                )
                if node.constants is not None:
                    child.constants = node.constants.clone()
                node.children[tok] = child
            if node.children:
                node = node.best_child(self.cfg.c_puct)

        # ── TRUE GPU BATCHED ROLLOUT INFERENCE ──
        batch_size = self.cfg.rollout_batch

        # FIX: The off-by-one bounds. +1 added so PyTorch matrix can hold the `[0]` pad sequence prefix natively.
        seq_len_limit = self.cfg.max_seq_len + 1
        vocab_size = self.cfg.action_dim

        # Initialize the entire batch directly on the GPU (No Python Lists)
        batch_seq = torch.zeros((batch_size, seq_len_limit), dtype=torch.long, device=self.device)

        # Insert the known prefix (the MCTS node state padded with starting zero)
        prefix = [0] + node.state
        prefix_len = len(prefix)
        batch_seq[:, :prefix_len] = torch.tensor(prefix, dtype=torch.long, device=self.device)

        # Track active agents and their RPN stack depths entirely on the GPU
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        depths = torch.full((batch_size,), max(0, self._stack_depth_after(node.state)), dtype=torch.long, device=self.device)

        with torch.no_grad():
            for step in range(prefix_len, seq_len_limit):
                if not active_mask.any():
                    break

                # Single forward pass for all active sequences
                logits, _ = self.explorer(batch_seq[:, :step])
                probs = torch.softmax(logits[:, -1, :] / self.cfg.temperature, dim=-1)

                # Vectorized Mask Generation
                valid_mask = torch.zeros((batch_size, vocab_size), dtype=torch.float64, device=self.device)
                n_vars = self.inputs.shape[1] if self.inputs.ndim > 1 else 1
                max_var = min(5, n_vars)

                valid_mask[:, 1:max_var+1] = 1.0
                valid_mask[:, Op.CONST] = 1.0
                valid_mask[:, 11:16] = (depths >= 1).unsqueeze(-1).float()  # SIN..CUBE unary
                valid_mask[:, 7:11]  = (depths >= 2).unsqueeze(-1).float()  # ADD..DIV binary

                masked_probs = probs * valid_mask
                prob_sums = masked_probs.sum(dim=-1, keepdim=True)

                # Deactivate dead sequences
                dead_ends = (prob_sums.squeeze(-1) < 1e-9)
                active_mask = active_mask & ~dead_ends

                masked_probs = torch.where(
                    dead_ends.unsqueeze(-1),
                    torch.zeros_like(masked_probs),
                    masked_probs / (prob_sums + 1e-9)
                )

                safe_probs = masked_probs.clone()
                safe_probs[dead_ends, Op.PAD] = 1.0

                # Single multinomial call bounds across all 128 agents concurrently
                next_tokens = torch.multinomial(safe_probs.float(), 1).squeeze(-1)

                batch_seq[:, step] = torch.where(
                    active_mask,
                    next_tokens,
                    torch.tensor(Op.PAD, device=self.device)
                )

                # Vectorized depth updates
                is_var_or_const = (next_tokens >= 1) & (next_tokens <= 6)
                is_binary = (next_tokens >= 7) & (next_tokens <= 10)

                depths = torch.where(active_mask & is_var_or_const, depths + 1, depths)
                depths = torch.where(active_mask & is_binary, depths - 1, depths)

        # Bring it back to CPU only at the very end to evaluate in IronDome
        rollouts_cpu = batch_seq[:, 1:].cpu().tolist()

        # Collect up to 20 unique valid rollouts (dedup by sequence)
        seen_seqs: set = set()
        valid_rollouts = []
        for r in rollouts_cpu:
            cln = [x for x in r if x != Op.PAD and 1 <= x < self.cfg.action_dim]
            if not cln:
                continue
            key = tuple(cln)
            if key not in seen_seqs:
                seen_seqs.add(key)
                valid_rollouts.append(cln)
            if len(valid_rollouts) >= 20:
                break

        if not valid_rollouts:
            node.backpropagate(0.0)
            return None

        # Quick-score each candidate: single forward pass (no L-BFGS).
        # Using inherited constants (or ones) gives an unbiased structural signal
        # so sequences containing cos(x3), cube(x2), etc. can compete fairly
        # against trivially-short single-variable sequences.
        best_clean = None
        best_score = -float('inf')
        with torch.no_grad():
            t_var = torch.var(current_target).item() + 1e-9
            for candidate in valid_rollouts:
                nc = candidate.count(Op.CONST)
                if nc == 0:
                    qc = torch.tensor([], dtype=torch.float64, device=self.device)
                elif node.constants is not None and len(node.constants) >= nc:
                    qc = node.constants[:nc]
                else:
                    qc = torch.ones(nc, dtype=torch.float64, device=self.device)
                pred_q = IronDome.execute(candidate, self.inputs, qc, self.ones)
                mse_q  = torch.mean((pred_q - current_target) ** 2).item()
                r2_q   = max(-1.0, 1.0 - mse_q / t_var)
                score  = r2_q - len(candidate) * self.cfg.parsimony_penalty
                if score > best_score:
                    best_score = score
                    best_clean = candidate

        clean = best_clean

        if not clean:
            node.backpropagate(0.0)
            return None

        c, r2, pred = self.alchemist.optimize(
            clean, self.inputs, current_target, self.ones,
            warm_start=node.constants
        )
        node.constants = c.clone()

        if node.parent and len(node.parent.children) >= 2:
            siblings = [
                s for s in node.parent.children.values()
                if s is not node
                and s.constants is not None
                and len(s.constants) == len(c)
            ]
            if siblings:
                sibling = siblings[0]
                grafted = ((1 - self.cfg.graft_alpha) * c
                           + self.cfg.graft_alpha * sibling.constants)
                node.constants = grafted.clone()
                sibling.is_informant = True

        if not self.check_orthogonality(pred, elite_preds):
            node.backpropagate(0.0)
            return None

        fitness = r2 - len(clean) * self.cfg.parsimony_penalty
        node.backpropagate(max(0.0, r2))
        return {'seq': clean, 'constants': c, 'fitness': fitness, 'r2': r2, 'pred': pred}

    def run_layer_search(
        self, current_target: torch.Tensor, layer_idx: int
    ) -> Optional[Dict]:
        logger.info(
            f"\n{'='*60}\n"
            f"  MCTS LAYER {layer_idx} | "
            f"{self.cfg.generations} gens × {self.cfg.mcts_simulations} sims\n"
            f"{'='*60}"
        )
        elite_preds: List[torch.Tensor] = []
        best_layer:  Optional[Dict]     = None
        stagnation:  int = 0

        root = MCTSNode(state=[])
        if self.discovered_layers:
            root.constants = self.discovered_layers[-1]['constants'].clone()

        global_elites: Dict[tuple, Dict] = {}

        for gen in range(1, self.cfg.generations + 1):
            if stagnation >= self.cfg.shadow_patience:
                self._shadow_mutate()
                stagnation = 0
                logger.info(f"[SHADOW] Neuro-mutation injected | Gen {gen} Layer {layer_idx}")

            for _ in range(self.cfg.mcts_simulations):
                result = self._mcts_simulate(root, current_target, elite_preds)
                if result:
                    seq_tuple = tuple(result['seq'])
                    if seq_tuple not in global_elites or result['fitness'] > global_elites[seq_tuple]['fitness']:
                        global_elites[seq_tuple] = result

            if not global_elites:
                stagnation += 1
                logger.info(
                    f"  Gen {gen:3d}/{self.cfg.generations} | Layer {layer_idx} | "
                    f"No valid sims this generation"
                )
                continue

            sorted_elites = sorted(global_elites.values(), key=lambda x: x['fitness'], reverse=True)
            global_elites = {tuple(sim['seq']): sim for sim in sorted_elites[:50]}
            best_sim = sorted_elites[0]

            eq_str = translate_to_sympy(best_sim['seq'], best_sim['constants'])
            logger.info(
                f"  Gen {gen:3d}/{self.cfg.generations} | Layer {layer_idx} | "
                f"R2={best_sim['r2']:.4f} | Pool={len(global_elites)} | {eq_str}"
            )

            base_r2, causality = self.alchemist.evaluate_terminal(
                best_sim['seq'], best_sim['constants'],
                self.inputs, current_target, self.ones,
                self.perturbation_engine
            )
            final_reward = base_r2 * causality

            if final_reward > self.cfg.fitness_threshold:
                if causality >= self.cfg.causality_threshold:
                    self.pareto.update(
                        best_sim['seq'], best_sim['constants'], base_r2, causality
                    )
                else:
                    self.graveyard.add(tuple(best_sim['seq']))

            if best_layer is None or final_reward > best_layer['fitness']:
                best_layer = {
                    'struct':    best_sim['seq'],
                    'constants': best_sim['constants'],
                    'fitness':   final_reward,
                    'r2':        base_r2,
                    'pred':      best_sim['pred']
                }
                elite_preds.append(best_sim['pred'].flatten())
                if len(elite_preds) > 5:
                    elite_preds.pop(0)
                stagnation = 0

                if base_r2 > 0.999:
                    logger.info(f"  [EARLY STOP] Perfect convergence achieved at Gen {gen}.")
                    break
            else:
                stagnation += 1

            self._ppo_update([(s['seq'], s['fitness']) for s in sorted_elites[:10]])

        if best_layer:
            eq_str = translate_to_sympy(best_layer['struct'], best_layer['constants'])
            logger.info(
                f"  LAYER {layer_idx} COMPLETE | "
                f"Best R2={best_layer['r2']:.5f} | {eq_str}"
            )
        else:
            logger.info(f"  LAYER {layer_idx} COMPLETE | No candidate found")

        return best_layer

    def run_protocol_zero(self, max_layers: int = 4) -> None:
        logger.info("Protocol Zero: Onion Peeling initiated...")
        current_target = self.original_target.clone()

        for i in range(1, max_layers + 1):
            layer = self.run_layer_search(current_target, i)
            if layer is None or layer['r2'] < 0.001:
                logger.info("Signal-to-noise too low — peeling complete.")
                break
            self.discovered_layers.append(layer)
            current_target = current_target - layer['pred']
            if layer['r2'] > 0.99:
                logger.info("Near-perfect convergence — peeling complete.")
                break

    def global_refinement(self) -> Tuple[float, List[torch.Tensor]]:
        logger.info("Global Refinement: harmonizing all layers...")
        if not self.discovered_layers:
            return 0.0, []

        structures = [l['struct'] for l in self.discovered_layers]
        init_c: List[float] = []
        for l in self.discovered_layers:
            init_c.extend(l['constants'].tolist())
        if not init_c:
            return self.discovered_layers[0]['r2'], []

        global_c = torch.tensor(
            init_c, requires_grad=True, device=self.device, dtype=torch.float64
        )
        opt = optim.LBFGS([global_c], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = IronDome.execute_multi_layer(
                structures, self.inputs, global_c, self.ones
            )
            loss = torch.mean((pred - self.original_target) ** 2)
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            final_pred = IronDome.execute_multi_layer(
                structures, self.inputs, global_c, self.ones
            )
            var      = torch.var(self.original_target).item() + 1e-9
            mse      = torch.mean((final_pred - self.original_target) ** 2).item()
            final_r2 = 1.0 - mse / var

        logger.info(f"Global Refinement complete. Unified R2: {final_r2:.5f}")

        refined, c_off = [], 0
        for i, struct in enumerate(structures):
            nc = struct.count(Op.CONST)
            lc = global_c[c_off: c_off + nc].detach()
            refined.append(lc)
            self.discovered_layers[i]['constants'] = lc
            c_off += nc

        return final_r2, refined

    def run(self, max_layers: int = 4) -> Optional[Dict]:
        logger.info("OMEGA Sovereign Discovery Engine igniting...")
        self.run_protocol_zero(max_layers=max_layers)

        if len(self.discovered_layers) > 1:
            self.global_refinement()

        best = self.pareto.best()
        logger.info(
            f"Discovery complete. "
            f"Pareto archive: {len(self.pareto.front)} laws. "
            f"Graveyard: {len(self.graveyard)} spurious equations banished."
        )
        return best if best else (
            self.discovered_layers[0] if self.discovered_layers else None
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def translate_to_sympy(
    structure:  List[int],
    constants:  torch.Tensor,
    var_names:  List[str] = None
) -> str:
    if var_names is None:
        var_names = ['x1', 'x2', 'x3', 'x4', 'x5']
    stack, c_idx = [], 0
    try:
        for act in structure:
            if 1 <= act <= 5:
                stack.append(sp.Symbol(var_names[act - 1]))
            elif act == Op.CONST:
                stack.append(round(constants[c_idx].item(), 4)); c_idx += 1
            elif act == Op.ADD:
                b, a = stack.pop(), stack.pop(); stack.append(a + b)
            elif act == Op.SUB:
                b, a = stack.pop(), stack.pop(); stack.append(a - b)
            elif act == Op.MUL:
                b, a = stack.pop(), stack.pop(); stack.append(a * b)
            elif act == Op.DIV:
                b, a = stack.pop(), stack.pop(); stack.append(a / b)
            elif act == Op.SIN:
                stack.append(sp.sin(stack.pop()))
            elif act == Op.COS:
                stack.append(sp.cos(stack.pop()))
            elif act == Op.EXP:
                stack.append(sp.exp(stack.pop()))
            elif act == Op.SQ:
                stack.append(stack.pop() ** 2)
            elif act == Op.CUBE:
                stack.append(stack.pop() ** 3)
        return str(sp.simplify(stack[-1]))
    except Exception:
        return "Translation Error"


# ─────────────────────────────────────────────────────────────────────────────
# 10. ENTRY POINT: CHALLENGE 1 - THE DUFFING OSCILLATOR + GRAPHING
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"OMEGA | Device: {device}")

    logger.info("\n" + "="*60)
    logger.info("  INITIATING CHALLENGE 1: THE DUFFING OSCILLATOR")
    logger.info("  Target: y = -0.2*w1 - 1.0*th1 - 0.5*th1^3 + 2.5*cos(th2)")
    logger.info("="*60 + "\n")

    torch.manual_seed(42)
    n = 500

    # Orthogonal datasets to prevent collinearity collapse
    w1  = (torch.rand(n, dtype=torch.float64) * 4) - 2
    th1 = (torch.rand(n, dtype=torch.float64) * 6) - 3
    th2 = (torch.rand(n, dtype=torch.float64) * 6) - 3

    # The True Physical Law
    y = -0.2 * w1 - 1.0 * th1 - 0.5 * (th1 ** 3) + 2.5 * torch.cos(th2)

    # Inject real-world measurement noise
    y = y + 0.05 * torch.randn(n, dtype=torch.float64)

    inputs = torch.stack([w1, th1, th2], dim=1).to(device)
    target = y.to(device)

    # ── THE CONFIGURATION ──
    cfg = OmegaConfig()
    engine = OmegaIndustrial(inputs, target, cfg, device)

    # Max layers = 4 (for 4 terms in the Duffing equation)
    result = engine.run(max_layers=4)

    logger.info("\n" + "="*60)
    if result is not None:
        struct    = result.get('struct') or result.get('seq', [])
        constants = result.get('constants', torch.tensor([]))

        # Pull final harmonized constants from Global Refinement across all layers
        final_structures = [layer['struct'] for layer in engine.discovered_layers]
        final_global_c   = torch.cat([layer['constants'] for layer in engine.discovered_layers])

        # Translate the aggregated equation parts for logging
        eq_parts = []
        c_offset = 0
        for st in final_structures:
            num_c = st.count(Op.CONST)
            c_subset = final_global_c[c_offset : c_offset + num_c]
            eq_parts.append(translate_to_sympy(st, c_subset, var_names=['w1', 'th1', 'th2']))
            c_offset += num_c

        full_discovered_eq = " + ".join(eq_parts)

        logger.info(f"  TARGET LAW       : -0.2*w1 - 1.0*th1 - 0.5*th1**3 + 2.5*cos(th2)")
        logger.info(f"  DISCOVERED LAW   : {full_discovered_eq}")
        logger.info(f"  Pareto archive   : {len(engine.pareto.front)} laws")
        logger.info(f"  Graveyard        : {len(engine.graveyard)} spurious banished")

        # ── MATPLOTLIB: EXPECTED VS PREDICTED GRAPH ──
        logger.info("  Generating Expected vs. Predicted Visualization...")

        with torch.no_grad():
            final_pred = IronDome.execute_multi_layer(
                final_structures, inputs, final_global_c, engine.ones
            ).cpu().numpy().flatten()

        expected_values = target.cpu().numpy().flatten()

        plt.figure(figsize=(10, 6))
        plt.scatter(expected_values, final_pred, alpha=0.6, color='cyan', edgecolor='darkblue', label='Discovered Data Points')

        # Plot the Perfect Fit Line (y=x)
        min_val = min(expected_values.min(), final_pred.min())
        max_val = max(expected_values.max(), final_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (y=x)')

        plt.title('OMEGA Discovery Engine: Target vs Predicted (Duffing Oscillator)', fontsize=14, pad=15)
        plt.xlabel('True Expected Values (Target)', fontsize=12)
        plt.ylabel('OMEGA Predicted Values', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Save output
        plt.savefig('omega_expected_vs_predicted.png', dpi=300)
        logger.info("  [PLOT SAVED] Graph successfully written to 'omega_expected_vs_predicted.png'.")

        # Display if in interactive environment
        plt.show()

    else:
        logger.info("  No law discovered. The chaos won this round.")
    logger.info("="*60)