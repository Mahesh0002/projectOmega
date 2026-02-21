"""
Project OMEGA: Singularity Benchmark (Model Alpha)

This module implements the "Consensus Learner" architecture designed to solve 
functions containing vertical asymptotes (e.g., y = 1 / (t - 2.5)). 
It utilizes an "Inlier Maximization" objective rather than standard MSE to 
prevent gradients from exploding when evaluating points near the singularity.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Optional, Dict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 0. LOGGING SETUP ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [ALPHA_SOLVER] | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---
@dataclass
class AlphaConfig:
    difficulty: str = "ASYMPTOTE"
    pop_size: int = 1000
    generations: int = 500
    action_dim: int = 10
    embedding_dim: int = 32
    hidden_dim: int = 64
    lr_student: float = 0.005
    
    # Adaptive Temperature Bounds
    heat_base: float = 0.05
    heat_max: float = 0.25
    heat_min: float = 0.01
    target_valid_ratio: float = 0.40

class Op(IntEnum):
    VAR_T = 0; CONST_A = 1; CONST_B = 2; ADD = 3
    SUB = 4; MUL = 5; DIV = 6; SQUARE = 7; SIN = 8; COS = 9

# --- 2. NUMERICAL SAFETY KERNEL ---
class NumericalSafeguards:
    @staticmethod
    def safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.abs(b) < 1e-3, torch.ones_like(b), a / b)

    @staticmethod
    def clamp_pow2(a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(a**2, -1e6, 1e6)

# --- 3. RECURRENT AGENT ---
class BatchedSwarmAgent(nn.Module):
    def __init__(self, config: AlphaConfig):
        super().__init__()
        self.embed = nn.Embedding(config.action_dim + 1, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

# --- 4. CONTINUOUS OPTIMIZATION ---
class ConstantOptimizer(nn.Module):
    def __init__(self, num_constants: int):
        super().__init__()
        self.coeffs = nn.Parameter(torch.ones(num_constants))

def fine_tune_constants(structure: List[int], t: torch.Tensor, y_target: torch.Tensor, steps: int = 50) -> Tuple[Optional[torch.Tensor], float]:
    num_c = sum(1 for a in structure if a in [Op.CONST_A, Op.CONST_B])
    if num_c == 0: 
        return None, 1e9

    best_loss = float('inf')
    best_coeffs = None
    grid_seeds = [1.0, 3.0, 0.5, 5.0]

    for seed in grid_seeds:
        model = ConstantOptimizer(num_c)
        with torch.no_grad():
            model.coeffs[0].fill_(seed)
            if num_c > 1: 
                nn.init.uniform_(model.coeffs[1:], -1.0, 1.0)

        opt = optim.Adam(model.parameters(), lr=0.1)

        for _ in range(steps):
            opt.zero_grad()
            stack = []
            c_idx = 0
            try:
                for act in structure:
                    if act == Op.VAR_T: 
                        stack.append(t)
                    elif act in [Op.CONST_A, Op.CONST_B]:
                        stack.append(model.coeffs[c_idx] * torch.ones_like(t))
                        c_idx += 1
                    elif act == Op.ADD: 
                        b, a = stack.pop(), stack.pop()
                        stack.append(a+b)
                    elif act == Op.SUB: 
                        b, a = stack.pop(), stack.pop()
                        stack.append(a-b)
                    elif act == Op.MUL: 
                        b, a = stack.pop(), stack.pop()
                        stack.append(a*b)
                    elif act == Op.DIV: 
                        b, a = stack.pop(), stack.pop()
                        stack.append(NumericalSafeguards.safe_div(a, b))
                    elif act == Op.SQUARE: 
                        stack.append(NumericalSafeguards.clamp_pow2(stack.pop()))
                    elif act == Op.SIN: 
                        stack.append(torch.sin(stack.pop()))
                    elif act == Op.COS: 
                        stack.append(torch.cos(stack.pop()))

                while len(stack) > 1: 
                    stack.append(stack.pop() + stack.pop())

                # Huber-style robust loss for local gradients near singularities
                diff = torch.abs(y_target - stack[-1])
                loss = torch.mean(torch.where(diff < 1.0, diff**2, diff))

                loss.backward()
                opt.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_coeffs = model.coeffs.detach().clone()
            except Exception: 
                break

    return best_coeffs, best_loss

# --- 5. BENCHMARK MANAGER ---
class AlphaSingularitySolver:
    def __init__(self, config: AlphaConfig, t_data: torch.Tensor, y_data: torch.Tensor):
        self.cfg = config
        self.t = t_data
        self.y = y_data
        self.student = BatchedSwarmAgent(config)
        self.opt = optim.Adam(self.student.parameters(), lr=config.lr_student)

        self.best_structure: Optional[Tuple[List[int], torch.Tensor]] = None
        self.best_fitness = 0.0

        self.trash_heap: List[Tuple[float, torch.Tensor]] = []
        self.structure_cache: Dict[Tuple[int, ...], Tuple[torch.Tensor, float]] = {}
        self.fitness_cache: Dict[Tuple[int, ...], Tuple[float, bool]] = {}

    def _generate_structured_chaos(self, count: int = 20) -> List[torch.Tensor]:
        random_structures = []
        for _ in range(count):
            seq = [0] if random.random() < 0.5 else [random.choice([1, 2])]
            for _ in range(random.randint(2, 6)):
                if random.random() < 0.6:
                    operand = 0 if random.random() < 0.4 else random.choice([1, 2])
                    op = random.choice([3, 4, 5, 6])
                    seq.extend([operand, op])
                else:
                    op = random.choice([7, 8, 9])
                    seq.append(op)
            random_structures.append(torch.tensor([[0] + seq]))
        return random_structures

    def evaluate_expression(self, actions: List[int]) -> Tuple[float, bool]:
        stack = []
        for act in actions:
            if act == Op.VAR_T: stack.append(self.t)
            elif act in [Op.CONST_A, Op.CONST_B]: stack.append(torch.ones_like(self.t))
            elif act == Op.ADD:
                if len(stack) < 2: return 0.0, False
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)
            elif act == Op.SUB:
                if len(stack) < 2: return 0.0, False
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
            elif act == Op.MUL:
                if len(stack) < 2: return 0.0, False
                b, a = stack.pop(), stack.pop()
                stack.append(a * b)
            elif act == Op.DIV:
                if len(stack) < 2: return 0.0, False
                b, a = stack.pop(), stack.pop()
                stack.append(NumericalSafeguards.safe_div(a, b))
            elif act == Op.SQUARE:
                if not stack: return 0.0, False
                stack.append(NumericalSafeguards.clamp_pow2(stack.pop()))
            elif act == Op.SIN:
                if not stack: return 0.0, False
                stack.append(torch.sin(stack.pop()))
            elif act == Op.COS:
                if not stack: return 0.0, False
                stack.append(torch.cos(stack.pop()))

        if not stack: return 0.0, False
        while len(stack) > 1: stack.append(stack.pop() + stack.pop())
        pred = stack[-1]

        if torch.isnan(pred).any() or torch.isinf(pred).any(): return 0.0, False

        # Inlier Maximization Objective (Replaces standard MSE)
        diff = torch.abs(self.y - pred)
        tolerance = 0.1 * torch.abs(self.y) + 0.1
        hits = (diff < tolerance).float()
        hit_rate = torch.mean(hits)
        
        final_score = (hit_rate ** 2)
        penalty = len(actions) * 0.005
        return max(0.001, final_score - penalty), True

    def run(self):
        logger.info(f"Initializing Asymptote Solver | Target: {self.cfg.difficulty}")
        stagnation_counter = 0
        last_best_fitness = 0.0
        current_mutation = self.cfg.heat_base

        chaos = self._generate_structured_chaos(50)
        for c in chaos: 
            self.trash_heap.append((0.0, c))

        for gen in range(self.cfg.generations):
            # 1. BATCH GENERATION
            batch_hist = torch.zeros(self.cfg.pop_size, 1, dtype=torch.long)
            batch_actions = [[] for _ in range(self.cfg.pop_size)]
            hidden = None
            stack_depths = torch.zeros(self.cfg.pop_size, dtype=torch.long)
            active_mask = torch.ones(self.cfg.pop_size, dtype=torch.bool)

            for step in range(15):
                logits, hidden = self.student(batch_hist, hidden)
                last_logits = logits[:, -1, :]

                mask_bin = (stack_depths < 2).unsqueeze(1).expand_as(last_logits)
                last_logits[mask_bin & (torch.arange(10).unsqueeze(0) >= 3) & (torch.arange(10).unsqueeze(0) <= 6)] = -float('inf')
                mask_un = (stack_depths < 1).unsqueeze(1).expand_as(last_logits)
                last_logits[mask_un & (torch.arange(10).unsqueeze(0) >= 7)] = -float('inf')

                probs = torch.softmax(last_logits, dim=-1)
                noise = torch.rand_like(probs) * current_mutation
                dists = torch.distributions.Categorical(probs + noise)
                actions = dists.sample()

                batch_hist = actions.unsqueeze(1)
                act_cpu = actions.cpu().numpy()
                
                for i in range(self.cfg.pop_size):
                    if not active_mask[i]: continue
                    a = act_cpu[i]
                    batch_actions[i].append(a)
                    if a in [0, 1, 2]: stack_depths[i] += 1
                    elif a in [3, 4, 5, 6]: stack_depths[i] -= 1
                    if stack_depths[i] < 1: active_mask[i] = False

            # 2. EVALUATION
            batch_scores = []
            batch_tensors = []
            valid_count = 0

            for i in range(self.cfg.pop_size):
                struct_tuple = tuple(batch_actions[i])
                if struct_tuple in self.fitness_cache:
                    score, is_valid = self.fitness_cache[struct_tuple]
                else:
                    score, is_valid = self.evaluate_expression(batch_actions[i])
                    self.fitness_cache[struct_tuple] = (score, is_valid)

                batch_scores.append(score)
                if is_valid: valid_count += 1
                batch_tensors.append(torch.tensor([[0] + batch_actions[i]]))

            valid_ratio = valid_count / self.cfg.pop_size
            error = valid_ratio - self.cfg.target_valid_ratio
            current_mutation += (error * 0.05)
            current_mutation = max(self.cfg.heat_min, min(self.cfg.heat_max, current_mutation))
            if self.best_fitness > 0.90: 
                current_mutation = 0.01

            # 3. DIVERSITY FILTER
            scores_t = torch.tensor(batch_scores)
            threshold = torch.quantile(scores_t, 0.8)
            winners_idx = [i for i in range(self.cfg.pop_size) if batch_scores[i] >= threshold]
            losers_indices = [i for i in range(self.cfg.pop_size) if batch_scores[i] < threshold and batch_scores[i] > 1e-3]

            for idx in losers_indices:
                cand_score = batch_scores[idx]
                cand_tensor = batch_tensors[idx]
                is_duplicate = False
                strictness = 1e-4 if len(self.trash_heap) > 200 else 1e-9

                if len(self.trash_heap) > 0:
                    sample_size = min(len(self.trash_heap), 50)
                    sample = random.sample(self.trash_heap, sample_size)
                    for item in sample:
                        if abs(item[0] - cand_score) < strictness:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    self.trash_heap.append((cand_score, cand_tensor))

            if len(self.trash_heap) > 500:
                self.trash_heap.sort(key=lambda x: x[0])
                self.trash_heap = self.trash_heap[-500:]

            # 4. COOPERATIVE MERGE
            grafted_winners = []
            if gen % 3 == 0 and winners_idx and self.trash_heap:
                for _ in range(50):
                    teacher = batch_tensors[random.choice(winners_idx)]
                    student = random.choice(self.trash_heap)[1]
                    if student.size(1) > 1:
                        child = torch.cat([teacher, student[:, 1:], torch.tensor([[Op.ADD]])], dim=1)
                        grafted_winners.append(child)

            # 5. TUNING
            final_train_set = [batch_tensors[i] for i in winners_idx] + grafted_winners
            if valid_count < 50: 
                final_train_set += [batch_tensors[i] for i in range(self.cfg.pop_size) if batch_scores[i] > 1e-3]

            if final_train_set:
                candidates = random.sample(final_train_set, min(50, len(final_train_set)))
                for cand in candidates:
                    struct = [a.item() for a in cand[0, 1:]]
                    struct_tuple = tuple(struct)

                    if struct_tuple in self.structure_cache:
                        coeffs, loss = self.structure_cache[struct_tuple]
                    else:
                        coeffs, loss = fine_tune_constants(struct, self.t, self.y, steps=30)
                        self.structure_cache[struct_tuple] = (coeffs, loss)

                    if loss < 1e8:
                        tuned_fit = 1.0 / (1.0 + np.log1p(loss)) 
                        if tuned_fit > self.best_fitness:
                            self.best_fitness = tuned_fit
                            c, l = fine_tune_constants(struct, self.t, self.y, steps=200)
                            self.best_structure = (struct, c)
                            self.structure_cache[struct_tuple] = (c, l)
                            logger.info(f"Generation {gen} | New Optimal Structure | Fitness: {self.best_fitness:.5f}")
                            stagnation_counter = 0

                self.opt.zero_grad()
                max_len = max(t.size(1) for t in final_train_set)
                padded = [torch.cat([t, torch.zeros(1, max_len - t.size(1), dtype=torch.long)], dim=1) for t in final_train_set]
                batch_t = torch.cat(padded, dim=0)
                logits, _ = self.student(batch_t[:, :-1])
                loss = nn.functional.cross_entropy(logits.reshape(-1, self.cfg.action_dim), batch_t[:, 1:].reshape(-1), ignore_index=0)
                loss.backward()
                self.opt.step()

            if self.best_fitness > last_best_fitness + 0.001:
                stagnation_counter = 0
                last_best_fitness = self.best_fitness
            else:
                stagnation_counter += 1

            if stagnation_counter > 20:
                current_mutation = min(current_mutation + 0.05, 0.25)

    def visualize(self):
        if not self.best_structure:
            logger.warning("No structure found to visualize.")
            return
            
        struct, coeffs = self.best_structure
        stack, eq_str = [], []
        c_idx = 0
        try:
            for act in struct:
                if act == Op.VAR_T:
                    stack.append(self.t)
                    eq_str.append("t")
                elif act in [Op.CONST_A, Op.CONST_B]:
                    val = coeffs[c_idx].item()
                    stack.append(val * torch.ones_like(self.t))
                    eq_str.append(f"{val:.2f}")
                    c_idx += 1
                elif act == Op.ADD:
                    b,a = stack.pop(), stack.pop(); eq_str.append(f"({eq_str.pop()}+{eq_str.pop()})"); stack.append(a+b)
                elif act == Op.SUB:
                    b,a = stack.pop(), stack.pop(); b_s, a_s = eq_str.pop(), eq_str.pop(); eq_str.append(f"({a_s}-{b_s})"); stack.append(a-b)
                elif act == Op.MUL:
                    b,a = stack.pop(), stack.pop(); b_s, a_s = eq_str.pop(), eq_str.pop(); eq_str.append(f"({a_s}*{b_s})"); stack.append(a*b)
                elif act == Op.DIV:
                    b,a = stack.pop(), stack.pop(); b_s, a_s = eq_str.pop(), eq_str.pop(); eq_str.append(f"({a_s}/{b_s})"); stack.append(NumericalSafeguards.safe_div(a,b))
                elif act == Op.SQUARE:
                    stack.append(NumericalSafeguards.clamp_pow2(stack.pop())); eq_str.append(f"({eq_str.pop()})^2")
                elif act == Op.SIN:
                    stack.append(torch.sin(stack.pop())); eq_str.append(f"sin({eq_str.pop()})")
                elif act == Op.COS:
                    stack.append(torch.cos(stack.pop())); eq_str.append(f"cos({eq_str.pop()})")

            while len(eq_str) > 1:
                b, a = eq_str.pop(), eq_str.pop()
                eq_str.append(f"({a}+{b})")
                stack.append(stack.pop() + stack.pop())

            logger.info(f"Final Discovered Equation: {eq_str[-1]}")
            
            plt.figure(figsize=(10,6))
            plt.scatter(self.t.numpy(), self.y.numpy(), label='Ground Truth Data', alpha=0.5, color='red')
            plt.plot(self.t.numpy(), stack[-1].detach().numpy(), 'b-', linewidth=2, label='Model Prediction')
            plt.ylim(-10, 10)
            plt.title(f"Singularity Reconstruction: {eq_str[-1]}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except Exception as e:
            logger.error(f"Visualization failed during syntax tree parsing: {e}")

if __name__ == "__main__":
    cfg = AlphaConfig(difficulty="ASYMPTOTE")
    t_vals = torch.linspace(0, 5, 100).view(-1, 1)

    # Singular target: Vertical asymptote at t = 2.5
    y_true = 1.0 / (t_vals - 2.5)

    engine = AlphaSingularitySolver(config=cfg, t_data=t_vals, y_data=y_true)
    engine.run()
    engine.visualize()