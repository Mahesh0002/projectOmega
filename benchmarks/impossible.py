"""
Project OMEGA: Titan Benchmark (The Impossible Phase)

This module implements a multi-agent "Island Model" optimization strategy to tackle 
highly non-linear, multi-frequency target functions (e.g., tan(t) + sin(exp(t))). 
Multiple independent search agents (Islands) operate with unique structural biases, 
periodically sharing their best discoveries to escape local minima.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Dict

# --- 0. SETUP ---
for handler in logging.root.handlers[:]: 
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S', stream=sys.stdout)
logger = logging.getLogger("OMEGA_TITAN")

class Op(IntEnum):
    VAR_T = 1; CONST_A = 2; CONST_B = 3; ADD = 4; SUB = 5
    MUL = 6; DIV = 7; SQUARE = 8; SIN = 9; COS = 10; EXP = 11

@dataclass
class OmegaConfig:
    action_dim: int = 12
    embedding_dim: int = 64
    hidden_dim: int = 256
    lr_student: float = 0.005
    pop_size: int = 150
    parsimony: float = 0.005
    max_seq_len: int = 16

@dataclass
class IslandConfig:
    name: str
    value: float
    op_bias: Dict[Op, float]

# --- 1. REGISTRY (Memoization & Caching) ---
class Registry:
    def __init__(self):
        self.pages: Dict[str, Tuple[List[int], torch.Tensor]] = {}
        self.last_used: Dict[str, int] = {}
        
    def add_entry(self, name: str, data: Tuple[List[int], torch.Tensor], gen: int):
        self.pages[name] = data
        self.last_used[name] = gen
        
    def consult(self) -> List[Tuple[List[int], torch.Tensor]]: 
        return list(self.pages.values())
        
    def erode(self, current_gen: int, horizon: int = 50):
        to_remove = [n for n, g in self.last_used.items() if current_gen - g > horizon]
        for name in to_remove: 
            del self.pages[name]
            del self.last_used[name]

LIBRARY = Registry()

# --- 2. PHYSICS LAYER ---
class SafeMath:
    @staticmethod
    def div(a, b): 
        return a / (b + 1e-8 * torch.sign(b) + 1e-9)

def dynamic_weighting(y_true):
    """Applies dynamic penalty weights to extreme asymptotic values."""
    weights = torch.ones_like(y_true)
    weights[torch.abs(y_true) > 10] *= 50.0
    weights[torch.abs(y_true) > 30] *= 200.0
    return weights

def evaluate_expression(structure: List[int], t, constants=None, ones_buffer=None):
    stack = []
    c_idx = 0
    if constants is None: 
        constants = torch.tensor([])
    if constants.dim() == 0: 
        constants = constants.unsqueeze(0)
        
    try:
        for act in structure:
            if act == Op.VAR_T: 
                stack.append(t)
            elif act in [Op.CONST_A, Op.CONST_B]:
                val = constants[c_idx] if c_idx < len(constants) else 1.0
                c_idx += 1
                stack.append(val * ones_buffer if ones_buffer is not None else val * torch.ones_like(t))
            elif act == Op.ADD: 
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)
            elif act == Op.SUB: 
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
            elif act == Op.MUL: 
                b, a = stack.pop(), stack.pop()
                stack.append(a * b)
            elif act == Op.DIV: 
                b, a = stack.pop(), stack.pop()
                stack.append(SafeMath.div(a, b))
            elif act == Op.SQUARE: 
                stack.append(torch.pow(torch.clamp(stack.pop(), -100, 100), 2))
            elif act == Op.SIN: 
                stack.append(torch.sin(stack.pop()))
            elif act == Op.COS: 
                stack.append(torch.cos(stack.pop()))
            elif act == Op.EXP: 
                stack.append(torch.exp(torch.clamp(stack.pop(), -10, 25)))
                
        res = stack[-1] if stack else torch.zeros_like(t)
        return res.view_as(t)
    except: 
        return torch.zeros_like(t)

# --- 3. SEARCH AGENT (ISLAND) ---

class Island:
    def __init__(self, config, t, y, g_cfg):
        self.cfg = config
        self.t = t
        self.y = y
        self.g_cfg = g_cfg
        
        self.student = nn.LSTM(g_cfg.embedding_dim, g_cfg.hidden_dim, num_layers=2, batch_first=True)
        self.embed = nn.Embedding(g_cfg.action_dim + 2, g_cfg.embedding_dim)
        self.fc = nn.Linear(g_cfg.hidden_dim, g_cfg.action_dim + 2)
        self.opt = optim.Adam(list(self.student.parameters()) + list(self.embed.parameters()) + list(self.fc.parameters()), lr=g_cfg.lr_student)
        
        self.best_local = None
        self.failed_heap = []
        
        self.w_c = dynamic_weighting(y).view(-1, 1)
        self.ones_buffer = torch.ones_like(t)

    def step(self, global_best, gen):
        # Jittered evaluation to force robust structural identities
        t_noise = self.t + torch.randn_like(self.t) * 0.0005

        candidates = []
        if self.best_local: 
            candidates.append((self.best_local[0], self.best_local[1], 0.05))
        if global_best: 
            candidates.append((global_best[0], global_best[1], 0.05))
        if self.failed_heap:
            for _ in range(min(3, len(self.failed_heap))):
                bad_s, bad_c = random.choice(self.failed_heap)
                candidates.append((bad_s, bad_c, 1.5)) 

        # Neural Generation
        curr_in = torch.zeros(self.g_cfg.pop_size, 1, dtype=torch.long)
        actions_list = [[] for _ in range(self.g_cfg.pop_size)]
        hidden = None
        
        for _ in range(self.g_cfg.max_seq_len):
            emb = self.embed(curr_in)
            output, hidden = self.student(emb, hidden)
            logits = self.fc(output[:, -1, :])
            
            for op, b in self.cfg.op_bias.items(): 
                logits[:, op] += b
                
            probs = torch.softmax(logits / 1.5, dim=-1)
            curr_in = torch.multinomial(probs, 1)
            
            for i, a in enumerate(curr_in.squeeze().tolist()):
                if a != 0: 
                    actions_list[i].append(a)

        # Build Evaluation Pool
        eval_pool = []
        for seq in actions_list: 
            eval_pool.append((seq, None, 1.0))
        for s, c, j in candidates: 
            eval_pool.append((s, c, j))

        batch = []
        for raw_s, guess_c, jitter in eval_pool:
            clean_s = [t for t in raw_s if t in range(1, 12)]
            if Op.VAR_T not in clean_s or not clean_s: 
                continue

            # Continuous Constant Optimization
            num_c = sum(1 for a in clean_s if a in [Op.CONST_A, Op.CONST_B])
            c_param = guess_c.clone() if guess_c is not None and len(guess_c) == num_c else torch.randn(num_c)
            if jitter > 0: 
                c_param += torch.randn_like(c_param) * jitter
            c_param.requires_grad_(True)

            iter_budget = 5 if guess_c is not None else 15
            optimizer = optim.LBFGS([c_param], lr=1, max_iter=iter_budget)
            
            def closure():
                optimizer.zero_grad()
                yp = evaluate_expression(clean_s, self.t, c_param, self.ones_buffer)
                l = torch.mean(((yp - self.y)**2) * self.w_c)
                l.backward()
                return l
                
            try: 
                optimizer.step(closure)
            except: 
                continue

            with torch.no_grad():
                yp_noise = evaluate_expression(clean_s, t_noise, c_param, self.ones_buffer)
                if torch.isnan(yp_noise).any() or torch.isinf(yp_noise).any():
                    if len(self.failed_heap) < 30: 
                        self.failed_heap.append((clean_s, c_param.detach()))
                    continue

                mse = torch.mean((yp_noise - self.y)**2).item()
                acc = 1.0 - (mse / (torch.var(self.y) + 1e-9)).item()

            fit = acc - (self.g_cfg.parsimony * len(clean_s))
            if yp_noise[-1] > yp_noise[-10]: 
                fit += 0.05
            batch.append((clean_s, c_param.detach(), fit, acc))

        batch.sort(key=lambda x: x[2], reverse=True)
        if batch:
            winner = batch[0]
            if self.best_local is None or winner[2] > self.best_local[2]:
                self.best_local = winner
                if winner[3] > 0.98: 
                    LIBRARY.add_entry(f"{self.cfg.name}_GEN_{gen}", (winner[0], winner[1]), gen)

            # Imitation Training 
            elites = batch[:10]
            self.opt.zero_grad()
            l_total = 0
            for e in elites:
                s_in = torch.tensor([0] + e[0])
                target = torch.tensor(e[0] + [0])
                emb = self.embed(s_in.unsqueeze(0))
                out, _ = self.student(emb)
                l_total += nn.functional.cross_entropy(self.fc(out).view(-1, 14), target)
            l_total.backward()
            self.opt.step()

# --- 4. MULTI-AGENT MANAGER ---
class IslandManager:
    def __init__(self, t_v, y_v):
        self.t = t_v
        self.y = y_v
        self.g_cfg = OmegaConfig()
        self.islands = [
            Island(IslandConfig("Cathedral", 1.0, {Op.SIN: 2.0, Op.COS: 1.5}), t_v, y_v, self.g_cfg),
            Island(IslandConfig("Academy", 1.2, {Op.EXP: 2.5, Op.MUL: 1.5}), t_v, y_v, self.g_cfg),
            Island(IslandConfig("Forge", 1.8, {Op.DIV: 3.0, Op.SQUARE: 2.0}), t_v, y_v, self.g_cfg)
        ]
        self.global_best = None

    def run(self):
        logger.info("Initializing Multi-Island Optimization for Titan Benchmark...")
        for gen in range(1, 401):
            if gen % 30 == 0: 
                LIBRARY.erode(gen)
                
            for island in self.islands:
                island.step(self.global_best, gen)
                if island.best_local:
                    if self.global_best is None or island.best_local[2] > self.global_best[2]:
                        self.global_best = island.best_local
                        logger.info(f"GEN {gen} | [{island.cfg.name}] Best Discovered | R2: {self.global_best[3]*100:.4f}%")

    def visualize(self):
        if not self.global_best: 
            return
            
        t_dense = torch.linspace(0, 1.565, 500).view(-1, 1)
        y_true = torch.tan(t_dense) + torch.sin(torch.exp(t_dense))
        y_p = evaluate_expression(self.global_best[0], t_dense, self.global_best[1])
        
        plt.figure(figsize=(12, 6))
        plt.plot(t_dense, y_true, 'r--', alpha=0.6, label='Ground Truth')
        plt.plot(t_dense, y_p, 'b-', label='OMEGA Extracted Model')
        plt.ylim(-10, 100)
        plt.title(f"Titan Benchmark Result | R2: {self.global_best[3]*100:.2f}%")
        plt.legend()
        plt.show()
        
        op_map = {i.value: i.name for i in Op}
        print("Final Extracted Equation Sequence:", [op_map.get(t, str(t)) for t in self.global_best[0]])

if __name__ == "__main__":
    t_v = torch.linspace(0, 1.56, 100).view(-1, 1)
    y_v = torch.tan(t_v) + torch.sin(torch.exp(t_v))
    
    mgr = IslandManager(t_v, y_v)
    mgr.run()
    mgr.visualize()