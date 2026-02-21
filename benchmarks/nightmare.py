"""
Project OMEGA: Exponential Oscillation Benchmark (The Nightmare Phase)

This module evaluates the engine's ability to recover highly nested, 
exponentially compounding frequencies (e.g., y = sin(exp(c * t))). 
It utilizes a Recurrent Neural Network (RNN) as a generative policy, 
guided by a temperature-annealed reinforcement learning loop with a 
semantic caching mechanism to prevent redundant L-BFGS evaluations.
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
from typing import List, Tuple, Optional, Dict

# --- 0. SETUP ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | [OMEGA_NIGHTMARE] | %(message)s', 
    datefmt='%H:%M:%S', 
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---
@dataclass
class OmegaConfig:
    difficulty: str = "Exponential Oscillation"
    pop_size: int = 2000       
    generations: int = 600     
    action_dim: int = 12
    embedding_dim: int = 48
    hidden_dim: int = 128
    lr_student: float = 0.005
    temperature_base: float = 1.5     
    parsimony_penalty: float = 0.003 

class Op(IntEnum):
    VAR_T = 1; CONST_A = 2; CONST_B = 3
    ADD = 4; SUB = 5; MUL = 6; DIV = 7
    SQUARE = 8; SIN = 9; COS = 10; EXP = 11

# --- 2. NUMERICAL SAFETY KERNEL ---
class NumericalSafeguards:
    @staticmethod
    def safe_div(a, b): 
        return a / (b + 1e-6 * torch.sign(b) + 1e-9)
    @staticmethod
    def safe_exp(a): 
        return torch.exp(torch.clamp(a, -10, 15)) 
    @staticmethod
    def clamp_pow2(a): 
        return torch.pow(torch.clamp(a, -100, 100), 2)

# --- 3. RECURRENT EXPLORER ---
class RecurrentExplorer(nn.Module):
    def __init__(self, config: OmegaConfig):
        super().__init__()
        self.embed = nn.Embedding(config.action_dim + 2, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(config.hidden_dim, config.action_dim + 2)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        logits[:, :, 0] = -1e9  # Mask padding token
        return logits, hidden

# --- 4. CONTINUOUS OPTIMIZATION (L-BFGS) ---
class ParameterOptimizer(nn.Module):
    def __init__(self, num_constants: int):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(num_constants))

def evaluate_expression(structure: List[int], t, constants=None):
    stack = []
    c_idx = 0
    try:
        for act in structure:
            if act == Op.VAR_T: 
                stack.append(t)
            elif act in [Op.CONST_A, Op.CONST_B]:
                val = constants[c_idx] if constants is not None else 1.0
                stack.append(val * torch.ones_like(t))
                c_idx += 1
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
                stack.append(NumericalSafeguards.safe_div(a, b))
            elif act == Op.SQUARE: 
                stack.append(NumericalSafeguards.clamp_pow2(stack.pop()))
            elif act == Op.SIN: 
                stack.append(torch.sin(stack.pop()))
            elif act == Op.COS: 
                stack.append(torch.cos(stack.pop()))
            elif act == Op.EXP: 
                stack.append(NumericalSafeguards.safe_exp(stack.pop()))
        
        if not stack: 
            return torch.zeros_like(t) + 1e9 
            
        result = stack[-1]
        if result.dim() == 0: 
            result = result.expand_as(t)
        return result

    except Exception: 
        return torch.zeros_like(t) + 1e9

def fine_tune_constants(structure: List[int], t, y_target):
    num_c = sum(1 for a in structure if a in [Op.CONST_A, Op.CONST_B])
    
    with torch.no_grad():
        y_pred = evaluate_expression(structure, t, constants=torch.ones(num_c) if num_c else None)
        loss = torch.nn.functional.mse_loss(y_pred, y_target).item()
        
    if torch.isnan(y_pred).any() or loss > 1000: 
        return torch.ones(num_c), 1e9
    if num_c == 0: 
        return [], loss

    model = ParameterOptimizer(num_c)
    nn.init.normal_(model.coeffs, mean=0.0, std=1.0)
    opt = optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        y_pred = evaluate_expression(structure, t, model.coeffs)
        loss = torch.nn.functional.mse_loss(y_pred, y_target)
        if torch.isnan(loss): 
            loss = torch.tensor(1e9, requires_grad=True)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        return model.coeffs.detach(), closure().item()
    except: 
        return model.coeffs.detach(), 1e9

# --- 5. BENCHMARK MANAGER ---
class NightmareBenchmark:
    def __init__(self, config: OmegaConfig, t_data, y_data):
        self.cfg = config
        self.t = t_data
        self.y = y_data
        self.student = RecurrentExplorer(config)
        self.opt = optim.Adam(self.student.parameters(), lr=config.lr_student)
        self.best_solution = None
        self.semantic_cache = {}

    def get_semantic_hash(self, y_pred):
        """Creates a discrete hash from continuous outputs to bypass redundant computations."""
        indices = torch.linspace(0, len(y_pred)-1, 20).long()
        sampled = y_pred[indices]
        return tuple(torch.round(sampled * 5).flatten().int().tolist())

    def run(self):
        logger.info(f"Initializing Benchmark. Target Environment: {self.cfg.difficulty}")
        temperature = self.cfg.temperature_base
        progress = 0.0

        for gen in range(self.cfg.generations):
            self.student.train()
            progress += (1.0 / self.cfg.generations)
            if progress > 1.0: 
                progress = 1.0
            
            # Curriculum Learning: Gradually expose more of the time series
            visible_idx = max(20, int(len(self.t) * (0.2 + 0.8 * progress)))
            t_curr = self.t[:visible_idx]
            y_curr = self.y[:visible_idx]
            t_shaken = t_curr + torch.randn_like(t_curr) * 0.005

            # 1. Generate Symbolic Structures
            current_input = torch.zeros(self.cfg.pop_size, 1, dtype=torch.long)
            batch_actions = [[] for _ in range(self.cfg.pop_size)]
            hidden = None
            stack_depths = torch.zeros(self.cfg.pop_size, dtype=torch.long)

            for step in range(12):
                logits, hidden = self.student(current_input, hidden)
                l = logits[:, -1, :]
                
                # Stack execution safeguards
                l[(stack_depths < 2), 4:8] = -1e9
                l[(stack_depths < 1), 8:12] = -1e9

                probs = torch.softmax(l / temperature, dim=-1)
                actions = torch.multinomial(probs, 1)
                current_input = actions
                cpu_acts = actions.squeeze().tolist()
                
                for i, a in enumerate(cpu_acts):
                    batch_actions[i].append(a)
                    if a in [1, 2, 3]: stack_depths[i] += 1
                    elif a in [4, 5, 6, 7]: stack_depths[i] -= 1

            # 2. Evaluate & Optimize
            batch_results = []
            candidates = [x for i, x in enumerate(batch_actions) if stack_depths[i] == 1]
            if self.best_solution: 
                candidates.append(list(self.best_solution[0]))
            
            unique_candidates = list(set(tuple(x) for x in candidates))

            for struct in unique_candidates:
                struct_list = list(struct)
                num_c = sum(1 for a in struct_list if a in [Op.CONST_A, Op.CONST_B])
                
                with torch.no_grad():
                    y_preview = evaluate_expression(struct_list, t_curr, torch.ones(num_c) if num_c else None)
                if torch.isnan(y_preview).any(): 
                    continue
                
                h = self.get_semantic_hash(y_preview)
                
                if h in self.semantic_cache:
                    c, _ = self.semantic_cache[h]
                    with torch.no_grad():
                        loss = torch.nn.functional.mse_loss(evaluate_expression(struct_list, t_curr, c), y_curr).item()
                else:
                    c, loss = fine_tune_constants(struct_list, t_curr, y_curr)
                    self.semantic_cache[h] = (c, loss)

                # Stability check
                with torch.no_grad():
                    y_shaken = evaluate_expression(struct_list, t_shaken, c)
                    stability = torch.nn.functional.mse_loss(y_shaken, y_curr).item()
                
                total_loss = loss + (stability * 0.2) + (self.cfg.parsimony_penalty * len(struct_list))
                batch_results.append((struct_list, c, 1.0/(1.0 + total_loss)))

            batch_results.sort(key=lambda x: x[2], reverse=True)

            # 3. Policy Update
            if batch_results:
                top = batch_results[0]
                if self.best_solution is None or top[2] > self.best_solution[2]:
                    self.best_solution = top
                    logger.info(f"Generation {gen} | New Best Model | Fitness: {top[2]:.5f} | Temp: {temperature:.2f}")

                elites = batch_results[:max(1, int(len(batch_results)*0.1))]
                self.opt.zero_grad()
                loss_train = 0
                for e in elites:
                    seq = torch.tensor([0] + e[0])
                    target = torch.tensor(e[0] + [0])
                    logits, _ = self.student(seq.unsqueeze(0))
                    loss_train += nn.functional.cross_entropy(logits.view(-1, self.cfg.action_dim + 2), target)
                loss_train.backward()
                self.opt.step()

            # Simulated Annealing
            if gen % 10 == 0: 
                temperature = max(0.1, temperature * 0.98)

    def visualize(self):
        if not self.best_solution: return
        struct, coeffs, fit = self.best_solution
        
        t_test = torch.linspace(0, 4.5, 400).view(-1, 1)
        y_test = torch.sin(torch.exp(1.7 * t_test))
        
        with torch.no_grad():
            y_pred = evaluate_expression(struct, t_test, coeffs)

        plt.figure(figsize=(10, 6))
        plt.plot(t_test.numpy(), y_test.numpy(), 'k--', label='Ground Truth')
        plt.plot(t_test.numpy(), y_pred.numpy(), 'r-', alpha=0.8, linewidth=2, label='Extracted Model')
        plt.axvline(x=3.2, color='g', linestyle=':', label='Training Data Limit')
        plt.title(f"Exponential Oscillation Result | Final Fitness: {fit:.5f}")
        plt.legend()
        plt.show()
        
        op_map = {i.value: i.name for i in Op}
        print("Final Extracted Sequence:", [op_map.get(x) for x in struct])

if __name__ == "__main__":
    # Generate exponential sine wave data
    t_v = torch.linspace(0, 3.2, 200).view(-1, 1)
    y_v = torch.sin(torch.exp(1.7 * t_v)) 
    
    mgr = NightmareBenchmark(OmegaConfig(), t_v, y_v)
    mgr.run()
    mgr.visualize()