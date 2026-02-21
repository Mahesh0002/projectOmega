"""
Project OMEGA: Multi-Body Orbital Dynamics (Planetary Movements)

This module demonstrates the Iterative Residual Fitting protocol on simulated 
multi-body gravitational systems. It isolates the primary gravitational 
influence (e.g., the Sun) in Stage 1, computes the residual variance, and 
then trains a secondary agent to extract minor perturbations (e.g., Jupiter).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sympy as sp
import logging
import hashlib
import matplotlib.pyplot as plt
from enum import IntEnum
from dataclasses import dataclass

# --- 1. SYSTEM CONFIGURATION ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.float64)

for handler in logging.root.handlers[:]: 
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', stream=sys.stdout, force=True)
logger = logging.getLogger("OMEGA_ASTRO")

class Op(IntEnum):
    VAR_SUN = 1; VAR_MARS = 2; VAR_JUP = 3; VAR_SAT = 4; VAR_URA = 5
    CONST = 6; ADD = 7; SUB = 8; MUL = 9; DIV = 10; SIN = 11; COS = 12

@dataclass
class OmegaConfig:
    action_dim: int = 12
    embedding_dim: int = 128
    hidden_dim: int = 256
    min_pop: int = 500
    max_pop: int = 4000
    start_pop: int = 2000
    generations: int = 40       
    orthogonality_limit: float = 0.90
    wolf_min_planets: int = 1   
    parsimony: float = 0.002    
    alchemist_elite_ratio: float = 0.20
    oracle_cache_limit: int = 20000

# --- 2. PHYSICS KERNEL ---
class IronDome:
    EPS = 1e-9
    
    @staticmethod
    def get_hash(structure): 
        return hashlib.md5(str(structure).encode()).hexdigest()
    
    @staticmethod
    def execute(structure, input_matrix, constants, ones_buffer):
        stack = []
        c_idx = 0
        try:
            for act in structure:
                if act <= 5: 
                    stack.append(input_matrix[:, act-1:act])
                elif act == Op.CONST:
                    val = constants[c_idx] if c_idx < len(constants) else 1.0
                    stack.append(val * ones_buffer)
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
                    stack.append(a / (b + IronDome.EPS * torch.sign(b) + IronDome.EPS))
                elif act == Op.SIN: 
                    stack.append(torch.sin(stack.pop()))
                elif act == Op.COS: 
                    stack.append(torch.cos(stack.pop()))
            return stack[-1] if stack else torch.zeros_like(ones_buffer)
        except: 
            return torch.zeros_like(ones_buffer)

# --- 3. ARCHITECTURE ---
class Explorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(len(Op)+2, cfg.embedding_dim)
        self.lstm = nn.LSTM(cfg.embedding_dim, cfg.hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(cfg.hidden_dim, len(Op)+2)
        
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.head(out), hidden

class OmegaEngine:
    def __init__(self, inputs, target, cfg, device):
        self.inputs = inputs
        self.target = target
        self.cfg = cfg
        self.device = device
        self.ones = torch.ones((len(target), 1), dtype=torch.float64).to(device)
        self.explorer = Explorer(cfg).to(device).double()
        self.opt = optim.Adam(self.explorer.parameters(), lr=0.005)
        self.best = None

    def optimize_constants(self, structure):
        num_c = structure.count(Op.CONST)
        c = torch.ones(num_c, requires_grad=True, device=self.device, dtype=torch.float64)
        if num_c > 0:
            optimizer = optim.LBFGS([c], lr=1, max_iter=20, line_search_fn="strong_wolfe")
            
            def closure():
                optimizer.zero_grad()
                pred = IronDome.execute(structure, self.inputs, c, self.ones)
                loss = torch.mean((pred - self.target)**2)
                loss.backward()
                return loss
                
            try: optimizer.step(closure)
            except: pass
        
        with torch.no_grad():
            pred = IronDome.execute(structure, self.inputs, c, self.ones)
            variance = torch.var(self.target) + 1e-9
            r2 = 1.0 - (torch.mean((pred - self.target)**2) / variance).item()
            
        return c.detach(), r2, pred.detach()

    def run_stage(self, stage_name):
        logger.info(f"Initiating {stage_name}...")
        
        for gen in range(1, self.cfg.generations + 1):
            pop_size = self.cfg.start_pop
            seqs = torch.zeros(pop_size, 15, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                inp = torch.zeros(pop_size, 1, dtype=torch.long).to(self.device)
                h = None
                for t in range(15):
                    logits, h = self.explorer(inp, h)
                    probs = torch.softmax(logits[:, -1, :] / 1.2, dim=-1)
                    inp = torch.multinomial(probs, 1)
                    seqs[:, t] = inp.squeeze()
            
            batch = []
            cpu_seqs = seqs.cpu().numpy()
            for i in range(pop_size):
                s = [x for x in cpu_seqs[i] if 1 <= x <= len(Op)]
                if not s: continue
                if len(set([o for o in s if o <= 5])) < self.cfg.wolf_min_planets: 
                    continue
                
                c, r2, _ = self.optimize_constants(s)
                fitness = r2 - (len(s) * self.cfg.parsimony)
                batch.append((s, c, fitness, r2))
            
            batch.sort(key=lambda x: x[2], reverse=True)
            if batch:
                current_best = batch[0]
                if self.best is None or current_best[2] > self.best[2]:
                    self.best = current_best
                    logger.info(f"Generation {gen} | Target R2: {self.best[3]:.4f}")
            
            if batch:
                self.opt.zero_grad()
                loss = 0
                for s, _, _, _ in batch[:100]:
                    t_seq = torch.tensor([0]+s+[0], device=self.device)
                    l, _ = self.explorer(t_seq[:-1].unsqueeze(0))
                    loss += nn.functional.cross_entropy(l.view(-1, len(Op)+2), t_seq[1:])
                loss.backward()
                self.opt.step()
                
        return self.best

# --- 4. PROTOCOL EXECUTION ---
def generate_synthetic_astro_data():
    """Generates proxy orbital data if physical dataset is missing."""
    np.random.seed(42)
    samples = 2000
    df = pd.DataFrame({
        'Sun': np.random.uniform(1.0, 5.0, samples),
        'Mars': np.random.uniform(0.1, 1.0, samples),
        'Jup': np.random.uniform(0.5, 3.0, samples),
        'Sat': np.random.uniform(0.2, 2.0, samples),
        'Ura': np.random.uniform(0.1, 1.5, samples)
    })
    # Target: Sun/Mars (Dominant) + Jupiter/Saturn (Perturbation)
    df['Target'] = (df['Sun'] / df['Mars']) + np.sin(df['Jup'] * df['Sat'])
    return df

def execute_orbital_protocol():
    if os.path.exists("real_astro_data.csv"):
        logger.info("Loading physical observation dataset.")
        df = pd.read_csv("real_astro_data.csv")
    else:
        logger.info("Dataset not found. Generating synthetic orbital mechanics data.")
        df = generate_synthetic_astro_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.tensor(df[['Sun','Mars','Jup','Sat','Ura']].values, dtype=torch.float64).to(device)
    target = torch.tensor(df['Target'].values, dtype=torch.float64).view(-1, 1).to(device)
    cfg = OmegaConfig()

    # --- STAGE 1: Primary Gravitational Influence ---
    logger.info("Stage 1: Extracting primary orbital dynamics...")
    engine1 = OmegaEngine(inputs, target, cfg, device)
    best1 = engine1.run_stage("STAGE 1")
    
    with torch.no_grad():
        ones = torch.ones_like(target)
        pred1 = IronDome.execute(best1[0], inputs, best1[1].to(device), ones)
        residual = target - pred1 
    
    logger.info(f"Stage 1 Complete. Primary R2: {best1[3]:.4f}. Isolating residuals...")

    # --- STAGE 2: Secondary Perturbations ---
    logger.info("Stage 2: Extracting secondary planetary perturbations...")
    cfg.generations = 40
    cfg.parsimony = 0.001 
    
    engine2 = OmegaEngine(inputs, residual, cfg, device)
    best2 = engine2.run_stage("STAGE 2")
    
    # --- SYNTHESIS ---
    logger.info("Synthesizing multi-body dynamic model...")
    with torch.no_grad():
        pred2 = IronDome.execute(best2[0], inputs, best2[1].to(device), ones)
        final_pred = pred1 + pred2
        final_mse = torch.mean((final_pred - target)**2).item()
        final_r2 = 1.0 - (final_mse / (torch.var(target).item() + 1e-9))
    
    print("\n" + "="*50)
    print(f"Final Multi-Body Model R2: {final_r2:.5f}")
    print(f"Layer 1 (Primary): {translate_equation(best1[0], best1[1])}")
    print(f"Layer 2 (Residual): {translate_equation(best2[0], best2[1])}")
    print("="*50)
    
    plt.figure(figsize=(15, 6))
    plt.plot(target.cpu().numpy()[:200], 'k', alpha=0.5, label='Observation Data')
    plt.plot(final_pred.cpu().numpy()[:200], 'c--', linewidth=2, label='Unified OMEGA Model')
    plt.plot(pred1.cpu().numpy()[:200], 'r:', alpha=0.6, label='Stage 1 Base Model')
    plt.legend()
    plt.title(f"Orbital Dynamics Reconstruction | Global R2: {final_r2:.4f}")
    plt.xlabel("Time Steps")
    plt.ylabel("System State")
    plt.show()

def translate_equation(structure, constants):
    syms = {1:'Sun', 2:'Mars', 3:'Jup', 4:'Sat', 5:'Ura'}
    stack = []
    c_idx = 0
    try:
        for act in structure:
            if act <= 5: 
                stack.append(sp.Symbol(syms[act]))
            elif act == Op.CONST: 
                stack.append(round(constants[c_idx].item(), 3))
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
                stack.append(a / b)
            elif act == Op.SIN: 
                stack.append(sp.sin(stack.pop()))
            elif act == Op.COS: 
                stack.append(sp.cos(stack.pop()))
        return str(sp.simplify(stack[-1]))
    except: 
        return "Translation Error"

if __name__ == "__main__":
    execute_orbital_protocol()