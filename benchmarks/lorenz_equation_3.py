"""
Project OMEGA: Iterative Discovery (Lorenz Equation 3)



This module demonstrates the "Protocol Zero" (Iterative Peeling) technique 
on the third differential equation of the Lorenz chaotic system (dz/dt). 
It first extracts the dominant nonlinear interaction and then trains a 
secondary search agent on the residual variance to recover the full law.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
import logging
import hashlib
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Set

# --- 0. SYSTEM CONFIGURATION ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.float64)
torch.set_num_threads(os.cpu_count()) 

for handler in logging.root.handlers[:]: 
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', stream=sys.stdout, force=True)
logger = logging.getLogger("OMEGA_ITERATIVE")

# --- 1. OPERATORS ---
class Op(IntEnum):
    VAR_X    = 1 
    VAR_Y    = 2 
    VAR_Z    = 3 
    CONST    = 4 
    ADD      = 5 
    SUB      = 6 
    MUL      = 7 

@dataclass
class OmegaConfig:
    action_dim: int = 10
    embedding_dim: int = 64
    hidden_dim: int = 128
    
    min_pop: int = 200
    max_pop: int = 1000
    start_pop: int = 500
    generations: int = 50 
    
    patience_limit: int = 8          
    chaos_threshold: float = 0.15    
    max_shadow_interventions: int = 5 
    
    orthogonality_threshold: float = 0.98
    parsimony: float = 0.005 
    alchemist_steps: int = 50
    wolf_min_features: int = 1

# --- 2. REGISTRY & PHYSICS KERNEL ---
class Civilization:
    def __init__(self):
        self.registry: Set[str] = set()
        self.grimoire_path = "omega_iterative_registry.json"
        self.elites = [] 

    def hash_structure(self, structure: List[int]) -> str:
        return hashlib.sha256(str(structure).encode()).hexdigest()

    def is_duplicate(self, structure: List[int]) -> bool:
        h = self.hash_structure(structure)
        if h in self.registry: 
            return True
        self.registry.add(h)
        return False

class IronDome:
    ORACLE_CACHE = {} 
    
    @staticmethod
    def execute(structure, input_matrix, constants, ones_buffer, use_oracle=True):
        struct_key = str(structure) + str(constants)
        if use_oracle and struct_key in IronDome.ORACLE_CACHE:
            return IronDome.ORACLE_CACHE[struct_key]

        stack = []
        c_idx = 0
        try:
            for act in structure:
                if act <= 3: 
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
            
            result = stack[-1] if stack else torch.zeros_like(ones_buffer)
            if use_oracle and len(IronDome.ORACLE_CACHE) < 20000:
                IronDome.ORACLE_CACHE[struct_key] = result
            return result
        except: 
            return torch.zeros_like(ones_buffer)

# --- 3. ARCHITECTURE ---
class Explorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(len(Op)+2, cfg.embedding_dim)
        self.lstm = nn.LSTM(cfg.embedding_dim, cfg.hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(cfg.hidden_dim, len(Op)+2)
        
    def forward(self, x, h=None):
        out, h_n = self.lstm(self.embed(x), h)
        return self.head(out), h_n
    
    def mutate(self, noise_std=0.1):
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)
        logger.info(f"Mutation applied: Injected {noise_std} Gaussian Noise.")

class OmegaEngine:
    def __init__(self, inputs, target, cfg, device):
        self.inputs = inputs
        self.target = target
        self.cfg = cfg
        self.device = device
        self.ones = torch.ones((len(target), 1), dtype=torch.float64).to(device)
        self.civ = Civilization()
        self.explorer = Explorer(cfg).to(device).double()
        self.opt = optim.Adam(self.explorer.parameters(), lr=0.005)
        self.best = None
        self.shadow_counter = 0
        self.interventions = 0

    def check_orthogonality(self, candidate_pred, elite_preds):
        if not elite_preds: return True
        cand_vec = candidate_pred.detach().flatten()
        for elite_vec in elite_preds:
            elite_vec_flat = elite_vec.flatten()
            dot = torch.dot(cand_vec, elite_vec_flat)
            sim = dot / (torch.norm(cand_vec) * torch.norm(elite_vec_flat) + 1e-9)
            if sim > self.cfg.orthogonality_threshold: return False
        return True

    def optimize(self, structure):
        num_c = structure.count(Op.CONST)
        c = torch.ones(num_c, requires_grad=True, device=self.device, dtype=torch.float64)
        if num_c > 0:
            optimizer = optim.LBFGS([c], lr=1, max_iter=self.cfg.alchemist_steps, line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                pred = IronDome.execute(structure, self.inputs, c, self.ones, use_oracle=False)
                loss = torch.mean((pred - self.target)**2)
                loss.backward()
                return loss
            try: optimizer.step(closure)
            except: pass
        
        with torch.no_grad():
            pred = IronDome.execute(structure, self.inputs, c, self.ones, use_oracle=False)
            variance = torch.var(self.target) + 1e-9
            r2 = 1.0 - (torch.mean((pred - self.target)**2) / variance).item()
        return c.detach(), r2, pred.detach()

    def run_civilization(self):
        elite_preds = [] 
        
        for gen in range(1, self.cfg.generations + 1):
            if self.shadow_counter >= self.cfg.patience_limit:
                if self.interventions < self.cfg.max_shadow_interventions:
                    logger.warning(f"Intervention triggered at Gen {gen} due to stagnation.")
                    self.explorer.mutate(self.cfg.chaos_threshold)
                    self.shadow_counter = 0
                    self.interventions += 1
                else: 
                    break

            pop_size = self.cfg.start_pop
            seqs = torch.zeros(pop_size, 12, dtype=torch.long).to(self.device)
            with torch.no_grad():
                inp = torch.zeros(pop_size, 1, dtype=torch.long).to(self.device)
                h = None
                for t in range(12):
                    logits, h = self.explorer(inp, h)
                    temp = 1.0 + (0.5 * (self.interventions + 1)) 
                    probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
                    inp = torch.multinomial(probs, 1)
                    seqs[:, t] = inp.squeeze()
            
            batch = []
            cpu_seqs = seqs.cpu().numpy()
            
            for i in range(pop_size):
                s = [x for x in cpu_seqs[i] if 1 <= x <= len(Op)]
                if not s: continue
                if self.civ.is_duplicate(s): continue
                if len([o for o in s if o <= 3]) < self.cfg.wolf_min_features: continue

                c, r2, pred = self.optimize(s)
                if not self.check_orthogonality(pred, elite_preds): continue
                
                fitness = r2 - (len(s) * self.cfg.parsimony)
                batch.append((s, c, fitness, r2, pred))

            batch.sort(key=lambda x: x[2], reverse=True)
            
            if batch:
                current_best = batch[0]
                if self.best is None or current_best[2] > self.best[2]:
                    self.best = current_best
                    elite_preds.append(current_best[4].detach().flatten()) 
                    if len(elite_preds) > 10: elite_preds.pop(0)
                    logger.info(f"Generation {gen} | New Best R2={self.best[3]:.5f}")
                    self.shadow_counter = 0 
                else: 
                    self.shadow_counter += 1
                
                self.opt.zero_grad()
                loss = 0
                for s, _, _, _, _ in batch[:50]: 
                    t_seq = torch.tensor([0]+s+[0], device=self.device)
                    l, _ = self.explorer(t_seq[:-1].unsqueeze(0))
                    loss += nn.functional.cross_entropy(l.view(-1, len(Op)+2), t_seq[1:])
                loss.backward()
                self.opt.step()
            else: 
                self.shadow_counter += 1
                
        if self.best:
            final_law = safe_translate(self.best[0], self.best[1])
            return self.best, final_law
        return None, "Null"

def safe_translate(structure, constants):
    syms = {1:'x', 2:'y', 3:'z'}
    stack = []
    c_idx = 0
    try:
        for act in structure:
            if act <= 3: 
                stack.append(sp.Symbol(syms[act]))
            elif act == Op.CONST: 
                val = round(constants[c_idx].item(), 4)
                stack.append(val)
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
        return str(sp.simplify(stack[-1]))
    except: 
        return "Translation Error"

# --- 4. DATA GENERATOR ---
def generate_lorenz_z_data():
    logger.info("Generating Lorenz Equation 3 data (dz/dt)...")
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    dt = 0.01
    steps = 3000
    
    xs, ys, zs = [], [], []
    x, y, z = 1.0, 1.0, 1.0
    
    for _ in range(steps):
        xs.append(x); ys.append(y); zs.append(z)
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z 
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    
    dz_true = xs * ys - beta * zs
    dz_noisy = dz_true + np.random.normal(0, np.std(dz_true)*0.01, size=len(dz_true))
    inputs = np.column_stack((xs, ys, zs))
    
    return torch.tensor(inputs, dtype=torch.float64), torch.tensor(dz_noisy, dtype=torch.float64).view(-1, 1)

# --- 5. ITERATIVE DISCOVERY (PROTOCOL ZERO) ---
def run_iterative_discovery(inputs, target, cfg, device):
    print("\nInitiating Iterative Discovery Protocol...")
    
    # PASS 1: Dominant Signal
    logger.info("Pass 1: Identifying dominant signal components...")
    engine1 = OmegaEngine(inputs, target, cfg, device)
    best1, law1 = engine1.run_civilization()
    
    if best1 is None: 
        return "Failure"
    
    with torch.no_grad():
        ones = torch.ones((len(target), 1), dtype=torch.float64).to(device)
        pred1 = IronDome.execute(best1[0], inputs, best1[1].to(device), ones, use_oracle=False)
        residual = target - pred1
        
        resid_var = torch.var(residual).item()
        target_var = torch.var(target).item()
        
        print("\nPass 1 Complete.")
        print(f"  >> Extracted Term: {law1}")
        print(f"  >> Accuracy (R2):  {best1[3]:.5f}")
        print(f"  >> Variance Left:  {resid_var:.4f} (Original: {target_var:.4f})")
    
    if best1[3] > 0.999: 
        return law1

    # PASS 2: Residual Target
    print("\nPass 2: Extracting residual dynamics...")
    cfg.generations = 30 
    cfg.start_pop = 1000
    
    engine2 = OmegaEngine(inputs, residual, cfg, device)
    best2, law2 = engine2.run_civilization()
    
    if best2:
        print("\nPass 2 Complete.")
        print(f"  >> Extracted Residual: {law2}")
        print(f"  >> Residual Fit (R2):  {best2[3]:.5f}")
        return f"({law1}) + ({law2})"
    else:
        return law1

# --- 6. VISUALIZATION ---
def visualize_discovery(inputs, target, equation_str, device):
    print(f"\nVisualizing Extracted Law: {equation_str}")
    
    x, y, z = sp.symbols('x y z')
    try:
        sym_expr = sp.sympify(equation_str)
        predict_func = sp.lambdify((x, y, z), sym_expr, "numpy")
    except Exception as e:
        print(f"Error parsing equation string: {e}")
        return

    X_np = inputs[:, 0].cpu().numpy()
    Y_np = inputs[:, 1].cpu().numpy()
    Z_np = inputs[:, 2].cpu().numpy()
    true_val = target.cpu().numpy().flatten()
    
    try:
        pred_val = predict_func(X_np, Y_np, Z_np)
        if np.isscalar(pred_val):
            pred_val = np.full_like(true_val, pred_val)
    except Exception as e:
        print(f"Math execution failed during plotting: {e}")
        return

    error = np.abs(true_val - pred_val)
    mse = np.mean(error**2)
    r2 = 1.0 - (np.sum((true_val - pred_val)**2) / np.sum((true_val - np.mean(true_val))**2))

    fig = plt.figure(figsize=(20, 8))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(X_np, Y_np, Z_np, c=error, cmap='viridis', s=3, alpha=0.7)
    ax1.set_title("Lorenz Attractor\nColor = Error Magnitude", fontsize=14)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=30, azim=45)
    plt.colorbar(sc, ax=ax1, label="|True - Pred|")
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(true_val[::5], pred_val[::5], alpha=0.3, s=10, c='purple', label="Model Prediction")
    
    min_v, max_v = np.min(true_val), np.max(true_val)
    ax2.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2, label="Perfect Alignment")
    
    ax2.set_title(f"Ground Truth vs Extracted Law (R² = {r2:.5f})", fontsize=14)
    ax2.set_xlabel("Ground Truth Target")
    ax2.set_ylabel("Extracted Equation Prediction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        inputs, target = generate_lorenz_z_data()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = inputs.to(device)
        target = target.to(device)
        
        cfg = OmegaConfig()
        final_law = run_iterative_discovery(inputs, target, cfg, device)
        
        print("\n" + "="*50)
        print("Final Report")
        print(f"Combined Law: {final_law}")
        print(f"Ground Truth: x*y - 2.6667*z")
        print("="*50)
        
        visualize_discovery(inputs, target, final_law, device)
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")