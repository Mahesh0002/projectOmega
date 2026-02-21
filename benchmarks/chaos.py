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
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.float64)
torch.set_num_threads(os.cpu_count()) 

for handler in logging.root.handlers[:]: 
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', stream=sys.stdout, force=True)
logger = logging.getLogger("OMEGA_LORENZ")

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
    parsimony: float = 0.001 
    alchemist_steps: int = 50
    wolf_min_features: int = 1

class Civilization:
    def __init__(self):
        self.registry: Set[str] = set()
        self.grimoire_path = "omega_lorenz_registry.json"
        self.elites = [] 
        self.load_grimoire()

    def hash_structure(self, structure: List[int]) -> str:
        return hashlib.sha256(str(structure).encode()).hexdigest()

    def is_duplicate(self, structure: List[int]) -> bool:
        h = self.hash_structure(structure)
        if h in self.registry: 
            return True
        self.registry.add(h)
        return False

    def save_grimoire(self, best_law_str: str, score: float):
        entry = {"law": best_law_str, "score": score}
        self.elites.append(entry)
        self.elites.sort(key=lambda x: x['score'], reverse=True)
        self.elites = self.elites[:5]
        try:
            with open(self.grimoire_path, 'w') as f:
                json.dump(self.elites, f, indent=4)
        except Exception as e:
            logger.debug(f"Failed to save registry: {e}")

    def load_grimoire(self):
        if os.path.exists(self.grimoire_path):
            try:
                with open(self.grimoire_path, 'r') as f:
                    self.elites = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load registry: {e}")

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
            if sim > self.cfg.orthogonality_threshold: 
                return False
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
            try: 
                optimizer.step(closure)
            except: 
                pass
        
        with torch.no_grad():
            pred = IronDome.execute(structure, self.inputs, c, self.ones, use_oracle=False)
            variance = torch.var(self.target) + 1e-9
            r2 = 1.0 - (torch.mean((pred - self.target)**2) / variance).item()
        return c.detach(), r2, pred.detach()

    def run(self):
        logger.info("Starting Lorenz attractor discovery process.")
        elite_preds = [] 
        
        for gen in range(1, self.cfg.generations + 1):
            if self.shadow_counter >= self.cfg.patience_limit:
                if self.interventions < self.cfg.max_shadow_interventions:
                    logger.warning(f"Shadow intervention triggered at Gen {gen}.")
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
                    logger.info(f"Generation {gen} | New Best R2: {self.best[3]:.5f}")
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
            self.civ.save_grimoire(final_law, self.best[3])
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

def generate_lorenz_data():
    logger.info("Generating Lorenz system data...")
    
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    dt = 0.01
    steps = 3000
    
    xs, ys, zs = [], [], []
    x, y, z = 1.0, 1.0, 1.0
    
    for _ in range(steps):
        xs.append(x)
        ys.append(y)
        zs.append(z)
        
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    
    dx_true = sigma * (ys - xs)
    dx_noisy = dx_true + np.random.normal(0, np.std(dx_true)*0.01, size=len(dx_true))
    
    inputs = np.column_stack((xs, ys, zs))
    target = dx_noisy
    
    input_tensor = torch.tensor(inputs, dtype=torch.float64)
    target_tensor = torch.tensor(target, dtype=torch.float64).view(-1, 1)
    
    logger.info("Data generation complete. Target: dx/dt = 10(y - x)")
    return input_tensor, target_tensor

def visualize_discovery(inputs, target, best_structure, best_constants, device):
    print("Visualizing discovery results...")
    
    with torch.no_grad():
        ones = torch.ones((len(target), 1), dtype=torch.float64).to(device)
        c_tensor = best_constants.clone().detach().to(device)
        pred = IronDome.execute(best_structure, inputs, c_tensor, ones, use_oracle=False)
        
    X = inputs[:, 0].cpu().numpy()
    Y = inputs[:, 1].cpu().numpy()
    Z = inputs[:, 2].cpu().numpy()
    
    true_dx = target.cpu().numpy().flatten()
    pred_dx = pred.cpu().numpy().flatten()
    error = np.abs(true_dx - pred_dx)
    
    fig = plt.figure(figsize=(20, 8))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc = ax1.scatter(X, Y, Z, c=error, cmap='inferno', s=2, alpha=0.6)
    ax1.set_title("The Lorenz Attractor\nColor = OMEGA Error Magnitude", fontsize=14)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    plt.colorbar(sc, ax=ax1, label="Absolute Error |True - Pred|")
    
    ax2 = fig.add_subplot(1, 2, 2)
    zoom = slice(0, 200)
    time_steps = np.arange(len(true_dx))[zoom]
    
    ax2.plot(time_steps, true_dx[zoom], 'k-', linewidth=3, alpha=0.3, label="True Physics (dx/dt)")
    ax2.plot(time_steps, pred_dx[zoom], 'r--', linewidth=1.5, label="OMEGA Law")
    
    ax2.set_title("Derivative Tracking (First 200 Steps)", fontsize=14)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("dx/dt (Velocity of X)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    mse = np.mean(error**2)
    print("\nDiagnostics:")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Max Error Spike:    {np.max(error):.4f}")
    
    if mse < 1e-4:
        print("Verdict: Convergence achieved.")
    else:
        print("Verdict: Drift detected. Model did not fully converge.")

if __name__ == "__main__":
    try:
        inputs, target = generate_lorenz_data()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = inputs.to(device)
        target = target.to(device)
        
        cfg = OmegaConfig()
        
        logger.info("Initializing OMEGA Engine...")
        engine = OmegaEngine(inputs, target, cfg, device)
        best, law = engine.run()
        
        print("\n" + "="*50)
        print("Final Report (Equation 1: dx/dt)")
        print(f"Final R2: {best[3]:.5f}")
        print(f"Discovered Law: {law}")
        print(f"Ground Truth:   10.0*y - 10.0*x")
        print("="*50 + "\n")
        
        if best is not None:
            visualize_discovery(inputs, target, best[0], best[1], device)
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")