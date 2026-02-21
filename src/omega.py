"""
Project OMEGA: Industrial Engine Core

This module contains the centralized architecture for the Sovereign Discovery Engine.
It implements the Transformer/LSTM Explorer, the L-BFGS Alchemist with Smart Initialization, 
the Iron Dome differentiable physics kernel, and the iterative Protocol Zero with 
Global Refinement.

Designed to be imported by benchmark scripts.
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
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# --- 0. SYSTEM CONFIGURATION ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.float64)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | [OMEGA_CORE] | %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- 1. GLOBAL OPERATORS ---
class Op(IntEnum):
    PAD = 0
    VAR_X1 = 1; VAR_X2 = 2; VAR_X3 = 3; VAR_X4 = 4; VAR_X5 = 5
    CONST = 6
    ADD = 7; SUB = 8; MUL = 9; DIV = 10
    SIN = 11; COS = 12; EXP = 13; SQ = 14

@dataclass
class OmegaConfig:
    action_dim: int = 15
    embedding_dim: int = 64
    hidden_dim: int = 128
    
    pop_size: int = 1000
    generations: int = 40
    max_seq_len: int = 16
    
    orthogonality_limit: float = 0.95
    parsimony_penalty: float = 0.005
    alchemist_steps: int = 30
    
    # Smart Init
    use_smart_init: bool = True
    
    # MCTS / Search specific
    temperature: float = 1.2

# --- 2. IRON DOME: PHYSICS KERNEL ---

class IronDome:
    EPS = 1e-8
    
    @staticmethod
    def safe_div(a, b):
        return a / (b + IronDome.EPS * torch.sign(b) + IronDome.EPS)
        
    @staticmethod
    def execute(structure: List[int], input_matrix: torch.Tensor, constants: torch.Tensor, ones_buffer: torch.Tensor) -> torch.Tensor:
        """Executes a discrete symbolic tree as a fully differentiable PyTorch graph."""
        stack = []
        c_idx = 0
        try:
            for act in structure:
                if 1 <= act <= 5: 
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
                    stack.append(IronDome.safe_div(a, b))
                elif act == Op.SIN: 
                    stack.append(torch.sin(stack.pop()))
                elif act == Op.COS: 
                    stack.append(torch.cos(stack.pop()))
                elif act == Op.EXP: 
                    stack.append(torch.exp(torch.clamp(stack.pop(), -10, 15)))
                elif act == Op.SQ:  
                    stack.append(torch.pow(torch.clamp(stack.pop(), -1e3, 1e3), 2))
                    
            return stack[-1] if stack else torch.zeros_like(ones_buffer)
        except Exception:
            return torch.zeros_like(ones_buffer) + 1e9

    @staticmethod
    def execute_multi_layer(structures: List[List[int]], input_matrix: torch.Tensor, global_constants: torch.Tensor, ones_buffer: torch.Tensor) -> torch.Tensor:
        """Executes multiple isolated layers and sums them (Used for Global Refinement)."""
        total_pred = torch.zeros_like(ones_buffer)
        c_offset = 0
        
        for struct in structures:
            num_c = struct.count(Op.CONST)
            layer_constants = global_constants[c_offset : c_offset + num_c]
            pred = IronDome.execute(struct, input_matrix, layer_constants, ones_buffer)
            total_pred += pred
            c_offset += num_c
            
        return total_pred

# --- 3. THE ALCHEMIST: OPTIMIZER ---
class TheAlchemist:
    def __init__(self, cfg: OmegaConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def optimize(self, structure: List[int], inputs: torch.Tensor, target: torch.Tensor, ones_buffer: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor]:
        num_c = structure.count(Op.CONST)
        
        # Smart Initialization Logic
        if num_c > 0:
            if self.cfg.use_smart_init:
                mag = torch.mean(torch.abs(target)).item()
                init_val = mag if mag > 1e-5 else 1.0
            else:
                init_val = 1.0
                
            c = torch.full((num_c,), init_val, requires_grad=True, device=self.device, dtype=torch.float64)
            optimizer = optim.LBFGS([c], lr=1.0, max_iter=self.cfg.alchemist_steps, line_search_fn="strong_wolfe")
            
            def closure():
                optimizer.zero_grad()
                pred = IronDome.execute(structure, inputs, c, ones_buffer)
                loss = torch.mean((pred - target)**2)
                loss.backward()
                return loss
                
            try: 
                optimizer.step(closure)
            except Exception: 
                pass
        else:
            c = torch.tensor([], device=self.device, dtype=torch.float64)

        with torch.no_grad():
            pred = IronDome.execute(structure, inputs, c, ones_buffer)
            variance = torch.var(target).item() + 1e-9
            mse = torch.mean((pred - target)**2).item()
            r2 = 1.0 - (mse / variance)
            
        return c.detach(), r2, pred.detach()

# --- 4. THE EXPLORER: NEURAL SEARCH ---
class ExplorerAgent(nn.Module):
    def __init__(self, cfg: OmegaConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.action_dim, cfg.embedding_dim)
        self.lstm = nn.LSTM(cfg.embedding_dim, cfg.hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.fc(out), hidden

# --- 5. THE SOVEREIGN ENGINE ---
class OmegaIndustrial:
    def __init__(self, inputs: torch.Tensor, target: torch.Tensor, cfg: OmegaConfig, device: torch.device):
        self.inputs = inputs
        self.original_target = target
        self.cfg = cfg
        self.device = device
        self.ones = torch.ones((len(target), 1), dtype=torch.float64).to(device)
        
        self.explorer = ExplorerAgent(cfg).to(device).double()
        self.opt = optim.Adam(self.explorer.parameters(), lr=0.005)
        self.alchemist = TheAlchemist(cfg, device)
        
        self.discovered_layers: List[Dict] = []

    def check_orthogonality(self, candidate_pred: torch.Tensor, elite_preds: List[torch.Tensor]) -> bool:
        if not elite_preds: return True
        cand_vec = candidate_pred.flatten()
        cand_norm = torch.norm(cand_vec)
        if cand_norm < 1e-9: return False
        
        for elite_vec in elite_preds:
            sim = torch.dot(cand_vec, elite_vec) / (cand_norm * torch.norm(elite_vec) + 1e-9)
            if sim > self.cfg.orthogonality_limit: 
                return False
        return True

    def run_layer_search(self, current_target: torch.Tensor, layer_idx: int) -> Optional[Dict]:
        """Runs the search and optimization loop for a single layer."""
        logger.info(f"Initiating Search for Layer {layer_idx}...")
        best_layer = None
        elite_preds = []
        
        for gen in range(1, self.cfg.generations + 1):
            seqs = torch.zeros(self.cfg.pop_size, self.cfg.max_seq_len, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                inp = torch.zeros(self.cfg.pop_size, 1, dtype=torch.long).to(self.device)
                h = None
                for t in range(self.cfg.max_seq_len):
                    logits, h = self.explorer(inp, h)
                    probs = torch.softmax(logits[:, -1, :] / self.cfg.temperature, dim=-1)
                    inp = torch.multinomial(probs, 1)
                    seqs[:, t] = inp.squeeze()
            
            batch = []
            cpu_seqs = seqs.cpu().numpy()
            
            for i in range(self.cfg.pop_size):
                s = [x for x in cpu_seqs[i] if 1 <= x < self.cfg.action_dim]
                if not s: continue
                
                c, r2, pred = self.alchemist.optimize(s, self.inputs, current_target, self.ones)
                if not self.check_orthogonality(pred, elite_preds): 
                    continue
                
                fitness = r2 - (len(s) * self.cfg.parsimony_penalty)
                batch.append((s, c, fitness, r2, pred))
                
            batch.sort(key=lambda x: x[2], reverse=True)
            
            if batch:
                current_best = batch[0]
                if best_layer is None or current_best[2] > best_layer['fitness']:
                    best_layer = {
                        'struct': current_best[0],
                        'constants': current_best[1],
                        'fitness': current_best[2],
                        'r2': current_best[3],
                        'pred': current_best[4]
                    }
                    elite_preds.append(current_best[4])
                    if len(elite_preds) > 5: elite_preds.pop(0)
                    logger.info(f"Layer {layer_idx} | Gen {gen} | R2: {best_layer['r2']:.4f}")
            
            if batch:
                self.opt.zero_grad()
                loss = 0
                for s, _, _, _, _ in batch[:50]:
                    t_seq = torch.tensor([0] + s + [0], device=self.device)
                    l, _ = self.explorer(t_seq[:-1].unsqueeze(0))
                    loss += nn.functional.cross_entropy(l.view(-1, self.cfg.action_dim), t_seq[1:])
                loss.backward()
                self.opt.step()
                
        return best_layer

    def run_protocol_zero(self, max_layers: int = 2) -> None:
        """Executes the iterative residual peeling algorithm."""
        logger.info("Executing Protocol Zero (Onion Peeling)...")
        current_target = self.original_target.clone()
        
        for i in range(1, max_layers + 1):
            layer_res = self.run_layer_search(current_target, i)
            
            if layer_res is None or layer_res['r2'] < 0.05:
                logger.info("Signal-to-noise ratio too low. Terminating peeling process.")
                break
                
            self.discovered_layers.append(layer_res)
            current_target = current_target - layer_res['pred']
            
            if layer_res['r2'] > 0.99:
                logger.info("Near-perfect convergence reached. Peeling complete.")
                break

    def global_refinement(self) -> Tuple[float, List[torch.Tensor]]:
        """Harmonizes all layers simultaneously using a global L-BFGS pass."""
        logger.info("Initiating Global Refinement (Wet-on-Wet Optimization)...")
        
        if not self.discovered_layers:
            return 0.0, []
            
        structures = [layer['struct'] for layer in self.discovered_layers]
        initial_constants = []
        for layer in self.discovered_layers:
            initial_constants.extend(layer['constants'].tolist())
            
        if not initial_constants:
            return self.discovered_layers[0]['r2'], []
            
        global_c = torch.tensor(initial_constants, requires_grad=True, device=self.device, dtype=torch.float64)
        optimizer = optim.LBFGS([global_c], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            total_pred = IronDome.execute_multi_layer(structures, self.inputs, global_c, self.ones)
            loss = torch.mean((total_pred - self.original_target)**2)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        # Recalculate Final Global R2
        with torch.no_grad():
            final_pred = IronDome.execute_multi_layer(structures, self.inputs, global_c, self.ones)
            variance = torch.var(self.original_target).item() + 1e-9
            mse = torch.mean((final_pred - self.original_target)**2).item()
            final_r2 = 1.0 - (mse / variance)
            
        logger.info(f"Global Refinement Complete. Unified R2: {final_r2:.5f}")
        
        # Package updated constants back to layers
        refined_constants = []
        c_offset = 0
        for i, struct in enumerate(structures):
            num_c = struct.count(Op.CONST)
            layer_c = global_c[c_offset : c_offset + num_c].detach()
            refined_constants.append(layer_c)
            self.discovered_layers[i]['constants'] = layer_c
            c_offset += num_c
            
        return final_r2, refined_constants

# --- 6. UTILITIES ---
def translate_to_sympy(structure: List[int], constants: torch.Tensor, var_names: List[str] = None):
    """Converts a discrete numeric structure into a readable mathematical string."""
    if var_names is None:
        var_names = ['x1', 'x2', 'x3', 'x4', 'x5']
        
    stack = []
    c_idx = 0
    try:
        for act in structure:
            if 1 <= act <= 5: 
                stack.append(sp.Symbol(var_names[act-1]))
            elif act == Op.CONST: 
                stack.append(round(constants[c_idx].item(), 4))
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
            elif act == Op.EXP: 
                stack.append(sp.exp(stack.pop()))
            elif act == Op.SQ:  
                stack.append(stack.pop()**2)
                
        return str(sp.simplify(stack[-1]))
    except Exception: 
        return "Translation Error"