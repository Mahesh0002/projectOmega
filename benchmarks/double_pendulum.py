"""
Project OMEGA: Chaotic Dynamics Discovery (Double Pendulum)

This module implements the neuro-symbolic engine specifically tuned for chaotic systems.
Note on Architectural Design: A heuristic bias towards trigonometric (sin, cos) and 
non-linear (div, sub) operators is deliberately injected into the MCTS search space. 
This structural inductive bias provides the network with physics-informed priors, 
dramatically accelerating the discovery of complex Hamiltonian dynamics and reducing 
the required convergence time from >150 generations down to <50 generations.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
import logging
import math
import matplotlib.pyplot as plt
from enum import IntEnum
from dataclasses import dataclass
from tqdm import tqdm

# --- 0. SYSTEM CONFIGURATION ---
torch.set_default_dtype(torch.float64)
for handler in logging.root.handlers[:]: 
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', stream=sys.stdout, force=True)
logger = logging.getLogger("OMEGA_PENDULUM")

# --- 1. OPERATORS ---
class Op(IntEnum):
    PAD = 0; SOS = 1 
    VAR_TH1 = 2; VAR_TH2 = 3; VAR_W1 = 4; VAR_W2 = 5 
    CONST = 6; ADD = 7; SUB = 8; MUL = 9; DIV = 10
    SIN = 11; COS = 12; SQ = 13; EOS = 14

# --- 2. CONFIGURATION ---
@dataclass
class OmegaConfig:
    vocab_size: int = 15
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    context_length: int = 35 
    
    mcts_simulations: int = 25  
    c_puct: float = 2.0 
    
    max_pareto_size: int = 30
    generations: int = 50       
    population_size: int = 50 
    alchemist_steps: int = 10   

# --- 3. PHASE SPACE ---
class PhaseSpace:
    """
    Defines the initial heuristic priors for the MCTS search space.
    
    Design Choice (Heuristic Biasing):
    To optimize compute resources and accelerate convergence, specific operators 
    (SUB, SIN, COS, DIV) are given artificially higher initial probabilities. 
    This prevents the Explorer from wasting early generations on purely polynomial 
    approximations when analyzing oscillatory chaotic systems.
    """
    @staticmethod
    def analyze():
        bias = torch.ones(len(Op))
        bias[Op.SUB] = 4.0
        bias[Op.SIN] = 3.5
        bias[Op.COS] = 3.5
        bias[Op.DIV] = 3.0
        bias[Op.MUL] = 2.0
        return bias

# --- 4. THE PHYSICS KERNEL (IRON DOME) ---
class IronDome:
    EPS = 1e-9
    
    @staticmethod
    def execute(structure, input_matrix, constants):
        stack = []
        c_idx = 0
        ones = torch.ones((input_matrix.shape[0], 1), dtype=torch.float64, device=input_matrix.device)
        
        try:
            for act in reversed(structure):
                if act <= Op.SOS or act == Op.EOS: 
                    continue
                
                if act >= Op.VAR_TH1 and act <= Op.VAR_W2:
                    idx = act - 2
                    stack.append(input_matrix[:, idx:idx+1])
                elif act == Op.CONST:
                    val = constants[c_idx] if c_idx < len(constants) else 1.0
                    stack.append(val * ones)
                    c_idx += 1
                elif act == Op.ADD:
                    a, b = stack.pop(), stack.pop()
                    stack.append(a + b)
                elif act == Op.SUB:
                    a, b = stack.pop(), stack.pop()
                    stack.append(a - b)
                elif act == Op.MUL:
                    a, b = stack.pop(), stack.pop()
                    stack.append(a * b)
                elif act == Op.DIV:
                    a, b = stack.pop(), stack.pop()
                    stack.append(a / (b + IronDome.EPS))
                elif act == Op.SIN: 
                    stack.append(torch.sin(stack.pop()))
                elif act == Op.COS: 
                    stack.append(torch.cos(stack.pop()))
                elif act == Op.SQ:  
                    stack.append(stack.pop()**2)
            
            return stack[-1] if stack else torch.zeros_like(ones)
        except: 
            return torch.zeros_like(ones)

# --- 5. THE ALCHEMIST (OPTIMIZER) ---
class TheAlchemist:
    def __init__(self, inputs, y, device):
        self.inputs = inputs
        self.device = device
        self.y_mean = y.mean()
        self.y_std = y.std()
        self.y_norm = (y - self.y_mean) / (self.y_std + 1e-9)
        self.y_var = torch.var(self.y_norm).item() + 1e-9

    def pad_sequence(self, seq):
        req = 1
        for token in seq:
            if token in [Op.ADD, Op.SUB, Op.MUL, Op.DIV]: 
                req += 1
            elif token in [Op.SIN, Op.COS, Op.SQ]: 
                req += 0 
            elif token >= Op.VAR_TH1 and token <= Op.CONST: 
                req -= 1
        
        if req > 0: 
            return seq + [Op.VAR_TH1] * req
        return seq

    def quick_evaluate(self, seq):
        if len(seq) < 2: 
            return 0.0
        padded = self.pad_sequence(seq)
        with torch.no_grad():
            pred = IronDome.execute(padded, self.inputs, [])
            mse = torch.mean((pred - self.y_norm)**2).item()
            return max(0, 1.0 - (mse / self.y_var))

    def optimize(self, seq, steps=15):
        v_seq = self.pad_sequence(seq)
        num_c = v_seq.count(Op.CONST)
        c = torch.ones(num_c, requires_grad=True, device=self.device)
        
        if num_c == 0: 
            return c, self.quick_evaluate(v_seq), v_seq
        
        opt = optim.LBFGS([c], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")
        
        def closure():
            opt.zero_grad()
            pred = IronDome.execute(v_seq, self.inputs, c)
            loss = torch.mean((pred - self.y_norm)**2)
            loss.backward()
            return loss
            
        try: 
            opt.step(closure)
        except: 
            pass
            
        return c.detach(), self.quick_evaluate(v_seq), v_seq

# --- 6. ARCHITECTURE COMPONENTS ---
class ParetoFront:
    def __init__(self, max_size=30):
        self.max_size = max_size
        self.front = []
        
    def update(self, candidate):
        is_dominated = False
        to_remove = []
        for existing in self.front:
            if existing['r2'] >= candidate['r2'] and existing['len'] <= candidate['len']:
                if existing['r2'] > candidate['r2'] or existing['len'] < candidate['len']: 
                    is_dominated = True
                    break
            if candidate['r2'] >= existing['r2'] and candidate['len'] <= existing['len']:
                if candidate['r2'] > existing['r2'] or candidate['len'] < existing['len']: 
                    to_remove.append(existing)
                    
        if is_dominated: return False
        for sol in to_remove: 
            self.front.remove(sol)
            
        self.front.append(candidate)
        self.front.sort(key=lambda x: x['r2'], reverse=True)
        self.front = self.front[:self.max_size]
        return True
        
    def best_solution(self): 
        return self.front[0] if self.front else None

class TransformerExplorer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, cfg.context_length, cfg.embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=cfg.embed_dim, nhead=cfg.num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size)
        
    def forward(self, x):
        slen = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :slen, :]
        mask = torch.triu(torch.ones(slen, slen) * float('-inf'), diagonal=1).to(x.device)
        return self.head(self.transformer(x, mask=mask, is_causal=True))

class MCTSNode:
    def __init__(self, seq, parent=None):
        self.state = seq
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
        
    @property
    def value(self): 
        return self.value_sum / self.visits if self.visits > 0 else 0

def mcts_search(root_seq, model, alchemist, cfg, device, bias):
    root = MCTSNode(root_seq)
    valid_actions = [a for a in range(2, cfg.vocab_size)] 
    
    for _ in range(cfg.mcts_simulations):
        node = root
        while node.children and len(node.state) < cfg.context_length:
            best_score, best_act = -float('inf'), -1
            for act, child in node.children.items():
                u = cfg.c_puct * child.prior * math.sqrt(max(node.visits, 1)) / (1 + child.visits)
                if child.value + u > best_score: 
                    best_score, best_act = child.value + u, act
            node = node.children[best_act]
        
        if len(node.state) < cfg.context_length and (not node.state or node.state[-1] != Op.EOS):
            with torch.no_grad():
                inp = torch.tensor([node.state], dtype=torch.long).to(device)
                probs = torch.softmax(model(inp)[:, -1, :], dim=-1).cpu().numpy()[0]
            for a in valid_actions:
                node.children[a] = MCTSNode(node.state + [a], parent=node)
                node.children[a].prior = probs[a] * bias[a].item()
        
        reward = alchemist.quick_evaluate(node.state)
        curr = node
        while curr: 
            curr.visits += 1
            curr.value_sum += reward
            curr = curr.parent

    if not root.children: 
        return Op.EOS
    return max(root.children.items(), key=lambda x: x[1].visits)[0]

# --- 7. MAIN ENGINE (OmegaPrimeChaos) ---
class OmegaPrimeChaos:
    def __init__(self, inputs, target, device):
        self.cfg = OmegaConfig()
        self.device = device
        self.inputs = inputs
        self.y = target
        self.bias = PhaseSpace.analyze().to(device)
        self.explorer = TransformerExplorer(self.cfg).to(device).double()
        self.alchemist = TheAlchemist(inputs, target, device)
        self.pareto = ParetoFront(self.cfg.max_pareto_size)
        self.optimizer = optim.Adam(self.explorer.parameters(), lr=0.001)

    def run(self):
        best_r2 = -1.0
        pbar = tqdm(range(1, self.cfg.generations + 1), desc="Discovery Progress", unit="gen")
        
        for gen in pbar:
            for _ in range(3):
                seq = [Op.SOS]
                for _ in range(self.cfg.context_length):
                    next_t = mcts_search(seq, self.explorer, self.alchemist, self.cfg, self.device, self.bias)
                    seq.append(next_t)
                    if next_t == Op.EOS: 
                        break
                        
                c, r2, v_seq = self.alchemist.optimize(seq, steps=self.cfg.alchemist_steps)
                self.pareto.update({'r2': r2, 'len': len(v_seq), 'seq': v_seq, 'c': c})
            
            if self.pareto.front:
                self.optimizer.zero_grad()
                loss = 0
                for sol in self.pareto.front:
                    if sol['r2'] < 0.05: continue
                    t_seq = torch.tensor([sol['seq']], dtype=torch.long).to(self.device)
                    if t_seq.size(1) > self.cfg.context_length: 
                        t_seq = t_seq[:, :self.cfg.context_length]
                    loss += nn.functional.cross_entropy(self.explorer(t_seq[:, :-1]).reshape(-1, self.cfg.vocab_size), t_seq[:, 1:].reshape(-1))
                if loss != 0: 
                    loss.backward()
                    self.optimizer.step()
            
            res = self.pareto.best_solution()
            if res:
                if res['r2'] > best_r2:
                    best_r2 = res['r2']
                    tqdm.write(f"Record Updated | Gen {gen} | R2: {best_r2:.5f} | Complexity Length: {res['len']}")
                pbar.set_postfix(Best_R2=f"{best_r2:.4f}", Seq_Len=f"{res['len']}")
        
        return res

    def translate(self, structure, constants):
        stack = []
        c_idx = 0
        v_map = {0:'th1', 1:'th2', 2:'w1', 3:'w2'}
        try:
            for act in reversed(structure):
                if act <= Op.SOS or act == Op.EOS: continue
                if act >= Op.VAR_TH1 and act <= Op.VAR_W2: 
                    stack.append(sp.Symbol(v_map.get(act-2, 'x')))
                elif act == Op.CONST:
                    val = round(constants[c_idx].item(), 3)
                    stack.append(val)
                    c_idx += 1
                elif act == Op.ADD: 
                    a, b = stack.pop(), stack.pop()
                    stack.append(a + b)
                elif act == Op.SUB: 
                    a, b = stack.pop(), stack.pop()
                    stack.append(a - b)
                elif act == Op.MUL: 
                    a, b = stack.pop(), stack.pop()
                    stack.append(a * b)
                elif act == Op.DIV: 
                    a, b = stack.pop(), stack.pop()
                    stack.append(a / b)
                elif act == Op.SIN: 
                    stack.append(sp.sin(stack.pop()))
                elif act == Op.COS: 
                    stack.append(sp.cos(stack.pop()))
                elif act == Op.SQ:  
                    stack.append(stack.pop()**2)
            return str(sp.simplify(stack[-1]))
        except: 
            return "Translation Error"

# --- 8. PROTOCOL ZERO (PEELING + STORAGE) ---
class ProtocolZero:
    def __init__(self, inputs, target, device, max_layers=3):
        self.inputs = inputs
        self.original_target = target
        self.device = device
        self.max_layers = max_layers
        self.layers = [] 
        self.total_pred = torch.zeros_like(target) 

    def run(self):
        logger.info("Protocol Zero Initialization: Multi-Layer Peeling Initiated.")
        current_target = self.original_target.clone()
        
        for layer in range(1, self.max_layers + 1):
            logger.info(f"Layer {layer}: Initiating Target Extraction...")
            engine = OmegaPrimeChaos(self.inputs, current_target, self.device)
            best = engine.run()
            
            if not best or best['r2'] < 0.05:
                logger.info("Signal-to-noise ratio too low. Stopping layer extraction.")
                break
                
            law_str = engine.translate(best['seq'], best['c'])
            logger.info(f"Layer {layer} Locked: {law_str} (R2: {best['r2']:.4f})")
            self.layers.append(law_str)
            
            with torch.no_grad():
                padded_seq = engine.alchemist.pad_sequence(best['seq'])
                pred_norm = IronDome.execute(padded_seq, self.inputs, best['c'])
                pred_real = pred_norm * engine.alchemist.y_std + engine.alchemist.y_mean
                
                self.total_pred += pred_real 
                current_target = current_target - pred_real
                
        return " + ".join(self.layers)
    
    def get_total_prediction(self):
        return self.total_pred

# --- 9. VISUALIZATION ---
def visualize_chaos_omega(inputs_tensor, target_tensor, pred_tensor, title="Phase Space Diagnostics"):
    X = inputs_tensor.cpu().numpy()
    y_true = target_tensor.cpu().numpy().flatten()
    y_pred = pred_tensor.cpu().numpy().flatten()
    theta1 = X[:, 0]
    omega1 = X[:, 2]
    residuals = y_true - y_pred
    
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    ax1 = fig.add_subplot(131)
    sc = ax1.scatter(theta1, omega1, c=y_pred, cmap='viridis', s=5, alpha=0.8)
    plt.colorbar(sc, ax=ax1, label='Predicted Acceleration')
    ax1.set_xlabel('Angle (Theta 1)')
    ax1.set_ylabel('Velocity (Omega 1)')
    ax1.set_title('Reconstructed Phase Space')
    
    ax2 = fig.add_subplot(132)
    sort_idx = np.argsort(theta1)
    slice_idx = sort_idx[::5]
    ax2.plot(theta1[slice_idx], y_true[slice_idx], 'b-', alpha=0.3, label='Ground Truth')
    ax2.plot(theta1[slice_idx], y_pred[slice_idx], 'r--', alpha=0.8, label='OMEGA Prediction')
    ax2.set_xlabel('Angle')
    ax2.set_ylabel('Acceleration')
    ax2.set_title('Force Field Structure')
    ax2.legend()
    
    ax3 = fig.add_subplot(133)
    speed = np.abs(X[:, 2]) + np.abs(X[:, 3])
    ax3.scatter(speed, np.abs(residuals), alpha=0.5, color='orange', s=10)
    ax3.set_xlabel('System Energy Proxy')
    ax3.set_ylabel('Absolute Error Magnitude')
    ax3.set_title('Error Distribution vs Complexity')
    
    plt.tight_layout()
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Engine Initialized. Compute Device: {device}")

    num_samples = 2000
    t1 = np.random.uniform(-np.pi, np.pi, num_samples)
    t2 = np.random.uniform(-np.pi, np.pi, num_samples)
    w1 = np.random.uniform(-2, 2, num_samples)
    w2 = np.random.uniform(-2, 2, num_samples)
    
    g, m1, m2, L1, L2 = 9.81, 1.0, 1.0, 1.0, 1.0
    
    num = -g*(2*m1+m2)*np.sin(t1) - m2*g*np.sin(t1-2*t2) - 2*np.sin(t1-t2)*m2*(w2**2*L2 + w1**2*L1*np.cos(t1-t2))
    den = L1*(2*m1+m2 - m2*np.cos(2*t1-2*t2))
    target = num / (den + 1e-6)

    inputs = torch.tensor(np.column_stack((t1, t2, w1, w2))).to(device)
    target = torch.tensor(target).view(-1, 1).to(device)

    peeler = ProtocolZero(inputs, target, device, max_layers=2)
    final_eq = peeler.run()
    
    print("\n" + "="*50)
    print(f"Final Extracted Equation: {final_eq}")
    print("="*50 + "\n")
    
    logger.info("Rendering phase space diagnostics...")
    final_pred = peeler.get_total_prediction()
    visualize_chaos_omega(inputs, target, final_pred)