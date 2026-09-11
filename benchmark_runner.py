import sys
import os
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'benchmarks'))

def run_chaos():
    import chaos
    print("Running chaos.py (Lorenz)...")
    start = time.time()
    
    inputs, target = chaos.generate_lorenz_data()
    device = chaos.torch.device('cpu') 
    inputs = inputs.to(device)
    target = target.to(device)
    
    cfg = chaos.OmegaConfig(generations=50) 
    engine = chaos.OmegaEngine(inputs, target, cfg, device)
    
    shadow_count = [0]
    if hasattr(engine, '_shadow_intervention'):
        orig_shadow = engine._shadow_intervention
        def mock_shadow(*args, **kwargs):
            shadow_count[0] += 1
            return orig_shadow(*args, **kwargs)
        engine._shadow_intervention = mock_shadow
    
    best, law = engine.run()
    end = time.time()
    
    cache_size = len(chaos.IronDome.ORACLE_CACHE) if hasattr(chaos.IronDome, 'ORACLE_CACHE') else 0
    
    return {
        "time_secs": round(end - start, 2),
        "cache_size": cache_size,
        "shadow_count": shadow_count[0],
        "best_r2": best[3] if best else 0
    }

def run_omega():
    import omega
    print("Running omega.py (Duffing)...")
    start = time.time()
    
    n = 500
    w1  = (omega.torch.rand(n, dtype=omega.torch.float64) * 4) - 2
    th1 = (omega.torch.rand(n, dtype=omega.torch.float64) * 6) - 3
    th2 = (omega.torch.rand(n, dtype=omega.torch.float64) * 6) - 3
    y = -0.2 * w1 - 1.0 * th1 - 0.5 * (th1 ** 3) + 2.5 * omega.torch.cos(th2)
    y = y + 0.05 * omega.torch.randn(n, dtype=omega.torch.float64)
    
    device = omega.torch.device('cpu')
    inputs = omega.torch.stack([w1, th1, th2], dim=1).to(device)
    target = y.to(device)
    
    cfg = omega.OmegaConfig(generations=50)
    engine = omega.OmegaIndustrial(inputs, target, cfg, device)
    
    shadow_count = [0]
    if hasattr(engine, '_shadow_mutate'):
        orig_shadow = engine._shadow_mutate
        def mock_shadow(*args, **kwargs):
            shadow_count[0] += 1
            return orig_shadow(*args, **kwargs)
        engine._shadow_mutate = mock_shadow
    
    result = engine.run(max_layers=2) # Reduced to 2 layers for faster profiling
    end = time.time()
    
    return {
        "time_secs": round(end - start, 2),
        "graveyard_size": len(engine.graveyard) if hasattr(engine, 'graveyard') else 0,
        "shadow_count": shadow_count[0],
        "pareto_size": len(engine.pareto.front) if hasattr(engine, 'pareto') else 0
    }

def run_pendulum():
    import double_pendulum
    print("Running double_pendulum.py...")
    start = time.time()
    
    num_samples = 2000
    t1 = double_pendulum.np.random.uniform(-double_pendulum.np.pi, double_pendulum.np.pi, num_samples)
    t2 = double_pendulum.np.random.uniform(-double_pendulum.np.pi, double_pendulum.np.pi, num_samples)
    w1 = double_pendulum.np.random.uniform(-2, 2, num_samples)
    w2 = double_pendulum.np.random.uniform(-2, 2, num_samples)
    
    g, m1, m2, L1, L2 = 9.81, 1.0, 1.0, 1.0, 1.0
    num = -g*(2*m1+m2)*double_pendulum.np.sin(t1) - m2*g*double_pendulum.np.sin(t1-2*t2) - 2*double_pendulum.np.sin(t1-t2)*m2*(w2**2*L2 + w1**2*L1*double_pendulum.np.cos(t1-t2))
    den = L1*(2*m1+m2-m2*double_pendulum.np.cos(2*t1-2*t2))
    alpha1 = num/den
    
    device = double_pendulum.torch.device('cpu')
    inputs = double_pendulum.torch.tensor(double_pendulum.np.column_stack((t1, t2, w1, w2)), dtype=double_pendulum.torch.float64, device=device)
    target = double_pendulum.torch.tensor(alpha1, dtype=double_pendulum.torch.float64, device=device)
    
    protocol = double_pendulum.ProtocolZero(inputs, target, device, max_layers=1) # 1 layer for profiling
    law = protocol.run()
    
    end = time.time()
    
    return {
        "time_secs": round(end - start, 2),
        "law_discovered": law != ""
    }

if __name__ == "__main__":
    results = {}
    import logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    try:
        results["chaos"] = run_chaos()
    except Exception as e:
        results["chaos"] = f"Error: {e}"
        
    try:
        results["omega"] = run_omega()
    except Exception as e:
        results["omega"] = f"Error: {e}"
        
    try:
        results["pendulum"] = run_pendulum()
    except Exception as e:
        results["pendulum"] = f"Error: {e}"
        
    print("\n" + "="*60)
    print("PHASE 0 BASELINE RESULTS (CPU Profiling)")
    print("="*60)
    for k, v in results.items():
        print(f"{k.upper()}: {v}")
