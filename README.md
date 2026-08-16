# OMEGA: Symbolic Equation Discovery Engine

A neuro-symbolic system that searches for closed-form mathematical equations describing noisy, chaotic data — combining a Transformer policy network, Monte Carlo Tree Search, and gradient-based constant optimization.

The goal: given raw time-series data from a physical system (like a driven oscillator), recover the actual governing equation rather than a black-box approximation of it.

**Status:** active research prototype. Core search loop runs end-to-end on the Duffing oscillator benchmark. Several components below are described at the level they're implemented — some are full systems, some are working first passes, and that's flagged explicitly rather than left for you to find out later.

---

## What this actually does

The search runs as a loop between two components:

* **The Explorer** — a small Transformer (2 layers, actor-critic heads) that proposes equation structures token-by-token, representing equations as Reverse Polish Notation sequences over a fixed operator set (`+ − × ÷ sin cos exp x² x³` and input variables). Structures are proposed via MCTS, using standard UCB1 selection over a tree of partial equations.
* **The Alchemist** — once a candidate structure is proposed, its numeric constants are fit with L-BFGS (Strong-Wolfe line search), warm-started from a parent node's constants where possible so related candidates converge faster instead of starting from scratch.

Equations are evaluated by executing them as a stack machine (`IronDome`) over the input tensors, scored by R², and kept in a Pareto front that balances fit quality against equation simplicity (parsimony).

The MCTS rollout phase is fully vectorized — up to 128 candidate rollouts are batched and evaluated on GPU in a single forward pass per step, rather than simulated one at a time. That part is genuinely load-bearing engineering, not a stated goal.

For multi-term equations, the engine uses **residual peeling**: it fits the dominant term first, subtracts its prediction from the target, then searches the residual for the next term, repeating until the signal is exhausted. This is how it recovers equations with several additive components (e.g. the four terms of the Duffing oscillator) without searching the full combined structure at once.

---

## What's implemented at a lighter level than the name suggests

Being direct about this, because the gap between the name and the mechanism is exactly what an experienced reader will check first — better it's stated here than discovered mid-review:

* **"Causal" filtering** is a finite-difference sensitivity check: each input variable is perturbed slightly and the output change is measured, to catch variables the equation depends on trivially or not at all. It's a genuinely useful pruning signal, but it isn't causal inference in the formal sense (no confounding control, no do-calculus, no counterfactual estimation). Documented here as "sensitivity filtering," not causal verification.
* **The policy update** is single-sample advantage-weighted actor-critic — no clipped surrogate objective or importance-sampling ratio, so it isn't PPO in the standard sense despite borrowing the actor-critic structure. It's a simpler, related update rule.
* **"Zero-allocation"** refers to reusing a fixed-size buffer for the stack pointer during equation evaluation, not to tensor memory — PyTorch still allocates a new tensor on each arithmetic op inside that loop. The buffer reuse avoids repeated Python-list allocation overhead across the stack machine's execution, which is a real (if smaller) win, just not literally zero-allocation end to end.

None of these make the underlying idea invalid — sensitivity filtering, advantage-weighted policy updates, and buffer-reuse are all legitimate techniques on their own. They're just not the specific formal methods their working names imply, and the working names are being corrected here rather than carried into anything client-facing.

---

## Benchmarks

| Target | Status |
| :--- | :--- |
| **Duffing oscillator** (4-term, noisy) | Runs end-to-end; recovers structure with visible fit on Expected vs Predicted plot |
| **Singularity case** (`1/(t−2.5)`) | Defined in benchmarks, not yet systematically evaluated |
| **N-body / inverse-square law** | Defined in benchmarks, not yet systematically evaluated |

Quantitative R² and convergence-generation numbers across repeated runs are the immediate next thing to add here — right now the Duffing result is demonstrated but not yet reported as a table of numbers across seeds, which is what's needed before treating it as validated rather than "runs successfully once."

---

## Open work

This repo's commit history currently understates the amount of iteration behind it — a chunk of active development happened in Colab and hasn't been synced here yet. That's being corrected.

Beyond that, the remaining open items are: 
* Reporting R² and convergence-generation numbers across multiple seeds rather than a single run.
* Evaluating the singularity and N-body benchmarks that are already defined in `benchmarks/` but not yet run systematically.
* Replacing the bare `except Exception` blocks in `IronDome.execute` and `TheAlchemist.optimize` with narrower exception handling so genuine bugs surface instead of silently scoring as a failed equation.
* Deciding whether the PPO-style update is worth extending into actual clipped PPO, or whether the simpler update is intentionally sufficient here — right now it's the latter by default, not by decision.

---

## Running it

**Prerequisites:** Python 3.8+, PyTorch, NumPy, SymPy, Matplotlib

```bash
python src/omega.py              # Duffing oscillator benchmark, saves Expected vs Predicted plot
python benchmarks/planetary_movements.py
