"""
Replicates the 51-state Markov chain RMSE experiment from:
Hutter, M., & Legg, S. (2007). Temporal Difference Updating without a Learning Rate. NIPS 2007.
"""

import numpy as np
import matplotlib.pyplot as plt
from algorithms import TDLambda, HLLambda

# ---- 1. Build the 51-state Markov chain as per the paper ----

def make_paper_chain51():
    n = 51
    P = np.zeros((n, n))
    R = np.zeros((n, n))
    for s in range(n):
        if s == 0:
            P[s, 25] = 1.0
            R[s, 25] = 1.0
        elif s == 50:
            P[s, 25] = 1.0
            R[s, 25] = -1.0
        else:
            P[s, s-1] = 0.5
            P[s, s+1] = 0.5
    return P, R

# ---- 2. Compute the true value function -----
def compute_true_values(P, R, gamma):
    n = P.shape[0]
    expected_rewards = (P * R).sum(axis=1)  # Expected reward for each state
    I = np.eye(n)
    v = np.linalg.solve(I - gamma*P, expected_rewards)
    return v

# ---- 3. RMSE utility ----
def compute_rmse(est, true):
    return np.sqrt(np.mean((est - true) ** 2))

# ---- 4. Run experiment ----
def run_td_experiment(
    n_runs=10,
    n_steps=20000,
    gamma=0.99,
    lam=0.9,
    alpha=0.1,
    eps=1e-8,
    log_interval=100,
    seed_base=2024,
    algorithm=TDLambda
):
    P, R = make_paper_chain51()
    true_values = compute_true_values(P, R, gamma)
    n_states = P.shape[0]
    rmse_curves = []

    for run in range(n_runs):
        rng = np.random.default_rng(seed_base + run)
        td = algorithm(num_states=n_states, gamma=gamma, lambda_=lam, alpha=alpha, eps=eps)
        state = 25  # Start in the middle, as often done
        rmse_this_run = []
        for step in range(n_steps):
            # Simulate transition
            probs = P[state]
            next_state = rng.choice(n_states, p=probs)
            reward = R[state, next_state]
            td.step(state, reward, next_state)
            state = next_state

            # Log RMSE
            if step % log_interval == 0:
                est = td.get_values()
                rmse_this_run.append(compute_rmse(est, true_values))
        rmse_curves.append(rmse_this_run)
        print(f"Run {run+1}/{n_runs} complete (α={alpha}). Final RMSE: {rmse_this_run[-1]:.4f}")

    rmse_curves = np.array(rmse_curves)  # shape (n_runs, n_log_points)
    mean_rmse = rmse_curves.mean(axis=0)
    std_rmse = rmse_curves.std(axis=0)
    steps = np.arange(0, n_steps, log_interval)
    return steps, mean_rmse, std_rmse, rmse_curves, true_values

# ---- 5. Plotting utility for multiple curves ----
def plot_multiple_rmse_curves(results_dict, save_path=None):
    plt.figure(figsize=(8, 6))
    
    for alpha, (steps, mean_rmse, std_rmse) in results_dict.items():
        label = f"TD(λ) α={alpha}, λ=0.9"
        plt.plot(steps, mean_rmse, label=label, linewidth=2)
        plt.fill_between(steps, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.2)
    
    plt.xlabel("Steps")
    plt.ylabel("RMSE")
    plt.title(f"TD(λ) on 51-state Markov Chain")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

# ---- 6. Save arrays ----
def save_results(mean_rmse, std_rmse, steps, out_prefix="results/paper_td"):
    np.save(f"{out_prefix}_mean_rmse.npy", mean_rmse)
    np.save(f"{out_prefix}_std_rmse.npy", std_rmse)
    np.save(f"{out_prefix}_steps.npy", steps)
    print(f"[INFO] RMSE curves and steps saved with prefix {out_prefix}")

# ---- 7. Main experiment run ----
if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    # Define alpha values to test
    alpha_values = [
        (0.1, 0.9, TDLambda),
        (0.2, 0.9, TDLambda),
        (0, 1.0, HLLambda),
    ]
    results = {}
    
    # Run experiments for each alpha value
    for alpha, lam, algorithm in alpha_values:
        print(f"\n{'='*50}")
        print(f"Running experiment with α = {alpha}")
        print(f"{'='*50}")
        
        steps, mean_rmse, std_rmse, rmse_curves, true_values = run_td_experiment(
            n_runs=10,     # Use 10 for quick test; use 300 for full Figure 2
            n_steps=40000, # 2e4 steps for quick test; 4e4 for full run
            gamma=0.99,
            lam=lam,
            alpha=alpha,
            eps=1e-8,
            log_interval=100,
            algorithm=algorithm
        )
        
        # Store results
        results[str(alpha)+algorithm.__name__] = (steps, mean_rmse, std_rmse)
        
        # Save individual results
        #save_results(mean_rmse, std_rmse, steps, out_prefix=f"results/paper_td_alpha_{alpha}")
        
        print(f"Final mean RMSE for α={alpha}: {mean_rmse[-1]:.4f} ± {std_rmse[-1]:.4f}")



    # Plot comparison of both alpha values
    plot_multiple_rmse_curves(results, save_path="results/paper_td_alpha_comparison_51.png")
    
    # Print comparison summary
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    for alpha, lam, algorithm in alpha_values:
        steps, mean_rmse, std_rmse = results[str(alpha)+algorithm.__name__]
        print(f"α = {alpha}, {algorithm.__name__}: Final RMSE = {mean_rmse[-1]:.4f} ± {std_rmse[-1]:.4f}")
    
    print("\nExperiment complete! Check results/paper_td_alpha_comparison_51.png for the comparison plot.")
