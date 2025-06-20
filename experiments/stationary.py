import numpy as np
import matplotlib.pyplot as plt
from algorithms import HLLambda, TDLambda

# ---- 1. Build the 21-state base Markov chain ----
def make_paper_chain21():
    n = 21
    P = np.zeros((n, n))
    R = np.zeros((n, n))
    for s in range(n):
        if s == 0:
            P[s, 10] = 1.0
            R[s, 10] = 1.0
        elif s == 20:
            P[s, 10] = 1.0
            R[s, 10] = -1.0
        else:
            P[s, s-1] = 0.5
            P[s, s+1] = 0.5
    return P, R

# ---- 2. Compute the true value function ----
def compute_true_values(P, R, gamma):
    n = P.shape[0]
    expected_rewards = (P * R).sum(axis=1)
    I = np.eye(n)
    v = np.linalg.solve(I - gamma * P, expected_rewards)
    print("True values:", v)
    print("Initial RMSE (all states):", compute_rmse(np.zeros_like(v), v))
    print("Initial RMSE (states 1-19):", np.sqrt(np.mean((v[1:-1]) ** 2)))
    return v

def monte_carlo_state_values(P, R, gamma, n_steps=500, n_episodes=200, seed=42):
    """
    Estimate the value of each state using Monte Carlo simulation.
    For each state, run n_episodes trajectories of n_steps each, and average the discounted returns.
    """
    n_states = P.shape[0]
    mc_values = np.zeros(n_states)
    rng = np.random.default_rng(seed)
    
    for s in range(n_states):
        returns = []
        for ep in range(n_episodes):
            state = s
            G = 0.0
            discount = 1.0
            for t in range(n_steps):
                probs = P[state]
                next_state = rng.choice(n_states, p=probs)
                reward = R[state, next_state]
                G += discount * reward
                discount *= gamma
                state = next_state
            returns.append(G)
        mc_values[s] = np.mean(returns)
        print(f"MC value for state {s}: {mc_values[s]:.4f}")
    return mc_values

# ---- 3. RMSE utility ----
def compute_rmse(est, true):
    return np.sqrt(np.mean((est[1:-1] - true[1:-1]) ** 2))

# ---- 4. General switching experiment ----
def run_switching_experiment(
    algorithm,
    lam,
    alpha,
    n_runs=100,
    n_steps=10000,
    gamma=0.9,
    eps=1e-8,
    switch_interval=5000,
    log_interval=1,
    seed_base=2024,
    true_vals=None
):
    P, R = make_paper_chain21()
    #true_vals = compute_true_values(P, R, gamma)
    #true_vals = monte_carlo_state_values(P, R, gamma, n_steps=500, n_episodes=500, seed=seed_base)
    n_states = P.shape[0]
    rmse_curves = []

    for run in range(n_runs):
        rng = np.random.default_rng(seed_base + run)
        agent = algorithm(
            num_states=n_states,
            gamma=gamma,
            lambda_=lam,
            alpha=alpha,
            eps=eps
        )
        state = 10
        rmse_this = []
        for step in range(n_steps):
            # pick which environment and true values
            R_current = R
            true_vals  = true_vals
            # sample transition
            probs = P[state]
            next_state = rng.choice(n_states, p=probs)
            reward = R_current[state, next_state]

            # update
            agent.step(state, reward, next_state)
            state = next_state

            # log RMSE
            if step % log_interval == 0:
                est = agent.get_values()
                rmse_this.append(compute_rmse(est, true_vals))

        rmse_curves.append(rmse_this)
        print(f"Run {run+1}/{n_runs} ({algorithm.__name__}, λ={lam}, α={alpha}) complete. Final RMSE: {rmse_this[-1]:.4f}")

    rmse_curves = np.array(rmse_curves)
    mean_rmse = rmse_curves.mean(axis=0)
    std_rmse  = rmse_curves.std(axis=0)
    steps = np.arange(0, n_steps, log_interval)
    return steps, mean_rmse, std_rmse

# ---- 5. Plotting multiple curves ----
def plot_multiple_rmse(
    results_dict,
    switch_interval=5000,
    save_path=None
):
    plt.figure(figsize=(8, 6))
    # plot each experiment
    for label, (steps, mean_rmse, std_rmse) in results_dict.items():
        plt.plot(steps, mean_rmse, label=label, linewidth=1)
        # plt.fill_between(
        #     steps,
        #     mean_rmse - std_rmse,
        #     mean_rmse + std_rmse,
        #     alpha=0.2
        # )
    # vertical dividers


    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.title(" 21-State Stationary Markov Chain")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

# ---- 6. Main execution ----
if __name__ == "__main__":
    # define experiments: label -> (algorithm, λ, α)
    experiments = {
        "HL(1.0)": (HLLambda, 1.0, 0.0),
        "TD(0.7), α=0.07":  (TDLambda, 0.7,    0.07),
        "TD(0.7), α=0.13":  (TDLambda, 0.7,    0.13),
    }
    results = {}
    P, R = make_paper_chain21()
    gamma = 0.9
    true_vals = monte_carlo_state_values(P, R, gamma, n_steps=500, n_episodes=1000, seed=2024)
    for label, (algo, lam, alpha) in experiments.items():
        print("\n" + "="*50)
        print(f"Starting {label}")
        print("="*50)
        steps, mean_rmse, std_rmse = run_switching_experiment(
            algorithm=algo,
            lam=lam,
            alpha=alpha,
            n_runs=10,
            n_steps=10000,
            gamma=0.9,
            eps=1e-8,
            switch_interval=5000,
            log_interval=1,
            seed_base=2024,
            true_vals=true_vals
        )
        results[label] = (steps, mean_rmse, std_rmse)
        print(f"{label}: Final RMSE = {mean_rmse[-1]:.4f} ± {std_rmse[-1]:.4f}")

    # plot comparison
    plot_multiple_rmse(
        results,
        switch_interval=5000,
        save_path="results/switching_chain_comparison.png"
    )
    print("\nDone! Check results/switching_chain_comparison.png for the overlayed RMSE curves.")
