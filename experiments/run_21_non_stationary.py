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

# ---- 1b. Create modified reward configuration ----
def make_switching_chain21():
    P, R_orig = make_paper_chain21()
    R_mod = R_orig.copy()
    # Change reward from last state (20) to middle (10) from -1.0 to 0.5
    R_mod[20, 10] = 0.5
    return P, R_orig, R_mod

# ---- 2. Compute the true value function ----
def compute_true_values(P, R, gamma):
    n = P.shape[0]
    expected_rewards = (P * R).sum(axis=1)
    I = np.eye(n)
    v = np.linalg.solve(I - gamma * P, expected_rewards)
    return v

# ---- 3. RMSE utility ----
def compute_rmse(est, true):
    return np.sqrt(np.mean((est[1:-1] - true[1:-1]) ** 2))

# ---- 4. General switching experiment ----
def run_switching_experiment(
    algorithm,
    lam,
    alpha,
    n_runs=10,
    n_steps=20000,
    gamma=0.9,
    eps=1e-8,
    switch_interval=5000,
    log_interval=100,
    seed_base=2024
):
    P, R_orig, R_mod = make_switching_chain21()
    true_orig = compute_true_values(P, R_orig, gamma)
    true_mod  = compute_true_values(P, R_mod,  gamma)
    n_states = P.shape[0]
    rmse_curves = []
    beta_histories = []

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

        # ---- WARM-UP PHASE ----
        warmup_steps = 5000
        print(f"Run {run+1}: Starting warm-up phase ({warmup_steps} steps)...")
        for warmup_step in range(warmup_steps):
            # Sample transition (using first environment)
            probs = P[state]
            next_state = rng.choice(n_states, p=probs)
            reward = R_orig[state, next_state]
            
            # Increment E and N without value updates
            if hasattr(agent, 'E') and hasattr(agent, 'N'):  # HLLambda
                agent.E[state] += 1.0
                agent.N[state] += 1.0
                # Decay traces and counts
                agent.E *= gamma * lam
                agent.N *= lam
            elif hasattr(agent, 'eligibility_traces'):  # TDLambda
                agent.eligibility_traces[state] += 1.0
                # Decay eligibility traces
                agent.eligibility_traces *= gamma * lam
            
            state = next_state
        
        print(f"Run {run+1}: Warm-up complete. Starting main experiment...")

        # ---- MAIN EXPERIMENT ----

        for step in range(n_steps):
            # pick which environment and true values
            env_idx = (step // switch_interval) % 2
            R_current = R_orig if env_idx == 0 else R_mod
            true_vals  = true_orig if env_idx == 0 else true_mod
            #if env_idx != ((step - 1) // switch_interval % 2) and algorithm.__name__ == "HLLambda":
                #agent.reset()

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

        # If HLLambda, store beta history for this run
        if hasattr(agent, 'get_beta_history'):
            beta_histories.append(agent.get_beta_history())

        rmse_curves.append(rmse_this)
        print(f"Run {run+1}/{n_runs} ({algorithm.__name__}, λ={lam}, α={alpha}) complete. Final RMSE: {rmse_this[-1]:.4f}")

    rmse_curves = np.array(rmse_curves)
    mean_rmse = rmse_curves.mean(axis=0)
    std_rmse  = rmse_curves.std(axis=0)
    steps = np.arange(0, n_steps, log_interval)
    if beta_histories:
        # Pad beta histories to the same length (in case of early termination)
        maxlen = max(len(b) for b in beta_histories)
        beta_histories = [np.pad(b, (0, maxlen - len(b)), 'edge') for b in beta_histories]
        beta_histories = np.array(beta_histories)
        mean_beta = beta_histories.mean(axis=0)
        std_beta = beta_histories.std(axis=0)
        return steps, mean_rmse, std_rmse, mean_beta, std_beta
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
    max_step = steps[-1]
    n_switches = max_step // switch_interval
    for k in range(1, n_switches + 1):
        plt.axvline(
            k * switch_interval,
            color='gray',
            linestyle=':',
            linewidth=1
        )

    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.title(" 21-State Non-Stationary Markov Chain (No Warm-Up)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_beta_curve(steps, mean_beta, std_beta, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(steps[:len(mean_beta)], mean_beta, label='Mean β', color='tab:blue')
    # plt.fill_between(steps[:len(mean_beta)], mean_beta - std_beta, mean_beta + std_beta, alpha=0.2, color='tab:blue')
    plt.xlabel('Time')
    plt.ylabel('Mean β')
    plt.title('Mean β over Time (HLLambda)')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Beta plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

# ---- 6. Main execution ----
if __name__ == "__main__":
    # define experiments: label -> (algorithm, λ, α)
    experiments = {
        "HL(0.9995)": (HLLambda, 0.9995, 0.0),
        #"TD(0.8), α=0.05":  (TDLambda, 0.8,    0.05),
        #"TD(0.9), α=0.05":  (TDLambda, 0.9,    0.05),
    }
    results = {}
    beta_plot_data = None

    for label, (algo, lam, alpha) in experiments.items():
        print("\n" + "="*50)
        print(f"Starting {label}")
        print("="*50)
        out = run_switching_experiment(
            algorithm=algo,
            lam=lam,
            alpha=alpha,
            n_runs=100,
            n_steps=20000,
            gamma=0.9,
            eps=1e-8,
            switch_interval=5000,
            log_interval=1,
            seed_base=2024
        )
        if len(out) == 5:
            steps, mean_rmse, std_rmse, mean_beta, std_beta = out
            if label.startswith("HL"):  # Only plot for HLLambda
                beta_plot_data = (steps, mean_beta, std_beta)
            results[label] = (steps, mean_rmse, std_rmse)
        else:
            steps, mean_rmse, std_rmse = out
            results[label] = (steps, mean_rmse, std_rmse)
        print(f"{label}: Final RMSE = {mean_rmse[-1]:.4f} ± {std_rmse[-1]:.4f}")

    # plot comparison
    plot_multiple_rmse(
        results,
        switch_interval=5000,
        save_path="results/switching_chain_comparison.png"
    )
    # Plot beta if available
    if beta_plot_data is not None:
        plot_beta_curve(
            beta_plot_data[0][:len(beta_plot_data[1])],
            beta_plot_data[1],
            beta_plot_data[2],
            save_path="results/beta.png"
        )
    print("\nDone! Check results/switching_chain_comparison.png for the overlayed RMSE curves.")
