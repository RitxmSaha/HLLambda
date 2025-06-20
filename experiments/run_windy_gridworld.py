import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from environments import WindyGridworld
from algorithms import Sarsa, HLS, Q, HLQ
from scipy import signal
# --- run one full continuous Q(λ) experiment ---
def run_single_experiment(args):
    """Run a single experiment - extracted for parallelization"""
    Algorithm, lam, alpha, gamma, epsilon, num_steps, run = args
    import random
    seed = random.randint(0, 1000000)
    
    # create env and agent
    env = WindyGridworld(seed=seed)
    agent = Algorithm(
        num_states=env.state_space,
        num_actions=env.action_space,
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma,
        lam=lam
    )
    agent.reset()
    state = env.reset()
    action = agent.select_action(state)
    rewards = np.zeros(num_steps)
    
    # run for fixed number of steps
    for t in range(num_steps):
        next_state, reward = env.step(action)
        rewards[t] = reward

        # select next action
        next_action = agent.select_action(next_state)

        agent.step(state, action, reward, next_state, next_action)

        state, action = next_state, next_action

    # compute empirical discounted return G_t for each time step
    returns = np.zeros(num_steps)
    G = 0
    for t in reversed(range(num_steps)):
        G = rewards[t] + gamma * G
        returns[t] = G
    
    return returns

def run_q_experiment(
    Algorithm,
    lam,
    alpha,
    num_runs=1,
    num_steps=50000,
    gamma=0.99,
    epsilon=0.1
):
    all_returns = np.zeros((num_runs, num_steps))
    
    # Prepare arguments for parallel execution
    args_list = [(Algorithm, lam, alpha, gamma, epsilon, num_steps, run) 
                 for run in range(num_runs)]
    
    # Run experiments in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_experiment, args_list))
    
    # Collect results
    for run, returns in enumerate(results):
        all_returns[run, :] = returns

    # aggregate
    mean_ret = all_returns.mean(axis=0)
    std_ret = all_returns.std(axis=0)
    
    # Apply 50-step moving average as mentioned in the paper
    window_size = 50
    if len(mean_ret) >= window_size:
        # Compute moving average
        smoothed_mean = np.convolve(mean_ret, np.ones(window_size)/window_size, mode='valid')
        smoothed_std = np.convolve(std_ret, np.ones(window_size)/window_size, mode='valid')
        # Adjust steps to match the smoothed data length
        steps = np.arange(window_size, len(mean_ret) + 1)
        return steps, smoothed_mean, smoothed_std
    else:
        steps = np.arange(1, len(mean_ret) + 1)
        return steps, mean_ret, std_ret

# --- main: compare TDLambdaQ vs HLLambdaQ continuously ---
if __name__ == "__main__":
    plt.figure(figsize=(8, 6))

    experiments_sarsa = [
        ("SARSA(λ)", Sarsa, 0.5, 0.4, 0.005),
        ("HLS(λ)", HLS, 0.995, None, 0.003),
    ]

    experiments_q = [
        ("Q(λ)", Q, 0.75, 0.99, 0.01),
        ("HLQ(λ)", HLQ, 0.99, None, 0.01),
    ]

    for label, Algo, lam, alpha, eps in experiments_sarsa:
        name = f"{label}, λ={lam}, ε={eps}"
        steps, mean_ret, std_ret = run_q_experiment(
            Algorithm=Algo,
            lam=lam,
            alpha=alpha if alpha is not None else 0.0,
            num_runs=500,  # Increased from 5 to 20 for smoother curves
            num_steps=50000,
            gamma=0.99,
            epsilon=eps
        )
        plt.plot(steps[:-1000], mean_ret[:-1000], label=name, linewidth=1)
        lower = np.clip(mean_ret - std_ret, 0, None)  # no negative lower bound
        upper = mean_ret + std_ret

        #plt.fill_between(steps, lower, upper, alpha=0.2)

    plt.xlabel("Time step")
    plt.ylabel("Empirical discounted return")
    plt.title("SARSA(λ) vs HLS(λ) on Windy Gridworld")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./results/windy_gridworld_comparison_sarsa_hls.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(8, 6))  # Create a NEW figure for the Q experiments

    for label, Algo, lam, alpha, eps in experiments_q:
        name = f"{label}, λ={lam}, ε={eps}"
        steps, mean_ret, std_ret = run_q_experiment(
            Algorithm=Algo,
            lam=lam,
            alpha=alpha if alpha is not None else 0.0,
            num_runs=100,  # Increased from 5 to 20 for smoother curves
            num_steps=50000,
            gamma=0.99,
            epsilon=eps
        )
        plt.plot(steps[:-1000], mean_ret[:-1000], label=name, linewidth=1)
        lower = np.clip(mean_ret - std_ret, 0, None)  # no negative lower bound
        upper = mean_ret + std_ret

        #plt.fill_between(steps, lower, upper, alpha=0.2)

    plt.xlabel("Time step")
    plt.ylabel("Empirical discounted return")
    plt.title("Q(λ) vs HLQ(λ) on Windy Gridworld")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./results/windy_gridworld_comparison_q_hlq.png', dpi=300, bbox_inches='tight')

