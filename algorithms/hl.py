import numpy as np
from algorithms import TDAlgorithm

class HLLambda(TDAlgorithm):
    """
    HL(λ) temporal-difference learning without a fixed learning rate,
    using per-transition β from Hutter & Legg (2007). 

    Maintains:
      V[s] : state-value estimates
      N[s] : decaying visit counts
      E[s] : eligibility traces
    """

    def __init__(self, num_states, gamma=0.90, lambda_=1.0, alpha=None, eps=1e-8):
        """
        Args:
            num_states (int): number of distinct states
            gamma (float): discount factor γ
            lambda_ (float): trace-decay λ
            eps (float): small constant to avoid division by zero
        """
        super().__init__(num_states)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = eps

        # initialize tables
        self.V = np.zeros(self.num_states, dtype=float)
        self.N = np.ones(self.num_states, dtype=float)   # start at 1 to avoid zero-divide
        self.E = np.zeros(self.num_states, dtype=float)
        self.beta_history = []  # Store mean beta at each step

    def reset(self):
        """Reset everything: value estimates, visit counts, and traces."""
        self.V.fill(0.0)
        self.N.fill(1.0)
        self.E.fill(0.0)

    def start_episode(self):
        """At the start of each episode, clear eligibility traces."""
        self.E.fill(0.0)

    def step(self, state, reward, next_state):
        """
        Perform one update of HL(λ):
          Δ = r + γ V[next_state] − V[state]
          E[state] += 1
          N[state] += 1
          β[s] = (N[next_state]/N[s]) / max(N[next_state] − γ E[next_state], eps)
          V[s] += β[s] * E[s] * Δ    for all s
          E[s] *= γ λ
          N[s] *= λ
        """
        # TD error
        delta = reward + self.gamma * self.V[next_state] - self.V[state]

        # increment eligibility & visit count of current state
        self.E[state] += 1.0
        self.N[state] += 1.0

        # compute denominator for β
        denom = self.N[next_state] - self.gamma * self.E[next_state]
        if denom < self.eps:
            denom = self.eps

        # vectorized β for all states
        beta = (self.N[next_state] / self.N) / denom

        # Track mean beta for this step
        self.beta_history.append(np.mean(beta))

        # update all V[s]
        self.V += beta * self.E * delta

        # decay traces and counts
        self.E *= self.gamma * self.lambda_
        self.N *= self.lambda_

    def get_values(self):
        """Return a copy of current state-value estimates."""
        return self.V.copy()

    def get_beta_history(self):
        """Return a copy of the mean beta values tracked at each step."""
        return np.array(self.beta_history)