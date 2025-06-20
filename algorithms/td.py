import numpy as np
from algorithms import TDAlgorithm

class TDLambda(TDAlgorithm):
    def __init__(self, num_states, gamma=0.99, lambda_=0.9, alpha=0.1, eps=None):
        """
        Classic TD(λ) with eligibility traces.
        Args:
            num_states: number of discrete states in the environment
            gamma: discount factor
            lambda_: eligibility trace decay
            alpha: learning rate
        """
        self.num_states = num_states
        self.gamma = gamma
        self.lambda_ = lambda_
        self.alpha = alpha

        self.value_function = np.zeros(num_states)
        self.eligibility_traces = np.zeros(num_states)

    def reset(self):
        """Reset value estimates and eligibility traces (for a new run/episode)."""
        self.value_function.fill(0)
        self.eligibility_traces.fill(0)

    def step(self, state, reward, next_state):
        """
        Perform a TD(λ) update for a single transition.
        Args:
            state: current state (int)
            reward: observed reward (float)
            next_state: next state (int)
        """
        # Calculate TD error
        td_error = reward + self.gamma * self.value_function[next_state] - self.value_function[state]

        # Increment eligibility trace for the current state
        self.eligibility_traces[state] += 1

        # Update value function for all states
        self.value_function += self.alpha * self.eligibility_traces * td_error

        # Decay eligibility traces
        self.eligibility_traces *= self.gamma * self.lambda_

    def get_values(self):
        """Returns the current value function (for analysis/plotting)."""
        return self.value_function.copy()