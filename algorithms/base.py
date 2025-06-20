from abc import ABC, abstractmethod
import numpy as np

class TDAlgorithm(ABC):
    """
    Abstract base class for all RL algorithms (tabular value-based).
    Ensures a standard interface for use in experiments.
    """

    def __init__(self, num_states, **kwargs):
        self.num_states = num_states

    @abstractmethod
    def reset(self):
        """Reset all internal state, including value estimates and traces."""
        pass

    def start_episode(self):
        """Reset per-episode variables, like eligibility traces (if episodic)."""
        pass

    @abstractmethod
    def step(self, state, reward, next_state):
        """
        Perform a learning step using observed transition.
        Args:
            state: current state (int or other format)
            reward: observed reward (float)
            next_state: next state (int or other format)
        """
        pass

    @abstractmethod
    def get_values(self):
        """Return current value function (or Q-table) for analysis."""
        pass