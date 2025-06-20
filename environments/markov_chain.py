import numpy as np

class MarkovChain:
    def __init__(self, transition_matrix, reward_matrix, initial_state=0):
        """
        transition_matrix: shape [num_states, num_states], rows sum to 1
        reward_matrix: shape [num_states, num_states] or [num_states], rewards for transitions
        initial_state: starting state index
        """
        self.transition_matrix = np.array(transition_matrix)
        self.reward_matrix = np.array(reward_matrix)
        self.num_states = self.transition_matrix.shape[0]
        self.initial_state = initial_state
        self.current_state = initial_state

    def reset(self, state=None):
        """Resets to initial or specified state."""
        self.current_state = self.initial_state if state is None else state
        return self.current_state

    def step(self, state=None):
        """
        Simulates one step from the current (or given) state.
        Returns: next_state, reward
        """
        if state is None:
            state = self.current_state
        probs = self.transition_matrix[state]
        next_state = np.random.choice(self.num_states, p=probs)
        # If reward matrix is 2D: per transition; else: per state
        if self.reward_matrix.ndim == 2:
            reward = self.reward_matrix[state, next_state]
        else:
            reward = self.reward_matrix[next_state]
        self.current_state = next_state
        return next_state, reward

    @classmethod
    def setup_random(cls, num_states, reward_range=(0, 1), zero_prob=0.9, seed=None):
        """
        Generate a random Markov chain.
        """
        rng = np.random.default_rng(seed)
        transition_matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            # Sparse transitions: set random subset to nonzero
            nonzero = rng.random(num_states) > zero_prob
            if not np.any(nonzero):
                nonzero[rng.integers(num_states)] = True
            transition_matrix[i, nonzero] = rng.random(np.sum(nonzero))
            transition_matrix[i] /= transition_matrix[i].sum()
        reward_matrix = rng.uniform(reward_range[0], reward_range[1], size=(num_states, num_states))
        return cls(transition_matrix, reward_matrix)

    @classmethod
    def setup_from_npz(cls, file_path):
        """
        Load transition and reward matrices from a .npz file.
        """
        data = np.load(file_path)
        return cls(data['transition_matrix'], data['reward_matrix'])

    def save_to_npz(self, file_path):
        """
        Save transition and reward matrices to a .npz file.
        """
        np.savez(file_path, transition_matrix=self.transition_matrix, reward_matrix=self.reward_matrix)
