import numpy as np

class Q:
    def __init__(self, num_states, num_actions, alpha=0.4, epsilon=0.005, gamma=0.99, lam=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.zeros((num_states, num_actions))
        self.E = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam

    def reset(self):
        self.Q[:] = 0
        self.E[:] = 0

    def reset_traces(self):
        self.E[:] = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.Q[state]
        max_q = np.max(q_values)
        actions_with_max_q = np.flatnonzero(q_values == max_q)
        return np.random.choice(actions_with_max_q)

    def step(self, state, action, reward, next_state, next_action):
        q_values = self.Q[state]
        max_q = np.max(q_values)
        actions_with_max_q = np.flatnonzero(q_values == max_q)
        next_action_optimal = np.argmax(self.Q[next_state])  # greedy action for Q(lambda)
        td_error = reward + self.gamma * self.Q[next_state, next_action_optimal] - self.Q[state, action]

        self.E[state, action] += 1
        self.Q += self.alpha * td_error * self.E

        if next_action == next_action_optimal:
            self.E *= self.gamma * self.lam
        else:
            self.E *= 0

    def get_values(self):
        return self.Q.copy()
