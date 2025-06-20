import numpy as np

# --- Environment: continuing windy gridworld matching Sutton & Barto ---
class WindyGridworld:
    def __init__(self, height=7, width=10, wind=None,
                 start=(3,0), goal=(3,7), seed=None):
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.rng = np.random.default_rng(seed)
        self.wind = wind if wind is not None else np.array([0,0,0,1,1,1,2,2,1,0])
        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]
        self.action_space = len(self.actions)
        self.state_space = self.width * self.height
        self.reset()

    def reset(self):
        self.agent = self.start
        return self._state_index(self.agent)

    def step(self, action):
        # Apply action first
        row, col = self.agent
        dr, dc = self.actions[action]


        new_row = row + dr
        new_col = col + dc
        
        # Clamp within grid bounds
        new_row = np.clip(new_row, 0, self.height-1)
        new_col = np.clip(new_col, 0, self.width-1)
        
        # Apply wind based on the OLD column (start column)
        w = self.wind[col]
        new_row -= w
        
        # Clamp row again after wind
        new_row = np.clip(new_row, 0, self.height-1)
        
        self.agent = (new_row, new_col)
        
        # Reward: +1 at goal, then teleport back to start; else 0
        if self.agent == self.goal:
            reward = 1
            self.agent = self.start
        else:
            reward = 0
        return self._state_index(self.agent), reward

    def _state_index(self, pos):
        return pos[0] * self.width + pos[1]