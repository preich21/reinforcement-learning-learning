# env_flappy.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # state = [y, v_y, pipe_dx, pipe_gap_y]
        low  = np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0,  1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(2)

        # physics params
        self.gravity      = -0.002
        self.flap_impulse =  0.03
        self.pipe_speed   =  0.01
        self.gap_half     =  0.1
        self.max_vy       =  0.3    # cap absolute vertical speed

        self.bird_x = 0.2   # fixed

        self.state = None
        self.score = 0

    def _reset_game(self):
        self.y           = 0.5
        self.v_y         = 0.0
        self.pipe_x      = 1.0
        self.pipe_gap_y  = np.random.uniform(0.1, 0.9)
        self.last_pipe_passed = False
        self.score = 0

    def _get_obs(self):
        pipe_dx = self.pipe_x - self.bird_x
        pipe_dx = np.clip(pipe_dx, 0.0, 1.0)
        return np.array([self.y, self.v_y, pipe_dx, self.pipe_gap_y], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_game()
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # apply flap
        if action == 1:
            self.v_y += self.flap_impulse

        # physics
        self.v_y += self.gravity

        # cap velocity
        self.v_y = np.clip(self.v_y, -self.max_vy, self.max_vy)

        self.y   += self.v_y

        # move pipe
        self.pipe_x -= self.pipe_speed

        # new pipe
        reward = 0.01  # alive reward
        done = False

        # check pipe passed
        if not self.last_pipe_passed and self.pipe_x < self.bird_x:
            reward += 1.0
            self.score += 1
            self.last_pipe_passed = True

        # respawn pipe if offscreen
        if self.pipe_x < 0.0:
            self.pipe_x = 1.0
            self.pipe_gap_y = np.random.uniform(0.3, 0.7)
            self.last_pipe_passed = False

        # collisions
        if self.y <= 0.0 or self.y >= 1.0:
            done = True
            # reward -= 0.5 # no extra penalty on death

        # check pipe collision (rough)
        if self.bird_x < self.pipe_x < self.bird_x + 0.05:  # in pipe column
            if abs(self.y - self.pipe_gap_y) > self.gap_half:
                done = True
                # reward -= 0.5 # no extra penalty on death

        obs = self._get_obs()
        info = {"score": self.score}

        return obs, reward, done, False, info

    def render(self):
        # stub – we’ll hook in pygame or terminal render later
        pass

    def close(self):
        pass
