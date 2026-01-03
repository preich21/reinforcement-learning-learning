# env_dino.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DinoEnv(gym.Env):
    """
    Simple Chrome-Dino-like runner with visual observations.

    Observation: 84x84 grayscale image, values in [0, 255], shape (84, 84, 1)
    Actions: 0 = do nothing, 1 = jump

    Reward:
        +1   per time step alive
        +10  when passing an obstacle
        -50  on collision (episode ends)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
            self,
            render_mode=None,
            screen_width: int = 84,
            screen_height: int = 84,
            max_steps: int = 5000,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_steps = max_steps

        # Observation: grayscale image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 1),
            dtype=np.uint8,
        )

        # Action: 0 = no-op, 1 = jump
        self.action_space = spaces.Discrete(2)

        # "World" coordinates are in pixels
        # Ground line a bit above bottom
        self.ground_y = int(self.screen_height * 0.7)

        # Dino properties
        self.dino_width = int(self.screen_width * 0.06)
        self.dino_height = int(self.screen_height * 0.18)
        self.dino_x = int(self.screen_width * 0.15)

        # Physics
        # y increases downward in image coordinates. Upward jump should be
        # a negative velocity and gravity should pull the dino back toward
        # increasing y (downwards).
        self.gravity = 0.5
        self.jump_velocity = -6.0
        self.max_fall_speed = 10.0

        # Obstacles
        self.obstacle_min_width = int(self.screen_width * 0.04)
        self.obstacle_max_width = int(self.screen_width * 0.08)
        self.obstacle_height = int(self.screen_height * 0.15)

        self.base_speed = 1.0
        self.speed_increase = 0.001  # speed += speed_increase * step

        # Internal state
        self._dino_y = None
        self._dino_vy = None
        self._obstacles = None  # list of dicts: {"x": float, "width": int}
        self._speed = None
        self._steps = None
        self._score = None

    # ------------- Gymnasium API -------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.np_random is set by super().reset(seed=seed)

        self._steps = 0
        self._score = 0
        self._speed = self.base_speed

        # Dino starts on ground
        self._dino_y = self.ground_y - self.dino_height
        self._dino_vy = 0.0

        # Single obstacle starting off-screen to the right
        self._obstacles = []
        self._spawn_obstacle(initial=True)

        obs = self._render_frame()
        info = {"score": self._score}

        return obs, info

    def step(self, action):
        self._steps += 1

        # --- Apply action ---
        # on_ground when the dino's top is at (or below) the ground-top
        # position. Since y grows downwards, being "on ground" means the
        # dino's y is greater-or-equal to the ground top position.
        on_ground = self._dino_y >= self.ground_y - self.dino_height - 1
        if action == 1 and on_ground:
            # apply negative upward velocity to jump
            self._dino_vy = self.jump_velocity

        # --- Physics update ---
        # integrate gravity (positive gravity pulls down, increasing y)
        self._dino_vy += self.gravity
        # clamp fall speed (positive is downward)
        if self._dino_vy > self.max_fall_speed:
            self._dino_vy = self.max_fall_speed

        # integrate velocity into position (negative vy moves up)
        self._dino_y += self._dino_vy

        # Clamp dino to ground (no double jump). If the dino moved below
        # the ground-top position (greater y), snap it back and zero vy.
        if self._dino_y > self.ground_y - self.dino_height:
            self._dino_y = self.ground_y - self.dino_height
            self._dino_vy = 0.0

        # Prevent the dino from going above the top of the screen. Negative
        # y would index from the end of the numpy array and cause the dino
        # to appear wrapped-around at the bottom. Clamp to 0 and stop upward
        # velocity when hitting the top.
        if self._dino_y < 0:
            self._dino_y = 0
            self._dino_vy = 0.0

        # --- Move obstacles ---
        self._speed += self.speed_increase
        dx = self._speed
        for obs in self._obstacles:
            obs["x"] -= dx

        # Remove obstacles that are off-screen
        self._obstacles = [o for o in self._obstacles if o["x"] + o["width"] > 0]

        # Spawn new obstacle if needed
        if len(self._obstacles) == 0 or (
                self._obstacles[-1]["x"] < self.screen_width * 0.6
        ):
            self._spawn_obstacle()

        # --- Reward + termination ---
        reward = 1.0  # alive bonus per step
        terminated = False
        truncated = self._steps >= self.max_steps

        # Reward for passing obstacles
        for obs in self._obstacles:
            if not obs.get("passed", False) and obs["x"] + obs["width"] < self.dino_x:
                obs["passed"] = True
                reward += 10.0
                self._score += 1

        # Collision detection (AABB)
        if self._check_collision():
            reward -= 50.0
            terminated = True

        obs = self._render_frame()
        info = {
            "score": self._score,
            "speed": self._speed,
            "steps": self._steps,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        For Gymnasium compatibility: returns RGB array if render_mode=='rgb_array'.
        No 'human' rendering implemented here to keep it simple and picklable.
        """
        if self.render_mode == "rgb_array":
            frame = self._render_frame()
            # Convert 1-channel grayscale to 3-channel RGB for gym-style rgb_array
            rgb = np.repeat(frame, 3, axis=2)
            return rgb
        else:
            # You can plug in cv2.imshow or pygame here if you really want human mode
            return None

    def close(self):
        pass

    # ------------- Internal helpers -------------

    def _spawn_obstacle(self, initial=False):
        """
        Spawn an obstacle to the right.
        If initial=True, spawn a bit further away.
        """
        width = int(
            self.np_random.integers(
                self.obstacle_min_width, self.obstacle_max_width + 1
            )
        )

        if initial:
            x = self.screen_width + self.screen_width * 0.3
        else:
            # Spawn at some random gap beyond the right edge
            gap_min = self.screen_width * 0.3
            gap_max = self.screen_width * 0.6
            gap = float(self.np_random.uniform(gap_min, gap_max))
            x = self.screen_width + gap

        self._obstacles.append(
            {
                "x": float(x),
                "width": width,
                "passed": False,
            }
        )

    def _check_collision(self):
        """
        Axis-aligned bounding-box (AABB) collision between dino and obstacles.
        """
        dino_left = self.dino_x
        dino_right = self.dino_x + self.dino_width
        dino_top = self._dino_y
        dino_bottom = self._dino_y + self.dino_height

        for obs in self._obstacles:
            obs_left = obs["x"]
            obs_right = obs["x"] + obs["width"]
            obs_top = self.ground_y - self.obstacle_height
            obs_bottom = self.ground_y

            # AABB overlap
            if (
                    dino_right > obs_left
                    and dino_left < obs_right
                    and dino_bottom > obs_top
                    and dino_top < obs_bottom
            ):
                return True

        return False

    def _render_frame(self):
        """
        Renders a simple 2D scene to a grayscale image (H, W, 1) uint8.
        White = 255, black = 0.
        """
        img = np.zeros(
            (self.screen_height, self.screen_width), dtype=np.uint8
        )

        # Ground line
        img[self.ground_y : self.ground_y + 2, :] = 255

        # Dino rectangle (clamped to image bounds to avoid numpy negative
        # indexing/wrap-around)
        dino_top = int(self._dino_y)
        dino_bottom = int(self._dino_y + self.dino_height)
        dino_left = self.dino_x
        dino_right = self.dino_x + self.dino_width
        dino_top_clamp = max(0, dino_top)
        dino_bottom_clamp = min(self.screen_height, dino_bottom)
        dino_left_clamp = max(0, dino_left)
        dino_right_clamp = min(self.screen_width, dino_right)
        if dino_bottom_clamp > dino_top_clamp and dino_right_clamp > dino_left_clamp:
            img[dino_top_clamp:dino_bottom_clamp, dino_left_clamp:dino_right_clamp] = 255

        # Obstacles
        for obs in self._obstacles:
            x1 = int(obs["x"])
            x2 = int(obs["x"] + obs["width"])
            y1 = self.ground_y - self.obstacle_height
            y2 = self.ground_y
            if x2 <= 0 or x1 >= self.screen_width:
                continue
            x1_clamp = max(0, x1)
            x2_clamp = min(self.screen_width, x2)
            img[y1:y2, x1_clamp:x2_clamp] = 255

        # Add channel dimension
        img = img[..., None]
        return img
