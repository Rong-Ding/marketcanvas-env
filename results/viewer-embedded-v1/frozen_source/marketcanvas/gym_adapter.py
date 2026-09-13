"""Gymnasium adapter with declared JSON-text spaces; not a tensor PPO policy adapter."""
import json
import string
import gymnasium as gym
from gymnasium import spaces
from .env import MarketCanvasEnv


class GymCanvasEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, **kwargs):
        super().__init__()
        self.core = MarketCanvasEnv(**kwargs)
        self.render_mode = render_mode
        self.action_space = spaces.Text(min_length=1, max_length=4096, charset=string.printable)
        self.observation_space = spaces.Text(min_length=1, max_length=262144, charset=string.printable)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.core.reset(seed=seed, options=options)
        return json.dumps(obs, sort_keys=True), info

    def step(self, action):
        try:
            parsed = json.loads(action)
        except (ValueError, TypeError):
            parsed = {"op": "invalid_json"}
        obs, reward, terminated, truncated, info = self.core.step(parsed)
        return json.dumps(obs, sort_keys=True), reward, terminated, truncated, info

    def render(self):
        return self.core.render()
