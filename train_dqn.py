"""
DQN 训练脚本（带经验回放）

使用 MeepMMIPBS 环境进行离散动作训练。
"""

from __future__ import annotations

import os
import sys
import ctypes
from typing import Any, Dict

# Ensure conda's libstdc++ is preferred to avoid CXXABI mismatch
_conda_prefix = os.environ.get("CONDA_PREFIX")
if not _conda_prefix:
    _exe_dir = os.path.dirname(sys.executable)
    _conda_prefix = os.path.dirname(_exe_dir)
if _conda_prefix:
    _conda_lib = os.path.join(_conda_prefix, "lib")
    _libstdcpp = os.path.join(_conda_lib, "libstdc++.so.6")
    if os.path.exists(_libstdcpp):
        try:
            ctypes.CDLL(_libstdcpp, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
if _conda_prefix:
    _conda_lib = os.path.join(_conda_prefix, "lib")
    _ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _conda_lib not in _ld_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{_conda_lib}:{_ld_library_path}" if _ld_library_path else _conda_lib
        )

import yaml
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from mmi_pbs_env import MeepMMIPBS


class MMIConvFeatures(BaseFeaturesExtractor):
    """针对 (C, X, Y) 观测的轻量 CNN 特征提取器"""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _make_env(config_path: str) -> gym.Env:
    env = MeepMMIPBS.from_config_yaml(config_path)

    cfg = _load_config(config_path)
    max_steps = (cfg.get("training", {}) or {}).get("max_episode_steps", 500)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    return env


def train(config_path: str = "./configs/default_config.yaml"):
    cfg = _load_config(config_path)
    training_cfg = cfg.get("training", {}) or {}
    dqn_cfg = cfg.get("dqn", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}

    base_log_dir = log_cfg.get("log_dir", "./results/rl_adjoint_results")
    log_dir = log_cfg.get("dqn_log_dir", os.path.join(base_log_dir, "dqn_results"))
    os.makedirs(log_dir, exist_ok=True)

    env = _make_env(config_path)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=dqn_cfg.get("learning_rate", training_cfg.get("learning_rate", 3e-4)),
        batch_size=dqn_cfg.get("batch_size", training_cfg.get("batch_size", 64)),
        gamma=dqn_cfg.get("gamma", training_cfg.get("gamma", 0.99)),
        buffer_size=dqn_cfg.get("buffer_size", 100_000),
        learning_starts=dqn_cfg.get("learning_starts", 5_000),
        train_freq=dqn_cfg.get("train_freq", 4),
        gradient_steps=dqn_cfg.get("gradient_steps", 1),
        target_update_interval=dqn_cfg.get("target_update_interval", 10_000),
        exploration_fraction=dqn_cfg.get("exploration_fraction", 0.1),
        exploration_final_eps=dqn_cfg.get("exploration_final_eps", 0.05),
        tensorboard_log=log_dir,
        verbose=1,
        policy_kwargs={
            "normalize_images": False,
            "features_extractor_class": MMIConvFeatures,
            "features_extractor_kwargs": {"features_dim": dqn_cfg.get("features_dim", 512)},
        },
    )

    total_timesteps = training_cfg.get("total_timesteps", 100000)
    model.learn(total_timesteps=total_timesteps)

    model.save(os.path.join(log_dir, "haac_dqn_model"))


if __name__ == "__main__":
    train()
