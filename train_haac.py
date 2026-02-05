"""
HAAC 训练脚本 (PPO)

使用 MeepMMIPBS 环境进行混合动作训练：像素翻转 + 伴随法超级动作。
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution

from mmi_pbs_env import MeepMMIPBS
from haac_network import HAAC_Network


class HAACPolicy(ActorCriticPolicy):
    """基于 HAAC_Network 的自定义 PPO Policy"""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )

        # 根据环境输入尺寸初始化
        n_cells_x = observation_space.shape[1]
        n_cells_y = observation_space.shape[2]
        self.haac = HAAC_Network(
            input_channels=observation_space.shape[0],
            base_channels=32,
            n_cells_x=n_cells_x,
            n_cells_y=n_cells_y,
        )

        # 动作分布 (离散)
        self.action_dist = CategoricalDistribution(self.action_space.n)

        # 重新构建优化器，确保包含 HAAC 参数
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        logits, values = self.haac(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values = self.haac(obs)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        logits, _ = self.haac(obs)
        return self.action_dist.proba_distribution(action_logits=logits)

    def predict_values(self, obs: torch.Tensor):
        _, values = self.haac(obs)
        return values


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _make_env(config_path: str) -> gym.Env:
    env = MeepMMIPBS.from_config_yaml(config_path)

    # 训练步长限制
    cfg = _load_config(config_path)
    max_steps = (cfg.get("training", {}) or {}).get("max_episode_steps", 500)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    return env


def train(config_path: str = "./configs/default_config.yaml"):
    cfg = _load_config(config_path)
    training_cfg = cfg.get("training", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}

    log_dir = log_cfg.get("log_dir", "./results/rl_adjoint_results")
    os.makedirs(log_dir, exist_ok=True)

    env = _make_env(config_path)

    model = PPO(
        policy=HAACPolicy,
        env=env,
        learning_rate=training_cfg.get("learning_rate", 3e-4),
        n_epochs=training_cfg.get("n_epochs", 10),
        batch_size=training_cfg.get("batch_size", 64),
        gamma=training_cfg.get("gamma", 0.99),
        ent_coef=training_cfg.get("entropy_coeff", 0.01),
        clip_range=training_cfg.get("clip_range", 0.2),
        tensorboard_log=log_dir,
        verbose=1,
    )

    total_timesteps = training_cfg.get("total_timesteps", 100000)
    model.learn(total_timesteps=total_timesteps)

    model.save(os.path.join(log_dir, "haac_ppo_model"))


if __name__ == "__main__":
    train()
