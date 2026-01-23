#!/usr/bin/env python
"""
PPO训练示例

使用Stable-Baselines3的PPO算法训练MMI PBS设计。
"""

import sys
import os
import numpy as np
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, '/home/xsdgm')

import pbs_env


def train_ppo(
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    save_freq: int = 10000,
    log_dir: str = None,
    model_name: str = "mmi_pbs_ppo"
):
    """
    使用PPO训练MMI PBS设计
    
    Args:
        total_timesteps: 总训练步数
        learning_rate: 学习率
        n_steps: 每次更新的步数
        batch_size: 批次大小
        n_epochs: 每次更新的epoch数
        gamma: 折扣因子
        save_freq: 保存模型的频率
        log_dir: 日志目录
        model_name: 模型名称
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.utils import set_random_seed
    
    # 设置日志目录
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"/home/xsdgm/pbs_env/logs/{model_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"日志目录: {log_dir}")
    
    # 创建环境
    def make_env(rank, seed=0):
        def _init():
            env = pbs_env.make("MeepMMIPBS-v0")
            env.reset(seed=seed + rank)
            return env
        set_random_seed(seed)
        return _init
    
    # 使用多进程环境加速
    num_envs = 4  # 并行环境数
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env(100)])
    
    # 定义回调
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // num_envs,
        save_path=log_dir,
        name_prefix=model_name
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=save_freq // num_envs,
        deterministic=True,
        render=False
    )
    
    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"总步数: {total_timesteps}")
    print(f"并行环境数: {num_envs}")
    print(f"学习率: {learning_rate}")
    print(f"批次大小: {batch_size}")
    
    # 训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = os.path.join(log_dir, f"{model_name}_final.zip")
    model.save(final_model_path)
    print(f"\n最终模型已保存: {final_model_path}")
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    return model, log_dir


def evaluate_model(model_path: str, num_episodes: int = 10):
    """
    评估训练好的模型
    
    Args:
        model_path: 模型文件路径
        num_episodes: 评估回合数
    """
    from stable_baselines3 import PPO
    
    print(f"\n加载模型: {model_path}")
    model = PPO.load(model_path)
    
    # 创建评估环境
    env = pbs_env.make("MeepMMIPBS-v0")
    
    print(f"\n评估 {num_episodes} 个回合...")
    
    all_rewards = []
    all_te_port1 = []
    all_tm_port2 = []
    all_crosstalk = []
    best_structure = None
    best_reward = -np.inf
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        episode_reward = 0
        
        for step in range(500):  # max steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_te_port1.append(info["te_port1"])
        all_tm_port2.append(info["tm_port2"])
        all_crosstalk.append(info["crosstalk"])
        
        if info["best_reward"] > best_reward:
            best_reward = info["best_reward"]
            best_structure = env.get_best_structure()
        
        print(f"  回合 {ep+1}: 奖励={episode_reward:.4f}, "
              f"TE→P1={info['te_port1']:.3f}, "
              f"TM→P2={info['tm_port2']:.3f}")
    
    # 打印统计
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"平均奖励: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"平均 TE→Port1: {np.mean(all_te_port1):.4f}")
    print(f"平均 TM→Port2: {np.mean(all_tm_port2):.4f}")
    print(f"平均串扰: {np.mean(all_crosstalk):.4f}")
    print(f"最佳奖励: {best_reward:.4f}")
    
    # 保存最佳结构
    if best_structure is not None:
        np.save("/home/xsdgm/pbs_env/examples/best_structure.npy", best_structure)
        print(f"\n最佳结构已保存: /home/xsdgm/pbs_env/examples/best_structure.npy")
    
    env.close()
    
    return {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "best_reward": best_reward,
        "best_structure": best_structure
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO训练MMI PBS设计")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "eval", "both"],
                       help="运行模式")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="训练步数")
    parser.add_argument("--model", type=str, default=None,
                       help="评估时的模型路径")
    parser.add_argument("--episodes", type=int, default=10,
                       help="评估回合数")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        print("\n" + "#" * 60)
        print("# PPO 训练 MMI PBS 设计")
        print("#" * 60)
        
        model, log_dir = train_ppo(total_timesteps=args.timesteps)
        model_path = os.path.join(log_dir, "mmi_pbs_ppo_final.zip")
    
    if args.mode in ["eval", "both"]:
        if args.mode == "eval" and args.model is None:
            print("错误: 评估模式需要指定 --model 参数")
            return
        
        model_path = args.model if args.mode == "eval" else model_path
        
        print("\n" + "#" * 60)
        print("# 模型评估")
        print("#" * 60)
        
        evaluate_model(model_path, num_episodes=args.episodes)


if __name__ == "__main__":
    main()
