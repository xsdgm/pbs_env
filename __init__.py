"""
MEEP MMI PBS 强化学习环境

基于MEEP FDTD仿真的1×2 MMI偏振分束器逆向设计环境。
"""

from gymnasium.envs.registration import register

from .core import Simulator, SimulationResult
from .mmi_pbs_env import MeepMMIPBS
from .meep_simulator import MMISimulator
from .utils import compute_reward, visualize_structure

__version__ = "0.1.0"
__all__ = [
    "MeepMMIPBS", 
    "MMISimulator", 
    "Simulator", 
    "SimulationResult", 
    "compute_reward", 
    "visualize_structure"
]

# 注册gymnasium环境
register(
    id="MeepMMIPBS-v0",
    entry_point="pbs_env.mmi_pbs_env:MeepMMIPBS",
    max_episode_steps=500,
)

register(
    id="MeepMMIPBS-Fast-v0",
    entry_point="pbs_env.mmi_pbs_env:MeepMMIPBS",
    max_episode_steps=200,
    kwargs={"resolution": 10, "run_time": 50},
)


def make(env_id: str, **kwargs):
    """
    创建环境的便捷函数
    
    Args:
        env_id: 环境ID，如 'MeepMMIPBS-v0'
        **kwargs: 传递给环境的额外参数
    
    Returns:
        gymnasium环境实例
    """
    import gymnasium as gym
    return gym.make(env_id, **kwargs)
