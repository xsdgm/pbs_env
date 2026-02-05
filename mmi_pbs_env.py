"""
MMI PBS 强化学习环境

基于MEEP FDTD仿真的1×2 MMI偏振分束器设计环境。
目标：TE模式输出到端口1，TM模式输出到端口2。
"""

import os
import numpy as np
import yaml
from typing import Optional, Tuple, Dict, Any
from functools import partial

import gymnasium as gym
from gymnasium import spaces


try:
    from .core import Simulator, SimulationResult
    from .meep_simulator import MMISimulator, SimulationConfig, load_simulation_config
    from .utils import compute_reward
except ImportError:
    from core import Simulator, SimulationResult
    from meep_simulator import MMISimulator, SimulationConfig, load_simulation_config
    from utils import compute_reward



class MeepMMIPBS(gym.Env):
    """
    MEEP MMI PBS 强化学习环境
    
    设计目标:
    - TE偏振输出到端口1
    - TM偏振输出到端口2
    
    观察空间:
    - 'structure': MMI区域的结构分布 (n_cells_x, n_cells_y)
    
    动作空间:
    - Discrete(n_cells_x * n_cells_y): 翻转某个单元的材料
    
    奖励函数:
    - 最大化目标端口效率，最小化串扰
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    @classmethod
    def from_config_yaml(cls, config_path: str = None, **kwargs):
        """
        从YAML配置创建环境
        
        Args:
            config_path: YAML配置路径
            **kwargs: 覆盖的环境参数
        
        Returns:
            MeepMMIPBS实例
        """
        config, num_workers = load_simulation_config(config_path)
        simulator = MMISimulator(config=config, num_workers=num_workers)

        extra_cfg = {}
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    extra_cfg = yaml.safe_load(f) or {}
            except FileNotFoundError:
                extra_cfg = {}
        
        env_kwargs = {
            'n_cells_x': config.n_cells_x,
            'n_cells_y': config.n_cells_y,
            'simulator': simulator,
        }

        # 读取可选的环境/训练参数
        reward_cfg = (extra_cfg.get("reward", {}) or {})
        env_kwargs.update({
            "init_mode": (extra_cfg.get("structure", {}) or {}).get("init_mode", "random"),
            "reward_alpha": reward_cfg.get("alpha", 1.0),
            "reward_beta": reward_cfg.get("beta", 1.0),
            "reward_gamma": reward_cfg.get("gamma", 0.5),
            "reward_type": reward_cfg.get("type", "absolute"),
        })

        rl_cfg = (extra_cfg.get("rl", {}) or {})
        logging_cfg = (extra_cfg.get("logging", {}) or {})
        env_kwargs.update({
            "adjoint_cost": rl_cfg.get("adjoint_cost", 0.5),
            "flip_cost": rl_cfg.get("flip_cost", 0.01),
            "adjoint_steps": rl_cfg.get("adjoint_steps", 5),
            "adjoint_lr": rl_cfg.get("adjoint_lr", 0.01),
            "adjoint_target_weights": rl_cfg.get("adjoint_target_weights", None),
            "log_dir": logging_cfg.get("log_dir", "./results/rl_adjoint_results"),
        })
        env_kwargs.update(kwargs)
        return cls(**env_kwargs)
    
    def __init__(
        self,
        # 结构参数
        n_cells_x: Optional[int] = None,
        n_cells_y: Optional[int] = None,
        init_mode: str = "random",  # "random", "ones", "zeros", "half", "test_uniform", "test_center_bar", "test_y_split"
        
        # 仿真参数
        simulator: Optional[Simulator] = None,
        wavelength: Optional[float] = None,
        mmi_width: Optional[float] = None,
        mmi_length: Optional[float] = None,
        resolution: Optional[int] = None,
        run_time: Optional[float] = None,

        
        # 并行参数
        num_workers: int = 12,
        
        # 奖励参数
        reward_alpha: float = 1.0,  # TE_port1权重
        reward_beta: float = 1.0,   # TM_port2权重
        reward_gamma: float = 0.5,  # 串扰惩罚权重
        reward_type: str = "absolute",  # "absolute" or "delta"
        reward_mode: str = "barrel",  # "barrel" | "sum" | "min_er"
        curriculum_steps: Tuple[int, int] = (1000, 3000),
        curriculum_beta: float = 0.5,

        # 伴随法（超级动作）参数
        adjoint_cost: float = 0.5,
        flip_cost: float = 0.01,
        adjoint_steps: int = 5,
        adjoint_lr: float = 0.01,
        adjoint_target_weights: Optional[Dict[str, float]] = None,

        # 日志
        log_dir: Optional[str] = "./results/rl_adjoint_results",
        
        # 渲染
        render_mode: Optional[str] = None,
    ):
        """
        初始化环境
        
        Args:
            n_cells_x: MMI区域X方向网格数
            n_cells_y: MMI区域Y方向网格数
            init_mode: 初始化模式
            simulator: 仿真器实例（可选），如果提供则忽略后续仿真配置参数
            wavelength: 工作波长 (μm)
            mmi_width: MMI宽度 (μm)
            mmi_length: MMI长度 (μm)
            resolution: 仿真分辨率 (pixels/μm)
            run_time: 仿真时间
            num_workers: 并行工作进程数
            reward_alpha: TE效率权重

            reward_beta: TM效率权重
            reward_gamma: 串扰惩罚权重
            reward_type: 奖励类型
            render_mode: 渲染模式
        """
        super().__init__()
        
        # 从YAML加载默认配置
        default_config, default_workers = load_simulation_config()
        
        # 保存参数（使用配置或传入值）
        self.n_cells_x = n_cells_x or default_config.n_cells_x
        self.n_cells_y = n_cells_y or default_config.n_cells_y
        self.n_cells = self.n_cells_x * self.n_cells_y
        self.init_mode = init_mode
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_gamma = reward_gamma
        self.reward_type = reward_type
        self.reward_mode = reward_mode
        self.curriculum_steps = curriculum_steps
        self.curriculum_beta = curriculum_beta
        self.render_mode = render_mode

        # 伴随法参数
        self.adjoint_cost = adjoint_cost
        self.flip_cost = flip_cost
        self.adjoint_steps = adjoint_steps
        self.adjoint_lr = adjoint_lr
        self.adjoint_target_weights = adjoint_target_weights or {
            "te_port1": 1.0,
            "tm_port2": 1.0,
        }

        # 日志
        self.log_dir = log_dir
        self.episode_index = 0
        
        if simulator is not None:
            self.simulator = simulator
        else:
            # 创建仿真器（用传入值覆盖默认配置）
            self.sim_config = SimulationConfig(
                wavelength=wavelength or default_config.wavelength,
                mmi_width=mmi_width or default_config.mmi_width,
                mmi_length=mmi_length or default_config.mmi_length,
                resolution=resolution or default_config.resolution,
                run_time=run_time or default_config.run_time,
                n_cells_x=self.n_cells_x,
                n_cells_y=self.n_cells_y,
            )
            self.simulator = MMISimulator(self.sim_config, num_workers=num_workers or default_workers)
        
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2, self.n_cells_x, self.n_cells_y),
            dtype=np.float32
        )
        
        # 定义动作空间 (离散：翻转某个像素)
        self.action_space = spaces.Discrete(self.n_cells + 1)
        
        # 状态变量
        self.structure = None
        self.current_results = None
        self.prev_reward = 0.0
        self.step_count = 0
        self.last_action_was_adjoint = False
        
        # 性能追踪
        self.best_reward = -np.inf
        self.best_structure = None
        self.episode_rewards = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
        
        Returns:
            observation: 初始观察
            info: 信息字典
        """
        super().reset(seed=seed)
        
        options = options or {}
        
        # 初始化结构
        if "structure" in options:
            self.structure = options["structure"].copy()
        else:
            self.structure = self._init_structure()
        
        # 运行初始仿真
        self.current_results = self.simulator.simulate(self.structure)
        
        # 重置计数器
        self.step_count = 0
        self.prev_reward = self._compute_reward()
        self.episode_rewards = []
        self.last_action_was_adjoint = False

        # 日志目录
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self._ensure_log_header()
            self.episode_index += 1
        
        # 获取观察和信息
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 要翻转的像素索引
        
        Returns:
            observation: 观察
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 信息字典
        """
        cost = 0.0
        action_type = "flip"

        # 分支 1: 伴随法超级动作
        if action == self.n_cells:
            self._run_adjoint_optimization(steps=self.adjoint_steps)
            cost = self.adjoint_cost
            action_type = "adjoint"
            self.last_action_was_adjoint = True
        else:
            # 分支 2: 翻转像素
            x = action // self.n_cells_y
            y = action % self.n_cells_y
            self.structure[x, y] = 1.0 - self.structure[x, y]
            cost = self.flip_cost
            self.last_action_was_adjoint = False
        
        # 运行仿真
        self.current_results = self.simulator.simulate(self.structure)
        
        # 计算奖励
        current_reward = self._compute_reward()
        if self.reward_type == "delta":
            reward = current_reward - self.prev_reward - cost
        else:
            reward = current_reward - cost
        
        self.prev_reward = current_reward
        self.episode_rewards.append(reward)
        
        # 更新最佳结果
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.best_structure = self.structure.copy()
            self._save_best_results()
        
        # 更新步数
        self.step_count += 1
        
        # 获取观察和信息
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            "action_type": action_type,
            "action_cost": cost,
            "last_action_was_adjoint": self.last_action_was_adjoint,
        })

        self._log_step(action=action, action_type=action_type, reward=reward, cost=cost)
        
        # terminated 和 truncated
        terminated = False  # 没有自然终止条件
        truncated = False   # 由 max_episode_steps 控制
        
        return obs, reward, terminated, truncated, info
    
    def _init_structure(self) -> np.ndarray:
        """初始化结构"""
        if self.init_mode == "random":
            return self.np_random.random((self.n_cells_x, self.n_cells_y)).astype(np.float32)
        elif self.init_mode == "ones":
            return np.ones((self.n_cells_x, self.n_cells_y), dtype=np.float32)
        elif self.init_mode == "zeros":
            return np.zeros((self.n_cells_x, self.n_cells_y), dtype=np.float32)
        elif self.init_mode == "half":
            struct = np.zeros((self.n_cells_x, self.n_cells_y), dtype=np.float32)
            struct[:, :self.n_cells_y // 2] = 1.0
            return struct
        elif self.init_mode == "test_uniform":
            return create_test_structure(self.n_cells_x, self.n_cells_y, mode="uniform")
        elif self.init_mode == "test_center_bar":
            return create_test_structure(self.n_cells_x, self.n_cells_y, mode="center_bar")
        elif self.init_mode == "test_y_split":
            return create_test_structure(self.n_cells_x, self.n_cells_y, mode="y_split")
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        gradients = self.simulator.compute_gradients(
            self.structure,
            target_weights=self.adjoint_target_weights
        ).astype(np.float32)
        obs = np.stack([self.structure.astype(np.float32), gradients], axis=0)
        return obs

    def _run_adjoint_optimization(self, steps: int = 5):
        """运行伴随法优化若干步"""
        for _ in range(max(1, steps)):
            gradients = self.simulator.compute_gradients(
                self.structure,
                target_weights=self.adjoint_target_weights
            )
            gradients = self._normalize_gradients(gradients)
            self.structure = np.clip(self.structure + self.adjoint_lr * gradients, 0.0, 1.0)

    def _normalize_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """归一化并裁剪梯度"""
        scale = np.percentile(np.abs(gradients), 99.9)
        if scale > 1e-8:
            gradients = gradients / scale
        gradients = np.clip(gradients, -1.0, 1.0)
        return gradients
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        if self.current_results is None:
            return 0.0

        te_p1 = self.current_results.te_port1
        tm_p2 = self.current_results.tm_port2
        te_p2 = self.current_results.te_port2
        tm_p1 = self.current_results.tm_port1

        # Curriculum Learning
        # 阶段1: 仅优化通光
        # 阶段2: 加入串扰惩罚
        # 阶段3: 预留（可加入二值化/工艺约束）
        step_1, step_2 = self.curriculum_steps
        if self.reward_mode == "barrel":
            if self.step_count < step_1:
                return float(te_p1 + tm_p2)
            if self.step_count < step_2:
                return float((te_p1 + tm_p2) - self.curriculum_beta * (te_p2 + tm_p1))

            # 第三阶段：使用“木桶效应”+串扰惩罚
            return float((te_p1 * tm_p2) - self.reward_gamma * (te_p2 + tm_p1))

        if self.reward_mode == "min_er":
            # min(T) * log10(ER) 变体
            eps = 1e-10
            er_te = (te_p1 + eps) / (te_p2 + eps)
            er_tm = (tm_p2 + eps) / (tm_p1 + eps)
            er = min(er_te, er_tm)
            return float(min(te_p1, tm_p2) * np.log10(er + eps))

        # 默认：求和 + 串扰惩罚
        return compute_reward(
            te_port1=te_p1,
            tm_port2=tm_p2,
            te_port2=te_p2,
            tm_port1=tm_p1,
            alpha=self.reward_alpha,
            beta=self.reward_beta,
            gamma=self.reward_gamma,
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """获取信息字典"""
        info = {
            "step_count": self.step_count,
            "best_reward": self.best_reward,
        }
        
        if self.current_results:
            info.update({
                "te_port1": self.current_results.te_port1,
                "te_port2": self.current_results.te_port2,
                "tm_port1": self.current_results.tm_port1,
                "tm_port2": self.current_results.tm_port2,
                "crosstalk": self.current_results.crosstalk,
                "total_efficiency": self.current_results.total_efficiency,
            })
            
            # 计算消光比 (dB)
            te_er = 10 * np.log10(
                (info["te_port1"] + 1e-10) / (info["te_port2"] + 1e-10)
            )
            tm_er = 10 * np.log10(
                (info["tm_port2"] + 1e-10) / (info["tm_port1"] + 1e-10)
            )
            info["te_extinction_ratio_dB"] = te_er
            info["tm_extinction_ratio_dB"] = tm_er
        
        return info

    def _ensure_log_header(self):
        """确保日志文件有表头"""
        log_path = os.path.join(self.log_dir, "rl_adjoint_run.csv")
        if not os.path.exists(log_path):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(
                    "episode,step,action,action_type,reward,cost,"
                    "te_port1,tm_port2,te_port2,tm_port1,total_efficiency,crosstalk\n"
                )

    def _log_step(self, action: int, action_type: str, reward: float, cost: float):
        """记录单步日志"""
        if not self.log_dir:
            return
        log_path = os.path.join(self.log_dir, "rl_adjoint_run.csv")
        te_p1 = self.current_results.te_port1 if self.current_results else 0.0
        tm_p2 = self.current_results.tm_port2 if self.current_results else 0.0
        te_p2 = self.current_results.te_port2 if self.current_results else 0.0
        tm_p1 = self.current_results.tm_port1 if self.current_results else 0.0
        total_eff = self.current_results.total_efficiency if self.current_results else 0.0
        crosstalk = self.current_results.crosstalk if self.current_results else 0.0

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{self.episode_index},{self.step_count},{action},{action_type},"
                f"{reward:.6f},{cost:.6f},{te_p1:.6f},{tm_p2:.6f},"
                f"{te_p2:.6f},{tm_p1:.6f},{total_eff:.6f},{crosstalk:.6f}\n"
            )

    def _save_best_results(self):
        """保存最佳结构与指标"""
        if not self.log_dir or self.best_structure is None or self.current_results is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        np.save(os.path.join(self.log_dir, "best_structure_rl.npy"), self.best_structure)
        with open(os.path.join(self.log_dir, "rl_best_performance.txt"), "w", encoding="utf-8") as f:
            f.write("RL + Adjoint 最佳结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best Reward: {self.best_reward:.6f}\n")
            f.write(f"TE→Port1: {self.current_results.te_port1:.6f}\n")
            f.write(f"TM→Port2: {self.current_results.tm_port2:.6f}\n")
            f.write(f"TE→Port2: {self.current_results.te_port2:.6f}\n")
            f.write(f"TM→Port1: {self.current_results.tm_port1:.6f}\n")
            f.write(f"Total Efficiency: {self.current_results.total_efficiency:.6f}\n")
            f.write(f"Crosstalk: {self.current_results.crosstalk:.6f}\n")
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_human()
    
    def _render_frame(self) -> np.ndarray:
        """渲染为RGB数组"""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # 绘制结构
        im = ax.imshow(
            self.structure.T,
            cmap="YlOrBr",
            aspect="auto",
            origin="lower"
        )
        ax.set_xlabel("X (cells)")
        ax.set_ylabel("Y (cells)")
        ax.set_title(f"MMI PBS Structure (Step {self.step_count})")
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, label="Material (0=SiO2, 1=Si)")
        
        # 添加性能信息
        if self.current_results:
            info_text = (
                f"TE→Port1: {self.current_results.te_port1:.3f}\n"
                f"TM→Port2: {self.current_results.tm_port2:.3f}\n"
                f"Crosstalk: {self.current_results.crosstalk:.3f}"
            )
            ax.text(
                1.02, 0.5, info_text,
                transform=ax.transAxes,
                verticalalignment='center',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        # 转换为数组
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        plt.close(fig)
        
        return buf[:, :, :3]  # 去掉alpha通道
    
    def _render_human(self):
        """人类可读渲染"""
        print(f"\n=== Step {self.step_count} ===")
        print(f"Structure shape: {self.structure.shape}")
        print(f"Fill factor: {np.mean(self.structure):.3f}")
        
        if self.current_results:
            print(f"TE → Port1: {self.current_results.te_port1:.4f}")
            print(f"TE → Port2: {self.current_results.te_port2:.4f}")
            print(f"TM → Port1: {self.current_results.tm_port1:.4f}")
            print(f"TM → Port2: {self.current_results.tm_port2:.4f}")
            print(f"Crosstalk: {self.current_results.crosstalk:.4f}")
            print(f"Current Reward: {self.prev_reward:.4f}")
            print(f"Best Reward: {self.best_reward:.4f}")
    
    def close(self):
        """关闭环境"""
        pass
    
    def get_best_structure(self) -> Optional[np.ndarray]:
        """获取最佳结构"""
        return self.best_structure.copy() if self.best_structure is not None else None
    
    def save_structure(self, path: str):
        """保存当前结构"""
        np.save(path, self.structure)
    
    def load_structure(self, path: str):
        """加载结构"""
        self.structure = np.load(path)


def create_test_structure(
    n_cells_x: int,
    n_cells_y: int,
    mode: str = "y_split"
) -> np.ndarray:
    """
    生成用于验证仿真的测试结构。

    模式说明:
    - uniform: 全1（硅填充）
    - center_bar: 中心直波导条带
    - y_split: 中心输入并逐渐分叉到上下两端口
    """
    struct = np.zeros((n_cells_x, n_cells_y), dtype=np.float32)

    if mode == "uniform":
        struct[:] = 1.0
        return struct

    if mode == "center_bar":
        bar_half = max(1, n_cells_y // 10)
        center = n_cells_y // 2
        struct[:, center - bar_half:center + bar_half] = 1.0
        return struct

    if mode == "y_split":
        bar_half = max(1, n_cells_y // 12)
        center = n_cells_y // 2
        top_center = (3 * n_cells_y) // 4
        bot_center = n_cells_y // 4
        x_split = n_cells_x // 3
        x_merge = (2 * n_cells_x) // 3

        # 输入段：中心直波导
        struct[:x_split, center - bar_half:center + bar_half] = 1.0

        # 分叉段：从中心平滑过渡到上下两端口
        for i, x in enumerate(range(x_split, x_merge)):
            t = i / max(1, (x_merge - x_split - 1))
            top_c = int(round(center + t * (top_center - center)))
            bot_c = int(round(center + t * (bot_center - center)))
            struct[x, top_c - bar_half:top_c + bar_half] = 1.0
            struct[x, bot_c - bar_half:bot_c + bar_half] = 1.0

        # 输出段：上下双波导
        struct[x_merge:, top_center - bar_half:top_center + bar_half] = 1.0
        struct[x_merge:, bot_center - bar_half:bot_center + bar_half] = 1.0
        return struct

    raise ValueError(f"Unknown test structure mode: {mode}")
