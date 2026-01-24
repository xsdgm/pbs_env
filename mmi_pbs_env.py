"""
MMI PBS 强化学习环境

基于MEEP FDTD仿真的1×2 MMI偏振分束器设计环境。
目标：TE模式输出到端口1，TM模式输出到端口2。
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from functools import partial

import gymnasium as gym
from gymnasium import spaces


from .core import Simulator, SimulationResult
from .meep_simulator import MMISimulator, SimulationConfig
from .utils import compute_reward



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
    
    def __init__(
        self,
        # 结构参数
        n_cells_x: int = 30,
        n_cells_y: int = 8,
        init_mode: str = "random",  # "random", "ones", "zeros", "half"
        
        # 仿真参数
        simulator: Optional[Simulator] = None,
        wavelength: float = 1.55,
        mmi_width: float = 4.0,
        mmi_length: float = 15.0,
        resolution: int = 10,  # 速度优先，降低分辨率
        run_time: float = 50,  # 仿真时间，单位 1/f

        
        # 并行参数
        num_workers: int = 12,
        
        # 奖励参数
        reward_alpha: float = 1.0,  # TE_port1权重
        reward_beta: float = 1.0,   # TM_port2权重
        reward_gamma: float = 0.5,  # 串扰惩罚权重
        reward_type: str = "absolute",  # "absolute" or "delta"
        
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
        
        # 保存参数
        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.n_cells = n_cells_x * n_cells_y
        self.init_mode = init_mode
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_gamma = reward_gamma
        self.reward_type = reward_type
        self.render_mode = render_mode
        
        if simulator is not None:
            self.simulator = simulator
            # 尝试从simulator获取配置
            if hasattr(simulator, 'config'):
                # 简单同步参数
                pass
        else:
            # 创建默认仿真器
            self.sim_config = SimulationConfig(
                wavelength=wavelength,
                mmi_width=mmi_width,
                mmi_length=mmi_length,
                resolution=resolution,
                run_time=run_time,
                n_cells_x=n_cells_x,
                n_cells_y=n_cells_y,
            )
            self.simulator = MMISimulator(self.sim_config, num_workers=num_workers)
        
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_cells_x, n_cells_y),
            dtype=np.float32
        )
        
        # 定义动作空间 (离散：翻转某个像素)
        self.action_space = spaces.Discrete(self.n_cells)
        
        # 状态变量
        self.structure = None
        self.current_results = None
        self.prev_reward = 0.0
        self.step_count = 0
        
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
        # 解析动作
        x = action // self.n_cells_y
        y = action % self.n_cells_y
        
        # 翻转像素
        self.structure[x, y] = 1.0 - self.structure[x, y]
        
        # 运行仿真
        self.current_results = self.simulator.simulate(self.structure)
        
        # 计算奖励
        current_reward = self._compute_reward()
        if self.reward_type == "delta":
            reward = current_reward - self.prev_reward
        else:
            reward = current_reward
        
        self.prev_reward = current_reward
        self.episode_rewards.append(reward)
        
        # 更新最佳结果
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.best_structure = self.structure.copy()
        
        # 更新步数
        self.step_count += 1
        
        # 获取观察和信息
        obs = self._get_observation()
        info = self._get_info()
        
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
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        return self.structure.copy()
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        if self.current_results is None:
            return 0.0
        
        # current_results 现在是 SimulationResult 对象
        return compute_reward(
            te_port1=self.current_results.te_port1,
            tm_port2=self.current_results.tm_port2,
            te_port2=self.current_results.te_port2,
            tm_port1=self.current_results.tm_port1,
            alpha=self.reward_alpha,
            beta=self.reward_beta,
            gamma=self.reward_gamma
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
