"""
伴随法（Adjoint Method）逆向设计示例

演示如何使用伴随法进行MMI PBS设计优化。
伴随法利用梯度信息进行高效的局部优化，是快速调优的有效方法。

使用项目的仿真环境进行优化。
"""

import sys
import os

# 添加父目录到路径以导入pbs_env模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from pathlib import Path
from utils import visualize_results

from meep_simulator import MMISimulator, load_simulation_config, SimulationResult


class AdjointOptimizer:
    """
    伴随法优化器 - 梯度下降优化
    
    利用MEEP伴随功能计算灵敏度/梯度，进行高效的结构优化。
    适合局部精细调优，收敛速度快。
    """
    
    def __init__(
        self,
        simulator: MMISimulator,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        damping: float = 0.5,
        early_stop_patience: int = 20,
        log_path: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        crosstalk_warmup_iters: int = 30,
        crosstalk_scale: float = 0.3
    ):
        """
        初始化伴随法优化器
        
        Args:
            simulator: 仿真器实例
            learning_rate: 学习率（梯度步长）
            num_iterations: 最大迭代次数
            damping: 阻尼系数（0-1，防止振荡）
            early_stop_patience: 无改进的耐心步数
            log_path: 日志文件路径（可选）
            seed: 随机种子
            verbose: 是否打印优化过程
        """
        self.simulator = simulator
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.damping = damping
        self.early_stop_patience = early_stop_patience
        self.verbose = verbose
        self.log_file = Path(log_path).expanduser() if log_path else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            # 清空旧日志文件，开始新的优化记录
            with self.log_file.open("w", encoding="utf-8") as f:
                f.write("")  # 创建空文件
        
        if seed is not None:
            np.random.seed(seed)
        
        # 优化目标权重：初始阶段仅关注拉升效率，无串扰惩罚 (两阶段优化)
        # 阶段1：TE/TM > 0.8 前，串扰权重为 0
        # 阶段2：满足条件后，串扰权重设为 -0.5
        self.target_weights = {
            "te_port1": 1.0,
            "tm_port2": 1.0,
            "te_port2": 0.0,
            "tm_port1": 0.0,
        }
        self.crosstalk_warmup_iters = max(0, int(crosstalk_warmup_iters))
        self.crosstalk_scale = float(crosstalk_scale)
        
        # 消光比惩罚系数（用于伴随法FOM）
        self.alpha_penalty = 1.0  # TE→P2 惩罚系数
        self.beta_penalty = 1.0   # TM→P1 惩罚系数
        
        # 优化历史
        self.history = {
            "fitness": [],
            "te_port1": [],
            "tm_port2": [],
            "total_efficiency": [],
            "crosstalk": [],
            "gradient_norm": [],
            "er_te_db": [],
            "er_tm_db": [],
        }
    
    def _log(self, msg: str):
        """同时写入控制台和文件的日志函数"""
        if self.verbose:
            print(msg)
        if self.log_file:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
    
    def _calculate_softmin(self, x: float, y: float, k: float = 15.0) -> tuple[float, float, float]:
        """
        计算 Softmin FOM 以及对应的梯度权重。
        Softmin = -1/k * log(e^(-k*x) + e^(-k*y)) 逼近 min(x, y)
        Returns: (fom, w_x, w_y)
        """
        # 数值稳定性：避免溢出
        max_val = max(x, y)
        exp_x = np.exp(-k * (x - max_val))
        exp_y = np.exp(-k * (y - max_val))
        denom = exp_x + exp_y
        
        fom = max_val - (1.0 / k) * np.log(denom)
        
        # Softmax 权重 (导数): 较小值获得更大权重
        w_x = exp_x / denom
        w_y = exp_y / denom
        
        return fom, w_x, w_y
    
    def _calculate_extinction_ratio_fom(self, result, k: float = 10.0) -> tuple[float, dict]:
        """
        计算基于消光比的FOM，直接优化PBS的核心指标。
        
        ER_TE = TE→Port1 / TE→Port2 (期望越高越好)
        ER_TM = TM→Port2 / TM→Port1 (期望越高越好)
        
        为了数值稳定，使用对数形式:
        log_ER_TE = log(TE→P1) - log(TE→P2)
        log_ER_TM = log(TM→P2) - log(TM→P1)
        
        FOM = Softmin(log_ER_TE, log_ER_TM) 确保两个偏振都达到高消光比
        
        Returns: (fom, gradient_weights_dict)
        """
        eps = 1e-8  # 防止log(0)
        
        # 计算对数消光比 (单位: nepers，可转换为dB: *10/ln(10))
        log_er_te = np.log(result.te_port1 + eps) - np.log(result.te_port2 + eps)
        log_er_tm = np.log(result.tm_port2 + eps) - np.log(result.tm_port1 + eps)
        
        # Softmin 作用于对数消光比
        fom, w_te, w_tm = self._calculate_softmin(log_er_te, log_er_tm, k=k)
        
        # 计算各端口的梯度权重
        # d(log_ER_TE)/d(te_port1) = 1/te_port1,  d(log_ER_TE)/d(te_port2) = -1/te_port2
        # d(log_ER_TM)/d(tm_port2) = 1/tm_port2,  d(log_ER_TM)/d(tm_port1) = -1/tm_port1
        gradient_weights = {
            "te_port1": w_te / (result.te_port1 + eps),
            "te_port2": -w_te / (result.te_port2 + eps),
            "tm_port2": w_tm / (result.tm_port2 + eps),
            "tm_port1": -w_tm / (result.tm_port1 + eps),
        }
        
        return fom, gradient_weights
    
    def _fitness(self, result: SimulationResult, weights: Optional[Dict[str, float]] = None) -> float:
        """计算适应度"""
        weights = weights or self.target_weights
        # === 改进的 FOM 定义 ===
        # 目标：最大化偏振分离效果
        # FOM = (TE→P1 - TE→P2) + (TM→P2 - TM→P1)
        #     = 目标端口效率 - 串扰效率
        # 
        # 这样的定义同时实现：
        # 1. 最大化目标端口的电场强度 |E|²
        # 2. 最小化串扰（泄漏到错误端口的能量）
        # 3. 鼓励偶振分离
        te_separation = result.te_port1 - result.te_port2  # TE 应去 P1
        tm_separation = result.tm_port2 - result.tm_port1  # TM 应去 P2
        return te_separation + tm_separation

    def _get_adaptive_crosstalk_weight(self, result, base_weight: float = 0.3) -> float:
        """
        自适应串扰权重：根据当前效率动态调整。
        效率越高，串扰惩罚越重，促进精细调优。
        """
        min_eff = min(result.te_port1, result.tm_port2)
        # 效率<0.5时权重为0，效率>0.9时权重达到base_weight
        scale = np.clip((min_eff - 0.5) / 0.4, 0.0, 1.0)
        return base_weight * scale
    
    def optimize(
        self,
        init_structure: Optional[np.ndarray] = None,
        init_mode: str = "random"
    ) -> Dict[str, Any]:
        """
        运行伴随法优化
        
        Args:
            init_structure: 初始结构（可选，若提供则忽略init_mode）
            init_mode: 初始化模式 "random", "zeros", "ones", "half"
        
        Returns:
            优化结果字典
        """
        
        # 初始化结构
        if init_structure is not None:
            structure = init_structure.copy().astype(np.float32)
        else:
            if init_mode == "random":
                structure = np.random.rand(
                    self.simulator.config.n_cells_x,
                    self.simulator.config.n_cells_y
                ).astype(np.float32)
            elif init_mode == "zeros":
                structure = np.zeros(
                    (self.simulator.config.n_cells_x, self.simulator.config.n_cells_y),
                    dtype=np.float32
                )
            elif init_mode == "ones":
                structure = np.ones(
                    (self.simulator.config.n_cells_x, self.simulator.config.n_cells_y),
                    dtype=np.float32
                )
            elif init_mode == "half":
                structure = 0.5 * np.ones(
                    (self.simulator.config.n_cells_x, self.simulator.config.n_cells_y),
                    dtype=np.float32
                )
            elif init_mode == "gray_noise":
                # 灰度背景 (0.5) + 微弱随机噪声 (±0.05)
                base = 0.5 * np.ones(
                    (self.simulator.config.n_cells_x, self.simulator.config.n_cells_y),
                    dtype=np.float32
                )
                noise = np.random.uniform(-0.05, 0.05, base.shape).astype(np.float32)
                structure = np.clip(base + noise, 0, 1)
            elif init_mode == "y_branch":
                # Y型分支初始化
                structure = np.zeros(
                    (self.simulator.config.n_cells_x, self.simulator.config.n_cells_y),
                    dtype=np.float32
                )
                nx, ny = structure.shape
                
                # 几何参数 (基于wg_width计算像素宽度，因为去掉了taper)
                cfg = self.simulator.config
                # 使用 wg_width 而不是 taper_width
                wg_width_ratio = cfg.wg_width / cfg.mmi_width
                radius_cells = (wg_width_ratio * ny) / 2
                bar_half = max(1, int(round(radius_cells)))
                
                center_y = ny // 2
                # 修改：输出端口移至角落
                top_y = ny - bar_half
                bot_y = bar_half
                
                # 修改：减少直波导长度
                # 输入直波导缩减为 1/8 长度 (原为 1/3)
                x_split = nx // 8
                # 输出直波导缩减为 1/8 长度
                x_merge = (7 * nx) // 8
                
                # 1. 输入直波导
                structure[:x_split, center_y - bar_half : center_y + bar_half] = 1.0
                
                # 2. 分叉段 (平滑过渡)
                for x in range(x_split, x_merge):
                    # 归一化进度 t: 0 -> 1
                    t = (x - x_split) / max(1, x_merge - x_split - 1)
                    
                    # 计算上下路径中心
                    cy_top = int(round(center_y + t * (top_y - center_y)))
                    cy_bot = int(round(center_y + t * (bot_y - center_y)))
                    
                    structure[x, cy_top - bar_half : cy_top + bar_half] = 1.0
                    structure[x, cy_bot - bar_half : cy_bot + bar_half] = 1.0
                
                # 3. 输出直波导
                structure[x_merge:, top_y - bar_half : top_y + bar_half] = 1.0
                structure[x_merge:, bot_y - bar_half : bot_y + bar_half] = 1.0
                
            else:
                raise ValueError(f"Unknown initialization mode: {init_mode}")
        
        self._log("启动伴随法优化")
        self._log(f"初始学习率: {self.learning_rate}")
        self._log(f"最大迭代次数: {self.num_iterations}")
        self._log(f"阻尼系数: {self.damping}")
        self._log(f"初始化模式: {init_mode}")
        self._log(f"结构形状: {structure.shape}")
        if self.log_file:
            self._log(f"日志文件: {self.log_file}")
            
        # 可视化初始结构
        if self.log_file:
            init_save_path = self.log_file.parent / "initial_structure.png"
            plot_structure(
                structure, 
                title=f"Initial Structure ({init_mode})", 
                save_path=str(init_save_path),
                config=self.simulator.config,
                show=False
            )
            self._log(f"初始结构已保存到: {init_save_path}")
            
        self._log("-" * 60)
        
        best_structure = structure.copy()
        best_fitness = -np.inf
        no_improve_count = 0
        
        # 自适应学习率相关参数
        current_lr = self.learning_rate
        min_lr = self.learning_rate * 0.01  # 最小学习率为初始的1%
        lr_decay = 0.5  # 学习率衰减因子
        prev_fitness = -np.inf
        prev_structure = structure.copy()
        consecutive_drops = 0  # 连续下降计数
        max_consecutive_drops = 3  # 允许的最大连续下降次数
        
        
        for iteration in range(self.num_iterations):
            # 1. 前向仿真：评估当前结构
            # 直接传入连续结构 (float array)，让 MEEP MaterialGrid 处理平滑过渡
            # 这确保梯度计算和目标函数评估在同一"连续空间"中进行
            result = self.simulator.simulate(structure, polarization="both")
            
            # === 效率Softmin + 自适应串扰惩罚 ===
            # === 修改后的 FOM 定义 ===
            # 目标：最大化目标端口的电场强度模平方 (即传输效率)
            # TE -> Port 1 (Upper)
            # TM -> Port 2 (Lower)
            # FOM = TE_Port1 + TM_Port2
            
            # === 消光比优化 FOM ===
            # 目标：在奖励目标信号的同时惩罚错误端口信号
            #
            # FOM = (T_TE→P1 - α·T_TE→P2) + (T_TM→P2 - β·T_TM→P1)
            #
            # 物理意义：
            # - 第一项：奖励TE去P1，同时惩罚TE去P2（消光比优化）
            # - 第二项：奖励TM去P2，同时惩罚TM去P1（消光比优化）
            #
            # 伴随源设置（相位控制技巧）：
            # - TE Case: Port1 发射 +1 幅度，Port2 发射 -α 幅度（相位相反）
            # - TM Case: Port2 发射 +1 幅度，Port1 发射 -β 幅度（相位相反）
            # 优点：一次伴随仿真即可同时计算"增加目标+减少串扰"的梯度
            
            te_separation = result.te_port1 - self.alpha_penalty * result.te_port2
            tm_separation = result.tm_port2 - self.beta_penalty * result.tm_port1
            fitness = te_separation + tm_separation
            
            # 梯度权重（对应伴随源配置）
            # d(FOM)/d(te_port1) = +1.0 (Port1 发射 +1)
            # d(FOM)/d(te_port2) = -α  (Port2 发射 -α)
            # d(FOM)/d(tm_port2) = +1.0 (Port2 发射 +1)
            # d(FOM)/d(tm_port1) = -β  (Port1 发射 -β)
            current_weights = {
                "te_port1": 1.0,
                "te_port2": -self.alpha_penalty,
                "tm_port2": 1.0,
                "tm_port1": -self.beta_penalty,
            }
            
            # 计算消光比（用于日志显示，单位dB）
            eps = 1e-8
            er_te_db = 10 * np.log10((result.te_port1 + eps) / (result.te_port2 + eps))
            er_tm_db = 10 * np.log10((result.tm_port2 + eps) / (result.tm_port1 + eps))
            
            # 更新 self.target_weights 供 compute_gradients 使用
            self.target_weights = current_weights.copy()
            
            # 记录历史
            self.history["fitness"].append(fitness)
            self.history["te_port1"].append(result.te_port1)
            self.history["tm_port2"].append(result.tm_port2)
            self.history["total_efficiency"].append(result.total_efficiency)
            self.history["crosstalk"].append(result.crosstalk)
            self.history["er_te_db"].append(er_te_db)
            self.history["er_tm_db"].append(er_tm_db)
            
            # 2. 检查是否改进
            # 注意：在自适应权重调整过程中（如Softmin），fitness定义会变化，直接比较fitness可能不准。
            # 这里我们依然比较fitness，因为我们希望Softmin值上升（代表最小效率上升）。
            # 为了避免微小波动导致no_improve_count累积，增加一个容差
            if fitness > best_fitness + 1e-6:
                best_fitness = fitness
                best_structure = structure.copy()
                no_improve_count = 0
            else:
                # 即使没有突破历史最佳，只要还在上升（或持平），也不算"无改进"
                # 只有当性能明显下降或长期停滞时才计数
                # 这里简化为：只要比上一步好（或者没变差太多），就重置局部计数
                # 但为了逻辑严谨，我们保持原义：no_improve_count 指的是"距离上一次打破历史记录"的步数
                no_improve_count += 1
            
            # === 自适应学习率：回退机制 ===
            # 检测是否性能显著下降（相比上一步）
            if iteration > 0 and fitness < prev_fitness - 0.05:  # 下降超过0.05视为显著
                consecutive_drops += 1
                if consecutive_drops >= max_consecutive_drops:
                    # 回退到上一步的结构，并降低学习率
                    structure = prev_structure.copy()
                    current_lr = max(current_lr * lr_decay, min_lr)
                    consecutive_drops = 0
                    self._log(f"  >>> 检测到梯度崩溃，回退结构并降低学习率至 {current_lr:.6f}")
            else:
                consecutive_drops = 0
            
            # 保存当前状态用于下一步比较
            prev_fitness = fitness
            prev_structure = structure.copy()
            
            # 3. 计算梯度
            gradients = self.simulator.compute_gradients(
                structure,
                target_weights=current_weights
            )
            gradient_norm = np.linalg.norm(gradients)
            self.history["gradient_norm"].append(gradient_norm)
            
            # 梯度归一化：控制最大更新步长 (使用分位数避免奇异值影响)
            # max_grad = np.max(np.abs(gradients))
            # 使用99.9%分位数作为缩放基准，防止个别极端梯度值抑制整体更新
            scale = np.percentile(np.abs(gradients), 99.9)
            if scale > 1e-8:
                gradients = gradients / scale
            
            # 裁剪过大的梯度值，确保稳定性
            gradients = np.clip(gradients, -1.0, 1.0)
            
            # 4. 梯度上升更新（使用自适应学习率）
            update = current_lr * gradients
            # 应用阻尼
            structure = structure + self.damping * update
            # 限制在[0, 1]
            structure = np.clip(structure, 0, 1)
            
            # 5. 打印进度
            # if iteration % max(1, self.num_iterations // 10) == 0 or iteration == self.num_iterations - 1:
            self._log(
                f"迭代 {iteration:3d} | FOM: {fitness:7.3f} | "
                f"TE→P1: {result.te_port1:.3f} | TM→P2: {result.tm_port2:.3f} | "
                f"ER_TE: {er_te_db:5.1f}dB | ER_TM: {er_tm_db:5.1f}dB | "
                f"LR: {current_lr:.5f}"
            )
            
            # 6. 早停
            if no_improve_count >= self.early_stop_patience:
                self._log(f"无改进达到耐心阈值 ({self.early_stop_patience})，提前停止")
                break
        
        # 最终二值化
        final_structure = (best_structure > 0.5).astype(int)
        best_result = self.simulator.simulate(final_structure, polarization="both")
        best_fitness_final = self._fitness(best_result)
        
        self._log("-" * 60)
        self._log("优化完成！")
        self._log(f"最终适应度: {best_fitness_final:.4f}")
        self._log(f"TE→Port1效率: {best_result.te_port1:.4f}")
        self._log(f"TM→Port2效率: {best_result.tm_port2:.4f}")
        self._log(f"TE→Port2串扰: {best_result.te_port2:.4f}")
        self._log(f"TM→Port1串扰: {best_result.tm_port1:.4f}")
        self._log(f"总效率: {best_result.total_efficiency:.4f}")
        self._log(f"总串扰: {best_result.crosstalk:.4f}")
        
        return {
            "best_structure": final_structure,
            "best_fitness": best_fitness_final,
            "best_result": best_result,
            "continuous_structure": best_structure,
            "history": self.history,
            "num_iterations": iteration + 1,
        }
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """获取优化历史"""
        return self.history.copy()


def setup_simulator(config_path: str = None) -> MMISimulator:
    """从YAML加载配置并构建仿真器"""
    config, num_workers = load_simulation_config(config_path)
    return MMISimulator(config=config, num_workers=num_workers)


def run_adjoint_optimization(
    simulator: MMISimulator,
    learning_rate: float = 0.01,
    num_iterations: int = 100,
    init_mode: str = "random",
    log_path: str = "./logs/adjoint_run.txt"
) -> Dict[str, Any]:
    """
    运行伴随法优化
    
    Args:
        simulator: 仿真器实例
        learning_rate: 学习率
        num_iterations: 迭代次数
        init_mode: 初始化模式
        log_path: 日志文件保存路径
    
    Returns:
        优化结果字典
    """
    print("\n" + "=" * 70)
    print("伴随法（Adjoint Method）- 设计优化")
    print("=" * 70)
    
    optimizer = AdjointOptimizer(
        simulator=simulator,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        damping=0.5,
        early_stop_patience=50,  # 增加耐心值
        log_path=log_path,
        seed=42,
        verbose=True,
        crosstalk_warmup_iters=30,
        crosstalk_scale=0.3
    )
    
    return optimizer.optimize(init_mode=init_mode)


def evaluate_structure(simulator: MMISimulator, structure: np.ndarray) -> Dict[str, float]:
    """
    评估结构的设计指标
    
    Args:
        simulator: 仿真器实例
        structure: 结构数组
    
    Returns:
        设计指标字典
    """
    result = simulator.simulate(structure, polarization="both")
    
    metrics = {
        "te_port1": result.te_port1,
        "te_port2": result.te_port2,
        "tm_port1": result.tm_port1,
        "tm_port2": result.tm_port2,
        "total_efficiency": result.total_efficiency,
        "crosstalk": result.crosstalk,
        "te_extinction_ratio": 10 * np.log10((result.te_port1 + 1e-10) / (result.te_port2 + 1e-10)),
        "tm_extinction_ratio": 10 * np.log10((result.tm_port2 + 1e-10) / (result.tm_port1 + 1e-10))
    }
    
    return metrics


def plot_optimization_history(adjoint_result: Dict[str, Any], save_path: str = None):
    """
    绘制优化历史
    
    Args:
        adjoint_result: 伴随法优化结果
        save_path: 保存路径（可选）
    """
    history = adjoint_result["history"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = range(len(history["fitness"]))
    
    # 适应度演化
    ax = axes[0, 0]
    ax.plot(iterations, history["fitness"], 'r-', label="Fitness", linewidth=2)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness")
    ax.set_title("Adjoint Method - Fitness Evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # TE→Port1和TM→Port2效率
    ax = axes[0, 1]
    ax.plot(iterations, history["te_port1"], 'b-', label="TE->Port1", linewidth=2)
    ax.plot(iterations, history["tm_port2"], 'g-', label="TM->Port2", linewidth=2)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Efficiency")
    ax.set_title("Port Efficiency Evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 总效率和串扰
    ax = axes[1, 0]
    ax.plot(iterations, history["total_efficiency"], 'g-', label="Total Eff", linewidth=2)
    ax.plot(iterations, history["crosstalk"], 'r-', label="Crosstalk", linewidth=2)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Efficiency")
    ax.set_title("Total Efficiency vs Crosstalk")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 梯度范数
    ax = axes[1, 1]
    ax.semilogy(iterations, history["gradient_norm"], 'purple', linewidth=2)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Gradient Norm (Log)")
    ax.set_title("Gradient Convergence")
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"优化历史已保存到: {save_path}")
    
    plt.show()


def plot_structure(
    structure: np.ndarray, 
    title: str = "Optimized Structure", 
    save_path: str = None,
    config: Optional[Any] = None,
    show: bool = True
):
    """
    绘制结构可视化 (支持物理尺寸坐标)
    
    Args:
        structure: 结构数组
        title: 图标题
        save_path: 保存路径（可选）
        config: 仿真配置对象（用于获取物理尺寸）
        show: 是否显示图像
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 转置以匹配坐标系
    structure_display = structure.T[::-1]  # Y轴反向
    
    # 确定坐标范围
    if config:
        # 如果有配置，使用微米单位
        extent = [
            -config.mmi_length / 2, 
            config.mmi_length / 2, 
            -config.mmi_width / 2, 
            config.mmi_width / 2
        ]
        xlabel = "X (μm)"
        ylabel = "Y (μm)"
    else:
        # 否则使用网格索引
        extent = [0, structure.shape[0], 0, structure.shape[1]]
        xlabel = "X (cells)"
        ylabel = "Y (cells)"
    
    im = ax.imshow(
        structure_display, 
        cmap='gray', 
        aspect='equal',  # 保持物理比例
        origin='lower',
        extent=extent
    )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Material (0=SiO2, 1=Si)")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结构图已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def compare_random_vs_optimized(simulator: MMISimulator, optimized_structure: np.ndarray):
    """
    对比随机结构与优化结构
    
    Args:
        simulator: 仿真器实例
        optimized_structure: 优化后的结构
    """
    print("\n" + "=" * 70)
    print("随机结构 vs 伴随法优化结构对比")
    print("=" * 70)
    
    # 随机结构
    random_structure = np.random.randint(0, 2, size=(simulator.config.n_cells_x, simulator.config.n_cells_y))
    random_metrics = evaluate_structure(simulator, random_structure)
    
    # 优化结构
    opt_metrics = evaluate_structure(simulator, optimized_structure)
    
    print(f"\n{'指标':<25} {'随机结构':<15} {'优化结构':<15} {'改进':<15}")
    print("-" * 70)
    
    for key in ["te_port1", "tm_port2", "total_efficiency", "crosstalk"]:
        rand_val = random_metrics[key]
        opt_val = opt_metrics[key]
        improvement = opt_val - rand_val
        
        if key == "crosstalk":
            print(f"{key:<25} {rand_val:>14.4f} {opt_val:>14.4f} {improvement:>+14.4f} (低更好)")
        else:
            print(f"{key:<25} {rand_val:>14.4f} {opt_val:>14.4f} {improvement:>+14.4f}")
    
    print("\n消光比对比 (dB):")
    print(f"{'TE消光比':<25} {random_metrics['te_extinction_ratio']:>6.2f} dB       {opt_metrics['te_extinction_ratio']:>6.2f} dB")
    print(f"{'TM消光比':<25} {random_metrics['tm_extinction_ratio']:>6.2f} dB       {opt_metrics['tm_extinction_ratio']:>6.2f} dB")


def save_results_to_file(adjoint_result: Dict[str, Any], save_dir: str = "./results/adjoint_results"):
    """
    保存伴随法优化结果到文件
    
    Args:
        adjoint_result: 优化结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    best_structure = adjoint_result["best_structure"]
    best_result = adjoint_result["best_result"]
    history = adjoint_result["history"]
    
    # 保存最优结构
    np.save(f"{save_dir}/best_structure_adjoint.npy", best_structure)
    
    # 保存连续结构（二值化前）
    if "continuous_structure" in adjoint_result:
        np.save(f"{save_dir}/continuous_structure_adjoint.npy", adjoint_result["continuous_structure"])
    
    # 保存优化历史
    np.save(f"{save_dir}/adjoint_history.npy", history)
    
    # 保存性能指标到文本文件
    with open(f"{save_dir}/adjoint_performance.txt", "w") as f:
        f.write("伴随法优化结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"最优适应度: {adjoint_result['best_fitness']:.4f}\n")
        f.write(f"TE→Port1效率: {best_result.te_port1:.4f}\n")
        f.write(f"TM→Port2效率: {best_result.tm_port2:.4f}\n")
        f.write(f"TE→Port2串扰: {best_result.te_port2:.4f}\n")
        f.write(f"TM→Port1串扰: {best_result.tm_port1:.4f}\n")
        f.write(f"总效率: {best_result.total_efficiency:.4f}\n")
        f.write(f"总串扰: {best_result.crosstalk:.4f}\n")
        f.write(f"\n迭代次数: {adjoint_result['num_iterations']}\n")
        
        # 计算消光比
        te_er = 10 * np.log10((best_result.te_port1 + 1e-10) / (best_result.te_port2 + 1e-10))
        tm_er = 10 * np.log10((best_result.tm_port2 + 1e-10) / (best_result.tm_port1 + 1e-10))
        f.write(f"TE消光比: {te_er:.2f} dB\n")
        f.write(f"TM消光比: {tm_er:.2f} dB\n")
    
    print(f"\n结果已保存到 {save_dir}:")
    print(f"  - 最优结构: {save_dir}/best_structure_adjoint.npy")
    print(f"  - 优化历史: {save_dir}/adjoint_history.npy")
    print(f"  - 性能指标: {save_dir}/adjoint_performance.txt")


def main():
    """主函数"""
    # 定义保存目录
    SAVE_DIR = "./results/adjoint_results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 初始化仿真器
    print("初始化MEEP仿真器...")
    simulator = setup_simulator()
    
    # 2. 运行伴随法优化
    print("\n运行伴随法优化...")
    adjoint_result = run_adjoint_optimization(
        simulator=simulator,
        learning_rate=0.01,      # 降低学习率，避免振荡
        num_iterations=200,      # 迭代次数
        init_mode="half",        # 使用平衡初始化 (0.5 灰度)
        log_path=f"{SAVE_DIR}/adjoint_run.txt"
    )
    
    # 3. 对比随机结构与优化结构
    compare_random_vs_optimized(simulator, adjoint_result["best_structure"])
    
    # 4. 绘制结果
    print("\n生成可视化...")
    plot_optimization_history(adjoint_result, save_path=f"{SAVE_DIR}/adjoint_optimization_history.png")
    plot_structure(
        adjoint_result["best_structure"],
        title="Adjoint Optimized PBS Structure",
        save_path=f"{SAVE_DIR}/adjoint_optimized_structure.png",
        config=simulator.config  # 传入配置以显示物理坐标
    )

    # 4.b 保存最佳结构和仿真结果组合图
    print(f"正在保存最佳结果可视化...")
    visualize_results(
        results=adjoint_result["best_result"],
        structure=adjoint_result["best_structure"],
        save_path=f"{SAVE_DIR}/adjoint_best_result.png",
        show=False
    )
    print(f"结果图已保存到: {SAVE_DIR}/adjoint_best_result.png")
    
    # 5. 保存结果到统一目录
    save_results_to_file(adjoint_result, save_dir=SAVE_DIR)
    
    print("\n" + "=" * 70)
    print("伴随法优化完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
