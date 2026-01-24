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
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        初始化伴随法优化器
        
        Args:
            simulator: 仿真器实例
            learning_rate: 学习率（梯度步长）
            num_iterations: 最大迭代次数
            damping: 阻尼系数（0-1，防止振荡）
            early_stop_patience: 无改进的耐心步数
            seed: 随机种子
            verbose: 是否打印优化过程
        """
        self.simulator = simulator
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.damping = damping
        self.early_stop_patience = early_stop_patience
        self.verbose = verbose
        
        if seed is not None:
            np.random.seed(seed)
        
        # 优化目标权重：与GA一致
        self.target_weights = {
            "te_port1": 1.0,
            "tm_port2": 1.0,
            "te_port2": -0.5,
            "tm_port1": -0.5,
        }
        
        # 优化历史
        self.history = {
            "fitness": [],
            "te_port1": [],
            "tm_port2": [],
            "total_efficiency": [],
            "crosstalk": [],
            "gradient_norm": [],
        }
    
    def _fitness(self, result: SimulationResult) -> float:
        """计算适应度"""
        return (
            self.target_weights["te_port1"] * result.te_port1
            + self.target_weights["tm_port2"] * result.tm_port2
            + self.target_weights["te_port2"] * result.te_port2
            + self.target_weights["tm_port1"] * result.tm_port1
        )
    
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
            else:
                raise ValueError(f"Unknown initialization mode: {init_mode}")
        
        if self.verbose:
            print("启动伴随法优化")
            print(f"学习率: {self.learning_rate}")
            print(f"最大迭代次数: {self.num_iterations}")
            print(f"阻尼系数: {self.damping}")
            print(f"初始化模式: {init_mode}")
            print(f"结构形状: {structure.shape}")
            print("-" * 60)
        
        best_structure = structure.copy()
        best_fitness = -np.inf
        no_improve_count = 0
        
        for iteration in range(self.num_iterations):
            # 1. 前向仿真：评估当前结构
            binary_structure = (structure > 0.5).astype(int)
            result = self.simulator.simulate(binary_structure, polarization="both")
            fitness = self._fitness(result)
            
            # 记录历史
            self.history["fitness"].append(fitness)
            self.history["te_port1"].append(result.te_port1)
            self.history["tm_port2"].append(result.tm_port2)
            self.history["total_efficiency"].append(result.total_efficiency)
            self.history["crosstalk"].append(result.crosstalk)
            
            # 2. 检查是否改进
            if fitness > best_fitness:
                best_fitness = fitness
                best_structure = structure.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 3. 计算梯度
            gradients = self.simulator.compute_gradients(
                structure,
                target_weights=self.target_weights
            )
            gradient_norm = np.linalg.norm(gradients)
            self.history["gradient_norm"].append(gradient_norm)
            
            # 4. 梯度上升更新
            update = self.learning_rate * gradients
            # 应用阻尼
            structure = structure + self.damping * update
            # 限制在[0, 1]
            structure = np.clip(structure, 0, 1)
            
            # 5. 打印进度
            if self.verbose and (iteration % max(1, self.num_iterations // 10) == 0 or iteration == self.num_iterations - 1):
                print(
                    f"迭代 {iteration:3d} | 适应度: {fitness:7.4f} | "
                    f"TE→P1: {result.te_port1:6.4f} | TM→P2: {result.tm_port2:6.4f} | "
                    f"梯度范数: {gradient_norm:7.4f} | 无改进计数: {no_improve_count}"
                )
            
            # 6. 早停
            if no_improve_count >= self.early_stop_patience:
                if self.verbose:
                    print(f"无改进达到耐心阈值 ({self.early_stop_patience})，提前停止")
                break
        
        # 最终二值化
        final_structure = (best_structure > 0.5).astype(int)
        best_result = self.simulator.simulate(final_structure, polarization="both")
        best_fitness_final = self._fitness(best_result)
        
        if self.verbose:
            print("-" * 60)
            print("优化完成！")
            print(f"最终适应度: {best_fitness_final:.4f}")
            print(f"TE→Port1效率: {best_result.te_port1:.4f}")
            print(f"TM→Port2效率: {best_result.tm_port2:.4f}")
            print(f"TE→Port2串扰: {best_result.te_port2:.4f}")
            print(f"TM→Port1串扰: {best_result.tm_port1:.4f}")
            print(f"总效率: {best_result.total_efficiency:.4f}")
            print(f"总串扰: {best_result.crosstalk:.4f}")
        
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
    init_mode: str = "random"
) -> Dict[str, Any]:
    """
    运行伴随法优化
    
    Args:
        simulator: 仿真器实例
        learning_rate: 学习率
        num_iterations: 迭代次数
        init_mode: 初始化模式
    
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
        early_stop_patience=15,
        seed=42,
        verbose=True
    )
    
    return optimizer.optimize(init_mode="half")


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


def plot_structure(structure: np.ndarray, title: str = "Optimized Structure", save_path: str = None):
    """
    绘制结构可视化
    
    Args:
        structure: 结构数组
        title: 图标题
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 转置以匹配坐标系
    structure_display = structure.T[::-1]  # Y轴反向
    
    im = ax.imshow(structure_display, cmap='gray', aspect='auto', origin='lower')
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Material (0=SiO2, 1=Si)")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结构图已保存到: {save_path}")
    
    plt.show()


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


def main():
    """主函数"""
    # 1. 初始化仿真器
    print("初始化MEEP仿真器...")
    simulator = setup_simulator()
    
    # 2. 运行伴随法优化
    print("\n运行伴随法优化...")
    adjoint_result = run_adjoint_optimization(
        simulator=simulator,
        learning_rate=0.01,
        num_iterations=100,
        init_mode="random"
    )
    
    # 3. 对比随机结构与优化结构
    compare_random_vs_optimized(simulator, adjoint_result["best_structure"])
    
    # 4. 绘制结果
    print("\n生成可视化...")
    plot_optimization_history(adjoint_result, save_path="./adjoint_optimization_history.png")
    plot_structure(
        adjoint_result["best_structure"],
        title="Adjoint Optimized PBS Structure",
        save_path="./adjoint_optimized_structure.png"
    )

    # 4.b 保存最佳结构和仿真结果组合图
    print(f"正在保存最佳结果可视化...")
    visualize_results(
        results=adjoint_result["best_result"],
        structure=adjoint_result["best_structure"],
        save_path="./adjoint_best_result.png",
        show=False
    )
    print(f"结果图已保存到: ./adjoint_best_result.png")
    
    # 5. 保存结果
    os.makedirs("./results", exist_ok=True)
    best_result = adjoint_result["best_result"]
    with open("./results/adjoint_performance.txt", "w") as f:
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
    
    print("\n结果已保存到 ./results/adjoint_performance.txt")
    
    print("\n" + "=" * 70)
    print("伴随法优化完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
