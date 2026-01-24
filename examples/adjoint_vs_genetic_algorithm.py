"""
遗传算法 vs 伴随法 对比分析

对比遗传算法与伴随法（adjoint method）的优化性能，
作为baseline算法评估框架。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, Tuple

from meep_simulator import MMISimulator, SimulationConfig, GeneticAlgorithmOptimizer
from utils import compute_reward


class AdjointOptimizer:
    """
    伴随法优化器 - 梯度下降方法
    
    用于与遗传算法进行性能对比
    """
    
    def __init__(
        self,
        simulator: MMISimulator,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        verbose: bool = True
    ):
        """
        初始化伴随法优化器
        
        Args:
            simulator: 仿真器实例
            learning_rate: 学习率
            num_iterations: 迭代次数
            verbose: 是否打印过程
        """
        self.simulator = simulator
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        
        self.history = {
            "fitness": [],
            "te_port1": [],
            "tm_port2": [],
            "total_efficiency": [],
            "crosstalk": []
        }
    
    def optimize(self, init_structure: np.ndarray = None) -> Dict[str, Any]:
        """
        运行伴随法优化
        
        Args:
            init_structure: 初始结构（可选）
        
        Returns:
            优化结果字典
        """
        if init_structure is None:
            structure = np.random.randint(0, 2, size=(30, 8))
        else:
            structure = init_structure.copy().astype(np.float32)
        
        if self.verbose:
            print(f"启动伴随法优化")
            print(f"学习率: {self.learning_rate}")
            print(f"迭代次数: {self.num_iterations}")
            print("-" * 60)
        
        for iteration in range(self.num_iterations):
            # 评估当前结构
            result = self.simulator.simulate(structure, polarization="both")
            
            # 计算适应度
            fitness = result.te_port1 + result.tm_port2 - 0.5 * (result.te_port2 + result.tm_port1)
            
            self.history["fitness"].append(fitness)
            self.history["te_port1"].append(result.te_port1)
            self.history["tm_port2"].append(result.tm_port2)
            self.history["total_efficiency"].append(result.total_efficiency)
            self.history["crosstalk"].append(result.crosstalk)
            
            if self.verbose and (iteration % max(1, self.num_iterations // 10) == 0 or iteration == self.num_iterations - 1):
                print(f"迭代 {iteration:3d} | 适应度: {fitness:7.4f} | "
                      f"TE→P1: {result.te_port1:6.4f} | TM→P2: {result.tm_port2:6.4f}")
            
            # 计算梯度
            target_weights = {"te_port1": 1.0, "tm_port2": 1.0}
            gradients = self.simulator.compute_gradients(structure, target_weights)
            
            # 更新结构 (梯度上升)
            structure = structure + self.learning_rate * gradients
            
            # 归一化到 [0, 1]
            structure = np.clip(structure, 0, 1)
        
        # 最终二值化
        best_structure = (structure > 0.5).astype(int)
        best_result = self.simulator.simulate(best_structure, polarization="both")
        best_fitness = best_result.te_port1 + best_result.tm_port2 - 0.5 * (best_result.te_port2 + best_result.tm_port1)
        
        if self.verbose:
            print("-" * 60)
            print("优化完成！")
            print(f"最优结构适应度: {best_fitness:.4f}")
        
        return {
            "best_structure": best_structure,
            "best_fitness": best_fitness,
            "best_result": best_result,
            "history": self.history,
            "final_structure": structure
        }


def compare_algorithms(
    simulator: MMISimulator,
    ga_params: Dict[str, Any] = None,
    adjoint_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    对比遗传算法与伴随法
    
    Args:
        simulator: 仿真器实例
        ga_params: 遗传算法参数
        adjoint_params: 伴随法参数
    
    Returns:
        对比结果
    """
    
    # 默认参数
    if ga_params is None:
        ga_params = {
            "pop_size": 30,
            "num_generations": 50,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elite_ratio": 0.1
        }
    
    if adjoint_params is None:
        adjoint_params = {
            "learning_rate": 0.01,
            "num_iterations": 50
        }
    
    results = {}
    
    # 1. 遗传算法
    print("\n" + "=" * 70)
    print("遗传算法（GA）优化")
    print("=" * 70)
    
    start_time = time.time()
    
    ga_optimizer = GeneticAlgorithmOptimizer(
        simulator=simulator,
        pop_size=ga_params["pop_size"],
        num_generations=ga_params["num_generations"],
        mutation_rate=ga_params["mutation_rate"],
        crossover_rate=ga_params["crossover_rate"],
        elite_ratio=ga_params["elite_ratio"],
        seed=42,
        verbose=True
    )
    
    ga_result = ga_optimizer.optimize(init_mode="random")
    ga_time = time.time() - start_time
    
    results["ga"] = {
        "result": ga_result,
        "time": ga_time,
        "params": ga_params
    }
    
    print(f"\nGA优化耗时: {ga_time:.2f}秒")
    
    # 2. 伴随法
    print("\n" + "=" * 70)
    print("伴随法（Adjoint Method）优化")
    print("=" * 70)
    
    start_time = time.time()
    
    adjoint_optimizer = AdjointOptimizer(
        simulator=simulator,
        learning_rate=adjoint_params["learning_rate"],
        num_iterations=adjoint_params["num_iterations"],
        verbose=True
    )
    
    adjoint_result = adjoint_optimizer.optimize()
    adjoint_time = time.time() - start_time
    
    results["adjoint"] = {
        "result": adjoint_result,
        "time": adjoint_time,
        "params": adjoint_params
    }
    
    print(f"\n伴随法优化耗时: {adjoint_time:.2f}秒")
    
    return results


def print_comparison_summary(comparison_results: Dict[str, Any]):
    """
    打印对比总结
    
    Args:
        comparison_results: 对比结果
    """
    print("\n" + "=" * 70)
    print("算法性能对比总结")
    print("=" * 70)
    
    ga_result = comparison_results["ga"]["result"]
    ga_time = comparison_results["ga"]["time"]
    
    adjoint_result = comparison_results["adjoint"]["result"]
    adjoint_time = comparison_results["adjoint"]["time"]
    
    ga_perf = ga_result["best_result"]
    adj_perf = adjoint_result["best_result"]
    
    print(f"\n{'指标':<30} {'遗传算法':<18} {'伴随法':<18} {'优劣':<15}")
    print("-" * 81)
    
    # TE到端口1
    ga_te_p1 = ga_perf.te_port1
    adj_te_p1 = adj_perf.te_port1
    winner = "GA胜" if ga_te_p1 > adj_te_p1 else ("伴随法胜" if adj_te_p1 > ga_te_p1 else "平局")
    print(f"{'TE→Port1效率':<30} {ga_te_p1:<18.4f} {adj_te_p1:<18.4f} {winner:<15}")
    
    # TM到端口2
    ga_tm_p2 = ga_perf.tm_port2
    adj_tm_p2 = adj_perf.tm_port2
    winner = "GA胜" if ga_tm_p2 > adj_tm_p2 else ("伴随法胜" if adj_tm_p2 > ga_tm_p2 else "平局")
    print(f"{'TM→Port2效率':<30} {ga_tm_p2:<18.4f} {adj_tm_p2:<18.4f} {winner:<15}")
    
    # 总效率
    ga_total = ga_perf.total_efficiency
    adj_total = adj_perf.total_efficiency
    winner = "GA胜" if ga_total > adj_total else ("伴随法胜" if adj_total > ga_total else "平局")
    print(f"{'总效率':<30} {ga_total:<18.4f} {adj_total:<18.4f} {winner:<15}")
    
    # 串扰
    ga_crosstalk = ga_perf.crosstalk
    adj_crosstalk = adj_perf.crosstalk
    winner = "GA胜" if ga_crosstalk < adj_crosstalk else ("伴随法胜" if adj_crosstalk < ga_crosstalk else "平局")
    print(f"{'总串扰（低更好）':<30} {ga_crosstalk:<18.4f} {adj_crosstalk:<18.4f} {winner:<15}")
    
    # 优化时间
    print(f"{'优化耗时(秒)':<30} {ga_time:<18.2f} {adjoint_time:<18.2f}")
    
    print("\n" + "-" * 81)


def plot_convergence_comparison(comparison_results: Dict[str, Any], save_path: str = None):
    """
    绘制收敛曲线对比
    
    Args:
        comparison_results: 对比结果
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ga_history = comparison_results["ga"]["result"]["history"]
    adj_history = comparison_results["adjoint"]["result"]["history"]
    
    ga_gens = range(len(ga_history["best_fitness"]))
    adj_iters = range(len(adj_history["fitness"]))
    
    # 适应度对比
    ax = axes[0, 0]
    ax.plot(ga_gens, ga_history["best_fitness"], 'b-', label="GA-最优", linewidth=2)
    ax.plot(adj_iters, adj_history["fitness"], 'r-', label="伴随法", linewidth=2)
    ax.set_xlabel("迭代/代数")
    ax.set_ylabel("适应度")
    ax.set_title("适应度演化对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TE→Port1效率
    ax = axes[0, 1]
    ax.plot(ga_gens, ga_history["te_port1"] if "te_port1" in ga_history else [0]*len(ga_gens), 
            'b-', label="GA", linewidth=2)
    ax.plot(adj_iters, adj_history["te_port1"], 'r-', label="伴随法", linewidth=2)
    ax.set_xlabel("迭代/代数")
    ax.set_ylabel("效率")
    ax.set_title("TE→Port1效率演化")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TM→Port2效率
    ax = axes[1, 0]
    ax.plot(ga_gens, ga_history["tm_port2"] if "tm_port2" in ga_history else [0]*len(ga_gens), 
            'b-', label="GA", linewidth=2)
    ax.plot(adj_iters, adj_history["tm_port2"], 'r-', label="伴随法", linewidth=2)
    ax.set_xlabel("迭代/代数")
    ax.set_ylabel("效率")
    ax.set_title("TM→Port2效率演化")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 串扰对比
    ax = axes[1, 1]
    ax.plot(ga_gens, [ga_history["crosstalk"][i] for i in range(len(ga_gens))] if "crosstalk" in ga_history else [0]*len(ga_gens), 
            'b-', label="GA", linewidth=2)
    ax.plot(adj_iters, adj_history["crosstalk"], 'r-', label="伴随法", linewidth=2)
    ax.set_xlabel("迭代/代数")
    ax.set_ylabel("串扰")
    ax.set_title("串扰演化对比（低更好）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n收敛曲线对比已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    print("初始化MEEP仿真器...")
    config = SimulationConfig(
        resolution=10,
        run_time=50,
        n_cells_x=30,
        n_cells_y=8
    )
    simulator = MMISimulator(config=config, num_workers=12)
    
    # GA参数
    ga_params = {
        "pop_size": 30,
        "num_generations": 50,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "elite_ratio": 0.1
    }
    
    # 伴随法参数
    adjoint_params = {
        "learning_rate": 0.01,
        "num_iterations": 50
    }
    
    # 运行对比
    comparison_results = compare_algorithms(
        simulator=simulator,
        ga_params=ga_params,
        adjoint_params=adjoint_params
    )
    
    # 打印对比总结
    print_comparison_summary(comparison_results)
    
    # 绘制收敛曲线
    plot_convergence_comparison(comparison_results, save_path="./ga_vs_adjoint_comparison.png")
    
    print("\n对比分析完成！")


if __name__ == "__main__":
    main()
