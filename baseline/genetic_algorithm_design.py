"""
遗传算法逆向设计示例

演示如何使用遗传算法进行MMI PBS设计，
并与伴随法（adjoint method）进行对比。

使用项目的仿真环境进行优化。
"""

import sys
import os

# 添加父目录到路径以导入pbs_env模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from tqdm import tqdm

import meep as mp
from meep_simulator import MMISimulator, load_simulation_config, GeneticAlgorithmOptimizer, SdfGeneticAlgorithmOptimizer
from utils import compute_reward


def setup_simulator(config_path: str = None) -> MMISimulator:
    """从YAML加载配置并构建仿真器。"""
    config, num_workers = load_simulation_config(config_path)
    return MMISimulator(config=config, num_workers=num_workers)


def run_genetic_algorithm_optimization(
    simulator: MMISimulator,
    pop_size: int = 50,
    num_generations: int = 100,
    mutation_rate: float = 0.1,
    init_mode: str = "random",
    use_parallel: bool = True,
    num_workers: Optional[int] = None,
    log_path: str = "./logs/ga_run.log"
) -> Dict[str, Any]:
    """
    运行遗传算法优化
    
    Args:
        simulator: 仿真器实例
        pop_size: 种群大小
        num_generations: 演化代数
        mutation_rate: 变异率
        init_mode: 初始化模式
        use_parallel: 是否使用多核并行评估
        num_workers: 并行进程数（None时自动使用simulator或CPU核心数）
        log_path: 日志文件保存路径
    
    Returns:
        优化结果字典
    """
    print("\n" + "=" * 70)
    print("遗传算法（Genetic Algorithm）- 设计优化")
    print("=" * 70)
    
    ga_optimizer = GeneticAlgorithmOptimizer(
        simulator=simulator,
        pop_size=pop_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        crossover_rate=0.8,
        elite_ratio=0.1,
        seed=42,
        verbose=True,
        use_parallel=use_parallel,
        num_workers=num_workers or simulator.num_workers,
        log_path=log_path
    )
    
    ga_result = ga_optimizer.optimize(init_mode=init_mode)
    
    return ga_result



def run_sdf_ga_optimization(
    simulator: MMISimulator,
    pop_size: int = 50,
    num_generations: int = 100,
    mutation_strength: float = 0.1,
    n_circles: int = 10,
    log_path: str = "./logs/sdf_ga_run.log"
) -> Dict[str, Any]:
    """
    运行基于SDF的遗传算法优化
    """
    print("\n" + "=" * 70)
    print("SDF + 遗传算法 - 设计优化")
    print("=" * 70)
    
    optimizer = SdfGeneticAlgorithmOptimizer(
        simulator=simulator,
        pop_size=pop_size,
        num_generations=num_generations,
        mutation_strength=mutation_strength,
        n_circles=n_circles,
        log_path=log_path,
        seed=42,
        verbose=True
    )
    
    result = optimizer.optimize()
    return result


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


def plot_optimization_history(ga_result: Dict[str, Any], save_path: str = None):
    """
    绘制优化历史
    
    Args:
        ga_result: 遗传算法优化结果
        save_path: 保存路径（可选）
    """
    history = ga_result["history"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 适应度演化
    ax1 = axes[0]
    generations = range(len(history["best_fitness"]))
    ax1.plot(generations, history["best_fitness"], 'b-', label="Best Fitness", linewidth=2)
    ax1.fill_between(
        generations,
        np.array(history["avg_fitness"]) - np.array(history["std_fitness"]),
        np.array(history["avg_fitness"]) + np.array(history["std_fitness"]),
        alpha=0.3,
        label="Mean +/- Std"
    )
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Genetic Algorithm - Fitness Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 性能指标
    ax2 = axes[1]
    best_result = ga_result["best_result"]
    
    metrics = [
        "TE->P1",
        "TE->P2",
        "TM->P1",
        "TM->P2"
    ]
    values = [
        best_result.te_port1,
        best_result.te_port2,
        best_result.tm_port1,
        best_result.tm_port2
    ]
    colors = ['green', 'red', 'red', 'green']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel("Efficiency")
    ax2.set_title("Best Structure Performance")
    ax2.set_ylim([0, 1])
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label="Target")
    
    # 在柱子上显示数值
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"优化历史已保存到: {save_path}")
    
    plt.show()


def compare_random_vs_optimized(simulator: MMISimulator, optimized_structure: np.ndarray):
    """
    对比随机结构与优化结构
    
    Args:
        simulator: 仿真器实例
        optimized_structure: 优化后的结构
    """
    print("\n" + "=" * 70)
    print("随机结构 vs 优化结构对比")
    print("=" * 70)
    
    # 随机结构
    random_structure = np.random.randint(
        0, 2, 
        size=(simulator.config.n_cells_x, simulator.config.n_cells_y)
    )
    random_metrics = evaluate_structure(simulator, random_structure)
    
    # 优化结构
    opt_metrics = evaluate_structure(simulator, optimized_structure)
    
    print(f"\n{'指标':<25} {'随机结构':<15} {'优化结构':<15} {'改进':<15}")
    print("-" * 70)
    
    for key in ["te_port1", "tm_port2", "total_efficiency", "te_extinction_ratio", "tm_extinction_ratio"]:
        rand_val = random_metrics[key]
        opt_val = opt_metrics[key]
        improvement = opt_val - rand_val
        
        if "ratio" in key:
            print(f"{key:<25} {rand_val:>6.2f} dB       {opt_val:>6.2f} dB       {improvement:>+6.2f} dB")
        else:
            print(f"{key:<25} {rand_val:>14.4f} {opt_val:>14.4f} {improvement:>+14.4f}")


def plot_structure(structure: np.ndarray, title: str = "MMI PBS Structure", save_path: str = None):
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


def save_results_to_file(ga_result: Dict[str, Any], save_dir: str = "./results"):
    """
    保存优化结果到文件
    
    Args:
        ga_result: 优化结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    best_individual = ga_result["best_individual"]
    best_result = ga_result["best_result"]
    history = ga_result["history"]
    
    # 保存最优结构
    np.save(f"{save_dir}/best_structure_ga.npy", best_individual)
    
    # 保存优化历史
    np.save(f"{save_dir}/ga_history.npy", history)
    
    # 保存性能指标到文本文件
    with open(f"{save_dir}/ga_performance.txt", "w") as f:
        f.write("遗传算法优化结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"最优适应度: {ga_result['best_fitness']:.4f}\n")
        f.write(f"TE→Port1效率: {best_result.te_port1:.4f}\n")
        f.write(f"TM→Port2效率: {best_result.tm_port2:.4f}\n")
        f.write(f"TE→Port2串扰: {best_result.te_port2:.4f}\n")
        f.write(f"TM→Port1串扰: {best_result.tm_port1:.4f}\n")
        f.write(f"总效率: {best_result.total_efficiency:.4f}\n")
        f.write(f"总串扰: {best_result.crosstalk:.4f}\n")
        f.write(f"\n演化代数: {history['generation_count']}\n")
        f.write(f"最终种群大小: {ga_result['final_population'].shape[0]}\n")
    
    print(f"\n结果已保存到 {save_dir}:")
    print(f"  - 最优结构: {save_dir}/best_structure_ga.npy")
    print(f"  - 优化历史: {save_dir}/ga_history.npy")
    print(f"  - 性能指标: {save_dir}/ga_performance.txt")


def main():
    """主函数"""
    # 1. 初始化仿真器
    print("初始化MEEP仿真器...")
    simulator = setup_simulator()
    
    # 定义 n_circles (保持一致)
    N_CIRCLES = 10  # 对于 4x3um 紧凑设计，10个圆足够
    
    # 2. 运行SDF遗传算法
    print("\n运行SDF + GA优化...")
    ga_result = run_sdf_ga_optimization(
        simulator=simulator,
        pop_size=30,            # 降低到合理大小 (30 * 50 = 1500次仿真 ≈ 20分钟)
        num_generations=50,     # 降低代数以快速验证
        mutation_strength=0.15,
        n_circles=N_CIRCLES,    # 使用统一的变量
        log_path="./logs/sdf_ga_run.log"
    )
    
    # 3. 结果处理...
    if ga_result and ga_result.get("best_individual") is not None:
        # 3. 绘制结果
        print("\n生成可视化...")
        plot_optimization_history(ga_result, save_path="./sdf_ga_optimization_history.png")
        
        # 4.b 保存最佳结构和仿真结果组合图 (使用集成方法)
        print(f"正在保存最佳结果可视化...")
        
        # 为了可视化，从SDF参数重新采样结构
        final_params = ga_result["best_individual"]
        best_sdf = SdfGeneticAlgorithmOptimizer.create_sdf_from_params(
            final_params, 
            n_circles=N_CIRCLES,  # 使用与优化器一致的参数
            domain_size=(simulator.config.mmi_length, simulator.config.mmi_width)
        )
        
        # 创建一个离散的结构图用于可视化
        xs = np.linspace(-simulator.config.mmi_length/2, simulator.config.mmi_length/2, simulator.config.n_cells_x)
        ys = np.linspace(-simulator.config.mmi_width/2, simulator.config.mmi_width/2, simulator.config.n_cells_y)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
        
        sampled_structure = np.zeros((simulator.config.n_cells_x, simulator.config.n_cells_y))
        for i in range(simulator.config.n_cells_x):
            for j in range(simulator.config.n_cells_y):
                p = mp.Vector3(grid_x[i, j], grid_y[i, j])
                if best_sdf(p) <= 0:
                    sampled_structure[i, j] = 1
        
        from utils import visualize_results
        visualize_results(
            results=ga_result["best_result"],
            structure=sampled_structure,
            save_path="./sdf_ga_best_result.png",
            show=False
        )
        print(f"结果图已保存到: ./sdf_ga_best_result.png")
        
        # 5. 保存结果
        save_results_to_file(ga_result, save_dir="./sdf_ga_results")
    
    print("\n" + "=" * 70)
    print("优化完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
