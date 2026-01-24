"""
贪心搜索逆向设计示例

使用贪心算法在项目仿真环境下进行MMI PBS优化，作为轻量级baseline。
"""

import sys
import os

# 添加父目录到路径以导入pbs_env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from meep_simulator import MMISimulator, load_simulation_config, GreedyOptimizer


def setup_simulator(config_path: str = None) -> MMISimulator:
    config, num_workers = load_simulation_config(config_path)
    return MMISimulator(config=config, num_workers=num_workers)


def run_greedy(simulator: MMISimulator, max_steps: int = 200, patience: int = 20) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("贪心算法（Greedy）- 设计优化")
    print("=" * 70)

    optimizer = GreedyOptimizer(
        simulator=simulator,
        max_steps=max_steps,
        patience=patience,
        seed=42,
        verbose=True,
    )

    return optimizer.optimize(init_mode="random")


def plot_history(history: Dict[str, Any], save_path: str = None):
    generations = range(len(history["best_fitness"]))

    plt.figure(figsize=(8, 5))
    plt.plot(generations, history["best_fitness"], label="最优适应度", linewidth=2)
    plt.xlabel("步数")
    plt.ylabel("适应度")
    plt.title("贪心算法收敛曲线")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"收敛曲线已保存到: {save_path}")

    plt.show()


def plot_structure(structure: np.ndarray, title: str = "贪心优化结构", save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    display = structure.T[::-1]
    im = ax.imshow(display, cmap="gray", aspect="auto", origin="lower")
    ax.set_xlabel("X方向单元索引")
    ax.set_ylabel("Y方向单元索引")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="材料 (0=SiO2, 1=Si)")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"结构图已保存到: {save_path}")

    plt.show()


def main():
    simulator = setup_simulator()

    result = run_greedy(simulator, max_steps=200, patience=20)

    plot_history(result["history"], save_path="./greedy_history.png")
    plot_structure(result["best_structure"], title="贪心算法优化后的MMI PBS结构", save_path="./greedy_structure.png")

    print("\n最优适应度: {:.4f}".format(result["best_fitness"]))
    best = result["best_result"]
    print("TE→Port1效率: {:.4f}".format(best.te_port1))
    print("TM→Port2效率: {:.4f}".format(best.tm_port2))
    print("TE→Port2串扰: {:.4f}".format(best.te_port2))
    print("TM→Port1串扰: {:.4f}".format(best.tm_port1))


if __name__ == "__main__":
    main()
