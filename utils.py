"""
工具函数

提供奖励计算、结构处理、可视化等功能。
"""

import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def compute_reward(
    te_port1: float,
    tm_port2: float,
    te_port2: float = 0.0,
    tm_port1: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5
) -> float:
    """
    计算PBS优化奖励
    
    目标: TE → Port1, TM → Port2
    
    Args:
        te_port1: TE模式在端口1的效率 (目标)
        tm_port2: TM模式在端口2的效率 (目标)
        te_port2: TE模式在端口2的效率 (串扰)
        tm_port1: TM模式在端口1的效率 (串扰)
        alpha: TE效率权重
        beta: TM效率权重
        gamma: 串扰惩罚权重
    
    Returns:
        奖励值
    """
    # 目标效率
    target_efficiency = alpha * te_port1 + beta * tm_port2
    
    # 串扰惩罚
    crosstalk = te_port2 + tm_port1
    
    reward = target_efficiency - gamma * crosstalk
    
    return float(reward)


def compute_extinction_ratio(target: float, crosstalk: float, eps: float = 1e-10) -> float:
    """
    计算消光比 (dB)
    
    Args:
        target: 目标端口效率
        crosstalk: 串扰端口效率
        eps: 防止除零的小量
    
    Returns:
        消光比 (dB)
    """
    return 10 * np.log10((target + eps) / (crosstalk + eps))


def compute_insertion_loss(efficiency: float, eps: float = 1e-10) -> float:
    """
    计算插入损耗 (dB)
    
    Args:
        efficiency: 传输效率
        eps: 防止除零的小量
    
    Returns:
        插入损耗 (dB，正值表示损耗)
    """
    return -10 * np.log10(efficiency + eps)


def normalize_structure(structure: np.ndarray) -> np.ndarray:
    """
    归一化结构数组
    
    Args:
        structure: 原始结构数组
    
    Returns:
        归一化后的数组 (0-1范围)
    """
    struct = structure.copy()
    struct = (struct > 0.5).astype(np.float32)
    return struct


def apply_minimum_feature_size(
    structure: np.ndarray,
    mfs: int = 2
) -> np.ndarray:
    """
    应用最小特征尺寸约束
    
    使用形态学操作确保结构满足制造约束
    
    Args:
        structure: 结构数组
        mfs: 最小特征尺寸（像素）
    
    Returns:
        处理后的结构
    """
    from scipy import ndimage
    
    # 二值化
    binary = (structure > 0.5).astype(np.float32)
    
    # 开运算（去除小于mfs的特征）
    struct_elem = np.ones((mfs, mfs))
    opened = ndimage.binary_opening(binary, struct_elem).astype(np.float32)
    
    # 闭运算（填充小于mfs的空洞）
    closed = ndimage.binary_closing(opened, struct_elem).astype(np.float32)
    
    return closed


def visualize_structure(
    structure: np.ndarray,
    title: str = "MMI PBS Structure",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    可视化结构
    
    Args:
        structure: 结构数组
        title: 图标题
        figsize: 图尺寸
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(
        structure.T,
        cmap="YlOrBr",
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Material (0=SiO₂, 1=Si)")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_results(
    results: dict,
    structure: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    可视化仿真结果
    
    Args:
        results: 仿真结果字典
        structure: 结构数组（可选）
        figsize: 图尺寸
        save_path: 保存路径
        show: 是否显示
    """
    if structure is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：结构
        ax1 = axes[0]
        im = ax1.imshow(
            structure.T,
            cmap="YlOrBr",
            aspect="auto",
            origin="lower"
        )
        ax1.set_xlabel("X (cells)")
        ax1.set_ylabel("Y (cells)")
        ax1.set_title("MMI Structure")
        plt.colorbar(im, ax=ax1, label="Material")
        
        # 右图：效率条形图
        ax2 = axes[1]
    else:
        fig, ax2 = plt.subplots(figsize=(6, 5))
    
    # 绘制效率
    labels = ["TE→P1", "TE→P2", "TM→P1", "TM→P2"]
    values = [
        results.get("te_port1", 0),
        results.get("te_port2", 0),
        results.get("tm_port1", 0),
        results.get("tm_port2", 0)
    ]
    colors = ["green", "red", "red", "green"]  # 绿色=目标，红色=串扰
    
    bars = ax2.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel("Efficiency")
    ax2.set_title("Port Transmission")
    ax2.set_ylim(0, 1)
    
    # 在条形图上添加数值
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # 添加图例
    ax2.text(
        0.95, 0.95,
        f"Total Eff: {results.get('total_efficiency', 0):.3f}\n"
        f"Crosstalk: {results.get('crosstalk', 0):.3f}",
        transform=ax2.transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_symmetric_structure(n_cells_x: int, n_cells_y: int) -> np.ndarray:
    """
    创建对称初始结构
    
    Args:
        n_cells_x: X方向网格数
        n_cells_y: Y方向网格数
    
    Returns:
        对称结构数组
    """
    structure = np.zeros((n_cells_x, n_cells_y), dtype=np.float32)
    
    # 创建对称模式
    for i in range(n_cells_x):
        for j in range(n_cells_y // 2):
            val = np.random.choice([0, 1])
            structure[i, j] = val
            structure[i, n_cells_y - 1 - j] = val
    
    return structure


def create_gradient_structure(
    n_cells_x: int,
    n_cells_y: int,
    direction: str = "x"
) -> np.ndarray:
    """
    创建渐变结构
    
    Args:
        n_cells_x: X方向网格数
        n_cells_y: Y方向网格数
        direction: 渐变方向 "x" 或 "y"
    
    Returns:
        渐变结构数组
    """
    if direction == "x":
        gradient = np.linspace(0, 1, n_cells_x)[:, np.newaxis]
        structure = np.tile(gradient, (1, n_cells_y))
    else:
        gradient = np.linspace(0, 1, n_cells_y)[np.newaxis, :]
        structure = np.tile(gradient, (n_cells_x, 1))
    
    return structure.astype(np.float32)
