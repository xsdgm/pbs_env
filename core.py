"""
核心接口定义

定义仿真器接口和通用数据结构。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationResult:
    """仿真结果数据类"""
    te_port1: float
    te_port2: float
    tm_port1: float
    tm_port2: float
    total_efficiency: float
    crosstalk: float
    metadata: Dict[str, Any] = None


class Simulator(ABC):
    """仿真器抽象基类"""
    
    @abstractmethod
    def simulate(self, structure: np.ndarray) -> SimulationResult:
        """
        运行仿真
        
        Args:
            structure: 结构数组 (0-1归一化)
            
        Returns:
            SimulationResult对象
        """
        pass
    
    @property
    @abstractmethod
    def config(self) -> Any:
        """获取配置"""
        pass

    @abstractmethod
    def compute_gradients(
        self, 
        structure: np.ndarray, 
        target_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        计算结构梯度
        
        Args:
            structure: 结构数组
            target_weights: 优化目标权重，例如 {"te_port1": 1.0, "tm_port2": 1.0}
            
        Returns:
            梯度数组，形状与structure相同
        """
        pass
