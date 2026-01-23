"""
MEEP 仿真器封装

提供MMI PBS结构的FDTD仿真功能，支持并行计算。
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


@dataclass
class SimulationConfig:
    """仿真配置"""
    # 几何参数 (单位: μm)
    wavelength: float = 1.55  # 工作波长
    mmi_width: float = 4.0  # MMI区域宽度
    mmi_length: float = 15.0  # MMI区域长度
    wg_width: float = 0.5  # 输入/输出波导宽度
    wg_length: float = 2.0  # 输入/输出波导长度
    thickness: float = 0.22  # 波导厚度 (SOI)
    
    # 材料参数
    n_si: float = 3.48  # 硅折射率 @1550nm
    n_sio2: float = 1.44  # 二氧化硅折射率
    
    # 仿真参数
    resolution: int = 20  # 网格分辨率 (pixels/μm)
    pml_thickness: float = 1.0  # PML厚度
    run_time: float = 100  # 仿真时间 (in units of 1/frequency)
    
    # 网格参数
    n_cells_x: int = 30  # MMI区域X方向网格数
    n_cells_y: int = 8  # MMI区域Y方向网格数


class MMISimulator:
    """
    MMI PBS MEEP仿真器
    
    支持TE和TM偏振的仿真，计算各端口的传输效率。
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None, num_workers: int = 12):
        """
        初始化仿真器
        
        Args:
            config: 仿真配置
            num_workers: 并行工作进程数
        """
        self.config = config or SimulationConfig()
        self.num_workers = num_workers
        self._check_meep()
    
    def _check_meep(self):
        """检查MEEP是否可用"""
        try:
            import meep as mp
            self.mp = mp
            self._meep_available = True
        except ImportError:
            print("Warning: MEEP not available, using mock simulation")
            self._meep_available = False
    
    def simulate(
        self,
        structure: np.ndarray,
        polarization: str = "both"
    ) -> Dict[str, float]:
        """
        运行FDTD仿真
        
        Args:
            structure: 结构数组，形状为(n_cells_x, n_cells_y)，值为0或1
                      0表示SiO2，1表示Si
            polarization: 偏振态 "te", "tm", 或 "both"
        
        Returns:
            包含传输效率的字典
        """
        if not self._meep_available:
            return self._mock_simulate(structure)
        
        results = {}
        
        if polarization in ["te", "both"]:
            results.update(self._run_single_pol(structure, "te"))
        
        if polarization in ["tm", "both"]:
            results.update(self._run_single_pol(structure, "tm"))
        
        # 计算综合指标
        if polarization == "both":
            results["crosstalk"] = results.get("te_port2", 0) + results.get("tm_port1", 0)
            results["total_efficiency"] = results.get("te_port1", 0) + results.get("tm_port2", 0)
        
        return results
    
    def _run_single_pol(self, structure: np.ndarray, pol: str) -> Dict[str, float]:
        """
        运行单一偏振仿真
        
        Args:
            structure: 结构数组
            pol: 偏振态 "te" 或 "tm"
        
        Returns:
            传输效率字典
        """
        mp = self.mp
        cfg = self.config
        
        # 计算仿真区域尺寸
        sx = cfg.mmi_length + 2 * cfg.wg_length + 2 * cfg.pml_thickness
        sy = cfg.mmi_width + 2 * cfg.pml_thickness
        
        # 创建材料
        Si = mp.Medium(index=cfg.n_si)
        SiO2 = mp.Medium(index=cfg.n_sio2)
        
        # 定义几何结构
        geometry = []
        
        # 背景是SiO2（通过default_material设置）
        
        # 输入波导
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.wg_length, cfg.wg_width, mp.inf),
            center=mp.Vector3(-sx/2 + cfg.pml_thickness + cfg.wg_length/2, 0, 0),
            material=Si
        ))
        
        # 输出波导1 (上方)
        output_y = cfg.mmi_width / 4
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.wg_length, cfg.wg_width, mp.inf),
            center=mp.Vector3(sx/2 - cfg.pml_thickness - cfg.wg_length/2, output_y, 0),
            material=Si
        ))
        
        # 输出波导2 (下方)
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.wg_length, cfg.wg_width, mp.inf),
            center=mp.Vector3(sx/2 - cfg.pml_thickness - cfg.wg_length/2, -output_y, 0),
            material=Si
        ))
        
        # MMI区域 - 根据structure数组构建
        cell_dx = cfg.mmi_length / cfg.n_cells_x
        cell_dy = cfg.mmi_width / cfg.n_cells_y
        mmi_start_x = -cfg.mmi_length / 2
        mmi_start_y = -cfg.mmi_width / 2
        
        for i in range(cfg.n_cells_x):
            for j in range(cfg.n_cells_y):
                if structure[i, j] > 0.5:  # Si
                    cx = mmi_start_x + (i + 0.5) * cell_dx
                    cy = mmi_start_y + (j + 0.5) * cell_dy
                    geometry.append(mp.Block(
                        size=mp.Vector3(cell_dx, cell_dy, mp.inf),
                        center=mp.Vector3(cx, cy, 0),
                        material=Si
                    ))
        
        # 光源设置
        fcen = 1 / cfg.wavelength  # 中心频率
        
        # 根据偏振选择源
        if pol == "te":
            # TE: Ez polarization (out of plane)
            src_component = mp.Ez
        else:
            # TM: Hz polarization
            src_component = mp.Hz
        
        sources = [mp.Source(
            mp.ContinuousSource(frequency=fcen),
            component=src_component,
            center=mp.Vector3(-sx/2 + cfg.pml_thickness + 0.5, 0, 0),
            size=mp.Vector3(0, cfg.wg_width * 2, 0)
        )]
        
        # 创建仿真
        sim = mp.Simulation(
            cell_size=mp.Vector3(sx, sy, 0),
            geometry=geometry,
            sources=sources,
            boundary_layers=[mp.PML(cfg.pml_thickness)],
            resolution=cfg.resolution,
            default_material=SiO2
        )
        
        # 设置通量监测器
        flux_freq = fcen
        nfreq = 1
        
        # 输入通量
        flux_in = sim.add_flux(
            flux_freq, 0, nfreq,
            mp.FluxRegion(
                center=mp.Vector3(-sx/2 + cfg.pml_thickness + cfg.wg_length + 0.2, 0, 0),
                size=mp.Vector3(0, cfg.wg_width * 2, 0)
            )
        )
        
        # 输出端口1通量 (上方)
        flux_out1 = sim.add_flux(
            flux_freq, 0, nfreq,
            mp.FluxRegion(
                center=mp.Vector3(sx/2 - cfg.pml_thickness - 0.5, output_y, 0),
                size=mp.Vector3(0, cfg.wg_width * 2, 0)
            )
        )
        
        # 输出端口2通量 (下方)
        flux_out2 = sim.add_flux(
            flux_freq, 0, nfreq,
            mp.FluxRegion(
                center=mp.Vector3(sx/2 - cfg.pml_thickness - 0.5, -output_y, 0),
                size=mp.Vector3(0, cfg.wg_width * 2, 0)
            )
        )
        
        # 运行仿真
        sim.run(until=cfg.run_time)
        
        # 获取通量值
        flux_in_val = mp.get_fluxes(flux_in)[0]
        flux_out1_val = mp.get_fluxes(flux_out1)[0]
        flux_out2_val = mp.get_fluxes(flux_out2)[0]
        
        # 计算传输效率
        if abs(flux_in_val) > 1e-10:
            eff_port1 = abs(flux_out1_val / flux_in_val)
            eff_port2 = abs(flux_out2_val / flux_in_val)
        else:
            eff_port1 = 0.0
            eff_port2 = 0.0
        
        # 清理
        sim.reset_meep()
        
        return {
            f"{pol}_port1": float(eff_port1),
            f"{pol}_port2": float(eff_port2),
            f"{pol}_total": float(eff_port1 + eff_port2)
        }
    
    def _mock_simulate(self, structure: np.ndarray) -> Dict[str, float]:
        """
        模拟仿真（当MEEP不可用时）
        
        用于测试环境逻辑
        """
        # 基于结构特征生成伪结果
        fill_factor = np.mean(structure)
        asymmetry = np.mean(structure[:, :structure.shape[1]//2]) - \
                    np.mean(structure[:, structure.shape[1]//2:])
        
        # 模拟效率
        base_eff = 0.3 + 0.2 * fill_factor
        te_port1 = base_eff + 0.1 * asymmetry + np.random.normal(0, 0.02)
        te_port2 = base_eff - 0.1 * asymmetry + np.random.normal(0, 0.02)
        tm_port1 = base_eff - 0.1 * asymmetry + np.random.normal(0, 0.02)
        tm_port2 = base_eff + 0.1 * asymmetry + np.random.normal(0, 0.02)
        
        # 确保非负
        te_port1 = max(0, min(1, te_port1))
        te_port2 = max(0, min(1, te_port2))
        tm_port1 = max(0, min(1, tm_port1))
        tm_port2 = max(0, min(1, tm_port2))
        
        return {
            "te_port1": te_port1,
            "te_port2": te_port2,
            "te_total": te_port1 + te_port2,
            "tm_port1": tm_port1,
            "tm_port2": tm_port2,
            "tm_total": tm_port1 + tm_port2,
            "crosstalk": te_port2 + tm_port1,
            "total_efficiency": te_port1 + tm_port2
        }
    
    def simulate_parallel(
        self,
        structures: list,
        polarization: str = "both"
    ) -> list:
        """
        并行仿真多个结构
        
        Args:
            structures: 结构数组列表
            polarization: 偏振态
        
        Returns:
            结果列表
        """
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.simulate, struct, polarization)
                for struct in structures
            ]
            results = [f.result() for f in futures]
        return results


def simulate_single_structure(args):
    """
    用于并行仿真的辅助函数
    
    Args:
        args: (structure, config, polarization) 元组
    
    Returns:
        仿真结果字典
    """
    structure, config, polarization = args
    simulator = MMISimulator(config, num_workers=1)
    return simulator.simulate(structure, polarization)
