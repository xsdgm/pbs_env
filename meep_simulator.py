"""
MEEP 仿真器封装

提供MMI PBS结构的FDTD仿真功能，支持并行计算。
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .core import Simulator, SimulationResult
from dataclasses import dataclass



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


class MMISimulator(Simulator):
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
        self.config_ = config or SimulationConfig()
        self.num_workers = num_workers
        self._check_meep()

    @property
    def config(self) -> SimulationConfig:
        return self.config_
    
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
    ) -> SimulationResult:
        """
        运行FDTD仿真
        
        Args:
            structure: 结构数组，形状为(n_cells_x, n_cells_y)，值为0或1
                      0表示SiO2，1表示Si
            polarization: 偏振态 "te", "tm", 或 "both"
        
        Returns:
            SimulationResult 对象
        """
        if not self._meep_available:
            return self._mock_simulate(structure)
        
        results = {}
        
        if polarization in ["te", "both"]:
            results.update(self._run_single_pol(structure, "te"))
        
        if polarization in ["tm", "both"]:
            results.update(self._run_single_pol(structure, "tm"))
        
        # 封装结果
        te_p1 = results.get("te_port1", 0.0)
        te_p2 = results.get("te_port2", 0.0)
        tm_p1 = results.get("tm_port1", 0.0)
        tm_p2 = results.get("tm_port2", 0.0)
        
        return SimulationResult(
            te_port1=te_p1,
            te_port2=te_p2,
            tm_port1=tm_p1,
            tm_port2=tm_p2,
            total_efficiency=te_p1 + tm_p2,
            crosstalk=te_p2 + tm_p1,
            metadata=results
        )
    
    def _run_single_pol(self, structure: np.ndarray, pol: str) -> Dict[str, float]:
        """
        运行单一偏振仿真
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
    
    def _mock_simulate(self, structure: np.ndarray) -> SimulationResult:
        """
        模拟仿真（当MEEP不可用时）
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
        te_port1 = max(0.0, min(1.0, te_port1))
        te_port2 = max(0.0, min(1.0, te_port2))
        tm_port1 = max(0.0, min(1.0, tm_port1))
        tm_port2 = max(0.0, min(1.0, tm_port2))
        
        return SimulationResult(
            te_port1=te_port1,
            te_port2=te_port2,
            tm_port1=tm_port1,
            tm_port2=tm_port2,
            total_efficiency=te_port1 + tm_port2,
            crosstalk=te_port2 + tm_port1,
            metadata={}
        )
    
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
    def compute_gradients(
        self,
        structure: np.ndarray,
        target_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        计算结构梯度 (使用伴随法)
        
        Args:
            structure: 结构数组
            target_weights: 目标权重，默认 {"te_port1": 1.0, "tm_port2": 1.0}
            
        Returns:
            梯度数组
        """
        if not self._meep_available:
            # Mock gradients
            return np.random.randn(*structure.shape) * 0.1
        
        if target_weights is None:
            target_weights = {"te_port1": 1.0, "tm_port2": 1.0}
            
        grad_accum = np.zeros_like(structure, dtype=np.float64)
        
        # 处理 TE 梯度
        if any(k.startswith("te") for k in target_weights.keys()):
            # 根据现有代码定义：TE 对应 Ez 分量
            # 1. Forward Simulation (Source at Input)
            fields_fwd = self._run_field_sim(structure, "te", source_port="in")
            
            # 2. Adjoint Simulations
            if target_weights.get("te_port1", 0.0) != 0.0:
                fields_adj = self._run_field_sim(structure, "te", source_port="out1")
                # Gradient = 2 * Re(E_fwd * E_adj)
                # 注意：这里简化了常数，因为优化通常只需要方向
                # 对于 Ez 偏振，直接相乘
                grad = 2 * np.real(fields_fwd["Ez"] * fields_adj["Ez"])
                grad_accum += target_weights["te_port1"] * grad
                
            if target_weights.get("te_port2", 0.0) != 0.0:
                fields_adj = self._run_field_sim(structure, "te", source_port="out2")
                grad = 2 * np.real(fields_fwd["Ez"] * fields_adj["Ez"])
                grad_accum += target_weights["te_port2"] * grad

        # 处理 TM 梯度
        if any(k.startswith("tm") for k in target_weights.keys()):
            # 根据现有代码定义：TM 对应 Hz 分量 (即存在 Ex, Ey)
            # 1. Forward Simulation
            fields_fwd = self._run_field_sim(structure, "tm", source_port="in")
            
            # 2. Adjoint Simulations
            if target_weights.get("tm_port1", 0.0) != 0.0:
                fields_adj = self._run_field_sim(structure, "tm", source_port="out1")
                # Gradient = 2 * Re(E_fwd . E_adj) = 2 * Re(Ex_f E_x_a + Ey_f E_y_a)
                grad = 2 * np.real(fields_fwd["Ex"] * fields_adj["Ex"] + 
                                 fields_fwd["Ey"] * fields_adj["Ey"])
                grad_accum += target_weights["tm_port1"] * grad
                
            if target_weights.get("tm_port2", 0.0) != 0.0:
                fields_adj = self._run_field_sim(structure, "tm", source_port="out2")
                grad = 2 * np.real(fields_fwd["Ex"] * fields_adj["Ex"] + 
                                 fields_fwd["Ey"] * fields_adj["Ey"])
                grad_accum += target_weights["tm_port2"] * grad
                
        return grad_accum

    def _run_field_sim(
        self, 
        structure: np.ndarray, 
        pol: str, 
        source_port: str
    ) -> Dict[str, np.ndarray]:
        """
        运行仿真并获取MMI区域的场分布
        """
        mp = self.mp
        cfg = self.config
        
        # 几何设置 (与 _run_single_pol 相同)
        sx = cfg.mmi_length + 2 * cfg.wg_length + 2 * cfg.pml_thickness
        sy = cfg.mmi_width + 2 * cfg.pml_thickness
        
        Si = mp.Medium(index=cfg.n_si)
        SiO2 = mp.Medium(index=cfg.n_sio2)
        
        # 使用连续值构建几何以获得准确梯度
        geometry = self._build_geometry(structure, sx, sy, Si, SiO2, continuous=True)
        
        # 光源设置
        fcen = 1 / cfg.wavelength
        output_y = cfg.mmi_width / 4
        
        if pol == "te":
            src_component = mp.Ez
        else:
            src_component = mp.Hz
            
        # 确定光源位置
        if source_port == "in":
            src_center = mp.Vector3(-sx/2 + cfg.pml_thickness + 0.5, 0, 0)
            src_size = mp.Vector3(0, cfg.wg_width * 2, 0)
        elif source_port == "out1":
            src_center = mp.Vector3(sx/2 - cfg.pml_thickness - 0.5, output_y, 0)
            src_size = mp.Vector3(0, cfg.wg_width * 2, 0)
        elif source_port == "out2":
            src_center = mp.Vector3(sx/2 - cfg.pml_thickness - 0.5, -output_y, 0)
            src_size = mp.Vector3(0, cfg.wg_width * 2, 0)
        else:
            raise ValueError(f"Unknown source port: {source_port}")
            
        sources = [mp.Source(
            mp.ContinuousSource(frequency=fcen),
            component=src_component,
            center=src_center,
            size=src_size
        )]

        sim = mp.Simulation(
            cell_size=mp.Vector3(sx, sy, 0),
            geometry=geometry,
            sources=sources,
            boundary_layers=[mp.PML(cfg.pml_thickness)],
            resolution=cfg.resolution,
            default_material=SiO2
        )
        
        # 定义 DFT 监视区域 (仅覆盖 MMI 区域)
        # MMI中心在 (0,0), 大小 (mmi_length, mmi_width)
        mmi_region = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(cfg.mmi_length, cfg.mmi_width, 0)
        )
        
        # 添加 DFT 监视器
        if pol == "te":
            dft_obj = sim.add_dft_fields([mp.Ez], fcen, 0, 1, where=mmi_region)
        else:
            dft_obj = sim.add_dft_fields([mp.Ex, mp.Ey], fcen, 0, 1, where=mmi_region)
            
        # 运行仿真
        sim.run(until=cfg.run_time)
        
        # 提取场并重采样到 structure 的形状
        # 注意: get_array 返回的数组可能与 structure 形状不完全匹配，需要插值
        fields = {}
        
        target_shape = structure.shape # (nx, ny)
        
        if pol == "te":
            # Ez
            ez_data = sim.get_dft_array(dft_obj, mp.Ez, 0)
            fields["Ez"] = self._resample_field(ez_data, target_shape)
        else:
            # Ex, Ey
            ex_data = sim.get_dft_array(dft_obj, mp.Ex, 0)
            ey_data = sim.get_dft_array(dft_obj, mp.Ey, 0)
            fields["Ex"] = self._resample_field(ex_data, target_shape)
            fields["Ey"] = self._resample_field(ey_data, target_shape)
            
        sim.reset_meep()
        return fields

    def _build_geometry(self, structure: np.ndarray, sx: float, sy: float, Si: Any, SiO2: Any, continuous: bool = False):
        """
        构建几何结构
        
        改进：使用 mp.MaterialGrid 替代大量的 mp.Block，显著提高性能并支持亚像素平滑。
        """
        mp = self.mp
        cfg = self.config
        geometry = []
        
        # 输入波导
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.wg_length, cfg.wg_width, mp.inf),
            center=mp.Vector3(-sx/2 + cfg.pml_thickness + cfg.wg_length/2, 0, 0),
            material=Si
        ))
        
        # 输出波导
        output_y = cfg.mmi_width / 4
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.wg_length, cfg.wg_width, mp.inf),
            center=mp.Vector3(sx/2 - cfg.pml_thickness - cfg.wg_length/2, output_y, 0),
            material=Si
        ))
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.wg_length, cfg.wg_width, mp.inf),
            center=mp.Vector3(sx/2 - cfg.pml_thickness - cfg.wg_length/2, -output_y, 0),
            material=Si
        ))
        
        # MMI区域
        # 准备 weights
        if continuous:
            weights = structure
        else:
            weights = (structure > 0.5).astype(np.float64)
            
        # 使用 MaterialGrid
        # grid 尺寸对应 structure 形状
        # 注意: MEEP 中的 weights 默认映射: 0->medium1, 1->medium2
        grid = mp.MaterialGrid(
            mp.Vector3(cfg.n_cells_x, cfg.n_cells_y, 0),
            SiO2,  # 对应 weights=0
            Si,    # 对应 weights=1
            weights=weights,
            beta=0 if continuous else 100 # 如果不是连续模式，使用大beta进行陡峭二值化(虽然我们已经预先二值化了输入)
        )
        
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.mmi_length, cfg.mmi_width, mp.inf),
            center=mp.Vector3(0, 0, 0),
            material=grid
        ))
        
        return geometry

    def _resample_field(self, field_data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """简单的最近邻或线性插值重采样"""
        # field_data shape: (Nx, Ny) from simulation resolution
        # target_shape: (n_cells_x, n_cells_y)
        
        from scipy.ndimage import zoom
        
        # MEEP 返回的数组 维度顺序通常是 X, Y
        # 计算缩放因子
        zoom_factors = (
            target_shape[0] / field_data.shape[0],
            target_shape[1] / field_data.shape[1]
        )
        
        # 实部虚部分别插值
        real_resampled = zoom(field_data.real, zoom_factors, order=1)
        imag_resampled = zoom(field_data.imag, zoom_factors, order=1)
        
        return real_resampled + 1j * imag_resampled


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


class GeneticAlgorithmOptimizer:
    """
    遗传算法优化器 - 用于MMI PBS设计
    
    使用遗传算法进行逆向设计，作为伴随法的基准对比。
    基于项目的仿真环境进行优化。
    """
    
    def __init__(
        self,
        simulator: MMISimulator,
        n_cells_x: int = 30,
        n_cells_y: int = 8,
        pop_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        初始化遗传算法优化器
        
        Args:
            simulator: MMISimulator实例
            n_cells_x: MMI区域X方向网格数
            n_cells_y: MMI区域Y方向网格数
            pop_size: 种群大小
            num_generations: 演化代数
            mutation_rate: 变异概率
            crossover_rate: 交叉概率
            elite_ratio: 精英保留比例
            seed: 随机种子
            verbose: 是否打印优化过程
        """
        self.simulator = simulator
        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.n_genes = n_cells_x * n_cells_y  # 每个结构的基因数
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.n_elites = max(1, int(pop_size * elite_ratio))
        self.verbose = verbose
        
        if seed is not None:
            np.random.seed(seed)
        
        # 优化目标权重：TE→Port1, TM→Port2
        self.target_weights = {
            "te_port1": 1.0,
            "tm_port2": 1.0,
            "te_port2": -0.5,  # 惩罚串扰
            "tm_port1": -0.5
        }
        
        # 记录优化历史
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "std_fitness": [],
            "best_individual": None,
            "generation_count": 0
        }
    
    def initialize_population(self, init_mode: str = "random") -> np.ndarray:
        """
        初始化种群
        
        Args:
            init_mode: 初始化模式 "random", "ones", "zeros", "half"
        
        Returns:
            种群数组，形状为(pop_size, n_cells_x, n_cells_y)
        """
        population = np.zeros((self.pop_size, self.n_cells_x, self.n_cells_y))
        
        if init_mode == "random":
            population = np.random.randint(0, 2, size=(self.pop_size, self.n_cells_x, self.n_cells_y))
        elif init_mode == "zeros":
            population = np.zeros((self.pop_size, self.n_cells_x, self.n_cells_y), dtype=int)
        elif init_mode == "ones":
            population = np.ones((self.pop_size, self.n_cells_x, self.n_cells_y), dtype=int)
        elif init_mode == "half":
            # 50%概率为1，50%概率为0
            population = np.random.randint(0, 2, size=(self.pop_size, self.n_cells_x, self.n_cells_y))
        else:
            raise ValueError(f"Unknown initialization mode: {init_mode}")
        
        return population
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        评估种群适应度
        
        Args:
            population: 种群数组
        
        Returns:
            适应度数组
        """
        fitness_scores = np.zeros(population.shape[0])
        
        for i, individual in enumerate(population):
            # 仿真该个体
            result = self.simulator.simulate(individual, polarization="both")
            
            # 计算适应度
            fitness = (
                self.target_weights["te_port1"] * result.te_port1 +
                self.target_weights["tm_port2"] * result.tm_port2 +
                self.target_weights["te_port2"] * result.te_port2 +
                self.target_weights["tm_port1"] * result.tm_port1
            )
            
            fitness_scores[i] = fitness
        
        return fitness_scores
    
    def selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择操作 - 使用轮盘赌选择
        
        Args:
            population: 当前种群
            fitness_scores: 适应度值
        
        Returns:
            选中的种群和对应的适应度
        """
        # 处理负适应度：平移使所有值为正
        fitness_min = np.min(fitness_scores)
        if fitness_min <= 0:
            fitness_scores_adjusted = fitness_scores - fitness_min + 1.0
        else:
            fitness_scores_adjusted = fitness_scores
        
        # 计算选择概率
        total_fitness = np.sum(fitness_scores_adjusted)
        if total_fitness <= 0:
            probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
        else:
            probabilities = fitness_scores_adjusted / total_fitness
        
        # 轮盘赌选择
        selected_indices = np.random.choice(
            len(population),
            size=self.pop_size - self.n_elites,
            p=probabilities,
            replace=True
        )
        
        selected_population = population[selected_indices]
        selected_fitness = fitness_scores[selected_indices]
        
        return selected_population, selected_fitness
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        交叉操作 - 单点交叉
        
        Args:
            parent1, parent2: 父代个体
        
        Returns:
            子代个体
        """
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 随机选择交叉点
        crossover_point = np.random.randint(1, self.n_genes)
        
        # 扁平化交叉
        flat_p1 = parent1.flatten()
        flat_p2 = parent2.flatten()
        
        child1_flat = np.concatenate([flat_p1[:crossover_point], flat_p2[crossover_point:]])
        child2_flat = np.concatenate([flat_p2[:crossover_point], flat_p1[crossover_point:]])
        
        child1 = child1_flat.reshape(self.n_cells_x, self.n_cells_y)
        child2 = child2_flat.reshape(self.n_cells_x, self.n_cells_y)
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        变异操作 - 位翻转变异
        
        Args:
            individual: 个体基因
        
        Returns:
            变异后的个体
        """
        mutated = individual.copy()
        
        # 计算变异基因数
        n_mutations = np.random.binomial(self.n_genes, self.mutation_rate)
        
        if n_mutations > 0:
            # 随机选择要变异的位置
            mutation_positions = np.random.choice(self.n_genes, size=n_mutations, replace=False)
            
            # 展平并变异
            flat_mutated = mutated.flatten()
            for pos in mutation_positions:
                flat_mutated[pos] = 1 - flat_mutated[pos]  # 翻转
            
            mutated = flat_mutated.reshape(self.n_cells_x, self.n_cells_y)
        
        return mutated
    
    def optimize(self, init_mode: str = "random") -> Dict[str, Any]:
        """
        运行遗传算法优化
        
        Args:
            init_mode: 初始化模式
        
        Returns:
            优化结果字典
        """
        # 初始化种群
        population = self.initialize_population(init_mode)
        
        if self.verbose:
            print(f"启动遗传算法优化")
            print(f"种群大小: {self.pop_size}")
            print(f"进化代数: {self.num_generations}")
            print(f"变异率: {self.mutation_rate}")
            print(f"交叉率: {self.crossover_rate}")
            print(f"精英保留比例: {self.elite_ratio} ({self.n_elites}个体)")
            print(f"初始化模式: {init_mode}")
            print("-" * 60)
        
        for generation in range(self.num_generations):
            # 评估当前种群
            fitness_scores = self.evaluate_population(population)
            
            # 找出最优个体
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_individual = population[best_idx].copy()
            
            # 记录历史
            self.history["best_fitness"].append(best_fitness)
            self.history["avg_fitness"].append(np.mean(fitness_scores))
            self.history["std_fitness"].append(np.std(fitness_scores))
            
            if generation == 0 or best_fitness > self.history["best_fitness"][0]:
                self.history["best_individual"] = best_individual.copy()
            
            if self.verbose and (generation % max(1, self.num_generations // 10) == 0 or generation == self.num_generations - 1):
                print(f"第 {generation:3d} 代 | 最优适应度: {best_fitness:7.4f} | "
                      f"平均适应度: {np.mean(fitness_scores):7.4f} | "
                      f"标准差: {np.std(fitness_scores):7.4f}")
            
            # 精英保留：保留最优的n_elites个个体
            elite_indices = np.argsort(fitness_scores)[-self.n_elites:]
            new_population = population[elite_indices].copy()
            
            # 选择
            selected_pop, _ = self.selection(population, fitness_scores)
            
            # 生成新个体
            offspring = []
            for i in range(0, len(selected_pop) - 1, 2):
                parent1 = selected_pop[i]
                parent2 = selected_pop[i + 1]
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.append(child1)
                offspring.append(child2)
            
            # 处理奇数情况
            if len(offspring) < self.pop_size - self.n_elites:
                parent = selected_pop[np.random.randint(len(selected_pop))]
                child = self.mutate(parent.copy())
                offspring.append(child)
            
            # 组建新种群
            population = np.vstack([new_population, np.array(offspring[:self.pop_size - self.n_elites])])
        
        # 最终评估
        final_fitness = self.evaluate_population(population)
        best_idx = np.argmax(final_fitness)
        best_individual = population[best_idx].copy()
        best_fitness = final_fitness[best_idx]
        
        # 获取最优解的详细仿真结果
        best_result = self.simulator.simulate(best_individual, polarization="both")
        
        if self.verbose:
            print("-" * 60)
            print("优化完成！")
            print(f"最优结构适应度: {best_fitness:.4f}")
            print(f"TE→Port1效率: {best_result.te_port1:.4f}")
            print(f"TM→Port2效率: {best_result.tm_port2:.4f}")
            print(f"TE→Port2串扰: {best_result.te_port2:.4f}")
            print(f"TM→Port1串扰: {best_result.tm_port1:.4f}")
            print(f"总效率: {best_result.total_efficiency:.4f}")
            print(f"总串扰: {best_result.crosstalk:.4f}")
        
        self.history["generation_count"] = self.num_generations
        
        return {
            "best_individual": best_individual,
            "best_fitness": best_fitness,
            "best_result": best_result,
            "history": self.history,
            "final_population": population,
            "final_fitness": final_fitness
        }
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """获取优化历史"""
        return self.history.copy()


class GreedyOptimizer:
    """
    贪心搜索优化器 (hill-climbing)

    逐步翻转单元以提升适应度，作为轻量级baseline算法。
    """

    def __init__(
        self,
        simulator: MMISimulator,
        n_cells_x: int = 30,
        n_cells_y: int = 8,
        max_steps: int = 200,
        patience: int = 20,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> None:
        self.simulator = simulator
        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.n_genes = n_cells_x * n_cells_y
        self.max_steps = max_steps
        self.patience = patience
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)

        # 与GA一致的目标权重
        self.target_weights = {
            "te_port1": 1.0,
            "tm_port2": 1.0,
            "te_port2": -0.5,
            "tm_port1": -0.5,
        }

    def _init_structure(self, init_mode: str) -> np.ndarray:
        if init_mode == "random":
            return np.random.randint(0, 2, size=(self.n_cells_x, self.n_cells_y))
        if init_mode == "zeros":
            return np.zeros((self.n_cells_x, self.n_cells_y), dtype=int)
        if init_mode == "ones":
            return np.ones((self.n_cells_x, self.n_cells_y), dtype=int)
        if init_mode == "half":
            return np.random.randint(0, 2, size=(self.n_cells_x, self.n_cells_y))
        raise ValueError(f"Unknown initialization mode: {init_mode}")

    def _fitness(self, result: SimulationResult) -> float:
        return (
            self.target_weights["te_port1"] * result.te_port1
            + self.target_weights["tm_port2"] * result.tm_port2
            + self.target_weights["te_port2"] * result.te_port2
            + self.target_weights["tm_port1"] * result.tm_port1
        )

    def optimize(self, init_mode: str = "random") -> Dict[str, Any]:
        """
        运行贪心搜索。

        每步尝试单点翻转，一旦找到提升即接受；无提升达到耐心阈值则停止。
        """

        structure = self._init_structure(init_mode)
        best_result = self.simulator.simulate(structure, polarization="both")
        best_fitness = self._fitness(best_result)

        history = {
            "best_fitness": [best_fitness],
            "step_fitness": [best_fitness],
            "te_port1": [best_result.te_port1],
            "tm_port2": [best_result.tm_port2],
            "crosstalk": [best_result.crosstalk],
        }

        if self.verbose:
            print("启动贪心搜索优化")
            print(f"最大步数: {self.max_steps}")
            print(f"耐心: {self.patience}")
            print(f"初始化模式: {init_mode}")
            print("-" * 60)

        no_improve = 0
        for step in range(1, self.max_steps + 1):
            improved = False
            flat = structure.flatten()

            # 随机遍历单点翻转顺序
            indices = np.arange(self.n_genes)
            np.random.shuffle(indices)

            for idx in indices:
                candidate_flat = flat.copy()
                candidate_flat[idx] = 1 - candidate_flat[idx]
                candidate = candidate_flat.reshape(self.n_cells_x, self.n_cells_y)

                cand_result = self.simulator.simulate(candidate, polarization="both")
                cand_fitness = self._fitness(cand_result)
                history["step_fitness"].append(cand_fitness)

                if cand_fitness > best_fitness:
                    best_fitness = cand_fitness
                    best_result = cand_result
                    structure = candidate
                    improved = True
                    no_improve = 0
                    break

            if not improved:
                no_improve += 1

            history["best_fitness"].append(best_fitness)
            history["te_port1"].append(best_result.te_port1)
            history["tm_port2"].append(best_result.tm_port2)
            history["crosstalk"].append(best_result.crosstalk)

            if self.verbose and (step % max(1, self.max_steps // 10) == 0 or improved):
                print(
                    f"步 {step:3d} | 最优适应度: {best_fitness:7.4f} | "
                    f"TE→P1: {best_result.te_port1:6.4f} | TM→P2: {best_result.tm_port2:6.4f} | "
                    f"无提升计数: {no_improve}"
                )

            if no_improve >= self.patience:
                if self.verbose:
                    print("无提升达到耐心阈值，提前停止。")
                break

        return {
            "best_structure": structure,
            "best_fitness": best_fitness,
            "best_result": best_result,
            "history": history,
        }

