"""
MEEP 仿真器封装

提供MMI PBS结构的FDTD仿真功能，支持并行计算。
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, Callable
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import yaml

from dataclasses import dataclass
try:
    from .core import Simulator, SimulationResult
except ImportError:
    from core import Simulator, SimulationResult



@dataclass
class SimulationConfig:
    """仿真配置"""
    # 几何参数 (单位: μm)
    wavelength: float = 1.55  # 工作波长
    mmi_width: float = 4.0  # MMI区域宽度
    mmi_length: float = 15.0  # MMI区域长度
    wg_width: float = 0.5  # 输入/输出波导宽度
    wg_length: float = 2.0  # 输入/输出波导长度
    taper_length: float = 1.0  # 锥形波导长度
    taper_width: float = 1.0   # 锥形波导在设计区域端的宽度
    thickness: float = 0.22  # 波导厚度 (SOI)
    
    # 材料参数
    n_si: float = 3.48  # 硅折射率 @1550nm
    n_sio2: float = 1.44  # 二氧化硅折射率
    
    # 仿真参数
    resolution: int = 10  # 网格分辨率 (pixels/μm)
    pml_thickness: float = 1.0  # PML厚度
    run_time: float = 200  # 仿真时间 (in units of 1/frequency)
    
    # 网格参数
    n_cells_x: int = 30  # MMI区域X方向网格数
    n_cells_y: int = 8  # MMI区域Y方向网格数
    
    # EIM 参数
    use_eim: bool = True  # 是否使用等效折射率方法
    slab_thickness: float = 0.22  # 平板波导厚度 (μm) (同 thickness, 明确语义)


def load_simulation_config(config_path: Optional[str] = None) -> Tuple[SimulationConfig, int]:
    """
    从YAML加载仿真配置，避免硬编码。

    Returns:
        (SimulationConfig实例, num_workers)
    """

    default_path = Path(__file__).resolve().parent / "configs" / "default_config.yaml"
    path = Path(config_path) if config_path else default_path

    if not path.exists():
        return SimulationConfig(), 12

    with path.open("r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f) or {}

    def g(section: str, key: str, default: Any):
        return (cfg_yaml.get(section, {}) or {}).get(key, default)

    config = SimulationConfig(
        wavelength=g("geometry", "wavelength", SimulationConfig.wavelength),
        mmi_width=g("geometry", "mmi_width", SimulationConfig.mmi_width),
        mmi_length=g("geometry", "mmi_length", SimulationConfig.mmi_length),
        wg_width=g("geometry", "wg_width", SimulationConfig.wg_width),
        wg_length=g("geometry", "wg_length", SimulationConfig.wg_length),
        taper_length=g("geometry", "taper_length", SimulationConfig.taper_length),
        taper_width=g("geometry", "taper_width", SimulationConfig.taper_width),
        thickness=g("geometry", "thickness", SimulationConfig.thickness),
        n_si=g("materials", "n_si", SimulationConfig.n_si),
        n_sio2=g("materials", "n_sio2", SimulationConfig.n_sio2),
        resolution=g("simulation", "resolution", SimulationConfig.resolution),
        pml_thickness=g("simulation", "pml_thickness", SimulationConfig.pml_thickness),
        run_time=g("simulation", "run_time", SimulationConfig.run_time),
        n_cells_x=g("structure", "n_cells_x", SimulationConfig.n_cells_x),
        n_cells_y=g("structure", "n_cells_y", SimulationConfig.n_cells_y),
        use_eim=g("simulation", "use_eim", SimulationConfig.use_eim),
        slab_thickness=g("geometry", "thickness", SimulationConfig.slab_thickness),
    )

    num_workers = g("parallel", "num_workers", 12)
    return config, num_workers


class MMISimulator(Simulator):
    """
    MMI PBS MEEP仿真器
    
    支持TE和TM偏振的仿真，计算各端口的传输效率。
    """
    
    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "MMISimulator":
        config, num_workers = load_simulation_config(config_path)
        return cls(config=config, num_workers=num_workers)
    
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
        structure: Optional[np.ndarray] = None,
        polarization: str = "both",
        sdf_func: Optional[Callable[[Any], float]] = None
    ) -> SimulationResult:
        """
        运行FDTD仿真
        
        Args:
            structure: (可选) 结构数组，形状为(n_cells_x, n_cells_y)，值为0或1。
                      0表示SiO2，1表示Si
            polarization: 偏振态 "te", "tm", 或 "both"
            sdf_func: (可选) 用于定义MMI区域几何的SDF函数。
                      如果提供，则忽略 `structure` 参数。
        
        Returns:
            SimulationResult 对象
        """
        if not self._meep_available:
            return self._mock_simulate(structure)

        if structure is None and sdf_func is None:
            raise ValueError("Either 'structure' or 'sdf_func' must be provided for simulation.")
        
        results = {}
        
        if polarization in ["te", "both"]:
            results.update(self._run_single_pol(structure, "te", sdf_func=sdf_func))
        
        if polarization in ["tm", "both"]:
            results.update(self._run_single_pol(structure, "tm", sdf_func=sdf_func))
        
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
    
    def get_effective_index(self, n_core: float, n_clad: float, thickness: float, polarization: str, wavelength: float) -> float:
        """
        计算平板波导的有效折射率 (Effective Index Method)
        
        Args:
            n_core: 芯层折射率
            n_clad: 包层折射率 (假定上下包层相同，对称波导)
            thickness: 波导厚度 (um)
            polarization: "te" or "tm" (Chip polarization)
                          TE: Electric field parallel to slab (Ey predominantly)
                          TM: Electric field perpendicular to slab (Ez predominantly, but in slab mode analysis usually called TM)
                          NOTE: 
                          - In slab analysis: 
                            - TE modes have E parallel to interface (Iy) -> n_eff ~ 2.8 for Si
                            - TM modes have H parallel to interface (Ix) -> n_eff ~ 1.8 for Si
            wavelength: 工作波长 (um)
            
        Returns:
            effective index
        """
        # 简单求解超越方程 or 使用近似公式
        # 这里使用简单的近似或二分法求解对称平板波导
        
        k0 = 2 * np.pi / wavelength
        
        # 定义色散方程残差函数
        def dispersion_func(n_eff):
            if n_eff <= n_clad or n_eff >= n_core:
                return 1.0 # Invalid
            
            k0_h = k0 * thickness
            u = k0_h * np.sqrt(n_core**2 - n_eff**2)
            v = k0_h * np.sqrt(n_eff**2 - n_clad**2)
            
            if polarization.lower() == "te":
                # TE mode (E parallel to slab)
                # tan(u/2) = v/u (even modes, fundamental is even)
                return np.tan(0.5 * u) - v / u
            else:
                # TM mode (H parallel to slab)
                # tan(u/2) = (n_core^2/n_clad^2) * (v/u)
                return np.tan(0.5 * u) - (n_core**2 / n_clad**2) * (v / u)

        # 二分查找 n_eff
        # n_eff 范围 (n_clad, n_core)
        low = n_clad + 1e-4
        high = n_core - 1e-4
        
        # 快速迭代
        for _ in range(20):
            mid = (low + high) / 2
            val = dispersion_func(mid)
            if val > 0:
                # 对于 tan(u/2), 当 n_eff 增大 -> u 减小 -> tan(u/2) 减小. v 增加.
                # func = tan - v/u.  n_eff UP => func DOWN.
                # val > 0 => func too big => n_eff too small
                if polarization.lower() == "te":
                    high = mid # Wait. Let's check slope.
                    # u decreases with n_eff. v increases with n_eff.
                    # R.H.S (v/u) increases with n_eff.
                    # L.H.S (tan(u/2)) decreases with n_eff.
                    # f = LHS - RHS. f decreases as n_eff increases.
                    # if f > 0, we need to increase n_eff to decrease f.
                    low = mid
                else:
                    low = mid
            else:
                high = mid
                
        return (low + high) / 2

    def _run_single_pol(self, structure: Optional[np.ndarray], pol: str, sdf_func: Optional[Callable[[Any], float]] = None) -> Dict[str, float]:
        """
        运行单一偏振仿真
        
        CRITICAL UPDATE FOR ON-CHIP SIMULATION:
        1. Polarization Mapping:
           - On-Chip TE (E parallel to chip) -> Meep Hz (TM mode in 2D)
           - On-Chip TM (E perp to chip)     -> Meep Ez (TE mode in 2D)
        
        2. Effective Index Method:
           - Materials are defined by n_eff calculated for the specific thickness and polarization.
        """
        mp = self.mp
        cfg = self.config
        
        # 减少Meep的日志输出
        mp.verbosity(0)
        
        # 计算有效折射率
        if cfg.use_eim:
            # Calculate n_eff for Core (Si) and Cladding (SiO2)
            # 注意: 这里计算的是平板模式的 n_eff
            # Core regions have Si slab. Cladding regions have thickness but SiO2 material? 
            # 通常 2D 仿真: 
            # - "Core" material = n_eff(Si slab in SiO2 bg)
            # - "Cladding" material = n_eff(SiO2 slab) ... 实际上就是 n_sio2 体材料
            #   或者近似为 n_sio2，因为通常刻蚀区域是完全刻穿或者剩下的也是SiO2
            
            n_eff_si = self.get_effective_index(cfg.n_si, cfg.n_sio2, cfg.thickness, pol, cfg.wavelength)
            n_eff_sio2 = cfg.n_sio2 # Background is typically SiO2 cladded
            
            # 使用计算出的有效折射率
            n_core_sim = n_eff_si
            n_bg_sim = n_eff_sio2
        else:
            n_core_sim = cfg.n_si
            n_bg_sim = cfg.n_sio2
        
        # 计算仿真区域尺寸 (包含锥形波导)
        total_wg_length = cfg.wg_length + cfg.taper_length
        sx = cfg.mmi_length + 2 * total_wg_length + 2 * cfg.pml_thickness
        sy = cfg.mmi_width + 2 * cfg.pml_thickness
        
        # 创建材料
        MatCore = mp.Medium(index=n_core_sim)
        MatBg = mp.Medium(index=n_bg_sim)
        
        # 定义几何结构
        geometry = []
        
        # 背景是 MatBg (通过default_material设置)
        
        # ============ 输入端 ============
        # 输入直波导 (直接连接到 MMI 边缘)
        # 长度 = wg_length + taper_length (保持总长度不变，或者只用 wg_length)
        # 既然用户要求固定宽度的直波导，我们去掉锥形，直接延伸直波导
        
        # 为了保持仿真区域总尺寸一致，我们将原有的 taper 区域替换为直波导
        # 总输入波导长度 = cfg.wg_length + cfg.taper_length
        input_wg_len_total = cfg.wg_length + cfg.taper_length
        input_wg_center_x = -cfg.mmi_length/2 - input_wg_len_total/2
        
        geometry.append(mp.Block(
            size=mp.Vector3(input_wg_len_total, cfg.wg_width, mp.inf),
            center=mp.Vector3(input_wg_center_x, 0, 0),
            material=MatCore
        ))
        
        # ============ 输出端 ============
        # 输出端口位于 mmi_width/4 位置（标准 1×2 MMI 配置）
        output_y = cfg.mmi_width / 4
        output_wg_len_total = cfg.wg_length + cfg.taper_length
        output_wg_center_x = cfg.mmi_length/2 + output_wg_len_total/2
        
        # 输出直波导1 (上方)
        geometry.append(mp.Block(
            size=mp.Vector3(output_wg_len_total, cfg.wg_width, mp.inf),
            center=mp.Vector3(output_wg_center_x, output_y, 0),
            material=MatCore
        ))
        
        # 输出直波导2 (下方)
        geometry.append(mp.Block(
            size=mp.Vector3(output_wg_len_total, cfg.wg_width, mp.inf),
            center=mp.Vector3(output_wg_center_x, -output_y, 0),
            material=MatCore
        ))
        
        # MMI区域 - 根据SDF或structure数组构建
        if sdf_func is not None:
            # 使用 MaterialGrid 实现 SDF 几何
            # 这比 UserDefinedGeometricObject 兼容性更好
            
            # 1. 生成网格坐标
            xs = np.linspace(-cfg.mmi_length/2, cfg.mmi_length/2, cfg.n_cells_x)
            ys = np.linspace(-cfg.mmi_width/2, cfg.mmi_width/2, cfg.n_cells_y)
            xv, yv = np.meshgrid(xs, ys, indexing='ij') # shape (nx, ny)
            
            # 2. 计算每个网格点的权重
            # weights=1 对应 medium2 (Air/Holes), weights=0 对应 medium1 (Si/Core)
            weights = np.zeros((cfg.n_cells_x, cfg.n_cells_y))
            
            for i in range(cfg.n_cells_x):
                for j in range(cfg.n_cells_y):
                    p = mp.Vector3(xv[i, j], yv[i, j])
                    # SDF <= 0 表示在圆(孔)内部 -> 设为空气 (weights=1)
                    if sdf_func(p) <= 0:
                        weights[i, j] = 1.0
                    else:
                        weights[i, j] = 0.0
            
            # 3. 创建 MaterialGrid
            # medium1: MatCore (Silicon slab)
            # medium2: mp.air (Holes)
            grid = mp.MaterialGrid(
                mp.Vector3(cfg.n_cells_x, cfg.n_cells_y, 0),
                MatCore, # weights=0
                mp.air,  # weights=1
                weights=weights,
                beta=100 # Sharp transition
            )
            
            geometry.append(mp.Block(
                size=mp.Vector3(cfg.mmi_length, cfg.mmi_width, mp.inf),
                center=mp.Vector3(0, 0, 0),
                material=grid
            ))

        elif structure is not None:
            # MMI区域 - 使用 MaterialGrid 高效构建
            # 注意: structure=1 表示 Si (Core), structure=0 表示 SiO2/Air (Bg)
            # MaterialGrid: weights=1 -> medium2, weights=0 -> medium1
            # 我们定义 medium1=MatBg, medium2=MatCore
            # 这样 structure 的值 (0/1) 直接对应权重
            
            # 确保 structure 是正确方向
            # 输入 structure shape: (n_cells_x, n_cells_y)
            # Meep grid 期望: x 对应长度, y 对应宽度
            
            # 使用 MaterialGrid 进行亚像素平滑
            grid = mp.MaterialGrid(
                mp.Vector3(cfg.n_cells_x, cfg.n_cells_y, 0),
                MatBg,    # weights=0 -> Background (SiO2)
                MatCore,  # weights=1 -> Core (Si)
                weights=structure, # 0.0~1.0
                beta=20 # Soft transition for optimization stability if using adjoint, or strict for binary
                # 对于 RL, 如果 structure 是 binary，这没影响。如果 continuous，这会有帮助。
            )
            
            geometry.append(mp.Block(
                size=mp.Vector3(cfg.mmi_length, cfg.mmi_width, mp.inf),
                center=mp.Vector3(0, 0, 0),
                material=grid
            ))

        
        
        # 光源设置
        fcen = 1 / cfg.wavelength  # 中心频率
        
        # === CRITICAL POLARIZATION MAPPING ===
        # ON-CHIP TE (E parallel to chip) -> MEEP 2D TM / Hz
        # ON-CHIP TM (E perp to chip)     -> MEEP 2D TE / Ez
        
        if pol == "te":
            # On-Chip TE -> Meep: Hz polarization (Magnetic field out of plane)
            # Corresponds to TE modes in integrated photonics (E_x, E_y, H_z)
            # In Meep 2D, this is often called TM because H is out of plane? 
            # No, wait. 
            # Meep 2D Conventions:
            # TE: Ez, Hx, Hy (E out of plane) -> Physical TM (E perp to chip)
            # TM: Hz, Ex, Ey (H out of plane) -> Physical TE (E parallel to chip)
            
            src_component = mp.Hz
        else:
            # On-Chip TM -> Meep: Ez polarization (Electric field out of plane)
            # Physical TM (E perp to chip) -> Meep TE (Ez)
            src_component = mp.Ez
        
        sources = [mp.Source(
            mp.ContinuousSource(frequency=fcen),
            component=src_component,
            # source位置: 放在左侧直波导的起始处附近
            center=mp.Vector3(input_wg_center_x - input_wg_len_total/2 + 0.5, 0, 0),
            size=mp.Vector3(0, cfg.wg_width * 2, 0)
        )]
        
        # 创建仿真
        sim = mp.Simulation(
            cell_size=mp.Vector3(sx, sy, 0),
            geometry=geometry,
            sources=sources,
            boundary_layers=[mp.PML(cfg.pml_thickness)],
            resolution=cfg.resolution,
            default_material=MatBg
        )
        
        # 设置通量监测器
        flux_freq = fcen
        nfreq = 1
        
        # 通量监视器位置调整：直接放在直波导上
        # 输入通量: 在源之后，MMI之前
        flux_in_x = input_wg_center_x + input_wg_len_total/2 - 0.5 # 靠近MMI入口
        
        flux_in = sim.add_flux(
            flux_freq, 0, nfreq,
            mp.FluxRegion(
                center=mp.Vector3(flux_in_x, 0, 0),
                size=mp.Vector3(0, cfg.wg_width * 2, 0)
            )
        )
        
        # 输出通量: 在MMI之后直波导上
        flux_out_x = output_wg_center_x - output_wg_len_total/2 + 0.5 # 靠近MMI出口
        
        # 输出端口1通量 (上方)
        flux_out1 = sim.add_flux(
            flux_freq, 0, nfreq,
            mp.FluxRegion(
                center=mp.Vector3(flux_out_x, output_y, 0),
                size=mp.Vector3(0, cfg.wg_width * 2, 0)
            )
        )
        
        # 输出端口2通量 (下方)
        flux_out2 = sim.add_flux(
            flux_freq, 0, nfreq,
            mp.FluxRegion(
                center=mp.Vector3(flux_out_x, -output_y, 0),
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
        
        # 模拟效率（改进：提高基础效率使其更接近真实PBS）
        base_eff = 0.5 + 0.3 * fill_factor  # 从 0.3 改为 0.5
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
        return results

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
            # 片上 TE: 电场分量是 Ex, Ey
            # 1. Forward Simulation (Source at Input)
            fields_fwd = self._run_field_sim(structure, "te", source_port="in")
            
            # 2. Adjoint Simulations
            if target_weights.get("te_port1", 0.0) != 0.0:
                fields_adj = self._run_field_sim(structure, "te", source_port="out1")
                # Gradient = 2 * Re(coeff* . E_fwd . E_adj)
                coeff = fields_fwd.get("coeff_out1", 1.0)
                grad = 2 * np.real(np.conj(coeff) * (fields_fwd["Ex"] * fields_adj["Ex"] + 
                                 fields_fwd["Ey"] * fields_adj["Ey"]))
                grad_accum += target_weights["te_port1"] * grad
                
            if target_weights.get("te_port2", 0.0) != 0.0:
                fields_adj = self._run_field_sim(structure, "te", source_port="out2")
                coeff = fields_fwd.get("coeff_out2", 1.0)
                grad = 2 * np.real(np.conj(coeff) * (fields_fwd["Ex"] * fields_adj["Ex"] + 
                                 fields_fwd["Ey"] * fields_adj["Ey"]))
                grad_accum += target_weights["te_port2"] * grad

        # 处理 TM 梯度
        if any(k.startswith("tm") for k in target_weights.keys()):
            # 片上 TM: 电场分量是 Ez
            # 1. Forward Simulation
            fields_fwd = self._run_field_sim(structure, "tm", source_port="in")
            
            # 2. Adjoint Simulations
            if target_weights.get("tm_port1", 0.0) != 0.0:
                fields_adj = self._run_field_sim(structure, "tm", source_port="out1")
                # Gradient = 2 * Re(coeff* * E_fwd * E_adj)
                coeff = fields_fwd.get("coeff_out1", 1.0)
                grad = 2 * np.real(np.conj(coeff) * (fields_fwd["Ez"] * fields_adj["Ez"]))
                grad_accum += target_weights["tm_port1"] * grad
                
            if target_weights.get("tm_port2", 0.0) != 0.0:
                fields_adj = self._run_field_sim(structure, "tm", source_port="out2")
                coeff = fields_fwd.get("coeff_out2", 1.0)
                grad = 2 * np.real(np.conj(coeff) * (fields_fwd["Ez"] * fields_adj["Ez"]))
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
        
        # 几何设置 (与 _run_single_pol 相同，包含锥形波导)
        total_wg_length = cfg.wg_length + cfg.taper_length
        sx = cfg.mmi_length + 2 * total_wg_length + 2 * cfg.pml_thickness
        sy = cfg.mmi_width + 2 * cfg.pml_thickness
        
        # 保持与正向仿真一致的材料模型 (EIM)
        if cfg.use_eim:
            n_eff_si = self.get_effective_index(cfg.n_si, cfg.n_sio2, cfg.thickness, pol, cfg.wavelength)
            n_eff_sio2 = cfg.n_sio2
            Si = mp.Medium(index=n_eff_si)
            SiO2 = mp.Medium(index=n_eff_sio2)
        else:
            Si = mp.Medium(index=cfg.n_si)
            SiO2 = mp.Medium(index=cfg.n_sio2)
        
        # 使用连续值构建几何以获得准确梯度
        geometry = self._build_geometry(structure, sx, sy, Si, SiO2, continuous=True)
        
        # 光源设置
        fcen = 1 / cfg.wavelength
        # 输出端口位于 mmi_width/4 位置
        output_y = cfg.mmi_width / 4
        
        if pol == "te":
            src_component = mp.Hz
        else:
            src_component = mp.Ez
            
        # 确定光源位置 (在直波导区域)
        if source_port == "in":
            src_center = mp.Vector3(-sx/2 + cfg.pml_thickness + cfg.wg_length/2, 0, 0)
            src_size = mp.Vector3(0, cfg.wg_width * 2, 0)
        elif source_port == "out1":
            src_center = mp.Vector3(sx/2 - cfg.pml_thickness - cfg.wg_length/2, output_y, 0)
            src_size = mp.Vector3(0, cfg.wg_width * 2, 0)
        elif source_port == "out2":
            src_center = mp.Vector3(sx/2 - cfg.pml_thickness - cfg.wg_length/2, -output_y, 0)
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

        # Add mode monitors if source_port == "in"
        mode_monitors = {}
        if source_port == "in":
             # Output port 1 (top, 在锥形波导之后的直波导区域)
             fr1 = mp.FluxRegion(
                center=mp.Vector3(sx/2 - cfg.pml_thickness - cfg.wg_length/2, output_y, 0),
                size=mp.Vector3(0, cfg.wg_width * 2, 0)
             )
             # Output port 2 (bottom)
             fr2 = mp.FluxRegion(
                center=mp.Vector3(sx/2 - cfg.pml_thickness - cfg.wg_length/2, -output_y, 0),
                size=mp.Vector3(0, cfg.wg_width * 2, 0)
             )
             
             mode_monitors["out1"] = sim.add_mode_monitor(fcen, 0, 1, fr1)
             mode_monitors["out2"] = sim.add_mode_monitor(fcen, 0, 1, fr2)
        
        # 定义 DFT 监视区域 (仅覆盖 MMI 区域)
        # MMI中心在 (0,0), 大小 (mmi_length, mmi_width)
        mmi_region = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(cfg.mmi_length, cfg.mmi_width, 0)
        )
        
        # 添加 DFT 监视器
        # 注意：片上 TE (Meep Hz 源) 的电场是 Ex, Ey
        #       片上 TM (Meep Ez 源) 的电场是 Ez
        if pol == "te":
            dft_obj = sim.add_dft_fields([mp.Ex, mp.Ey], fcen, 0, 1, where=mmi_region)
        else:
            dft_obj = sim.add_dft_fields([mp.Ez], fcen, 0, 1, where=mmi_region)
            
        # 运行仿真
        sim.run(until=cfg.run_time)
        
        # 提取场并重采样到 structure 的形状
        # 注意: get_array 返回的数组可能与 structure 形状不完全匹配，需要插值
        fields = {}
        
        # Get mode coefficients
        if source_port == "in":
             res1 = sim.get_eigenmode_coefficients(mode_monitors["out1"], [1])
             res2 = sim.get_eigenmode_coefficients(mode_monitors["out2"], [1])
             
             # Extract alpha (amplitude of forward going wave)
             fields["coeff_out1"] = res1.alpha[0, 0, 0]
             fields["coeff_out2"] = res2.alpha[0, 0, 0]
        
        target_shape = structure.shape # (nx, ny)
        
        if pol == "te":
            # TE: Ex, Ey (面内电场)
            ex_data = sim.get_dft_array(dft_obj, mp.Ex, 0)
            ey_data = sim.get_dft_array(dft_obj, mp.Ey, 0)
            fields["Ex"] = self._resample_field(ex_data, target_shape)
            fields["Ey"] = self._resample_field(ey_data, target_shape)
        else:
            # TM: Ez (面外电场)
            ez_data = sim.get_dft_array(dft_obj, mp.Ez, 0)
            fields["Ez"] = self._resample_field(ez_data, target_shape)
            
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
        
        # ============ 输入端 ============
        # 输入直波导
        input_wg_center_x = -sx/2 + cfg.pml_thickness + cfg.wg_length/2
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.wg_length, cfg.wg_width, mp.inf),
            center=mp.Vector3(input_wg_center_x, 0, 0),
            material=Si
        ))
        
        # 输入锥形波导 (从 wg_width 渐变到 taper_width)
        input_taper_center_x = -cfg.mmi_length/2 - cfg.taper_length/2
        input_taper_vertices = [
            mp.Vector3(input_taper_center_x - cfg.taper_length/2, -cfg.wg_width/2),
            mp.Vector3(input_taper_center_x - cfg.taper_length/2, cfg.wg_width/2),
            mp.Vector3(input_taper_center_x + cfg.taper_length/2, cfg.taper_width/2),
            mp.Vector3(input_taper_center_x + cfg.taper_length/2, -cfg.taper_width/2),
        ]
        geometry.append(mp.Prism(
            vertices=input_taper_vertices,
            height=mp.inf,
            material=Si
        ))
        
        # ============ 输出端 ============
        output_y = cfg.mmi_width / 4
        
        # 输出锥形波导1 (上方)
        output_taper_center_x = cfg.mmi_length/2 + cfg.taper_length/2
        output_taper1_vertices = [
            mp.Vector3(output_taper_center_x - cfg.taper_length/2, output_y - cfg.taper_width/2),
            mp.Vector3(output_taper_center_x - cfg.taper_length/2, output_y + cfg.taper_width/2),
            mp.Vector3(output_taper_center_x + cfg.taper_length/2, output_y + cfg.wg_width/2),
            mp.Vector3(output_taper_center_x + cfg.taper_length/2, output_y - cfg.wg_width/2),
        ]
        geometry.append(mp.Prism(
            vertices=output_taper1_vertices,
            height=mp.inf,
            material=Si
        ))
        
        # 输出直波导1 (上方)
        output_wg_center_x = sx/2 - cfg.pml_thickness - cfg.wg_length/2
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.wg_length, cfg.wg_width, mp.inf),
            center=mp.Vector3(output_wg_center_x, output_y, 0),
            material=Si
        ))
        
        # 输出锥形波导2 (下方)
        output_taper2_vertices = [
            mp.Vector3(output_taper_center_x - cfg.taper_length/2, -output_y - cfg.taper_width/2),
            mp.Vector3(output_taper_center_x - cfg.taper_length/2, -output_y + cfg.taper_width/2),
            mp.Vector3(output_taper_center_x + cfg.taper_length/2, -output_y + cfg.wg_width/2),
            mp.Vector3(output_taper_center_x + cfg.taper_length/2, -output_y - cfg.wg_width/2),
        ]
        geometry.append(mp.Prism(
            vertices=output_taper2_vertices,
            height=mp.inf,
            material=Si
        ))
        
        # 输出直波导2 (下方)
        geometry.append(mp.Block(
            size=mp.Vector3(cfg.wg_length, cfg.wg_width, mp.inf),
            center=mp.Vector3(output_wg_center_x, -output_y, 0),
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


class SdfGeneticAlgorithmOptimizer:
    """
    基于SDF（有向距离函数）参数化的遗传算法优化器。
    
    这种方法不是优化离散的像素网格，而是优化定义
    连续几何形状的一组参数（例如，圆心和半径）。
    """
    
    def __init__(
        self,
        simulator: MMISimulator,
        pop_size: int = 50,
        num_generations: int = 100,
        mutation_strength: float = 0.1,
        n_circles: int = 10,
        log_path: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        self.simulator = simulator
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_strength = mutation_strength
        self.n_circles = n_circles
        self.n_params = n_circles * 3  # 每个圆3个参数 (x, y, r)
        self.verbose = verbose
        self.log_file = Path(log_path).expanduser() if log_path else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            np.random.seed(seed)
            
        # 定义参数边界
        self.domain = (simulator.config.mmi_length, simulator.config.mmi_width)
        self.bounds = np.array(
            [[-self.domain[0]/2, self.domain[0]/2],  # x
             [-self.domain[1]/2, self.domain[1]/2],  # y
             [0.01, self.domain[1]/4]] * self.n_circles # r - Allow very small radius
        ).T

        self.target_weights = {
            "te_port1": 1.0, "tm_port2": 1.0,
            "te_port2": -0.5, "tm_port1": -0.5  # 加强对串扰的惩罚
        }
        
        self.history = {
            "best_fitness": [], "avg_fitness": [], "std_fitness": [],
            "best_individual": None, "generation_count": 0
        }

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
        if self.log_file:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

    @staticmethod
    def create_sdf_from_params(params: np.ndarray, n_circles: int, domain_size: Tuple[float, float]):
        """从参数向量创建一个SDF函数。"""
        import meep as mp
        
        circles = params.reshape((n_circles, 3))
        
        def sdf(p: mp.Vector3) -> float:
            # 初始化为最大距离（在形状外部）
            min_dist = 1e6
            # 计算到所有圆的SDF的并集（最小距离）
            for x, y, r in circles:
                dist_sq = (p.x - x)**2 + (p.y - y)**2
                min_dist = min(min_dist, np.sqrt(dist_sq) - r)
            return min_dist
            
        return sdf

    def _fitness_from_result(self, result: SimulationResult) -> float:
        """根据仿真结果计算适应度。"""
        return (
            self.target_weights["te_port1"] * result.te_port1 +
            self.target_weights["tm_port2"] * result.tm_port2 +
            self.target_weights["te_port2"] * result.te_port2 +
            self.target_weights["tm_port1"] * result.tm_port1
        )
        
    def initialize_population(self) -> np.ndarray:
        """初始化种群，每个个体是一组SDF参数。"""
        population = np.random.rand(self.pop_size, self.n_params)
        low, high = self.bounds
        population = low + population * (high - low)
        return population

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """并行评估种群的适应度。"""
        from tqdm import tqdm
        tasks = [(
            ind, self.simulator.config, self.n_circles, self.domain
        ) for ind in population]
        
        results = []
        with ProcessPoolExecutor(max_workers=self.simulator.num_workers) as executor:
            # 使用 list(tqdm(...)) 来显示进度条
            # 注意：executor.map 会按顺序返回结果，可能会有等待
            # 为了更平滑的进度条，我们使用 submit + as_completed (但这会打乱顺序，需要重新排序)
            # 为了简单起见，这里仍用 map 但加上 tqdm 监控进度
            results = list(tqdm(executor.map(simulate_single_sdf, tasks), total=len(tasks), desc="Evaluating Population"))
            
        return np.array([self._fitness_from_result(res) for res in results])

    def selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """锦标赛选择。"""
        tournament_size = 5
        selection_indices = np.random.randint(0, len(population), size=(len(population), tournament_size))
        tournament_fitness = fitness[selection_indices]
        winner_indices = selection_indices[np.arange(len(population)), tournament_fitness.argmax(axis=1)]
        return population[winner_indices]

    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """模拟二进制交叉 (SBX)。"""
        offspring = np.zeros_like(parents)
        eta = 2.0  # 交叉分布指数
        
        # 确保parents数量是偶数，如果是奇数则跳过最后一个
        n_pairs = len(parents) // 2
        
        for i in range(n_pairs):
            idx1 = i * 2
            idx2 = i * 2 + 1
            
            p1, p2 = parents[idx1], parents[idx2]
            
            u = np.random.rand(self.n_params)
            beta = np.where(u <= 0.5, (2 * u)**(1/(eta+1)), (1/(2 - 2*u))**(1/(eta+1)))
            
            offspring[idx1] = 0.5 * ((1 + beta)*p1 + (1 - beta)*p2)
            offspring[idx2] = 0.5 * ((1 - beta)*p1 + (1 + beta)*p2)
        
        # 如果parents数量是奇数，最后一个直接复制
        if len(parents) % 2 == 1:
            offspring[-1] = parents[-1].copy()
            
        return offspring

    def mutate(self, population: np.ndarray) -> np.ndarray:
        """高斯变异。"""
        mutation_mask = np.random.rand(*population.shape) < 0.1  # 10% 变异概率
        
        low, high = self.bounds
        scale = (high - low) * self.mutation_strength
        
        noise = np.random.normal(0, scale, size=population.shape)
        population += mutation_mask * noise
        
        # 确保参数在边界内
        np.clip(population, low, high, out=population)
        return population

    def optimize(self) -> Dict[str, Any]:
        """运行完整的优化流程。"""
        self._log("启动SDF+GA优化...")
        
        population = self.initialize_population()
        
        for gen in range(self.num_generations):
            fitness = self.evaluate_population(population)
            
            # 记录统计数据
            best_idx = np.argmax(fitness)
            best_fit = fitness[best_idx]
            self.history["best_fitness"].append(best_fit)
            self.history["avg_fitness"].append(np.mean(fitness))
            self.history["std_fitness"].append(np.std(fitness))
            
            self._log(
                f"第 {gen:3d} 代 | 最优适应度: {best_fit:8.4f} | "
                f"平均适应度: {np.mean(fitness):8.4f} | 标准差: {np.std(fitness):8.4f}"
            )

            # 精英主义
            elites = population[np.argsort(fitness)[-2:]] # 保留最好的2个
            
            # 演化
            parents = self.selection(population, fitness)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            
            # 形成新一代
            population[:-2] = offspring[:-2]
            population[-2:] = elites
            
        # 最终评估
        final_fitness = self.evaluate_population(population)
        best_idx = np.argmax(final_fitness)
        best_individual = population[best_idx]
        
        # 获取最佳个体的详细仿真结果
        best_sdf = self.create_sdf_from_params(best_individual, self.n_circles, self.domain)
        best_result = self.simulator.simulate(structure=None, sdf_func=best_sdf)
        
        self.history["best_individual"] = best_individual
        self.history["generation_count"] = self.num_generations
        
        return {
            "best_individual": best_individual,
            "best_fitness": final_fitness[best_idx],
            "best_result": best_result,
            "history": self.history,
            "final_population": population
        }

    
    def visualize_results(
        self,
        results: Any,
        structure: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        可视化仿真结果 (Efficiency Bar Chart + Optional Structure)
        
        Args:
            results: 仿真结果字典或对象
            structure: 结构数组（可选）
            figsize: 图尺寸
            save_path: 保存路径
            show: 是否显示
        """
        import matplotlib.pyplot as plt
        
        if structure is not None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Left Plot: Structure
            ax1 = axes[0]
            # Transpose for correct visualization orientation relative to meep (if needed)
            display_struct = structure.T[::-1] # Common convention for meep grids
            im = ax1.imshow(
                display_struct,
                cmap="gray",
                aspect="auto",
                origin="lower",
                vmin=0, 
                vmax=1
            )
            ax1.set_xlabel("X (cells)")
            ax1.set_ylabel("Y (cells)")
            ax1.set_title("Optimized Structure")
            plt.colorbar(im, ax=ax1, label="Material (0=SiO2, 1=Si)")
            
            # Right Plot: Efficiency Bar Chart
            ax2 = axes[1]
        else:
            fig, ax2 = plt.subplots(figsize=(6, 5))
        
        # Draw Efficiency
        labels = ["TE->P1", "TE->P2", "TM->P1", "TM->P2"]
        
        # Compatible with SimulationResult object and dict
        if hasattr(results, "te_port1"):
            values = [
                results.te_port1,
                results.te_port2,
                results.tm_port1,
                results.tm_port2
            ]
            total_eff = results.total_efficiency
            crosstalk = results.crosstalk
        else:
            values = [
                results.get("te_port1", 0),
                results.get("te_port2", 0),
                results.get("tm_port1", 0),
                results.get("tm_port2", 0)
            ]
            total_eff = results.get("total_efficiency", 0)
            crosstalk = results.get("crosstalk", 0)
        
        colors = ["green", "red", "red", "green"]  # Green=Target, Red=Crosstalk
        
        bars = ax2.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel("Efficiency")
        ax2.set_title("Port Transmission Metrics")
        ax2.set_ylim(0, 1.05)
        
        # Add values on bars
        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Add Summary Text
        ax2.text(
            0.95, 0.95,
            f"Total Eff: {total_eff:.3f}\n"
            f"Crosstalk: {crosstalk:.3f}",
            transform=ax2.transAxes,
            ha='right',
            va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig

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


def simulate_single_sdf(args):
    """
    用于并行SDF仿真的辅助函数
    """
    params, config, n_circles, domain_size = args
    simulator = MMISimulator(config, num_workers=1)
    
    sdf_func = SdfGeneticAlgorithmOptimizer.create_sdf_from_params(
        params, n_circles, domain_size
    )
    
    return simulator.simulate(structure=None, sdf_func=sdf_func)


class GeneticAlgorithmOptimizer:
    """
    遗传算法优化器 - 用于MMI PBS设计
    
    使用遗传算法进行逆向设计，作为伴随法的基准对比。
    基于项目的仿真环境进行优化。
    """
    
    def __init__(
        self,
        simulator: MMISimulator,
        n_cells_x: Optional[int] = None,
        n_cells_y: Optional[int] = None,
        pop_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        use_parallel: bool = False,
        num_workers: Optional[int] = None,
        log_path: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        初始化遗传算法优化器
        
        Args:
            simulator: MMISimulator实例
            n_cells_x: MMI区域X方向网格数（None时从simulator.config读取）
            n_cells_y: MMI区域Y方向网格数（None时从simulator.config读取）
            pop_size: 种群大小
            num_generations: 演化代数
            mutation_rate: 变异概率
            crossover_rate: 交叉概率
            elite_ratio: 精英保留比例
            use_parallel: 是否使用多进程并行评估种群
            num_workers: 并行进程数（None时使用simulator.num_workers或CPU核数）
            log_path: 日志文件路径（None时仅控制台输出）
            seed: 随机种子
            verbose: 是否打印优化过程
        """
        self.simulator = simulator
        self.n_cells_x = n_cells_x or simulator.config.n_cells_x
        self.n_cells_y = n_cells_y or simulator.config.n_cells_y
        self.n_genes = self.n_cells_x * self.n_cells_y  # 每个结构的基因数
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.n_elites = max(1, int(pop_size * elite_ratio))
        self.verbose = verbose
        self.use_parallel = use_parallel
        self.num_workers = num_workers or simulator.num_workers or (mp.cpu_count() or 1)
        self.log_file = Path(log_path).expanduser() if log_path else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if seed is not None:
            np.random.seed(seed)
        
        # 优化目标权重：TE→Port1, TM→Port2
        self.target_weights = {
            "te_port1": 1.0,
            "tm_port2": 1.0,
            "te_port2": -0.5,  # 增强串扰惩罚
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

    def _log(self, msg: str):
        """同时写入控制台和文件的简单日志函数。"""
        if self.verbose:
            print(msg)
        if self.log_file:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def _fitness_from_result(self, result: SimulationResult) -> float:
        """根据仿真结果计算适应度。"""
        return (
            self.target_weights["te_port1"] * result.te_port1 +
            self.target_weights["tm_port2"] * result.tm_port2 +
            self.target_weights["te_port2"] * result.te_port2 +
            self.target_weights["tm_port1"] * result.tm_port1
        )
    
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

        if self.use_parallel:
            tasks = [
                (individual, self.simulator.config, "both")
                for individual in population
            ]
            # 显式使用 list 收集结果，避免生成器在 map 中可能的卡顿
            # 如果需要进度条，可以在外部控制
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(simulate_single_structure, tasks))

            for i, result in enumerate(results):
                fitness_scores[i] = self._fitness_from_result(result)
        else:
            for i, individual in enumerate(population):
                result = self.simulator.simulate(individual, polarization="both")
                fitness_scores[i] = self._fitness_from_result(result)

        return fitness_scores
    
    def selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择操作 - 锦标赛选择 (Tournament Selection)
        比轮盘赌更稳健，尤其是在处理可能为负的适应度时。
        """
        tournament_size = 5
        selected_population = []
        selected_fitness = []
        
        n_select = self.pop_size - self.n_elites
        
        for _ in range(n_select):
            # 随机选择参赛者
            candidates_indices = np.random.choice(len(population), size=tournament_size, replace=True)
            candidates_fitness = fitness_scores[candidates_indices]
            
            # 选出赢家 (适应度最大者)
            winner_idx = candidates_indices[np.argmax(candidates_fitness)]
            
            selected_population.append(population[winner_idx])
            selected_fitness.append(fitness_scores[winner_idx])
            
        return np.array(selected_population), np.array(selected_fitness)
    
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
        
        self._log("启动遗传算法优化")
        self._log(f"种群大小: {self.pop_size}")
        self._log(f"进化代数: {self.num_generations}")
        self._log(f"变异率: {self.mutation_rate}")
        self._log(f"交叉率: {self.crossover_rate}")
        self._log(f"精英保留比例: {self.elite_ratio} ({self.n_elites}个体)")
        self._log(f"初始化模式: {init_mode}")
        self._log(f"并行评估: {self.use_parallel}, workers: {self.num_workers}")
        if self.log_file:
            self._log(f"日志文件: {self.log_file}")
        self._log("-" * 60)
        
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
            
            if self.verbose or self.log_file:
                self._log(
                    f"第 {generation:3d} 代 | 最优适应度: {best_fitness:7.4f} | "
                    f"平均适应度: {np.mean(fitness_scores):7.4f} | "
                    f"标准差: {np.std(fitness_scores):7.4f}"
                )
            
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
        
        if self.verbose or self.log_file:
            self._log("-" * 60)
            self._log("优化完成！")
            self._log(f"最优结构适应度: {best_fitness:.4f}")
            self._log(f"TE→Port1效率: {best_result.te_port1:.4f}")
            self._log(f"TM→Port2效率: {best_result.tm_port2:.4f}")
            self._log(f"TE→Port2串扰: {best_result.te_port2:.4f}")
            self._log(f"TM→Port1串扰: {best_result.tm_port1:.4f}")
            self._log(f"总效率: {best_result.total_efficiency:.4f}")
            self._log(f"总串扰: {best_result.crosstalk:.4f}")
        
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
        n_cells_x: Optional[int] = None,
        n_cells_y: Optional[int] = None,
        max_steps: int = 200,
        patience: int = 20,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> None:
        self.simulator = simulator
        self.n_cells_x = n_cells_x or simulator.config.n_cells_x
        self.n_cells_y = n_cells_y or simulator.config.n_cells_y
        self.n_genes = self.n_cells_x * self.n_cells_y
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


class PBSAdjointSetup:
    """
    Helper class to set up Adjoint Optimization for PBS.
    Manages two separate simulations for TE and TM polarizations
    that share the same geometric pattern (weights).
    """
    
    def __init__(self, config):
        self.config = config
        self._check_meep()
        
    def _check_meep(self):
        try:
            import meep as mp
            self.mp = mp
        except ImportError:
            raise ImportError("MEEP is required for Adjoint Optimization")

    def get_effective_indices(self, pol: str):
        """Calculate effective indices for a given polarization."""
        # Use a temporary simulator instance to access the calculation method
        # This avoids code duplication
        temp_sim = MMISimulator(self.config)
        n_si = self.config.n_si
        n_sio2 = self.config.n_sio2
        thickness = self.config.thickness
        wavelength = self.config.wavelength
        
        n_eff_core = temp_sim.get_effective_index(n_si, n_sio2, thickness, pol, wavelength)
        n_eff_bg = n_sio2 
        
        return n_eff_core, n_eff_bg

    def create_single_pol_problem(self, pol: str, design_shape: tuple):
        """
        Create OptimizationProblem for a single polarization.
        pol: 'te' or 'tm'
        design_shape: (nx, ny) tuple for the grid
        """
        mp = self.mp
        mpa = meep.adjoint
        cfg = self.config
        
        # 1. Physics & Geometry setup
        n_core, n_bg = self.get_effective_indices(pol)
        MatCore = mp.Medium(index=n_core)
        MatBg = mp.Medium(index=n_bg)
        
        # Simulation domain size
        total_wg_length = cfg.wg_length + cfg.taper_length
        sx = cfg.mmi_length + 2 * total_wg_length + 2 * cfg.pml_thickness
        sy = cfg.mmi_width + 2 * cfg.pml_thickness
        cell_size = mp.Vector3(sx, sy, 0)
        
        pml_layers = [mp.PML(cfg.pml_thickness)]
        
        # 2. Design Region (MaterialGrid)
        # Note: weights=0 -> MatBg, weights=1 -> MatCore
        design_grid = mp.MaterialGrid(
            mp.Vector3(cfg.n_cells_x, cfg.n_cells_y),
            MatBg, MatCore, weights=np.ones(design_shape)*0.5,
            beta=1 # Will be updated during optimization
        )
        
        design_region = mpa.DesignRegion(
            design_grid,
            volume=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(cfg.mmi_length, cfg.mmi_width, 0)
            )
        )
        
        # 3. Static Geometry (Waveguides)
        geometry = [
            mp.Block(center=design_region.center, size=design_region.size, material=design_grid)
        ]
        
        # Define coordinates
        input_x_center = -sx/2 + cfg.pml_thickness + cfg.wg_length/2
        output_x_center = sx/2 - cfg.pml_thickness - cfg.wg_length/2
        input_taper_x = -cfg.mmi_length/2 - cfg.taper_length/2
        output_taper_x = cfg.mmi_length/2 + cfg.taper_length/2
        output_y_offset = cfg.mmi_width / 4
        
        # Input WG (Port 0)
        geometry.append(mp.Block(size=mp.Vector3(cfg.wg_length*2, cfg.wg_width, mp.inf),
                                 center=mp.Vector3(input_x_center, 0, 0), material=MatCore))
        # Input Taper
        geometry.append(mp.Prism(
            vertices=[
                mp.Vector3(input_taper_x - cfg.taper_length/2, -cfg.wg_width/2),
                mp.Vector3(input_taper_x - cfg.taper_length/2, cfg.wg_width/2),
                mp.Vector3(input_taper_x + cfg.taper_length/2, cfg.taper_width/2),
                mp.Vector3(input_taper_x + cfg.taper_length/2, -cfg.taper_width/2),
            ], height=mp.inf, material=MatCore))
            
        # Output Top (Port 1)
        geometry.append(mp.Block(size=mp.Vector3(cfg.wg_length*2, cfg.wg_width, mp.inf),
                                 center=mp.Vector3(output_x_center, output_y_offset, 0), material=MatCore))
        geometry.append(mp.Prism(
            vertices=[
                mp.Vector3(output_taper_x - cfg.taper_length/2, output_y_offset - cfg.taper_width/2),
                mp.Vector3(output_taper_x - cfg.taper_length/2, output_y_offset + cfg.taper_width/2),
                mp.Vector3(output_taper_x + cfg.taper_length/2, output_y_offset + cfg.wg_width/2),
                mp.Vector3(output_taper_x + cfg.taper_length/2, output_y_offset - cfg.wg_width/2),
            ], height=mp.inf, material=MatCore))

        # Output Bottom (Port 2)
        geometry.append(mp.Block(size=mp.Vector3(cfg.wg_length*2, cfg.wg_width, mp.inf),
                                 center=mp.Vector3(output_x_center, -output_y_offset, 0), material=MatCore))
        geometry.append(mp.Prism(
            vertices=[
                mp.Vector3(output_taper_x - cfg.taper_length/2, -output_y_offset - cfg.taper_width/2),
                mp.Vector3(output_taper_x - cfg.taper_length/2, -output_y_offset + cfg.taper_width/2),
                mp.Vector3(output_taper_x + cfg.taper_length/2, -output_y_offset + cfg.wg_width/2),
                mp.Vector3(output_taper_x + cfg.taper_length/2, -output_y_offset - cfg.wg_width/2),
            ], height=mp.inf, material=MatCore))

        # 4. Sources
        fcen = 1 / cfg.wavelength
        width = 0.2 * fcen 
        fwidth = width
        source_center = mp.Vector3(input_x_center, 0, 0)
        source_size = mp.Vector3(0, cfg.wg_width*2, 0)
        
        # Source component: TE -> Hz, TM -> Ez
        # Note: In 2D, TE usually means Hz polarization, TM means Ez polarization
        src_comp = mp.Hz if pol.lower() == 'te' else mp.Ez
        
        sources = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                             component=src_comp,
                             center=source_center,
                             size=source_size)]
        
        # 5. Simulation
        sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=sources,
            default_material=MatBg,
            resolution=cfg.resolution,
        )
        
        # 6. Monitors
        # Monitor component should match source
        monitor_pos_1 = mp.Vector3(output_x_center, output_y_offset, 0)
        monitor_pos_2 = mp.Vector3(output_x_center, -output_y_offset, 0)
        monitor_size = mp.Vector3(0, cfg.wg_width*2, 0)
        
        monitor1 = mpa.FourierFields(sim, mp.Volume(center=monitor_pos_1, size=monitor_size), src_comp)
        monitor2 = mpa.FourierFields(sim, mp.Volume(center=monitor_pos_2, size=monitor_size), src_comp)
        
        # 7. Objectives
        # TE -> Maximize Port 1 (Top)
        # TM -> Maximize Port 2 (Bottom)
        
        if pol.lower() == 'te':
            def J_te(field1, field2):
                return npa.sum(npa.abs(field1)**2) # Maximize Port 1
            obj_funcs = [J_te]
        else:
            def J_tm(field1, field2):
                return npa.sum(npa.abs(field2)**2) # Maximize Port 2
            obj_funcs = [J_tm]

        # Optimization Problem
        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=obj_funcs,
            objective_arguments=[monitor1, monitor2],
            design_regions=[design_region],
            frequencies=[fcen],
            decay_by=1e-3
        )
        
        return opt, design_grid

    def create_adjoint_problems(self):
        shape = (self.config.n_cells_x, self.config.n_cells_y)
        opt_te, grid_te = self.create_single_pol_problem('te', shape)
        opt_tm, grid_tm = self.create_single_pol_problem('tm', shape)
        return opt_te, opt_tm, grid_te, grid_tm
