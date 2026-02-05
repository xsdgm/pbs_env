# MEEP MMI PBS 强化学习环境

基于 MEEP FDTD 仿真的 1×2 MMI（多模干涉）偏振分束器（PBS）逆向设计强化学习环境。

## 设计目标

- **TE 偏振** → 输出端口 1
- **TM 偏振** → 输出端口 2

```
                    ┌──────────────────┐
                    │                  ├──► 输出1 (TE)
    输入 ──────────►│   MMI 区域       │
    (TE+TM)         │  (可优化结构)     ├──► 输出2 (TM)
                    └──────────────────┘
```

## 安装

### 前置依赖

确保已安装以下包（在 `mmi-rl` conda 环境中）：

```bash
conda activate mmi-rl
# 应该已经有：pymeep, gymnasium, stable-baselines3, torch
```

### 安装环境

```bash
cd /home/xsdgm/pbs_env
pip install -e .
```

或者直接将父目录添加到 Python 路径：

```python
import sys
sys.path.insert(0, '/home/xsdgm')
import pbs_env
```

## 环境检查

安装完成后，建议运行以下脚本验证环境配置：

```bash
# 检查 Python 环境及依赖
python env_checks/verify_env.py

# 验证物理仿真计算 (有效折射率法)
python env_checks/verify_physics.py
```

## 快速开始

### 基本使用

```python
import pbs_env

# 创建环境
env = pbs_env.make("MeepMMIPBS-v0")

# 重置环境
obs, info = env.reset(seed=42)

# 执行随机动作
for step in range(100):
    action = env.action_space.sample()  # 随机选择一个像素翻转
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step}: TE→P1={info['te_port1']:.3f}, "
          f"TM→P2={info['tm_port2']:.3f}, reward={reward:.4f}")

env.close()
```

### 使用 PPO 训练

```bash
cd /home/xsdgm/pbs_env/examples
python train_ppo.py --timesteps 50000
```

### 评估模型

```bash
python train_ppo.py --mode eval --model path/to/model.zip --episodes 10
```

## 环境规格

### 观察空间

- **类型**: `Box(0, 1, shape=(n_cells_x, n_cells_y))`
- **含义**: MMI 区域的材料分布（0=SiO₂, 1=Si）

### 动作空间

- **类型**: `Discrete(n_cells_x * n_cells_y)`
- **含义**: 翻转指定位置的材料

### 奖励函数

```
reward = α × TE_port1 + β × TM_port2 - γ × crosstalk
```

其中：
- `TE_port1`: TE 模式在端口 1 的传输效率（目标）
- `TM_port2`: TM 模式在端口 2 的传输效率（目标）
- `crosstalk`: 串扰 = TE_port2 + TM_port1

默认权重：`α=1.0, β=1.0, γ=0.5`

### 性能指标

环境 `info` 字典包含：

| 键 | 含义 |
|---|---|
| `te_port1` | TE 在端口 1 的效率 |
| `te_port2` | TE 在端口 2 的效率（串扰） |
| `tm_port1` | TM 在端口 1 的效率（串扰） |
| `tm_port2` | TM 在端口 2 的效率 |
| `crosstalk` | 总串扰 |
| `te_extinction_ratio_dB` | TE 消光比 (dB) |
| `tm_extinction_ratio_dB` | TM 消光比 (dB) |
| `best_reward` | 回合中的最佳奖励 |

## 配置

### 环境参数

```python
env = pbs_env.make(
    "MeepMMIPBS-v0",
    n_cells_x=30,           # X方向网格数
    n_cells_y=8,            # Y方向网格数
    wavelength=1.55,        # 工作波长 (μm)
    mmi_width=4.0,          # MMI宽度 (μm)
    mmi_length=15.0,        # MMI长度 (μm)
    resolution=10,          # 仿真分辨率 (速度优先)
    run_time=50,            # 仿真时间
    num_workers=12,         # 并行工作进程数
    reward_alpha=1.0,       # TE效率权重
    reward_beta=1.0,        # TM效率权重
    reward_gamma=0.5,       # 串扰惩罚权重
    init_mode="random",     # 初始化: random/ones/zeros/half
)
```

### 快速版本

使用低分辨率快速测试：

```python
env = pbs_env.make("MeepMMIPBS-Fast-v0")
```

## 基准算法 (Baselines)

本项目提供了两种传统的逆向设计方法作为对比基准：

### 1. 伴随法 (Adjoint Method)

利用梯度信息进行高效的局部优化。

```bash
python baseline/adjoint_design.py
```

### 2. 遗传算法 (Genetic Algorithm)

全局优化算法，适合探索更大的搜索空间。

```bash
python baseline/genetic_algorithm_design.py
```

## 强化学习 + 伴随法 (Hybrid)

先用强化学习进行全局探索，再用伴随法进行局部精细优化：

```bash
conda activate mmi-rl
python baseline/rl_adjoint_design.py --timesteps 20000 --adjoint-iters 120
```

可选参数示例：

```bash
python baseline/rl_adjoint_design.py \
    --config configs/default_config.yaml \
    --conda-env mmi-rl \
    --timesteps 50000 \
    --adjoint-iters 200 \
    --adjoint-lr 0.01
```

## 文件结构

```
pbs_env/
├── __init__.py          # 包初始化
├── core.py              # 核心逻辑
├── mmi_pbs_env.py       # Gym 环境实现
├── meep_simulator.py    # MEEP 仿真接口
├── utils.py             # 通用工具
├── configs/
│   └── default_config.yaml
├── baseline/            # 基准算法
│   ├── adjoint_design.py
│   └── genetic_algorithm_design.py
├── env_checks/          # 环境验证脚本
│   ├── verify_env.py
│   └── verify_physics.py
├── examples/            # 使用示例
│   ├── test_env.py
│   └── train_ppo.py
├── results/             # 仿真结果
└── README.md
```

## 测试

```bash
cd /home/xsdgm/pbs_env/examples
python test_env.py
```

## 注意事项

1. **仿真速度**: FDTD 仿真较慢，每步可能需要几秒。建议使用低分辨率测试。
2. **并行计算**: 支持 12 核并行，可根据 CPU 调整 `num_workers`。
3. **MEEP 不可用**: 当 MEEP 未安装时，环境会使用模拟仿真进行测试。

## License

MIT
