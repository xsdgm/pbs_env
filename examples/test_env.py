#!/usr/bin/env python
"""
环境测试脚本

测试MeepMMIPBS环境的基本功能。
"""

import sys
import numpy as np

# 添加父目录到路径
sys.path.insert(0, '/home/xsdgm')

def test_env_creation():
    """测试环境创建"""
    print("=" * 50)
    print("测试1: 环境创建")
    print("=" * 50)
    
    import pbs_env
    
    # 使用make函数创建
    env = pbs_env.make("MeepMMIPBS-v0")
    print(f"✓ 环境创建成功")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    
    env.close()
    return True


def test_env_reset():
    """测试环境重置"""
    print("\n" + "=" * 50)
    print("测试2: 环境重置")
    print("=" * 50)
    
    import pbs_env
    
    env = pbs_env.make("MeepMMIPBS-v0")
    obs, info = env.reset(seed=42)
    
    print(f"✓ 环境重置成功")
    print(f"  观察形状: {obs.shape}")
    print(f"  信息: {info}")
    
    env.close()
    return True


def test_env_step():
    """测试环境步进"""
    print("\n" + "=" * 50)
    print("测试3: 环境步进")
    print("=" * 50)
    
    import pbs_env
    
    env = pbs_env.make("MeepMMIPBS-v0")
    obs, info = env.reset(seed=42)
    
    # 执行几步
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  步骤 {i+1}: action={action}, reward={reward:.4f}")
    
    print(f"✓ 步进测试完成")
    print(f"  最终信息: {info}")
    
    env.close()
    return True


def test_random_episode():
    """测试随机回合"""
    print("\n" + "=" * 50)
    print("测试4: 随机回合 (20步)")
    print("=" * 50)
    
    import pbs_env
    
    env = pbs_env.make("MeepMMIPBS-v0")
    obs, info = env.reset(seed=42)
    
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (step + 1) % 5 == 0:
            print(f"  步骤 {step+1}: TE→P1={info['te_port1']:.3f}, "
                  f"TM→P2={info['tm_port2']:.3f}, "
                  f"串扰={info['crosstalk']:.3f}")
    
    print(f"✓ 回合完成")
    print(f"  总奖励: {total_reward:.4f}")
    print(f"  最佳奖励: {info['best_reward']:.4f}")
    
    env.close()
    return True


def test_visualization():
    """测试可视化"""
    print("\n" + "=" * 50)
    print("测试5: 可视化")
    print("=" * 50)
    
    import pbs_env
    from pbs_env.utils import visualize_structure, visualize_results
    
    env = pbs_env.make("MeepMMIPBS-v0")
    obs, info = env.reset(seed=42)
    
    # 保存结构可视化
    visualize_structure(
        obs,
        title="Initial MMI Structure",
        save_path="/home/xsdgm/pbs_env/examples/initial_structure.png",
        show=False
    )
    print("✓ 结构可视化已保存")
    
    # 执行几步后保存结果
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    
    results = {
        "te_port1": info["te_port1"],
        "te_port2": info["te_port2"],
        "tm_port1": info["tm_port1"],
        "tm_port2": info["tm_port2"],
        "total_efficiency": info["total_efficiency"],
        "crosstalk": info["crosstalk"]
    }
    
    visualize_results(
        results,
        structure=obs,
        save_path="/home/xsdgm/pbs_env/examples/results.png",
        show=False
    )
    print("✓ 结果可视化已保存")
    
    env.close()
    return True


def main():
    """运行所有测试"""
    print("\n" + "#" * 60)
    print("# MeepMMIPBS 环境测试")
    print("#" * 60)
    
    tests = [
        ("环境创建", test_env_creation),
        ("环境重置", test_env_reset),
        ("环境步进", test_env_step),
        ("随机回合", test_random_episode),
        ("可视化", test_visualization),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ 测试失败: {e}")
    
    # 打印摘要
    print("\n" + "#" * 60)
    print("# 测试摘要")
    print("#" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ 通过" if success else f"✗ 失败: {error}"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
