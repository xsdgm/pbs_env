
import sys
import os
import platform
import subprocess

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def check_environment():
    print_section("环境检查 (Environment Check)")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check conda env
    try:
        # This might not work if conda is not in path or shell, but we can try to infer
        env_path = os.environ.get('CONDA_PREFIX', 'Not set')
        print(f"Conda Prefix: {env_path}")
        if 'mmi-rl' in env_path:
            print("Status: \033[92m√ Correct Conda Environment (mmi-rl)\033[0m")
        else:
            print(f"Status: \033[93m! Warning: Current environment seems to be '{os.path.basename(env_path)}', expected 'mmi-rl'\033[0m")
    except Exception as e:
        print(f"Error checking conda env: {e}")

    # Check MEEP
    print_section("MEEP 仿真器检查")
    try:
        import meep as mp
        print(f"MEEP Version: {mp.__version__}")
        print("Status: \033[92m√ MEEP installed and importable\033[0m")
    except ImportError:
        print("Status: \033[91m× MEEP not found. Simulation will run in Mock mode.\033[0m")
    except Exception as e:
        print(f"Status: \033[91m× MEEP import error: {e}\033[0m")

def test_simulation():
    print_section("仿真流程测试 (Simulation Test)")
    
    # Setup path to import pbs_env
    # Current file: .../pbs_env/env_checks/verify_env.py
    # We want to add .../ (parent of pbs_env) to sys.path to import pbs_env
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # pbs_env
    user_root = os.path.dirname(project_root)   # home/xsdgm
    
    if user_root not in sys.path:
        sys.path.append(user_root)
        print(f"Added {user_root} to sys.path")
    
    try:
        from pbs_env.mmi_pbs_env import MeepMMIPBS
        print("Successfully imported MeepMMIPBS")
    except ImportError as e:
        print(f"\033[91mImport Error: {e}\033[0m")
        print("Attempting to import without package prefix (if local)...")
        if project_root not in sys.path:
             sys.path.append(project_root)
        try:
             # Try hacky import if package structure fails
             import mmi_pbs_env
             MeepMMIPBS = mmi_pbs_env.MeepMMIPBS
             print("Imported using local path fallback.")
        except ImportError as e2:
             print(f"\033[91mFatal Import Error: {e2}\033[0m")
             return

    # Initialize Environment
    try:
        print("Initializing Environment...")
        # Use low parameters for quick check
        env = MeepMMIPBS(
            n_cells_x=10, 
            n_cells_y=4, 
            resolution=10, 
            run_time=10.0, 
            num_workers=1,
            init_mode="random"
        )
        print("Environment initialized.")
        
        # Reset
        print("Running env.reset()...")
        obs, info = env.reset()
        print(f"Reset complete. Observation shape: {obs.shape}")
        
        # Step
        print("Running env.step()...")
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step complete.")
        print(f"Reward: {reward}")
        print(f"Info keys: {list(info.keys())}")
        
        env.close()
        print("\n\033[92m√ Simulation Test Passed!\033[0m")
        
    except Exception as e:
        print(f"\n\033[91m× Simulation Test Failed: {e}\033[0m")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_environment()
    test_simulation()
