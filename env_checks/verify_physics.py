
import numpy as np
import sys
import os

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())

from meep_simulator import MMISimulator, SimulationConfig

def test_eim_calculation():
    print("=== Testing Effective Index Method ===")
    
    sim = MMISimulator(num_workers=1)
    
    # Common parameters for SOI @ 1550nm
    n_si = 3.48
    n_sio2 = 1.44
    thickness = 0.22 # um
    wavelength = 1.55 # um

    print(f"Parameters: Si/SiO2 ({n_si}/{n_sio2}), Thickness={thickness}um, Wavelength={wavelength}um")
    
    # Calculate TE neff
    neff_te = sim.get_effective_index(n_si, n_sio2, thickness, "te", wavelength)
    print(f"TE neff: {neff_te:.4f}")
    
    # Calculate TM neff
    neff_tm = sim.get_effective_index(n_si, n_sio2, thickness, "tm", wavelength)
    print(f"TM neff: {neff_tm:.4f}")
    
    # Checks
    if not (n_sio2 < neff_te < n_si):
        print("FAIL: TE neff out of bounds")
        return False
        
    if not (n_sio2 < neff_tm < n_si):
        print("FAIL: TM neff out of bounds")
        return False
        
    if neff_te <= neff_tm:
        print("FAIL: Expected TE neff > TM neff for slab waveguide")
        return False
        
    print("PASS: EIM calculations look reasonable")
    return True

def test_simulation_setup():
    print("\n=== Testing Simulation Setup (Dry Run) ===")
    
    # Configure for mock or real run
    config = SimulationConfig(
        use_eim=True,
        slab_thickness=0.22,
        run_time=10 # Short run
    )
    
    sim = MMISimulator(config, num_workers=1)
    
    # Create dummy structure
    structure = np.zeros((config.n_cells_x, config.n_cells_y))
    
    if not sim._meep_available:
        print("WARNING: MEEP not installed, skipping real simulation test")
        return True
        
    try:
        print("Running TE simulation...")
        res_te = sim.simulate(structure, polarization="te")
        print(f"TE Results: {res_te}")
        
        print("Running TM simulation...")
        res_tm = sim.simulate(structure, polarization="tm")
        print(f"TM Results: {res_tm}")
        
        print("PASS: Simulation runs without errors")
        return True
    except Exception as e:
        print(f"FAIL: Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_eim_calculation()
    test_simulation_setup()
