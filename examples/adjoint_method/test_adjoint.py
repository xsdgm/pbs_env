
"""
Adjoint Method Optimization Test

This script demonstrates how to use the adjoint method interface to calculate gradients
and optimize the MMI structure for the PBS task.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path (parent of pbs_env)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from pbs_env.meep_simulator import MMISimulator, SimulationConfig
from pbs_env.utils import visualize_structure

def optimize_structure(steps=5, step_size=0.5):
    """
    Simple Gradient Ascent Optimization
    """
    print("Setting up simulator...")
    config = SimulationConfig(
        resolution=15,       # Slightly higher resolution for gradient
        n_cells_x=30,
        n_cells_y=10,
        run_time=200.0        # Ensure steady state (Length ~20um * n_g ~4 = ~80 time units required)
    )
    simulator = MMISimulator(config, num_workers=1)
    
    # Initialize random structure (0.5 grey scale for continuous optimization)
    # Adjoint method works best with continuous variables (density)
    # We start with 0.5 (Sin) everywhere plus some noise
    structure = np.ones((config.n_cells_x, config.n_cells_y)) * 0.5
    structure += np.random.normal(0, 0.05, structure.shape)
    structure = np.clip(structure, 0, 1)
    
    # Optimization loop
    for i in range(steps):
        print(f"\n--- Step {i+1}/{steps} ---")
        
        # 1. Compute Gradients
        # We want to maximize TE->Port1 and TM->Port2
        # Sensitivity to epsilon: grad > 0 means increasing epsilon (adding Si) increases objective
        print("Computing gradients...")
        grads = simulator.compute_gradients(
            structure, 
            target_weights={"te_port1": 1.0, "tm_port2": 1.0}
        )
        
        # Normalize grads for update
        grad_norm = np.max(np.abs(grads))
        if grad_norm > 0:
            grads_normalized = grads / grad_norm
        else:
            grads_normalized = grads
            
        print(f"Max Gradient: {grad_norm:.4e}")
        
        # 2. Update Structure (Gradient Ascent)
        # Update: p_new = p_old + step_size * grad
        # Note: Gradient is w.r.t epsilon.
        # Since epsilon is monotonic with p (density), direction is same.
        structure = structure + step_size * grads_normalized
        
        # Project to [0, 1]
        structure = np.clip(structure, 0, 1)
        
        # 3. Evaluate Performance
        # Need to binarize for 'real' performance check if simulator assumes binary, 
        # but MMISimulator can interpret continuous values as effective medium (via subpixel averaging in Meep)
        # Actually our current _build_geometry uses `if structure[i, j] > 0.5` thresholding!
        # This is a problem for gradient optimization.
        # For this test, we accept that we are optimizing the 'latent' variable, and we evaluate the thresholded one.
        
        binary_structure = (structure > 0.5).astype(float)
        result = simulator.simulate(binary_structure)
        
        print(f"Result (Thresholded):")
        print(f"  TE->P1: {result.te_port1:.4f}")
        print(f"  TM->P2: {result.tm_port2:.4f}")
        print(f"  Total:  {result.total_efficiency:.4f}")
        
        # Visualize
        if i == 0 or i == steps - 1:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].imshow(structure.T, origin='lower', cmap='RdBu', vmin=0, vmax=1)
            ax[0].set_title(f"Structure (Step {i})")
            
            # Visualize Gradient
            # Normalize for visualization
            vmax = np.max(np.abs(grads))
            ax[1].imshow(grads.T, origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
            ax[1].set_title("Gradient")
            
            plt.tight_layout()
            plt.savefig(f"examples/adjoint_method/step_{i}.png")
            plt.close()
            print(f"Saved visualization to examples/adjoint_method/step_{i}.png")

if __name__ == "__main__":
    optimize_structure()
