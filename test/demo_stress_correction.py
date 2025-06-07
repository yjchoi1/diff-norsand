#!/usr/bin/env python3
"""
Demo script showcasing the differentiable stress correction implementation.

This script demonstrates:
1. Basic usage of the differentiable stress correction function
2. Comparison with the numpy version
3. Automatic differentiation capabilities
4. Gradient computation for optimization tasks
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diff_norsand_stress_correction import stressCorrection as stressCorrection_diff
from norsand_functions import stressCorrection as stressCorrection_np

def main():
    print("=" * 80)
    print("DIFFERENTIABLE STRESS CORRECTION DEMO")
    print("=" * 80)
    
    # Define material parameters (typical sand parameters)
    params_dict = {
        'Lambda': 0.06,     # Compression index slope 
        'M_tc': 1.4,        # Critical state stress ratio in triaxial compression
        'N': 0.2,           # Material constant
        'H0': 0.5,          # Hardening parameter
        'Hy': 0.1,          # Hardening parameter
        'index_5': 0.0,     # Unused (index 5)
        'nu': 0.3,          # Poisson's ratio
        'chi_i': 0.3,       # Dilatancy parameter
        'index_8': 0.0,     # Unused (index 8)
        'Gamma': 2.0,       # Material constant for critical state line
        'e': 0.7,           # Current void ratio
        'G_max': 100.0,     # Maximum shear modulus
        'p_ref': 100.0,     # Reference pressure
        'm': 0.5            # Pressure exponent
    }
    
    # Convert to parameter arrays
    params_np = np.array([
        params_dict['Lambda'], params_dict['M_tc'], params_dict['N'], 
        params_dict['H0'], params_dict['Hy'], params_dict['index_5'],
        params_dict['nu'], params_dict['chi_i'], params_dict['index_8'],
        params_dict['Gamma'], params_dict['e'], params_dict['G_max'],
        params_dict['p_ref'], params_dict['m']
    ])
    
    # Test stress state (triaxial compression)
    sigma_np = np.array([150.0, 100.0, 100.0, 0.0, 0.0, 0.0])
    p_i_initial = 120.0
    F_initial = -0.948  # Initial yield function value
    
    # Tolerances
    FTOL = 1e-6
    MAXITS = 10
    
    print("\n1. BASIC FUNCTIONALITY COMPARISON")
    print("-" * 50)
    print(f"Initial stress state: {sigma_np}")
    print(f"Initial p_i: {p_i_initial}")
    print(f"Initial yield function value: {F_initial}")
    
    # Test numpy version
    print("\nNumPy version:")
    sigma_corr_np, p_i_corr_np = stressCorrection_np(
        params_np, F_initial, sigma_np, p_i_initial, FTOL, MAXITS
    )
    print(f"  Corrected stress: {sigma_corr_np}")
    print(f"  Corrected p_i: {p_i_corr_np:.8f}")
    
    # Test torch version
    print("\nTorch version:")
    params_torch = torch.tensor(params_np, dtype=torch.float32)
    sigma_torch = torch.tensor(sigma_np, dtype=torch.float32)
    p_i_torch = torch.tensor(p_i_initial, dtype=torch.float32)
    F_torch = torch.tensor(F_initial, dtype=torch.float32)
    
    sigma_corr_torch, p_i_corr_torch = stressCorrection_diff(
        params_torch, F_torch, sigma_torch, p_i_torch, FTOL, MAXITS
    )
    print(f"  Corrected stress: {sigma_corr_torch.detach().numpy()}")
    print(f"  Corrected p_i: {p_i_corr_torch.detach().numpy():.8f}")
    
    # Compare results
    stress_diff = np.abs(sigma_corr_torch.detach().numpy() - sigma_corr_np)
    p_i_diff = abs(p_i_corr_torch.detach().numpy() - p_i_corr_np)
    print(f"\nDifferences:")
    print(f"  Max stress difference: {np.max(stress_diff):.8f}")
    print(f"  p_i difference: {p_i_diff:.8f}")
    
    print("\n2. DIFFERENTIABILITY DEMONSTRATION")
    print("-" * 50)
    
    # Create tensors that require gradients
    params_grad = torch.tensor(params_np, dtype=torch.float32, requires_grad=True)
    sigma_grad = torch.tensor(sigma_np, dtype=torch.float32, requires_grad=True)
    p_i_grad = torch.tensor(p_i_initial, dtype=torch.float32, requires_grad=True)
    F_grad = torch.tensor(F_initial, dtype=torch.float32, requires_grad=True)
    
    # Run stress correction
    sigma_result, p_i_result = stressCorrection_diff(
        params_grad, F_grad, sigma_grad, p_i_grad, FTOL, MAXITS
    )
    
    # Create a scalar objective (e.g., sum of corrected stress components)
    objective = torch.sum(sigma_result) + p_i_result
    
    # Compute gradients
    objective.backward()
    
    print("Gradient computation successful!")
    print(f"Objective value: {objective.detach().numpy():.6f}")
    print("Gradient norms:")
    print(f"  d(objective)/d(params): {torch.norm(params_grad.grad).detach().numpy():.6f}")
    print(f"  d(objective)/d(sigma): {torch.norm(sigma_grad.grad).detach().numpy():.6f}")
    print(f"  d(objective)/d(p_i): {abs(p_i_grad.grad).detach().numpy():.6f}")
    print(f"  d(objective)/d(F): {abs(F_grad.grad).detach().numpy():.6f}")
    
    print("\n3. PARAMETER SENSITIVITY ANALYSIS")
    print("-" * 50)
    
    # Analyze sensitivity to specific parameters
    sensitive_params = ['Lambda', 'M_tc', 'N', 'chi_i']
    param_indices = [0, 1, 2, 7]
    
    print("Parameter sensitivities (gradient magnitudes):")
    for name, idx in zip(sensitive_params, param_indices):
        grad_val = abs(params_grad.grad[idx].detach().numpy())
        print(f"  {name:8s}: {grad_val:.6f}")
    
    print("\n4. OPTIMIZATION EXAMPLE")
    print("-" * 50)
    
    # Simple optimization example: minimize the correction magnitude
    def stress_correction_objective(params_opt):
        """Objective function for optimization"""
        sigma_result, p_i_result = stressCorrection_diff(
            params_opt, F_grad, sigma_grad, p_i_grad, FTOL, MAXITS
        )
        # Minimize the magnitude of correction
        correction_magnitude = torch.sum((sigma_result - sigma_grad)**2) + (p_i_result - p_i_grad)**2
        return correction_magnitude
    
    # Initialize parameters for optimization
    params_opt = torch.tensor(params_np, dtype=torch.float32, requires_grad=True)
    
    print("Initial correction magnitude:")
    initial_obj = stress_correction_objective(params_opt)
    print(f"  {initial_obj.detach().numpy():.6f}")
    
    # Simple gradient step
    initial_obj.backward()
    with torch.no_grad():
        # Small step in negative gradient direction for key parameters
        learning_rate = 0.001
        params_opt[0] -= learning_rate * params_opt.grad[0]  # Lambda
        params_opt[1] -= learning_rate * params_opt.grad[1]  # M_tc
    
    # Clear gradients and re-evaluate
    params_opt.grad.zero_()
    final_obj = stress_correction_objective(params_opt)
    final_obj.backward()
    
    print("After one optimization step:")
    print(f"  Correction magnitude: {final_obj.detach().numpy():.6f}")
    print(f"  Change: {(final_obj - initial_obj).detach().numpy():.6f}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nThe differentiable stress correction implementation provides:")
    print("• Accurate results matching the NumPy version")
    print("• Full differentiability for gradient-based optimization")
    print("• Parameter sensitivity analysis capabilities")
    print("• Integration with PyTorch's automatic differentiation")

if __name__ == "__main__":
    main() 