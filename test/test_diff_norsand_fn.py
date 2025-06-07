import torch
import pytest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diff_norsand_functions import findM, findM_i, findM_itc, findchi_i, findpsipsii, findp_imax, findF, finddFdsigma, findp_ipsi_iM_i, findC_p, finddFdepsilon_p, find_dlambda_p
from diff_utils import stress_decomp, find_sJ2J3, lode_angle, vol_dev
from norsand_functions import findp_ipsi_iM_i as findp_ipsi_iM_i_np
from norsand_functions import find_dlambda_p as find_dlambda_p_np

def test_findM_differentiable():
    """Test that findM is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    theta = torch.tensor(0.1, requires_grad=True)
    M_tc = torch.tensor(1.2, requires_grad=True)
    
    # Call the function
    M = findM(theta, M_tc)
    
    # Verify output requires gradients
    assert M.requires_grad, "Output M should require gradients"
    
    # Backpropagate
    M.backward()
    
    # Verify gradients exist for both inputs
    assert theta.grad is not None, "Gradients were not computed for theta"
    assert M_tc.grad is not None, "Gradients were not computed for M_tc"
    
    # Verify gradients are non-zero (for this specific input)
    assert theta.grad != 0, "Gradient with respect to theta should be non-zero"
    assert M_tc.grad != 0, "Gradient with respect to M_tc should be non-zero"
    
    # Test multiple cases including edge cases
    test_cases = [
        (0.0, 1.2),    # Zero Lode angle
        (0.52, 1.2),   # Approximately π/6 (maximum Lode angle)
        (-0.52, 1.2),  # Approximately -π/6 (minimum Lode angle)
        (0.1, 0.5),    # Small M_tc
        (0.1, 2.5)     # Large M_tc
    ]
    
    for i, (theta_val, M_tc_val) in enumerate(test_cases):
        # Create input tensors with requires_grad=True
        theta_i = torch.tensor(theta_val, requires_grad=True)
        M_tc_i = torch.tensor(M_tc_val, requires_grad=True)
        
        # Call the function
        M_i = findM(theta_i, M_tc_i)
        
        # Verify output requires gradients
        assert M_i.requires_grad, f"Output M should require gradients for case {i}"
        
        # Create a scalar that depends on the output for backpropagation
        result = M_i * M_i
        
        # Backpropagate
        result.backward()
        
        # Verify gradients exist and are finite
        assert theta_i.grad is not None, f"Gradients were not computed for theta in case {i}"
        assert M_tc_i.grad is not None, f"Gradients were not computed for M_tc in case {i}"
        assert torch.isfinite(theta_i.grad), f"Gradient should be finite for theta in case {i}"
        assert torch.isfinite(M_tc_i.grad), f"Gradient should be finite for M_tc in case {i}"
        
        # Print gradients for inspection
        print(f"Case {i}: theta={theta_val}, M_tc={M_tc_val}")
        print(f"  dM/dtheta = {theta_i.grad.item()}")
        print(f"  dM/dM_tc = {M_tc_i.grad.item()}")

def test_findM_i_differentiable():
    """Test that findM_i is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    M = torch.tensor(1.2, requires_grad=True)
    M_tc = torch.tensor(1.4, requires_grad=True)
    chi_i = torch.tensor(0.3, requires_grad=True)
    psi_i = torch.tensor(-0.05, requires_grad=True)
    N = torch.tensor(0.2, requires_grad=True)
    
    # Call the function
    M_i = findM_i(M, M_tc, chi_i, psi_i, N)
    
    # Verify output requires gradients
    assert M_i.requires_grad, "Output M_i should require gradients"
    
    # Backpropagate
    M_i.backward()
    
    # Verify gradients exist for all inputs
    assert M.grad is not None, "Gradients were not computed for M"
    assert M_tc.grad is not None, "Gradients were not computed for M_tc"
    assert chi_i.grad is not None, "Gradients were not computed for chi_i"
    assert psi_i.grad is not None, "Gradients were not computed for psi_i"
    assert N.grad is not None, "Gradients were not computed for N"
    
    # Verify gradients are non-zero where expected
    assert M.grad != 0, "Gradient with respect to M should be non-zero"
    assert M_tc.grad != 0, "Gradient with respect to M_tc should be non-zero"
    assert chi_i.grad != 0, "Gradient with respect to chi_i should be non-zero"
    assert psi_i.grad != 0, "Gradient with respect to psi_i should be non-zero"
    assert N.grad != 0, "Gradient with respect to N should be non-zero"
    
    # Test with positive psi_i
    psi_i_pos = torch.tensor(0.05, requires_grad=True)
    M_i_pos = findM_i(M, M_tc, chi_i, psi_i_pos, N)
    M_i_pos.backward()
    assert psi_i_pos.grad is not None, "Gradients were not computed for positive psi_i"
    assert torch.isfinite(psi_i_pos.grad), "Gradient should be finite for positive psi_i"
    
    # Test with zero psi_i (edge case for abs function)
    psi_i_zero = torch.tensor(0.0, requires_grad=True)
    M_i_zero = findM_i(M, M_tc, chi_i, psi_i_zero, N)
    M_i_zero.backward()
    assert psi_i_zero.grad is not None, "Gradients were not computed for zero psi_i"
    assert torch.isfinite(psi_i_zero.grad), "Gradient should be finite for zero psi_i"

def test_findM_itc_differentiable():
    """Test that findM_itc is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    N = torch.tensor(0.2, requires_grad=True)
    chi_i = torch.tensor(0.3, requires_grad=True)
    psi_i = torch.tensor(-0.05, requires_grad=True)
    M_tc = torch.tensor(1.4, requires_grad=True)
    
    # Call the function
    M_itc = findM_itc(N, chi_i, psi_i, M_tc)
    
    # Verify output requires gradients
    assert M_itc.requires_grad, "Output M_itc should require gradients"
    
    # Backpropagate
    M_itc.backward()
    
    # Verify gradients exist for all inputs
    assert N.grad is not None, "Gradients were not computed for N"
    assert chi_i.grad is not None, "Gradients were not computed for chi_i"
    assert psi_i.grad is not None, "Gradients were not computed for psi_i"
    assert M_tc.grad is not None, "Gradients were not computed for M_tc"
    
    # Verify gradients are non-zero (for this specific input)
    assert N.grad != 0, "Gradient with respect to N should be non-zero"
    assert chi_i.grad != 0, "Gradient with respect to chi_i should be non-zero"
    assert psi_i.grad != 0, "Gradient with respect to psi_i should be non-zero"
    assert M_tc.grad != 0, "Gradient with respect to M_tc should be non-zero"
    
    # Test with positive psi_i
    psi_i_pos = torch.tensor(0.05, requires_grad=True)
    M_itc_pos = findM_itc(N, chi_i, psi_i_pos, M_tc)
    M_itc_pos.backward()
    assert psi_i_pos.grad is not None, "Gradients were not computed for positive psi_i"
    assert torch.isfinite(psi_i_pos.grad), "Gradient should be finite for positive psi_i"

def test_findchi_i_differentiable():
    """Test that findchi_i is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    M_tc = torch.tensor(1.4, requires_grad=True)
    Chi_tc = torch.tensor(0.8, requires_grad=True)
    Lambda = torch.tensor(0.06, requires_grad=True)
    
    # Call the function
    chi_i_result = findchi_i(M_tc, Chi_tc, Lambda)
    
    # Verify output requires gradients
    assert chi_i_result.requires_grad, "Output chi_i should require gradients"
    
    # Backpropagate
    chi_i_result.backward()
    
    # Verify gradients exist for all inputs
    assert M_tc.grad is not None, "Gradients were not computed for M_tc"
    assert Chi_tc.grad is not None, "Gradients were not computed for Chi_tc" 
    assert Lambda.grad is not None, "Gradients were not computed for Lambda"
    
    # Verify gradients are non-zero (for this specific input)
    assert M_tc.grad != 0, "Gradient with respect to M_tc should be non-zero"
    assert Chi_tc.grad != 0, "Gradient with respect to Chi_tc should be non-zero"
    assert Lambda.grad != 0, "Gradient with respect to Lambda should be non-zero"

def test_findpsipsii_psi_differentiable():
    """Test that the psi output of findpsipsii is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    Gamma = torch.tensor(2.0, requires_grad=True)
    Lambda = torch.tensor(0.06, requires_grad=True)
    p = torch.tensor(100.0, requires_grad=True)
    p_i = torch.tensor(110.0, requires_grad=True)
    e = torch.tensor(0.7, requires_grad=True)
    
    # Call the function to get the psi output
    psi, _ = findpsipsii(Gamma, Lambda, p, p_i, e)
    
    # Verify output requires gradients
    assert psi.requires_grad, "Output psi should require gradients"
    
    # Backpropagate
    psi.backward()
    
    # Verify gradients exist for the inputs that affect psi
    assert Gamma.grad is not None, "Gradients were not computed for Gamma"
    assert Lambda.grad is not None, "Gradients were not computed for Lambda"
    assert p.grad is not None, "Gradients were not computed for p"
    assert e.grad is not None, "Gradients were not computed for e"
    
    # Verify gradients are non-zero (for these specific inputs)
    assert Gamma.grad != 0, "Gradient with respect to Gamma should be non-zero"
    assert Lambda.grad != 0, "Gradient with respect to Lambda should be non-zero"
    assert p.grad != 0, "Gradient with respect to p should be non-zero"
    assert e.grad != 0, "Gradient with respect to e should be non-zero"
    
    # p_i should not affect psi, so its gradient should be None
    assert p_i.grad is None, "p_i should not affect psi, gradient should be None"

def test_findpsipsii_psi_i_differentiable():
    """Test that the psi_i output of findpsipsii is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    Gamma = torch.tensor(2.0, requires_grad=True)
    Lambda = torch.tensor(0.06, requires_grad=True)
    p = torch.tensor(100.0, requires_grad=True)
    p_i = torch.tensor(110.0, requires_grad=True)
    e = torch.tensor(0.7, requires_grad=True)
    
    # Call the function to get the psi_i output
    _, psi_i = findpsipsii(Gamma, Lambda, p, p_i, e)
    
    # Verify output requires gradients
    assert psi_i.requires_grad, "Output psi_i should require gradients"
    
    # Backpropagate
    psi_i.backward()
    
    # Verify gradients exist for inputs that affect psi_i
    assert Gamma.grad is not None, "Gradients were not computed for Gamma"
    assert Lambda.grad is not None, "Gradients were not computed for Lambda"
    assert p_i.grad is not None, "Gradients were not computed for p_i"
    assert e.grad is not None, "Gradients were not computed for e"
    
    # p.grad should be None since psi_i doesn't depend on p in the implemented formula:
    # psi_i = e - Gamma + Lambda * log(p_i)
    assert p.grad is None, "Gradient should not be computed for p since psi_i doesn't depend on it"
    
    # Verify gradients are non-zero for inputs that affect psi_i
    assert Gamma.grad != 0, "Gradient with respect to Gamma should be non-zero"
    assert Lambda.grad != 0, "Gradient with respect to Lambda should be non-zero"
    assert p_i.grad != 0, "Gradient with respect to p_i should be non-zero"
    assert e.grad != 0, "Gradient with respect to e should be non-zero"

def test_findp_imax_differentiable():
    """Test that findp_imax is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    chi_i = torch.tensor(0.3, requires_grad=True)
    psi_i = torch.tensor(-0.05, requires_grad=True)
    p = torch.tensor(100.0, requires_grad=True)
    M_itc = torch.tensor(1.3, requires_grad=True)
    
    # Call the function
    p_imax = findp_imax(chi_i, psi_i, p, M_itc)
    
    # Verify output requires gradients
    assert p_imax.requires_grad, "Output p_imax should require gradients"
    
    # Backpropagate
    p_imax.backward()
    
    # Verify gradients exist for all inputs
    assert chi_i.grad is not None, "Gradients were not computed for chi_i"
    assert psi_i.grad is not None, "Gradients were not computed for psi_i"
    assert p.grad is not None, "Gradients were not computed for p"
    assert M_itc.grad is not None, "Gradients were not computed for M_itc"
    
    # Verify gradients are non-zero where expected
    assert chi_i.grad != 0, "Gradient with respect to chi_i should be non-zero"
    assert psi_i.grad != 0, "Gradient with respect to psi_i should be non-zero"
    assert p.grad != 0, "Gradient with respect to p should be non-zero"
    assert M_itc.grad != 0, "Gradient with respect to M_itc should be non-zero"
    
    # Test with positive psi_i (for completeness)
    chi_i_pos = torch.tensor(0.3, requires_grad=True)
    psi_i_pos = torch.tensor(0.05, requires_grad=True)
    p_pos = torch.tensor(100.0, requires_grad=True)
    M_itc_pos = torch.tensor(1.3, requires_grad=True)
    
    p_imax_pos = findp_imax(chi_i_pos, psi_i_pos, p_pos, M_itc_pos)
    p_imax_pos.backward()
    
    assert psi_i_pos.grad is not None, "Gradients were not computed for positive psi_i"
    assert torch.isfinite(psi_i_pos.grad), "Gradient should be finite for positive psi_i"

def test_findF_differentiable():
    """Test that findF is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    p = torch.tensor(100.0, requires_grad=True)
    q = torch.tensor(50.0, requires_grad=True)
    M_i = torch.tensor(1.2, requires_grad=True)
    p_i = torch.tensor(110.0, requires_grad=True)
    
    # Call the function
    F = findF(p, q, M_i, p_i)
    
    # Verify output requires gradients
    assert F.requires_grad, "Output F should require gradients"
    
    # Backpropagate
    F.backward()
    
    # Verify gradients exist for all inputs
    assert p.grad is not None, "Gradients were not computed for p"
    assert q.grad is not None, "Gradients were not computed for q"
    assert M_i.grad is not None, "Gradients were not computed for M_i" 
    assert p_i.grad is not None, "Gradients were not computed for p_i"
    
    # Verify gradients are non-zero (for this specific input)
    assert p.grad != 0, "Gradient with respect to p should be non-zero"
    assert q.grad != 0, "Gradient with respect to q should be non-zero"
    assert M_i.grad != 0, "Gradient with respect to M_i should be non-zero"
    assert p_i.grad != 0, "Gradient with respect to p_i should be non-zero"
    
    # Test edge case: p = p_i (when log(p/p_i) = 0)
    p_equal = torch.tensor(110.0, requires_grad=True)
    q_equal = torch.tensor(50.0, requires_grad=True)
    M_i_equal = torch.tensor(1.2, requires_grad=True)
    p_i_equal = torch.tensor(110.0, requires_grad=True)
    
    F_equal = findF(p_equal, q_equal, M_i_equal, p_i_equal)
    F_equal.backward()
    
    assert p_equal.grad is not None, "Gradients were not computed for p in edge case"
    assert q_equal.grad is not None, "Gradients were not computed for q in edge case"
    assert M_i_equal.grad is not None, "Gradients were not computed for M_i in edge case"
    assert p_i_equal.grad is not None, "Gradients were not computed for p_i in edge case"
    assert torch.isfinite(p_equal.grad), "Gradients should be finite for edge case"

def test_finddFdsigma_differentiable():
    """Test that finddFdsigma is differentiable with respect to input stress tensor."""
    # Create a stress tensor with requires_grad=True
    sigma = torch.tensor([100.0, 50.0, 50.0, 10.0, 5.0, 2.0], requires_grad=True)
    
    # Create parameters for the yield function
    M_i = torch.tensor(0.8, requires_grad=True)
    p_i = torch.tensor(150.0, requires_grad=True)
    
    # Call the function
    dfdsigma = finddFdsigma(sigma, M_i, p_i)
    
    # Verify output requires gradients
    assert dfdsigma.requires_grad, "Output dfdsigma should require gradients"
    
    # Create a scalar output from dfdsigma (sum of all elements)
    result = torch.sum(dfdsigma)
    
    # Backpropagate
    result.backward()
    
    # Verify gradients exist for all input parameters
    assert sigma.grad is not None, "Gradients were not computed for sigma"
    assert M_i.grad is not None, "Gradients were not computed for M_i"
    assert p_i.grad is not None, "Gradients were not computed for p_i"
    
    # Verify gradients are non-zero where expected
    assert torch.any(sigma.grad != 0), "sigma should have non-zero gradient"
    assert M_i.grad != 0, "M_i should have non-zero gradient"
    assert p_i.grad != 0, "p_i should have non-zero gradient"

def test_compare_finddFdsigma_methods():
    """Test that compares the gradients from finddFdsigma and numerical differentiation."""
    # NOTE: improve this test that can check physical consistency
    # Define multiple test cases
    test_cases = [
        # Regular case with varied components
        torch.tensor([100.0, 50.0, 75.0, 25.0, 10.0, 5.0]),
        
        # Hydrostatic stress case (all normal components equal)
        torch.tensor([50.0, 50.0, 50.0, 0.0, 0.0, 0.0]),
        
        # Deviatoric stress case (large deviator)
        torch.tensor([100.0, 50.0, 0.0, 20.0, 15.0, 10.0]),
        
        # Uniaxial tension case
        torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        
        # Pure shear case
        torch.tensor([0.0, 0.0, 0.0, 30.0, 20.0, 10.0]),
        
        # Biaxial stress with shear
        torch.tensor([100.0, 100.0, 0.0, 50.0, 25.0, 10.0])
    ]
    
    case_names = [
        "Regular case",
        "Hydrostatic stress",
        "Deviatoric stress",
        "Uniaxial tension",
        "Pure shear",
        "Biaxial with shear"
    ]
    
    # Common parameters for yield function
    M_i = 0.8
    p_i = 150.0
    
    # Numerical differentiation function
    def compute_numerical_grad(sigma, M_i, p_i, eps=1e-2):
        numerical_grad = torch.zeros_like(sigma)
        
        for i in range(len(sigma)):
            # Forward perturbation
            sigma_plus = sigma.clone()
            sigma_plus[i] += eps
            
            # Get p, q from perturbed stress tensor
            p_plus, q_plus = stress_decomp(sigma_plus)
            
            # Compute yield function F_plus
            F_plus = findF(p_plus, q_plus, torch.tensor(M_i), torch.tensor(p_i))
            
            # Backward perturbation
            sigma_minus = sigma.clone()
            sigma_minus[i] -= eps
            
            # Get p, q from perturbed stress tensor
            p_minus, q_minus = stress_decomp(sigma_minus)
            
            # Compute yield function F_minus
            F_minus = findF(p_minus, q_minus, torch.tensor(M_i), torch.tensor(p_i))
            
            # Central difference
            numerical_grad[i] = (F_plus - F_minus) / (2 * eps)
        
        return numerical_grad
    
    # Process each test case
    for case_idx, sigma in enumerate(test_cases):
        print(f"\n\n{'='*80}")
        print(f"Test Case {case_idx+1}: {case_names[case_idx]}")
        print(f"Stress tensor: {sigma}")
        print(f"{'='*80}")
        
        # Convert to tensors for different methods
        sigma_torch = sigma.clone().detach().requires_grad_(True)
        
        # Get gradients using different methods
        dfdsig_auto = finddFdsigma(sigma_torch, torch.tensor(M_i), torch.tensor(p_i))
        dfdsig_numerical = compute_numerical_grad(sigma, M_i, p_i)
        
        # Print gradients
        print("\nGradients of F with respect to sigma:")
        print("-" * 80)
        print(f"{'Component':<10} {'Autodiff':<15} {'Numerical':<15} {'Rel Diff %':<15}")
        print("-" * 80)
        
        # Check if gradients are similar
        rel_diffs = []
        
        for i in range(len(sigma)):
            auto_val = dfdsig_auto[i].detach().numpy()
            num_val = dfdsig_numerical[i].detach().numpy()
            
            # Calculate relative difference as percentage
            if abs(num_val) > 1e-6:  # Avoid division by zero or very small numbers
                rel_diff = 100 * abs(auto_val - num_val) / abs(num_val)
            else:
                # If numerical gradient is very small, check if autodiff is also small
                rel_diff = 100.0 if abs(auto_val) > 1e-4 else 0.0
            
            rel_diffs.append(rel_diff)
            
            component = f"sigma[{i}]"
            print(f"{component:<10} {auto_val:<15.6f} {num_val:<15.6f} {rel_diff:<15.2f}")
            
        # Assert that relative differences are within acceptable limits
        # Use higher tolerance for cases where numerical differentiation may be less accurate
        if case_idx == 1:  # Hydrostatic stress case
            # Hydrostatic case can have higher numerical error due to stress state
            max_allowed_diff = 20.0  # Allow up to 20% relative difference
        else:
            max_allowed_diff = 10.0  # Allow up to 10% relative difference for other cases
        
        # Check maximum relative difference
        max_rel_diff = max(rel_diffs)
        print(f"\nMaximum relative difference: {max_rel_diff:.2f}%")
        
        # For zero or near-zero components, skip the assertion
        # Focus on significant components where both gradients are non-negligible
        significant_diffs = []
        for i in range(len(sigma)):
            auto_val = abs(dfdsig_auto[i].detach().numpy())
            num_val = abs(dfdsig_numerical[i].detach().numpy())
            # Only include if both are significant
            if auto_val > 1e-4 and num_val > 1e-4:
                significant_diffs.append(rel_diffs[i])
        
        if significant_diffs:
            max_significant_diff = max(significant_diffs)
            print(f"Maximum significant relative difference: {max_significant_diff:.2f}%")
            assert max_significant_diff < max_allowed_diff, f"Gradients differ by more than {max_allowed_diff}% in {case_names[case_idx]}"
        else:
            print("No significant components to compare")
            # If no significant components, check that both gradients are small overall
            assert torch.allclose(dfdsig_auto, dfdsig_numerical, atol=1e-4), f"Gradients don't match for {case_names[case_idx]}"


def test_findp_ipsi_iM_i_differentiable():
    """Test that findp_ipsi_iM_i_torch is differentiable and compares with NumPy implementation."""
    # Create input tensors with requires_grad=True
    N = torch.tensor(0.2, requires_grad=True)
    chi_i = torch.tensor(0.3, requires_grad=True)
    lambd = torch.tensor(0.06, requires_grad=True)
    M_tc = torch.tensor(1.4, requires_grad=True)
    psi = torch.tensor(-0.05, requires_grad=True)
    p = torch.tensor(100.0, requires_grad=True)
    q = torch.tensor(50.0, requires_grad=True)
    M = torch.tensor(1.2, requires_grad=True)
    
    # Call the PyTorch function
    p_i, psi_i, M_i = findp_ipsi_iM_i(N, chi_i, lambd, M_tc, psi, p, q, M)
    
    # Verify output requires gradients
    assert p_i.requires_grad, "Output p_i should require gradients"
    assert psi_i.requires_grad, "Output psi_i should require gradients"
    assert M_i.requires_grad, "Output M_i should require gradients"
    
    # Backpropagate through all outputs
    loss = p_i + psi_i + M_i
    loss.backward()
    
    # Verify gradients exist for all inputs
    assert N.grad is not None, "Gradients were not computed for N"
    assert chi_i.grad is not None, "Gradients were not computed for chi_i"
    assert lambd.grad is not None, "Gradients were not computed for lambd"
    assert M_tc.grad is not None, "Gradients were not computed for M_tc"
    assert psi.grad is not None, "Gradients were not computed for psi"
    assert p.grad is not None, "Gradients were not computed for p"
    assert q.grad is not None, "Gradients were not computed for q"
    assert M.grad is not None, "Gradients were not computed for M"
    
    # Compare with NumPy implementation
    test_cases = [
        # N, chi_i, lambd, M_tc, psi, p, q, M
        (0.2, 0.3, 0.06, 1.4, -0.05, 100.0, 50.0, 1.2),  # Standard case
        (0.2, 0.3, 0.06, 1.4, 0.05, 100.0, 50.0, 1.2),   # Positive psi
        (0.2, 0.3, 0.06, 1.4, -0.05, 100.0, 10.0, 1.2),  # Low q/p ratio
        (0.2, 0.3, 0.06, 1.4, -0.05, 100.0, 90.0, 1.2),  # High q/p ratio
    ]
    
    print("\nComparison of PyTorch and NumPy implementations:")
    print("-" * 80)
    print(f"{'Case':<5} {'Variable':<10} {'PyTorch':<15} {'NumPy':<15} {'Rel Diff %':<15}")
    print("-" * 80)
    
    for i, (N_val, chi_i_val, lambd_val, M_tc_val, psi_val, p_val, q_val, M_val) in enumerate(test_cases):
        # PyTorch version (with detached gradients)
        N_t = torch.tensor(N_val, requires_grad=True)
        chi_i_t = torch.tensor(chi_i_val, requires_grad=True)
        lambd_t = torch.tensor(lambd_val, requires_grad=True)
        M_tc_t = torch.tensor(M_tc_val, requires_grad=True)
        psi_t = torch.tensor(psi_val, requires_grad=True)
        p_t = torch.tensor(p_val, requires_grad=True)
        q_t = torch.tensor(q_val, requires_grad=True)
        M_t = torch.tensor(M_val, requires_grad=True)
        
        try:
            # PyTorch version
            p_i_torch, psi_i_torch, M_i_torch = findp_ipsi_iM_i(
                N_t, chi_i_t, lambd_t, M_tc_t, psi_t, p_t, q_t, M_t
            )
            
            # NumPy version
            p_i_numpy, psi_i_numpy, M_i_numpy = findp_ipsi_iM_i_np(
                N_val, chi_i_val, lambd_val, M_tc_val, psi_val, p_val, q_val, M_val
            )
            
            # Extract values from PyTorch tensors
            p_i_torch_val = p_i_torch.detach().item()
            psi_i_torch_val = psi_i_torch.detach().item()
            M_i_torch_val = M_i_torch.detach().item()
            
            # Calculate relative differences
            rel_diff_p_i = 100.0 * abs(p_i_torch_val - p_i_numpy) / abs(p_i_numpy) if abs(p_i_numpy) > 1e-10 else 0.0
            rel_diff_psi_i = 100.0 * abs(psi_i_torch_val - psi_i_numpy) / abs(psi_i_numpy) if abs(psi_i_numpy) > 1e-10 else 0.0
            rel_diff_M_i = 100.0 * abs(M_i_torch_val - M_i_numpy) / abs(M_i_numpy) if abs(M_i_numpy) > 1e-10 else 0.0
            
            # Print results
            print(f"{i+1:<5} {'p_i':<10} {p_i_torch_val:<15.6f} {p_i_numpy:<15.6f} {rel_diff_p_i:<15.2f}")
            print(f"{'':<5} {'psi_i':<10} {psi_i_torch_val:<15.6f} {psi_i_numpy:<15.6f} {rel_diff_psi_i:<15.2f}")
            print(f"{'':<5} {'M_i':<10} {M_i_torch_val:<15.6f} {M_i_numpy:<15.6f} {rel_diff_M_i:<15.2f}")
            
            # Check that values are close (within 1% relative error)
            assert rel_diff_p_i < 1.0, f"p_i values differ by more than 1%: PyTorch={p_i_torch_val}, NumPy={p_i_numpy}"
            assert rel_diff_psi_i < 1.0, f"psi_i values differ by more than 1%: PyTorch={psi_i_torch_val}, NumPy={psi_i_numpy}"
            assert rel_diff_M_i < 1.0, f"M_i values differ by more than 1%: PyTorch={M_i_torch_val}, NumPy={M_i_numpy}"
            
        except RuntimeError as e:
            # Handle the case where no valid roots are found
            print(f"{i+1:<5} {'ERROR':<10} {str(e)}")
            # If NumPy version also fails, that's expected
            numpy_success = False
            try:
                findp_ipsi_iM_i_np(N_val, chi_i_val, lambd_val, M_tc_val, psi_val, p_val, q_val, M_val)
                numpy_success = True
            except:
                pass
            
            if numpy_success:
                assert False, f"PyTorch version failed but NumPy version succeeded for test case {i+1}"

def test_findC_p_differentiable():
    """Test that findC_p is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    C = torch.tensor([
        [1000.0, 200.0, 200.0, 0.0, 0.0, 0.0],
        [200.0, 1000.0, 200.0, 0.0, 0.0, 0.0],
        [200.0, 200.0, 1000.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 400.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 400.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 400.0]
    ], requires_grad=True, dtype=torch.float64)
    
    dfdsigma = torch.tensor([0.3, 0.2, 0.1, 0.05, 0.03, 0.02], requires_grad=True, dtype=torch.float64)
    dfdepsilon_dfdsigma = torch.tensor(50.0, requires_grad=True, dtype=torch.float64)
    
    # Call the function
    Cp = findC_p(C, dfdsigma, dfdepsilon_dfdsigma)
    
    # Verify output requires gradients
    assert Cp.requires_grad, "Output Cp should require gradients"
    
    # Create a scalar function of the output for backpropagation
    result = torch.sum(Cp)
    
    # Backpropagate
    result.backward()
    
    # Verify gradients exist for all inputs
    assert C.grad is not None, "Gradients were not computed for C"
    assert dfdsigma.grad is not None, "Gradients were not computed for dfdsigma"
    assert dfdepsilon_dfdsigma.grad is not None, "Gradients were not computed for dfdepsilon_dfdsigma"
    
    # Verify gradients are non-zero (for this specific input)
    assert torch.any(C.grad != 0), "Gradient with respect to C should be non-zero somewhere"
    assert torch.any(dfdsigma.grad != 0), "Gradient with respect to dfdsigma should be non-zero somewhere"
    assert dfdepsilon_dfdsigma.grad != 0, "Gradient with respect to dfdepsilon_dfdsigma should be non-zero"
    
    # Test various input values to ensure stability
    test_cases = [
        # C scale, dfdsigma scale, dfdepsilon_dfdsigma value
        (1000.0, 0.1, 50.0),   # Standard case
        (1000.0, 0.1, 10.0),   # Smaller dfdepsilon_dfdsigma
        (500.0, 0.2, 50.0),    # Different stiffness and gradient
        (2000.0, 0.05, 100.0), # Larger stiffness, smaller gradient
    ]
    
    for i, (c_scale, dfd_scale, dfde_val) in enumerate(test_cases):
        # Create scaled input tensors
        C_i = torch.tensor([
            [c_scale, 0.2*c_scale, 0.2*c_scale, 0.0, 0.0, 0.0],
            [0.2*c_scale, c_scale, 0.2*c_scale, 0.0, 0.0, 0.0],
            [0.2*c_scale, 0.2*c_scale, c_scale, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.4*c_scale, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.4*c_scale, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.4*c_scale]
        ], requires_grad=True, dtype=torch.float64)
        
        dfdsigma_i = torch.tensor([
            dfd_scale*3, dfd_scale*2, dfd_scale*1, 
            dfd_scale*0.5, dfd_scale*0.3, dfd_scale*0.2
        ], requires_grad=True, dtype=torch.float64)
        
        dfdepsilon_dfdsigma_i = torch.tensor(dfde_val, requires_grad=True, dtype=torch.float64)
        
        # Call the function and get a scalar output
        Cp_i = findC_p(C_i, dfdsigma_i, dfdepsilon_dfdsigma_i)
        result_i = torch.sum(Cp_i)
        
        # Reset gradients
        if C_i.grad is not None:
            C_i.grad.zero_()
        if dfdsigma_i.grad is not None:
            dfdsigma_i.grad.zero_()
        if dfdepsilon_dfdsigma_i.grad is not None:
            dfdepsilon_dfdsigma_i.grad.zero_()
        
        # Backpropagate
        result_i.backward()
        
        # Verify gradients exist and are finite for all inputs
        assert C_i.grad is not None, f"Gradients were not computed for C in case {i}"
        assert dfdsigma_i.grad is not None, f"Gradients were not computed for dfdsigma in case {i}"
        assert dfdepsilon_dfdsigma_i.grad is not None, f"Gradients were not computed for dfdepsilon_dfdsigma in case {i}"
        
        assert torch.all(torch.isfinite(C_i.grad)), f"Gradient should be finite for C in case {i}"
        assert torch.all(torch.isfinite(dfdsigma_i.grad)), f"Gradient should be finite for dfdsigma in case {i}"
        assert torch.isfinite(dfdepsilon_dfdsigma_i.grad), f"Gradient should be finite for dfdepsilon_dfdsigma in case {i}"
        
        print(f"Case {i}: c_scale={c_scale}, dfd_scale={dfd_scale}, dfde_val={dfde_val}")
        print(f"  Max grad C: {torch.max(torch.abs(C_i.grad)).item()}")
        print(f"  Max grad dfdsigma: {torch.max(torch.abs(dfdsigma_i.grad)).item()}")
        print(f"  Grad dfdepsilon_dfdsigma: {dfdepsilon_dfdsigma_i.grad.item()}")

def test_finddFdepsilon_p_differentiable():
    """Test that finddFdepsilon_p is differentiable with respect to its inputs."""
    # Create input tensors with requires_grad=True
    sigma_vec = torch.tensor([100.0, 50.0, 50.0, 10.0, 5.0, 2.0], requires_grad=True)
    
    # Create parameters for computing dF/depsilon_p
    # [H0, Hy, psi, N, M, M_i, M_tc, M_itc, p_i, p_imax, chi_i, psi_i, lambda]
    dfdep_params = torch.tensor([
        500.0,     # H0
        100.0,     # Hy
        -0.05,     # psi 
        0.2,       # N
        1.2,       # M
        0.8,       # M_i
        1.4,       # M_tc
        1.3,       # M_itc
        150.0,     # p_i
        200.0,     # p_imax
        0.3,       # chi_i
        -0.07,     # psi_i
        0.06       # lambda
    ], requires_grad=True)
    
    # Create dfdsigma by using finddFdsigma function
    M_i = dfdep_params[5]
    p_i = dfdep_params[8]
    dfdsigma = finddFdsigma(sigma_vec, M_i, p_i)
    
    # Call the function
    dfdep_dfdsig, dpideps_pd, dfdpi, dfdq_ = finddFdepsilon_p(sigma_vec, dfdep_params, dfdsigma)
    
    # Verify output requires gradients
    assert dfdep_dfdsig.requires_grad, "Output dfdep_dfdsig should require gradients"
    assert dpideps_pd.requires_grad, "Output dpideps_pd should require gradients"
    assert dfdpi.requires_grad, "Output dfdpi should require gradients"
    assert dfdq_.requires_grad, "Output dfdq_ should require gradients"
    
    # Create a scalar output for backpropagation
    loss = dfdep_dfdsig + dpideps_pd + dfdpi + torch.sum(dfdq_)
    
    # Backpropagate
    loss.backward()
    
    # Verify gradients exist for input tensors
    assert sigma_vec.grad is not None, "Gradients were not computed for sigma_vec"
    assert dfdep_params.grad is not None, "Gradients were not computed for dfdep_params"
    
    # Check that gradients are finite
    assert torch.all(torch.isfinite(sigma_vec.grad)), "Gradients for sigma_vec should be finite"
    assert torch.all(torch.isfinite(dfdep_params.grad)), "Gradients for dfdep_params should be finite"
    
    # Test with positive psi_i (different sign case)
    dfdep_params_pos = torch.tensor([
        500.0,     # H0
        100.0,     # Hy
        0.05,      # psi 
        0.2,       # N
        1.2,       # M
        0.8,       # M_i
        1.4,       # M_tc
        1.3,       # M_itc
        150.0,     # p_i
        200.0,     # p_imax
        0.3,       # chi_i
        0.07,      # psi_i (positive)
        0.06       # lambda
    ], requires_grad=True)
    
    # Reset gradients
    if sigma_vec.grad is not None:
        sigma_vec.grad.zero_()
        
    # Call the function again
    dfdsigma = finddFdsigma(sigma_vec, dfdep_params_pos[5], dfdep_params_pos[8])
    dfdep_dfdsig_pos, dpideps_pd_pos, dfdpi_pos, dfdq_pos = finddFdepsilon_p(
        sigma_vec, dfdep_params_pos, dfdsigma)
    
    # Create scalar output and backpropagate
    loss_pos = dfdep_dfdsig_pos + dpideps_pd_pos + dfdpi_pos + torch.sum(dfdq_pos)
    loss_pos.backward()
    
    # Verify gradients for positive psi_i
    assert dfdep_params_pos.grad is not None, "Gradients were not computed for positive psi_i case"
    assert torch.all(torch.isfinite(dfdep_params_pos.grad)), "Gradients should be finite for positive psi_i case"
    
    # Test with zero psi_i (edge case)
    dfdep_params_zero = torch.tensor([
        500.0,     # H0
        100.0,     # Hy
        0.0,       # psi 
        0.2,       # N
        1.2,       # M
        0.8,       # M_i
        1.4,       # M_tc
        1.3,       # M_itc
        150.0,     # p_i
        200.0,     # p_imax
        0.3,       # chi_i
        0.0,       # psi_i (zero)
        0.06       # lambda
    ], requires_grad=True)
    
    # Reset gradients again
    if sigma_vec.grad is not None:
        sigma_vec.grad.zero_()
    
    # Call the function for zero psi_i
    dfdsigma = finddFdsigma(sigma_vec, dfdep_params_zero[5], dfdep_params_zero[8])
    dfdep_dfdsig_zero, dpideps_pd_zero, dfdpi_zero, dfdq_zero = finddFdepsilon_p(
        sigma_vec, dfdep_params_zero, dfdsigma)
    
    # Create scalar output and backpropagate
    loss_zero = dfdep_dfdsig_zero + dpideps_pd_zero + dfdpi_zero + torch.sum(dfdq_zero)
    loss_zero.backward()
    
    # Verify gradients for zero psi_i
    assert dfdep_params_zero.grad is not None, "Gradients were not computed for zero psi_i case"
    assert torch.all(torch.isfinite(dfdep_params_zero.grad)), "Gradients should be finite for zero psi_i case"
    
    # Print gradient statistics for debugging
    print("\nGradient statistics for finddFdepsilon_p:")
    print(f"  Negative psi_i case: max grad = {torch.max(torch.abs(dfdep_params.grad)).item()}")
    print(f"  Positive psi_i case: max grad = {torch.max(torch.abs(dfdep_params_pos.grad)).item()}")
    print(f"  Zero psi_i case: max grad = {torch.max(torch.abs(dfdep_params_zero.grad)).item()}")

def test_find_dlambda_p_differentiable():
    """Test that find_dlambda_p is differentiable with respect to its inputs and compare with NumPy version."""
    # Create input tensors with requires_grad=True
    C = torch.tensor([
        [1000.0, 200.0, 200.0, 0.0, 0.0, 0.0],
        [200.0, 1000.0, 200.0, 0.0, 0.0, 0.0],
        [200.0, 200.0, 1000.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 400.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 400.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 400.0]
    ], requires_grad=True, dtype=torch.float64)
    
    dfdsigma = torch.tensor([0.3, 0.2, 0.1, 0.05, 0.03, 0.02], requires_grad=True, dtype=torch.float64)
    dfdepsilon_dfdsigma = torch.tensor(50.0, requires_grad=True, dtype=torch.float64)
    depsilon = torch.tensor([0.001, -0.0005, -0.0005, 0.0002, 0.0001, 0.0001], requires_grad=True, dtype=torch.float64)
    
    # Call the function
    dlambda_p = find_dlambda_p(C, dfdsigma, dfdepsilon_dfdsigma, depsilon)
    
    # Verify output requires gradients
    assert dlambda_p.requires_grad, "Output dlambda_p should require gradients"
    
    # Backpropagate
    dlambda_p.backward()
    
    # Verify gradients exist for all inputs
    assert C.grad is not None, "Gradients were not computed for C"
    assert dfdsigma.grad is not None, "Gradients were not computed for dfdsigma"
    assert dfdepsilon_dfdsigma.grad is not None, "Gradients were not computed for dfdepsilon_dfdsigma"
    assert depsilon.grad is not None, "Gradients were not computed for depsilon"
    
    # Verify gradients are finite (no NaN or Inf)
    assert torch.all(torch.isfinite(C.grad)), "Gradients for C should be finite"
    assert torch.all(torch.isfinite(dfdsigma.grad)), "Gradients for dfdsigma should be finite"
    assert torch.isfinite(dfdepsilon_dfdsigma.grad), "Gradient for dfdepsilon_dfdsigma should be finite"
    assert torch.all(torch.isfinite(depsilon.grad)), "Gradients for depsilon should be finite"
    
    # Compare with NumPy implementation
    # Convert PyTorch tensors to NumPy arrays
    C_np = C.detach().numpy()
    dfdsigma_np = dfdsigma.detach().numpy()
    dfdepsilon_dfdsigma_np = dfdepsilon_dfdsigma.detach().numpy()
    depsilon_np = depsilon.detach().numpy()
    
    # Call NumPy function
    dlambda_p_np = find_dlambda_p_np(C_np, dfdsigma_np, dfdepsilon_dfdsigma_np, depsilon_np)
    
    # Verify results are close
    torch_result = dlambda_p.detach().item()
    numpy_result = dlambda_p_np
    rel_diff = abs(torch_result - numpy_result) / abs(numpy_result) if abs(numpy_result) > 1e-10 else 0.0
    
    print(f"\nComparing find_dlambda_p implementations:")
    print(f"  PyTorch: {torch_result}")
    print(f"  NumPy:   {numpy_result}")
    print(f"  Relative difference: {rel_diff:.6%}")
    
    # Results should be very close (less than 0.1% different)
    assert rel_diff < 0.001, f"PyTorch and NumPy results differ by more than 0.1%: {rel_diff:.6%}"
    
    # Test various cases to ensure robust differentiation
    test_cases = [
        # C scale, dfdsigma scale, dfdepsilon_dfdsigma value, depsilon scale
        (1000.0, 0.1, 50.0, 0.001),   # Standard case
        (1000.0, 0.1, 10.0, 0.001),   # Smaller dfdepsilon_dfdsigma
        (500.0, 0.2, 50.0, 0.002),    # Different stiffness and gradient
        (2000.0, 0.05, 100.0, 0.0005) # Larger stiffness, smaller gradient
    ]
    
    for i, (c_scale, dfd_scale, dfde_val, deps_scale) in enumerate(test_cases):
        # Create scaled input tensors
        C_i = torch.tensor([
            [c_scale, 0.2*c_scale, 0.2*c_scale, 0.0, 0.0, 0.0],
            [0.2*c_scale, c_scale, 0.2*c_scale, 0.0, 0.0, 0.0],
            [0.2*c_scale, 0.2*c_scale, c_scale, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.4*c_scale, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.4*c_scale, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.4*c_scale]
        ], requires_grad=True, dtype=torch.float64)
        
        dfdsigma_i = torch.tensor([
            dfd_scale*3, dfd_scale*2, dfd_scale*1, 
            dfd_scale*0.5, dfd_scale*0.3, dfd_scale*0.2
        ], requires_grad=True, dtype=torch.float64)
        
        dfdepsilon_dfdsigma_i = torch.tensor(dfde_val, requires_grad=True, dtype=torch.float64)
        
        depsilon_i = torch.tensor([
            deps_scale, -0.5*deps_scale, -0.5*deps_scale, 
            0.2*deps_scale, 0.1*deps_scale, 0.1*deps_scale
        ], requires_grad=True, dtype=torch.float64)
        
        # Reset gradients
        if C_i.grad is not None:
            C_i.grad.zero_()
        if dfdsigma_i.grad is not None:
            dfdsigma_i.grad.zero_()
        if dfdepsilon_dfdsigma_i.grad is not None:
            dfdepsilon_dfdsigma_i.grad.zero_()
        if depsilon_i.grad is not None:
            depsilon_i.grad.zero_()
        
        # Call the function
        dlambda_p_i = find_dlambda_p(C_i, dfdsigma_i, dfdepsilon_dfdsigma_i, depsilon_i)
        
        # Backpropagate
        dlambda_p_i.backward()
        
        # Verify gradients exist and are finite for all inputs
        assert C_i.grad is not None, f"Gradients were not computed for C in case {i}"
        assert dfdsigma_i.grad is not None, f"Gradients were not computed for dfdsigma in case {i}"
        assert dfdepsilon_dfdsigma_i.grad is not None, f"Gradients were not computed for dfdepsilon_dfdsigma in case {i}"
        assert depsilon_i.grad is not None, f"Gradients were not computed for depsilon in case {i}"
        
        assert torch.all(torch.isfinite(C_i.grad)), f"Gradients for C should be finite in case {i}"
        assert torch.all(torch.isfinite(dfdsigma_i.grad)), f"Gradients for dfdsigma should be finite in case {i}"
        assert torch.isfinite(dfdepsilon_dfdsigma_i.grad), f"Gradient for dfdepsilon_dfdsigma should be finite in case {i}"
        assert torch.all(torch.isfinite(depsilon_i.grad)), f"Gradients for depsilon should be finite in case {i}"
        
        print(f"\nCase {i}: c_scale={c_scale}, dfd_scale={dfd_scale}, dfde_val={dfde_val}, deps_scale={deps_scale}")
        print(f"  dlambda_p value: {dlambda_p_i.item()}")
        print(f"  Max grad C: {torch.max(torch.abs(C_i.grad)).item()}")
        print(f"  Max grad dfdsigma: {torch.max(torch.abs(dfdsigma_i.grad)).item()}")
        print(f"  Grad dfdepsilon_dfdsigma: {dfdepsilon_dfdsigma_i.grad.item()}")
        print(f"  Max grad depsilon: {torch.max(torch.abs(depsilon_i.grad)).item()}")
