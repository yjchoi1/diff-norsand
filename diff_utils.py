import numpy as np
import torch
from typing import Tuple, Union
import warnings


def stress_decomp(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean pressure p and von Mises stress q from stress vector
    
    Args:
        sigma: Stress vector in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23], 
        1D tensor of shape (6,)
        
    Returns:
        p_: Mean pressure (1/3 of first invariant)
        q_: von Mises stress (square root of 3/2 of second invariant of deviatoric stress)
    """
    # Apply tension cutoffs differentiably using torch.clamp
    sigma_copy = torch.clamp(sigma, min=0.1)
    
    # Calculate mean pressure
    p_ = (1/3) * torch.sum(sigma_copy[0:3])
    
    # Calculate von Mises stress
    q_ = torch.sqrt(0.5 * ((sigma_copy[0] - sigma_copy[1])**2 + 
                          (sigma_copy[1] - sigma_copy[2])**2 + 
                          (sigma_copy[2] - sigma_copy[0])**2 + 
                          6 * (sigma_copy[3]**2 + sigma_copy[4]**2 + sigma_copy[5]**2)))
    
    return p_, q_


def find_sJ2J3(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute deviatoric stress tensor s, and J2, J3 invariants
    
    Args:
        sigma: Stress vector in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23], 
        1D tensor of shape (6,)
        
    Returns:
        s: Deviatoric stress vector, 1D tensor of shape (6,)
        J_2: Second invariant of deviatoric stress
        J_3: Third invariant of deviatoric stress
    """
    p, _ = stress_decomp(sigma)
    
    # Create deviatoric stress tensor
    s = sigma.clone()  # Clone to avoid modifying the input tensor
    s[0:3] = s[0:3] - p  # Subtract mean pressure from normal components
    
    # Calculate J2 (second invariant of deviatoric stress)
    J_2 = (1/2) * (s[0]**2 + s[1]**2 + s[2]**2 + 
                  2 * (s[3]**2 + s[4]**2 + s[5]**2))
    
    # Calculate J3 (third invariant of deviatoric stress)
    J_3 = (s[0] * s[1] * s[2] - 
           s[0] * s[5]**2 - 
           s[2] * s[3]**2 - 
           s[1] * s[4]**2 + 
           2 * s[3] * s[4] * s[5])
    
    return s, J_2, J_3


def vol_dev(sigma: torch.Tensor, epsilon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute volumetric strain and deviatoric strain in Voigt notation
    NOTE: This function needs check.
    
    Args:
        sigma: Stress vector in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23], 
        1D tensor of shape (6,) - NOTE: This parameter is not used in computation
        but kept for backward compatibility
        epsilon: Strain vector in Voigt notation [ε11, ε22, ε33, ε12, ε13, ε23], 
        1D tensor of shape (6,)
        
    Returns:
        e_v: Volumetric strain
        e_q: Deviatoric strain (standard magnitude, depends only on strain)
        
    Note:
        This function computes the standard deviatoric strain magnitude which depends
        only on the strain tensor. The sigma parameter is kept for backward compatibility
        but is not used in the computation. For stress-dependent deviatoric strain,
        use vol_dev_stress_weighted() instead.
    """
    # # Issue a deprecation-style warning about unused sigma parameter
    # warnings.warn(
    #     "The 'sigma' parameter in vol_dev() is not used in computation. "
    #     "This function computes standard deviatoric strain magnitude which depends only on strain. "
    #     "Use vol_dev_stress_weighted() if you need stress-dependent computation.",
    #     UserWarning,
    #     stacklevel=2
    # )
    
    # --- NOTE: This is the original implementation. ---    
    # If q is zero, the following e_q will be undefined (division by zero). 
    # We avoid this by using a small epsilon value.
    # NOTE: Need to ensure that this method works fine for norsand.
    #   The original implementation just force `e_q` to be zero when `q` is close to zero.
    #   Why do we compute the deviatoric strain `e_q` in this way?
    # e_v = torch.sum(epsilon[0:3])
    # p, q = stress_decomp(sigma)  # This line is not needed
    # safe_q = torch.clamp(q, min=1e-6)
    # if q < 1e-4:
    #     raise ValueError(
    #         f"Von Mises stress q={q.item():.2e} is too small (< 1e-4). "
    #         "This may cause numerical instability in deviatoric strain calculation. "
    #         "Consider handling this special case in your differentiable model."
    #     )
    
    # # Calculate deviatoric strain using the explicit form
    # e_q = ((sigma[0] - p) / safe_q * epsilon[0] + 
    #        (sigma[1] - p) / safe_q * epsilon[1] + 
    #        (sigma[2] - p) / safe_q * epsilon[2] + 
    #        2 * sigma[3] / safe_q * epsilon[3] + 
    #        2 * sigma[4] / safe_q * epsilon[4] + 
    #        2 * sigma[5] / safe_q * epsilon[5])
    
    # # --- NOTE: This is the implementation based on well known formula. ---
    # Add small epsilon to avoid sqrt(0) gradient issues when deviatoric strain is zero
    e_v = torch.sum(epsilon[0:3])
    eps_numerical = torch.tensor(1e-12, device=epsilon.device, dtype=epsilon.dtype)
    e_q = torch.sqrt(
        2/3 * (
            (epsilon[0] - e_v/3)**2 +
            (epsilon[1] - e_v/3)**2 +
            (epsilon[2] - e_v/3)**2 +
            2 * (epsilon[3]**2 + epsilon[4]**2 + epsilon[5]**2)
        ) + eps_numerical
    )
    
    return e_v, e_q


def vol_dev_stress_weighted(sigma: torch.Tensor, epsilon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute volumetric strain and stress-weighted deviatoric strain in Voigt notation
    
    This function computes a stress-weighted deviatoric strain, which represents the strain
    increment in the direction of the stress deviator. This is used in some plasticity models.
    
    Args:
        sigma: Stress vector in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23], 
        1D tensor of shape (6,)
        epsilon: Strain vector in Voigt notation [ε11, ε22, ε33, ε12, ε13, ε23], 
        1D tensor of shape (6,)
        
    Returns:
        e_v: Volumetric strain
        e_q_weighted: Stress-weighted deviatoric strain
    """
    p, q = stress_decomp(sigma)
    e_v = torch.sum(epsilon[0:3])
    
    # If q is zero, the following e_q will be undefined (division by zero). 
    # We avoid this by using a small epsilon value.
    safe_q = torch.clamp(q, min=1e-6)
    
    # Calculate stress-weighted deviatoric strain using the explicit form
    # This represents the strain increment in the direction of the stress deviator
    e_q_weighted = ((sigma[0] - p) / safe_q * epsilon[0] + 
                   (sigma[1] - p) / safe_q * epsilon[1] + 
                   (sigma[2] - p) / safe_q * epsilon[2] + 
                   2 * sigma[3] / safe_q * epsilon[3] + 
                   2 * sigma[4] / safe_q * epsilon[4] + 
                   2 * sigma[5] / safe_q * epsilon[5])
    
    return e_v, e_q_weighted


def lode_angle(sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute Lode angle from stress vector
    
    Args:
        sigma: Stress vector in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23], 
        1D tensor of shape (6,)
        
    Returns:
        theta: Lode angle in radians
        
    NOTE: This function is not fully rigorous at the threshold values about differentiability.
    """
    _, J_2, J_3 = find_sJ2J3(sigma)
    
    # Handle the case where J_2 is close to zero differentiably
    eps = torch.finfo(sigma.dtype).eps
    safe_J2 = torch.clamp(J_2, min=eps)
    
    # Calculate sin(3θ) with safe division
    factor = 3 * torch.sqrt(torch.tensor(3.0, device=sigma.device, dtype=sigma.dtype)) / 2
    sin3theta = factor * (J_3 / safe_J2**(3/2))
    
    # Clamp sin(3θ) to [-1, 1] range to ensure valid arcsin input
    # Ensure the sin3theta to be between -1 and 1.
    sin3theta_clamped = torch.clamp(
        sin3theta, -1 + eps, 1 - eps)
    
    # Calculate θ = arcsin(sin(3θ))/3
    theta = (1/3) * torch.arcsin(sin3theta_clamped)
    
    # Clamp θ to [-π/6, π/6] range
    # theta_clamped = torch.clamp(theta, -torch.pi/6, torch.pi/6)
    
    return theta


def findCe(
    G_max: torch.Tensor, 
    p: torch.Tensor, 
    p_ref: torch.Tensor, 
    m: torch.Tensor, 
    nu: torch.Tensor,
) -> torch.Tensor:
    """
    Compute elastic tangent operator
    
    Args:
        G_max: Maximum shear modulus, 1D tensor
        p: Mean pressure, 1D tensor
        p_ref: Reference pressure, 1D tensor
        m: Pressure exponent, 1D tensor
        nu: Poisson's ratio, 1D tensor
        
    Returns:
        C_e: Elastic tangent operator matrix, 2D tensor of shape (6, 6)
    """
    G = G_max * (p / p_ref) ** m
    K = ((2 * (1 + nu)) / (3 * (1 - 2 * nu))) * G
    C_e = torch.zeros((6, 6), device=p.device, dtype=p.dtype)
    d = K + (4/3) * G
    non_d = K - (2/3) * G
    
    for i in range(3):
        for j in range(3):
            if i == j:
                C_e[i, j] = d
            else:
                C_e[i, j] = non_d
    
    for k in range(3, 6):
        C_e[k, k] = G
    
    return C_e


def voigt_norm(voigt_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 norm of Voigt vector with proper scaling for tensor operations.

    Args:
        voigt_vec (torch.Tensor): Voigt notation tensor [σ11, σ22, σ33, σ12, σ13, σ23], shape (6,)

    Returns:
        torch.Tensor: Frobenius norm of the corresponding full tensor.
    """
    scale = torch.ones_like(voigt_vec)
    scale[3:6] = torch.sqrt(torch.tensor(2.0, device=voigt_vec.device, dtype=voigt_vec.dtype))
    scaled_vec = scale * voigt_vec
    norm_val = torch.norm(scaled_vec)
    return norm_val


def dJ2J3(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of J2, J3 invariants with respect to sigma using PyTorch's autodiff
    
    Args:
        sigma: Stress vector in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23], 1D tensor of shape (6,)
        
    Returns:
        dJ2_dsigma: Derivative of J2 with respect to sigma, 1D tensor of shape (6,)
        dJ3_dsigma: Derivative of J3 with respect to sigma, 1D tensor of shape (6,)
        
    # NOTE: Complete the combined test for this function.
    """
    # Create a copy of the input tensor that requires gradients
    sigma_grad = sigma.detach().clone().requires_grad_(True)
    
    # Compute J2 and J3 using the existing function
    _, J2, J3 = find_sJ2J3(sigma_grad)
    
    # Compute gradient of J2 with respect to sigma using torch.autograd.grad
    # We need `retain_graph=True` to avoid the graph being deleted after the first backward pass.
    # This is because the same computation graph will be used for J3. 
    dJ2_dsigma = torch.autograd.grad(J2, sigma_grad, create_graph=True, retain_graph=True)[0]
    
    # Compute gradient of J3 with respect to sigma using torch.autograd.grad
    dJ3_dsigma = torch.autograd.grad(J3, sigma_grad, create_graph=True)[0]
    
    return dJ2_dsigma, dJ3_dsigma


def dJ2J3_analytical(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of J2, J3 invariants with respect to sigma using analytical formulas
    
    Args:
        sigma: Stress vector in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23], 1D tensor of shape (6,)
        
    Returns:
        dJ2_dsigma: Derivative of J2 with respect to sigma, 1D tensor of shape (6,)
        dJ3_dsigma: Derivative of J3 with respect to sigma, 1D tensor of shape (6,)
    
    """
    s, _, _ = find_sJ2J3(sigma)
    dJ2_dsigma = torch.tensor([s[0], s[1], s[2], 2*sigma[3], 2*sigma[4], 2*sigma[5]], 
                             device=sigma.device, dtype=sigma.dtype)

    # NOTE: dJ3_disgma is wrong. See the test case in test/test_diff_utils.py
    dJ3_dsigma = torch.tensor([
        -(1/3) * s[0] * s[1] - (1/3) * s[0] * s[2] + (2/3) * s[1] * s[2] - 
        (2/3) * s[5]**2 + (1/3) * s[4]**2 + (1/3) * s[3]**2,

        -(1/3) * s[0] * s[1] + (2/3) * s[0] * s[2] - (1/3) * s[1] * s[2] + 
        (1/3) * s[5]**2 - (2/3) * s[4]**2 + (1/3) * s[3]**2,

        (2/3) * s[0] * s[1] - (1/3) * s[0] * s[2] - (1/3) * s[1] * s[2] + 
        (1/3) * s[5]**2 + (1/3) * s[4]**2 - (2/3) * s[3]**2,
        
        -2 * s[2] * s[3] + 2 * s[5] * s[4],

        -2 * s[1] * s[4] + 2 * s[3] * s[5],

        -2 * s[0] * s[5] + 2 * s[3] * s[4]
    ], device=sigma.device, dtype=sigma.dtype)
    
    return dJ2_dsigma, dJ3_dsigma 