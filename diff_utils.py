import numpy as np
import torch
from typing import Tuple, Union
import warnings


def stress_decomp(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute mean pressure p and von Mises stress q from stress vector
    
    The mean pressure and von Mises stress are computed as:
    
    .. math::
        p = \frac{1}{3}(\sigma_{11} + \sigma_{22} + \sigma_{33}) = \frac{1}{3}I_1
        
    .. math::
        q = \sqrt{\frac{3}{2}J_2} = \sqrt{\frac{1}{2}[(\sigma_{11}-\sigma_{22})^2 + (\sigma_{22}-\sigma_{33})^2 + (\sigma_{33}-\sigma_{11})^2 + 6(\sigma_{12}^2 + \sigma_{13}^2 + \sigma_{23}^2)]}
    
    where :math:`I_1` is the first stress invariant and :math:`J_2` is the second deviatoric stress invariant.
    
    Args:
        sigma: Stress vector in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]`, 
        1D tensor of shape (6,)
        
    Returns:
        tuple: A tuple containing:
        
            - **p_** (*torch.Tensor*) - Mean pressure :math:`p` (1/3 of first invariant)
            - **q_** (*torch.Tensor*) - von Mises stress :math:`q` (square root of 3/2 of second invariant of deviatoric stress)
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
    r"""
    Compute deviatoric stress tensor s, and J2, J3 invariants
    
    The deviatoric stress tensor and its invariants are computed as:
    
    .. math::
        \mathbf{s} = \boldsymbol{\sigma} - p\mathbf{I}
        
    .. math::
        J_2 = \frac{1}{2}\mathbf{s}:\mathbf{s} = \frac{1}{2}(s_{11}^2 + s_{22}^2 + s_{33}^2 + 2s_{12}^2 + 2s_{13}^2 + 2s_{23}^2)
        
    .. math::
        J_3 = \det(\mathbf{s}) = s_{11}s_{22}s_{33} - s_{11}s_{23}^2 - s_{33}s_{12}^2 - s_{22}s_{13}^2 + 2s_{12}s_{13}s_{23}
    
    where :math:`\mathbf{I}` is the identity tensor.
    
    Args:
        sigma: Stress vector in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]`, 
        1D tensor of shape (6,)
        
    Returns:
        tuple: A tuple containing:
        
            - **s** (*torch.Tensor*) - Deviatoric stress vector :math:`\mathbf{s}`, 1D tensor of shape (6,)
            - **J_2** (*torch.Tensor*) - Second invariant of deviatoric stress :math:`J_2`
            - **J_3** (*torch.Tensor*) - Third invariant of deviatoric stress :math:`J_3`
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
    r"""
    Compute volumetric strain and deviatoric strain in Voigt notation
    
    The volumetric and deviatoric strain are computed as:
    
    .. math::
        \varepsilon_v = \varepsilon_{11} + \varepsilon_{22} + \varepsilon_{33} = \text{tr}(\boldsymbol{\varepsilon})
        
    .. math::
        \varepsilon_q = \sqrt{\frac{2}{3}[(\varepsilon_{11} - \frac{\varepsilon_v}{3})^2 + (\varepsilon_{22} - \frac{\varepsilon_v}{3})^2 + (\varepsilon_{33} - \frac{\varepsilon_v}{3})^2 + 2(\varepsilon_{12}^2 + \varepsilon_{13}^2 + \varepsilon_{23}^2)]}
    
    Note:
        This function computes the standard deviatoric strain magnitude which depends
        only on the strain tensor. The sigma parameter is kept for backward compatibility
        but is not used in the computation. For stress-dependent deviatoric strain,
        use vol_dev_stress_weighted() instead.
        
    Args:
        sigma: Stress vector in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]`, 
        1D tensor of shape (6,) - NOTE: This parameter is not used in computation
        but kept for backward compatibility
        epsilon: Strain vector in Voigt notation :math:`[\varepsilon_{11}, \varepsilon_{22}, \varepsilon_{33}, \varepsilon_{12}, \varepsilon_{13}, \varepsilon_{23}]`, 
        1D tensor of shape (6,)
        
    Returns:
        tuple: A tuple containing:
        
            - **e_v** (*torch.Tensor*) - Volumetric strain :math:`\varepsilon_v`
            - **e_q** (*torch.Tensor*) - Deviatoric strain :math:`\varepsilon_q` (standard magnitude, depends only on strain)
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
    r"""
    Compute volumetric strain and stress-weighted deviatoric strain in Voigt notation
    
    This function computes a stress-weighted deviatoric strain:
    
    .. math::
        \varepsilon_v = \varepsilon_{11} + \varepsilon_{22} + \varepsilon_{33}
        
    .. math::
        \varepsilon_q^{weighted} = \frac{1}{q}\left[(\sigma_{11}-p)\varepsilon_{11} + (\sigma_{22}-p)\varepsilon_{22} + (\sigma_{33}-p)\varepsilon_{33} + 2\sigma_{12}\varepsilon_{12} + 2\sigma_{13}\varepsilon_{13} + 2\sigma_{23}\varepsilon_{23}\right]
    
    where :math:`p` is the mean stress, :math:`q` is the von Mises stress.
    This represents the strain increment in the direction of the stress deviator.
    
    Args:
        sigma: Stress vector in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]`, 
        1D tensor of shape (6,)
        epsilon: Strain vector in Voigt notation :math:`[\varepsilon_{11}, \varepsilon_{22}, \varepsilon_{33}, \varepsilon_{12}, \varepsilon_{13}, \varepsilon_{23}]`, 
        1D tensor of shape (6,)
        
    Returns:
        tuple: A tuple containing:
        
            - **e_v** (*torch.Tensor*) - Volumetric strain :math:`\varepsilon_v`
            - **e_q_weighted** (*torch.Tensor*) - Stress-weighted deviatoric strain :math:`\varepsilon_q^{weighted}`
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
    r"""
    Compute Lode angle from stress vector
    
    The Lode angle is computed as:
    
    .. math::
        \sin(3\theta) = \frac{3\sqrt{3}}{2} \frac{J_3}{J_2^{3/2}}
        
    .. math::
        \theta = \frac{1}{3}\arcsin(\sin(3\theta))
    
    where :math:`\theta \in [-\pi/6, \pi/6]` is the Lode angle, :math:`J_2` and :math:`J_3` are the second 
    and third deviatoric stress invariants.
    
    Args:
        sigma: Stress vector in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]`, 
        1D tensor of shape (6,)
        
    Returns:
        torch.Tensor: Lode angle :math:`\theta` in radians
        
    Note:
        This function is not fully rigorous at the threshold values about differentiability.
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
    r"""
    Compute elastic tangent operator
    
    The elastic tangent operator is computed using:
    
    .. math::
        G = G_{max} \left(\frac{p}{p_{ref}}\right)^m
        
    .. math::
        K = \frac{2(1+\nu)}{3(1-2\nu)} G
        
    .. math::
        \mathbf{C}^e = \begin{bmatrix}
        K + \frac{4}{3}G & K - \frac{2}{3}G & K - \frac{2}{3}G & 0 & 0 & 0 \\
        K - \frac{2}{3}G & K + \frac{4}{3}G & K - \frac{2}{3}G & 0 & 0 & 0 \\
        K - \frac{2}{3}G & K - \frac{2}{3}G & K + \frac{4}{3}G & 0 & 0 & 0 \\
        0 & 0 & 0 & G & 0 & 0 \\
        0 & 0 & 0 & 0 & G & 0 \\
        0 & 0 & 0 & 0 & 0 & G
        \end{bmatrix}
    
    where :math:`G` is the shear modulus, :math:`K` is the bulk modulus, and :math:`\nu` is Poisson's ratio.
    
    Args:
        G_max: Maximum shear modulus :math:`G_{max}`, 1D tensor
        p: Mean pressure :math:`p`, 1D tensor
        p_ref: Reference pressure :math:`p_{ref}`, 1D tensor
        m: Pressure exponent :math:`m`, 1D tensor
        nu: Poisson's ratio :math:`\nu`, 1D tensor
        
    Returns:
        torch.Tensor: Elastic tangent operator matrix :math:`\mathbf{C}^e`, 2D tensor of shape (6, 6)
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
    Compute L2 norm of stress tensor in Voigt notation with proper scaling.
    
    The Frobenius norm of a stress tensor represented in Voigt notation is:
    
    .. math::
        ||\boldsymbol{\sigma}||_F = \sqrt{\sum_{i=1}^{3} \sigma_{ii}^2 + 2\sum_{i=4}^{6} \sigma_{ij}^2}
        
    .. math::
        = \sqrt{\sigma_{11}^2 + \sigma_{22}^2 + \sigma_{33}^2 + 2(\sigma_{12}^2 + \sigma_{13}^2 + \sigma_{23}^2)}
    
    where the factor of 2 accounts for the off-diagonal shear stress terms appearing twice in the full tensor.

    Args:
        voigt_vec (torch.Tensor): Stress tensor in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]`, shape (6,)

    Returns:
        torch.Tensor: Frobenius norm :math:`||\boldsymbol{\sigma}||_F` of the stress tensor.
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
        sigma: Stress vector in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]`, 1D tensor of shape (6,)
        
    Returns:
        dJ2_dsigma: Derivative of :math:`J_2` with respect to :math:`\boldsymbol{\sigma}`, 1D tensor of shape (6,)
        dJ3_dsigma: Derivative of :math:`J_3` with respect to :math:`\boldsymbol{\sigma}`, 1D tensor of shape (6,)
        
    Note:
        Complete the combined test for this function.
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
    r"""
    Compute derivatives of J2, J3 invariants with respect to sigma using analytical formulas
    
    Args:
        sigma: Stress vector in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]`, 1D tensor of shape (6,)
        
    Returns:
        tuple: A tuple containing:
        
            - **dJ2_dsigma** (*torch.Tensor*) - Derivative of :math:`J_2` with respect to :math:`\boldsymbol{\sigma}`, 1D tensor of shape (6,)
            - **dJ3_dsigma** (*torch.Tensor*) - Derivative of :math:`J_3` with respect to :math:`\boldsymbol{\sigma}`, 1D tensor of shape (6,)
    
    Warning:
        The current implementation of :math:`\frac{\partial J_3}{\partial \boldsymbol{\sigma}}` contains errors. 
        See test cases in test/test_diff_utils.py. Use :func:`dJ2J3` for reliable gradient computation.
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