import numpy as np
import torch
from typing import Tuple, List, Union
from diff_utils import (
    stress_decomp, find_sJ2J3, vol_dev, lode_angle, 
    findCe, voigt_norm, dJ2J3)
from diff_norsand_functions import (
    findM, findM_i, findM_itc, 
    findchi_i, findpsipsii, findp_imax, findF, 
    finddFdsigma, findp_ipsi_iM_i, findC_p, finddFdepsilon_p, find_dlambda_p
)

def stressCorrection(params: torch.Tensor, F: torch.Tensor, sigma_vec: torch.Tensor, 
                     p_i: torch.Tensor, FTOL: float, MAXITS: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable stress correction algorithm using PyTorch
    
    Args:
        params: Material parameters (tensor of shape (14,))
            [Lambda, M_tc, N, H0, Hy, _, nu, chi_i, _, Gamma, e, G_max, p_ref, m]
        F: Yield function value (scalar tensor)
        sigma_vec: Stress vector in Voigt notation (tensor of shape (6,))
        p_i: Image mean pressure (scalar tensor)
        FTOL: Tolerance for yield function (float)
        MAXITS: Maximum number of iterations (int)
        
    Returns:
        sigma_corr: Corrected stress vector (tensor of shape (6,))
        p_i_corr: Corrected image mean pressure (scalar tensor)
    """
    # Unpack parameters
    Lambda = params[0]
    M_tc = params[1]
    N = params[2]
    H0 = params[3]
    Hy = params[4]
    nu = params[6]
    chi_i = params[7]
    Gamma = params[9]
    e = params[10]
    G_max = params[11]
    p_ref = params[12]
    m = params[13]

    # Initialize variables
    sigma_0 = sigma_vec.clone()
    p_i0 = p_i.clone()
    F0 = F.clone()

    # Begin loop
    for i in range(1, MAXITS + 1):
        p0, q0 = stress_decomp(sigma_0)
        theta_0 = lode_angle(sigma_0)
        M0 = findM(theta_0, M_tc)
        psi0, psi_i0 = findpsipsii(Gamma, Lambda, p0, p_i0, e)
        M_i0 = findM_i(M0, M_tc, chi_i, psi_i0, N)

        if i == 1:
            # Use provided F for first iteration
            F0_current = F0
        else:
            # Compute F for subsequent iterations
            F0_current = findF(p0, q0, M_i0, p_i0)
        
        # Values for cap and maximum yield surface
        M_itc0 = findM_itc(N, chi_i, psi_i0, M_tc)
        p_imax0 = findp_imax(chi_i, psi_i0, p0, M_itc0)
        
        # Compute C_e (to compute denominator)
        C_e = findCe(G_max, p0, p_ref, m, nu)
    
        # dF/dsigma using automatic differentiation
        # Create a copy that requires gradients for auto-diff
        sigma_for_grad = sigma_0.clone().requires_grad_(True)
        dfdsigma0 = finddFdsigma(sigma_for_grad, M_i0, p_i0)
        
        # dF/depsilonp
        dfdep_params0 = torch.stack([H0, Hy, psi0, N, M0, M_i0, M_tc, M_itc0, p_i0, p_imax0, chi_i, psi_i0, Lambda])
        dfdepsilon0_dfdsigma0, dpi_depspd0, dfdpi0, dfdq_0 = finddFdepsilon_p(sigma_0, dfdep_params0, dfdsigma0)
    
        # Compute denominator
        denom = -dfdepsilon0_dfdsigma0 + torch.matmul(dfdsigma0, torch.matmul(C_e, dfdsigma0))
        
        # Add small epsilon to avoid division by zero
        eps = torch.finfo(denom.dtype).eps
        denom_safe = torch.clamp(torch.abs(denom), min=eps) * torch.sign(denom + eps)
    
        # Compute del_lambda_p
        del_lambda = F0_current / denom_safe
        
        # Compute df/dpi using the same approach as numpy version
        # First term
        parfparpi = -p0 * M_i0 / p_i0
    
        # Second term
        dfdMi = -q0 / M_i0
    
        # Third term - handle sign function differentiably
        if torch.abs(psi_i0) > 1e-10:
            psi_sign = torch.sign(psi_i0)
        else:
            psi_sign = torch.tensor(0.0, device=psi_i0.device, dtype=psi_i0.dtype)
        
        dMidpi = -(M0 / M_tc) * N * chi_i * psi_sign * Lambda / p_i0
    
        # Combine terms
        dfdpi = parfparpi + dfdMi * dMidpi
    
        # Compute B0 = A0 / (df/dH) ~ A0 / (df/dpi) 
        # Add small epsilon to avoid division by zero
        dfdpi_safe = torch.clamp(torch.abs(dfdpi), min=eps) * torch.sign(dfdpi + eps)
        B0 = dfdepsilon0_dfdsigma0 / dfdpi_safe
        
        # Update stresses, PIV (match numpy version exactly)
        sigma_1 = sigma_0.clone()  # No stress update in numpy version
        p_i1 = p_i0 + 20 * del_lambda * B0

        # Compute new yield surface
        theta_1 = lode_angle(sigma_1)
        M1 = findM(theta_1, M_tc)
        p1, q1 = stress_decomp(sigma_1)
        _, psi_i1 = findpsipsii(Gamma, Lambda, p1, p_i1, e)
        M_i1 = findM_i(M1, M_tc, chi_i, psi_i1, N)
        F1 = findF(p1, q1, M_i1, p_i1)
    
        # Check if stresses land on yield surface (return condition)
        if torch.abs(F1) <= FTOL or i == MAXITS:
            return sigma_1, p_i1
        
        # Check for non-convergence of current scheme
        if torch.abs(F1) > torch.abs(F0_current):
            # Alternative update scheme
            dfdsigma_norm_sq = torch.matmul(dfdsigma0, dfdsigma0)
            dfdsigma_norm_sq_safe = torch.clamp(dfdsigma_norm_sq, min=eps)
            del_lambda_alt = F0_current / dfdsigma_norm_sq_safe
            
            sigma_1 = sigma_0.clone()  # No stress update
            p_i1 = p_i0  # No p_i update in alternative scheme
            
            theta_1 = lode_angle(sigma_1)
            M1 = findM(theta_1, M_tc)
            p1, q1 = stress_decomp(sigma_1)
            _, psi_i1 = findpsipsii(Gamma, Lambda, p1, p_i1, e)
            M_i1 = findM_i(M1, M_tc, chi_i, psi_i1, N)
            F1 = findF(p1, q1, M_i1, p_i1)
            
            if torch.abs(F1) <= FTOL or i == MAXITS:
                return sigma_1, p_i1

        # Update for next iteration
        sigma_0 = sigma_1.clone()
        p_i0 = p_i1.clone()
        F0 = F1.clone()

    # If we reach here without convergence, return the last values
    return sigma_0, p_i0