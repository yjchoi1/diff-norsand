import numpy as np
import torch
from typing import Tuple, List, Union
from diff_utils import (
    stress_decomp, find_sJ2J3, vol_dev, lode_angle, 
    findCe, voigt_norm, dJ2J3
)

def findM(theta: torch.Tensor, M_tc: torch.Tensor) -> torch.Tensor:
    r"""
    Compute M based on Lode angle and M_tc
    
    The critical state stress ratio M varies with the Lode angle according to:
    
    .. math::
        g(\theta) = 1 - \frac{M_{tc}}{3 + M_{tc}} \cos\left(\frac{3\theta}{2} + \frac{\pi}{4}\right)
        
    .. math::
        M = M_{tc} \cdot g(\theta)
    
    Args:
        theta: Lode angle :math:`\theta`
        M_tc: Critical state stress ratio in triaxial compression :math:`M_{tc}`
        
    Returns:
        torch.Tensor: Critical state stress ratio at given Lode angle :math:`M`
    """
    g_theta = 1 - (M_tc / (3 + M_tc)) * torch.cos((3 * theta) / 2 + torch.pi / 4)
    M_ = M_tc * g_theta
    return M_


def findM_i(M: torch.Tensor, M_tc: torch.Tensor, 
           chi_i: torch.Tensor, psi_i: torch.Tensor, 
           N: torch.Tensor) -> torch.Tensor:
    r"""
    Compute image stress ratio M_i
    
    The image stress ratio is computed as:
    
    .. math::
        M_i = M \left(1 - \frac{\chi_i N |\psi_i|}{M_{tc}}\right)
    
    Args:
        M: Critical state stress ratio :math:`M`
        M_tc: Critical state stress ratio in triaxial compression :math:`M_{tc}`
        chi_i: Dilatancy parameter :math:`\chi_i`
        psi_i: State parameter at image state :math:`\psi_i`
        N: Material constant :math:`N`
        
    Returns:
        torch.Tensor: Image stress ratio :math:`M_i`
        
    Note:
        The absolute value :math:`|\psi_i|` may require smooth approximations 
        for gradient-based optimization if :math:`\psi_i` frequently crosses zero.
    """
    Mi_ = M * (1 - chi_i * N * torch.abs(psi_i) / M_tc)
    return Mi_



def findM_itc(N: torch.Tensor, chi_i: torch.Tensor, psi_i: torch.Tensor, M_tc: torch.Tensor) -> torch.Tensor:
    """
    Compute M_itc
    
    The triaxial compression image stress ratio is:
    
    .. math::
        M_{itc} = M_{tc} - N \chi_i |\psi_i|
    
    Args:
        N: Material constant :math:`N`
        chi_i: Dilatancy parameter :math:`\chi_i`
        psi_i: State parameter at image state :math:`\psi_i`
        M_tc: Critical state stress ratio in triaxial compression :math:`M_{tc}`
        
    Returns:
        Mitc: M_itc parameter :math:`M_{itc}`
    """
    Mitc = M_tc - N * chi_i * torch.abs(psi_i)
    return Mitc


def findchi_i(M_tc: torch.Tensor, Chi_tc: torch.Tensor, Lambda: torch.Tensor) -> torch.Tensor:
    r"""
    Compute chi_i parameter
    
    The dilatancy parameter is computed as:
    
    .. math::
        \chi_i = \frac{\chi_{tc}}{1 - \frac{\Lambda \chi_{tc}}{M_{tc}}}
    
    Args:
        M_tc: Critical state stress ratio in triaxial compression :math:`M_{tc}`
        Chi_tc: Chi parameter in triaxial compression :math:`\chi_{tc}`
        Lambda: Material constant :math:`\Lambda`
        
    Returns:
        torch.Tensor: Chi_i parameter :math:`\chi_i`
    """
    chii = Chi_tc / (1 - Lambda * Chi_tc / M_tc)
    return chii


def findpsipsii(Gamma: torch.Tensor, Lambda: torch.Tensor, p: torch.Tensor, 
                p_i: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute psi and psi_i parameters
    
    The state parameters are computed as:
    
    .. math::
        e_c = \Gamma - \Lambda \ln(p)
        
    .. math::
        \psi = e - e_c
        
    .. math::
        \psi_i = e - \Gamma + \Lambda \ln(p_i)
    
    Args:
        Gamma: Material constant :math:`\Gamma`
        Lambda: Material constant :math:`\Lambda`
        p: Mean pressure :math:`p`
        p_i: Image mean pressure :math:`p_i`
        e: Void ratio :math:`e`
        
    Returns:
        psi: State parameter :math:`\psi`
        psi_i: State parameter at image state :math:`\psi_i`
    """
    e_c = Gamma - Lambda * torch.log(p)
    psi = e - e_c
    
    # psi_i = psi + Lambda * torch.log(p_i / p)
    # Mathematically simplifies to:
    psi_i = e - Gamma + Lambda * torch.log(p_i)
    
    return psi, psi_i


def findp_imax(chi_i: torch.Tensor, psi_i: torch.Tensor, p: torch.Tensor, M_itc: torch.Tensor) -> torch.Tensor:
    """
    Compute p_imax parameter
    
    The maximum image pressure is computed as:
    
    .. math::
        D_{min} = \chi_i \psi_i
        
    .. math::
        p_{imax} = p \exp( -\frac{D_{min}}{M_{itc}} )
    
    Args:
        chi_i: Dilatancy parameter :math:`\chi_i`
        psi_i: State parameter at image state :math:`\psi_i`
        p: Mean pressure :math:`p`
        M_itc: M_itc parameter :math:`M_{itc}`
        
    Returns:
        p_imax: Maximum image pressure :math:`p_{imax}`
    """
    D_min = chi_i * psi_i
    p_imax = p * torch.exp(-D_min / M_itc)
    return p_imax


def findF(p: torch.Tensor, q: torch.Tensor, M_i: torch.Tensor, p_i: torch.Tensor) -> torch.Tensor:
    r"""
    Compute yield surface value F
    
    The yield surface is defined as:
    
    .. math::
        F = \frac{q}{p} - M_i \left(1 - \ln\left(\frac{p}{p_i}\right)\right)
    
    Args:
        p: Mean pressure :math:`p`
        q: von Mises stress :math:`q`
        M_i: Image stress ratio :math:`M_i`
        p_i: Image mean pressure :math:`p_i`
        
    Returns:
        torch.Tensor: Yield surface value :math:`F`
    """
    F_ = q / p - M_i * (1 - torch.log(p / p_i))
    return F_


def finddFdsigma(sigma_vec: torch.Tensor, M_i: torch.Tensor, p_i: torch.Tensor) -> torch.Tensor:
    r"""
    Compute derivative of yield function with respect to stress using automatic differentiation
    
    Computes:
    
    .. math::
        \frac{\partial F}{\partial \boldsymbol{\sigma}} = \nabla_{\boldsymbol{\sigma}} F
    
    where :math:`F` is the yield function and :math:`\boldsymbol{\sigma}` is the stress vector.
    
    Args:
        sigma_vec: Stress vector in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]` (tensor of shape (6,))
        M_i: Image stress ratio :math:`M_i` (scalar tensor)
        p_i: Image mean pressure :math:`p_i` (scalar tensor)
        
    Returns:
        torch.Tensor: Derivative of yield function with respect to stress :math:`\frac{\partial F}{\partial \boldsymbol{\sigma}}` (tensor of shape (6,))
    """
    # Make sure sigma_vec requires gradients
    if not sigma_vec.requires_grad:
        raise ValueError("sigma_vec must require gradients")
    
    # Get p, q from stress vector
    p, q = stress_decomp(sigma_vec)

    # Compute yield function F
    F = findF(p, q, M_i, p_i)

    # Use autograd.grad to compute dF/dsigma_vec
    dfdsig = torch.autograd.grad(F, sigma_vec, create_graph=True)[0]

    return dfdsig


def findp_ipsi_iM_i(N, chi_i, lambd, M_tc, psi, p, q, M):
    """
    Find p_i, psi_i and M_i using PyTorch (differentiable version)
    
    Solves the system of equations for the image state parameters by finding roots of:
    
    .. math::
        F = 0 \quad \text{(yield surface condition)}
        
    where the unknowns are solved iteratively through quadratic equations involving 
    the dimensionless parameter :math:`x = \ln(p_i/p)`.
    
    The solution involves:
    
    .. math::
        a = \frac{N \chi_i \lambda}{M_{tc}}
        
    .. math::
        ax^2 + bx + c = 0
    
    with different coefficients depending on the sign of :math:`\psi_i`.
    
    Args:
        N: Material constant :math:`N` (scalar)
        chi_i: Dilatancy parameter :math:`\chi_i` (scalar)
        lambd: Lambda parameter :math:`\lambda` (scalar)
        M_tc: Critical state stress ratio in triaxial compression :math:`M_{tc}` (scalar)
        psi: State parameter :math:`\psi` (scalar)
        p: Mean pressure :math:`p` (scalar)
        q: von Mises stress :math:`q` (scalar)
        M: Critical state stress ratio :math:`M` (scalar)
        
    Returns:
        p_i: Image mean pressure :math:`p_i` (scalar)
        psii: State parameter at image state :math:`\psi_i` (scalar)
        Mi: Image stress ratio :math:`M_i` (scalar)
        
    Raises:
        RuntimeError: If no valid roots are found
        ValueError: If input parameters have non-reasonable values
    """
    # Check for non-reasonable input values
    if p <= 0:
        raise ValueError(f"Mean pressure p must be positive, got {p}")
    if M_tc <= 0:
        raise ValueError(f"M_tc must be positive, got {M_tc}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    
    # Set up terms to be used in quadratic equations
    a = N * chi_i * lambd / M_tc
    
    # Check if a is near zero which would make quadratic solution unreliable
    if torch.abs(a) < 1e-6:
        raise ValueError(f"Parameter 'a' is too small ({a}), causing numerical instability")
    
    psi_term = N * chi_i * psi / M_tc
    b1, b2 = a - 1 + psi_term, a + 1 + psi_term
    c1 = psi_term + (q / p) / M - 1
    c2 = psi_term - (q / p) / M + 1

    # Check discriminants before clamping to identify potential issues
    disc1_raw = b1**2 - 4 * a * c1
    disc2_raw = b2**2 - 4 * a * c2
    
    if disc1_raw < 0 and torch.abs(disc1_raw) > 1e-6:
        raise ValueError(f"First discriminant is significantly negative: {disc1_raw}")
    if disc2_raw < 0 and torch.abs(disc2_raw) > 1e-6:
        raise ValueError(f"Second discriminant is significantly negative: {disc2_raw}")

    # Calculate discriminants with clamping to ensure non-negative values
    disc1 = torch.clamp(disc1_raw, min=0.0)
    disc2 = torch.clamp(disc2_raw, min=0.0)

    # Calculate all possible roots
    x = torch.stack([
        (-b1 + torch.sqrt(disc1)) / (2 * a),
        (-b1 - torch.sqrt(disc1)) / (2 * a),
        (-b2 + torch.sqrt(disc2)) / (2 * a),
        (-b2 - torch.sqrt(disc2)) / (2 * a),
    ])  # shape [4]

    # Check for NaN or Inf in roots
    if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
        raise ValueError(f"Found NaN or Inf in computed roots: {x}")

    # Calculate corresponding values for each root
    psii = x * lambd + psi
    p_i = p * torch.exp(x)
    Mi = M * (1 - N * chi_i * torch.abs(psii) / M_tc)

    # Check if any Mi value is invalid
    if torch.any(torch.isnan(Mi)) or torch.any(torch.isinf(Mi)):
        raise ValueError(f"Found NaN or Inf in computed Mi values: {Mi}")

    # Calculate yield function residuals based on sign of psii
    F_pos = a * x**2 + (a - 1 + psi_term) * x + c1
    F_neg = a * x**2 + (a + 1 + psi_term) * x + c2
    F = torch.where(psii > 0, F_pos, F_neg)

    # Determine which roots are valid
    ok = (torch.abs(F) < 1e-6) & (Mi < M) & (Mi > 0.1 * M)
    
    # Raise an error if no valid roots are found
    if not torch.any(ok):
        # Try with relaxed tolerance to match NumPy implementation behavior
        # NOTE: Why Mi > 0.1 * M?
        ok_relaxed = (torch.abs(F) < 1e-4) & (Mi < M) & (Mi > 0.1 * M)
        if not torch.any(ok_relaxed):
            raise RuntimeError(f"No valid roots found in findp_ipsi_iM_i. Values: x={x}, F={F}, Mi={Mi}")
        ok = ok_relaxed
    
    # Choose the best root that most closely matches the NumPy implementation's behavior
    # In the NumPy version, roots are checked in order and the first valid one is returned
    # We'll mimic this by prioritizing roots with smallest residual
    F_abs = torch.abs(F)
    # Create a large penalty for invalid roots
    F_masked = torch.where(ok, F_abs, 1e10 * torch.ones_like(F_abs))
    # Find root with smallest residual
    best_idx = torch.argmin(F_masked)
    
    p_i = p_i[best_idx]
    psii = psii[best_idx]
    Mi = Mi[best_idx]
    
    # Final check for reasonable values
    if p_i <= 0:
        raise ValueError(f"Computed p_i is non-positive: {p_i}")
    if Mi <= 0:
        raise ValueError(f"Computed Mi is non-positive: {Mi}")

    return p_i, psii, Mi


def findC_p(C: torch.Tensor, dfdsigma: torch.Tensor, dfdepsilon_dfdsigma: torch.Tensor) -> torch.Tensor:
    r"""
    Compute elasto-plastic tangent stiffness modulus
    
    The elasto-plastic tangent operator is computed as:
    
    .. math::
        \mathbf{C}^{ep} = \mathbf{C} - \frac{\mathbf{C} \frac{\partial F}{\partial \boldsymbol{\sigma}} \otimes \frac{\partial F}{\partial \boldsymbol{\sigma}} \mathbf{C}}{\frac{\partial F}{\partial \boldsymbol{\sigma}} : \mathbf{C} : \frac{\partial F}{\partial \boldsymbol{\sigma}} - \frac{\partial F}{\partial \boldsymbol{\varepsilon}^p} : \frac{\partial F}{\partial \boldsymbol{\sigma}}}
    
    Args:
        C: Elastic tangent operator :math:`\mathbf{C}` (tensor of shape (6, 6))
        dfdsigma: Derivative of yield function with respect to stress :math:`\frac{\partial F}{\partial \boldsymbol{\sigma}}` (tensor of shape (6,))
        dfdepsilon_dfdsigma: :math:`\frac{\partial F}{\partial \boldsymbol{\varepsilon}^p} : \frac{\partial F}{\partial \boldsymbol{\sigma}}` (scalar tensor)
        
    Returns:
        torch.Tensor: Elasto-plastic tangent stiffness modulus :math:`\mathbf{C}^{ep}` (tensor of shape (6, 6))
    """
    # Compute dF/dsigma ^ T : C : dF/dsigma
    dfdsig_C_dfdsig = torch.matmul(torch.matmul(dfdsigma, C), dfdsigma)

    denom = dfdsig_C_dfdsig - dfdepsilon_dfdsigma
    
    Cp = C - torch.outer(torch.matmul(C, dfdsigma), torch.matmul(C, dfdsigma)) / denom
    return Cp


def finddFdepsilon_p(
    sigma_vec: torch.Tensor, 
    dfdep_params: torch.Tensor, 
    dfdsigma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Compute derivative of yield function with respect to plastic strain
    
    Computes the hardening law derivatives:
    
    .. math::
        \frac{\partial F}{\partial \boldsymbol{\varepsilon}^p} : \frac{\partial F}{\partial \boldsymbol{\sigma}} = \frac{\partial F}{\partial p_i} \frac{\partial p_i}{\partial \varepsilon_d^p} \frac{\partial F}{\partial q}
    
    where:
    
    .. math::
        \frac{\partial p_i}{\partial \varepsilon_d^p} = H \frac{M_i}{M_{itc}} (p_{imax} - p_i) \frac{p}{p_i}
        
    .. math::
        H = H_0 - H_y \psi
    
    Args:
        sigma_vec: Stress vector in Voigt notation :math:`[\sigma_{11}, \sigma_{22}, \sigma_{33}, \sigma_{12}, \sigma_{13}, \sigma_{23}]` (tensor of shape (6,))
        dfdep_params: Parameters for computing :math:`\frac{\partial F}{\partial \boldsymbol{\varepsilon}^p}` (tensor of shape (13,))
                     :math:`[H_0, H_y, \psi, N, M, M_i, M_{tc}, M_{itc}, p_i, p_{imax}, \chi_i, \psi_i, \lambda]`
        dfdsigma: Derivative of yield function with respect to stress :math:`\frac{\partial F}{\partial \boldsymbol{\sigma}}` (tensor of shape (6,))
        
    Returns:
        tuple: A tuple containing:
        
            - **dfdep_dfdsig** (*torch.Tensor*) - :math:`\frac{\partial F}{\partial \boldsymbol{\varepsilon}^p} : \frac{\partial F}{\partial \boldsymbol{\sigma}}` (scalar tensor)
            - **dpideps_pd** (*torch.Tensor*) - :math:`\frac{\partial p_i}{\partial \varepsilon_d^p}` (scalar tensor)
            - **dfdpi** (*torch.Tensor*) - :math:`\frac{\partial F}{\partial p_i}` (scalar tensor)
            - **dfdq_** (*torch.Tensor*) - :math:`\frac{\partial F}{\partial q}` (scalar tensor)
    """
    # Unpack parameters
    H0 = dfdep_params[0]
    Hy = dfdep_params[1]
    psi = dfdep_params[2]
    N = dfdep_params[3]
    M = dfdep_params[4]
    M_i = dfdep_params[5]
    M_tc = dfdep_params[6]
    M_itc = dfdep_params[7]
    p_i = dfdep_params[8]
    p_imax = dfdep_params[9]
    chi_i = dfdep_params[10]
    psi_i = dfdep_params[11]
    lamb = dfdep_params[12]

    H = H0 - Hy*psi
    p, q = stress_decomp(sigma_vec)
    
    # Compute dF/dpi = dF/dpi + dF/dMi * dMi/dpi
    # First term
    parfparpi = -p * M_i / p_i

    # Second term
    dfdMi = -q / M_i

    # Third term - make differentiable using torch.sign
    # Handle the case where psi_i is zero by using a small epsilon
    eps = 1e-10
    psi_term = torch.sign(psi_i + eps)

    dMidpi = -(M / M_tc) * N * chi_i * psi_term * lamb / p_i

    # Combine terms
    dfdpi = parfparpi + dfdMi * dMidpi

    # Compute dpi/depsilon_d^p
    dpideps_pd = H * (M_i / M_itc) * (p_imax - p_i) * (p / p_i)

    # Compute df/dsigma : depsilon_d^p/depsilon^p, which is the deviatoric
    # projection of df/dsigma
    _, dfdq_ = vol_dev(sigma_vec, dfdsigma)  # Deviatoric projection of dfdsigma

    # Use terms to find dF/dsigma : dF/depsilon^p
    # NOTE: Why this line don't use dF/dsigma directly?
    # NOTE: Why we don't compute df/depsilon^p directly?
    dfdep_dfdsig = dfdpi * dpideps_pd * dfdq_
    
    return dfdep_dfdsig, dpideps_pd, dfdpi, dfdq_


def find_dlambda_p(C: torch.Tensor, dfdsigma: torch.Tensor, dfdepsilon_dfdsigma: torch.Tensor, depsilon: torch.Tensor) -> torch.Tensor:
    r"""
    Compute change in plastic multiplier (differentiable PyTorch version)
    
    The plastic multiplier increment is computed from the consistency condition:
    
    .. math::
        d\lambda = \frac{\frac{\partial F}{\partial \boldsymbol{\sigma}} : \mathbf{C} : d\boldsymbol{\varepsilon}}{\frac{\partial F}{\partial \boldsymbol{\sigma}} : \mathbf{C} : \frac{\partial F}{\partial \boldsymbol{\sigma}} - \frac{\partial F}{\partial \boldsymbol{\varepsilon}^p} : \frac{\partial F}{\partial \boldsymbol{\sigma}}}
    
    Args:
        C: Elastic tangent operator :math:`\mathbf{C}` (tensor of shape (6, 6))
        dfdsigma: Derivative of yield function with respect to stress :math:`\frac{\partial F}{\partial \boldsymbol{\sigma}}` (tensor of shape (6,))
        dfdepsilon_dfdsigma: :math:`\frac{\partial F}{\partial \boldsymbol{\varepsilon}^p} : \frac{\partial F}{\partial \boldsymbol{\sigma}}` (scalar tensor)
        depsilon: Strain increment :math:`d\boldsymbol{\varepsilon}` (tensor of shape (6,))
        
    Returns:
        torch.Tensor: Change in plastic multiplier :math:`d\lambda` (scalar tensor)
    """
    # Compute numerator: dfdsigma : C : depsilon
    term1 = torch.matmul(dfdsigma, torch.matmul(C, depsilon))
    
    # Compute denominator: dfdsigma : C : dfdsigma - dfdepsilon_dfdsigma
    term2 = torch.matmul(dfdsigma, torch.matmul(C, dfdsigma)) - dfdepsilon_dfdsigma

    # Compute dlambda_p
    dlambda_p = term1 / term2
    
    return dlambda_p

