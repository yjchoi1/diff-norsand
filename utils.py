import numpy as np
from typing import Tuple

def stress_decomp(sigma: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean pressure p and von Mises stress q from stress vector
    
    Args:
        sigma: Stress vector in Voigt notation [σ11, σ22, σ33, σ12, σ13, σ23]
        
    Returns:
        p_: Mean pressure (1/3 of first invariant)
        q_: von Mises stress (square root of 3/2 of second invariant of deviatoric stress)
    """
    # Tension cutoffs
    sigma_copy = sigma.copy()
    for i in range(3):
        if sigma_copy[i] < 0.1:
            sigma_copy[i] = 0.1
    
    p_ = (1/3) * np.sum(sigma_copy[0:3])
    q_ = np.sqrt(0.5 * ((sigma_copy[0] - sigma_copy[1])**2 + 
                        (sigma_copy[1] - sigma_copy[2])**2 + 
                        (sigma_copy[2] - sigma_copy[0])**2 + 
                        6 * (sigma_copy[3]**2 + sigma_copy[4]**2 + sigma_copy[5]**2)))
    return p_, q_

def find_sJ2J3(sigma: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Compute deviatoric stress tensor s, and J2, J3 invariants
    
    Args:
        sigma: Stress vector in Voigt notation
        
    Returns:
        s: Deviatoric stress vector
        J_2: Second invariant of deviatoric stress
        J_3: Third invariant of deviatoric stress
    """
    p, _ = stress_decomp(sigma)
    s = sigma.copy().astype(float)  # Ensure we're working with float arrays
    s[0:3] -= p
    
    J_2 = (1/2) * (s[0]**2 + s[1]**2 + s[2]**2 + 2 * (s[3]**2 + s[4]**2 + s[5]**2))
    J_3 = (s[0] * s[1] * s[2] - s[0] * s[5]**2 - s[2] * s[3]**2 - 
           s[1] * s[4]**2 + 2 * s[3] * s[4] * s[5])
    
    return s, J_2, J_3

def vol_dev(sigma: np.ndarray, epsilon: np.ndarray) -> Tuple[float, float]:
    """
    Compute volumetric strain and deviatoric strain in Voigt notation
    
    Args:
        sigma: Stress vector in Voigt notation
        epsilon: Strain vector in Voigt notation
        
    Returns:
        e_v: Volumetric strain
        e_q: Deviatoric strain
    """
    p, q = stress_decomp(sigma)
    e_v = np.sum(epsilon[0:3])
    
    # Handle special case to avoid division by zero
    if q <= 1e-6:
        e_q = 0
        return e_v, e_q
    
    e_q = ((sigma[0] - p) / q * epsilon[0] + 
           (sigma[1] - p) / q * epsilon[1] + 
           (sigma[2] - p) / q * epsilon[2] + 
           2 * sigma[3] / q * epsilon[3] + 
           2 * sigma[4] / q * epsilon[4] + 
           2 * sigma[5] / q * epsilon[5])
    
    return e_v, e_q

def lode_angle(sigma: np.ndarray) -> float:
    """
    Compute Lode angle from stress vector
    
    Args:
        sigma: Stress vector in Voigt notation
        
    Returns:
        theta: Lode angle
    """
    _, J_2, J_3 = find_sJ2J3(sigma)
    
    if J_2 == 0:
        return 0
    
    sin3theta = np.real(3 * np.sqrt(3) / 2) * (J_3 / (J_2)**(3/2))
    
    if sin3theta > 0.99:
        sin3theta = 1
    if sin3theta < -0.99:
        sin3theta = -1
    
    theta = np.real((1/3) * np.arcsin(sin3theta))
    
    if theta > np.pi/6:
        theta = np.pi/6
    if theta < -np.pi/6:
        theta = -np.pi/6
    
    return theta

def findCe(G_max: float, p: float, p_ref: float, m: float, nu: float) -> np.ndarray:
    """
    Compute elastic tangent operator
    
    Args:
        G_max: Maximum shear modulus
        p: Mean pressure
        p_ref: Reference pressure
        m: Pressure exponent
        nu: Poisson's ratio
        
    Returns:
        C_e: Elastic tangent operator matrix
    """
    G = G_max * (p / p_ref) ** m
    K = ((2 * (1 + nu)) / (3 * (1 - 2 * nu))) * G
    C_e = np.zeros((6, 6))
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

def voigt_norm(vec: np.ndarray) -> float:
    """
    Compute L2 norm of Voigt vector with proper scaling
    
    Args:
        vec: Vector in Voigt notation
        
    Returns:
        norm_val: L2 norm of vector
    """
    scaled_vec = vec.copy()
    scaled_vec[3:6] = np.sqrt(2) * vec[3:6]  # Shear components scaled by sqrt(2)
    norm_val = np.sqrt(np.sum(scaled_vec**2))
    return norm_val

def dJ2J3(sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute derivatives of J2, J3 invariants with respect to sigma
    
    Args:
        sigma: Stress vector in Voigt notation
        
    Returns:
        dJ2_dsigma: Derivative of J2 with respect to sigma
        dJ3_dsigma: Derivative of J3 with respect to sigma
    """
    s, _, _ = find_sJ2J3(sigma)
    dJ2_dsigma = np.array([s[0], s[1], s[2], 2*sigma[3], 2*sigma[4], 2*sigma[5]])

    dJ3_dsigma = np.array([
        -(1/3) * s[0] * s[1] - (1/3) * s[0] * s[2] + (2/3) * s[1] * s[2] - 
        (2/3) * s[5]**2 + (1/3) * s[4]**2 + (1/3) * s[3]**2,

        -(1/3) * s[0] * s[1] + (2/3) * s[0] * s[2] - (1/3) * s[1] * s[2] + 
        (1/3) * s[5]**2 - (2/3) * s[4]**2 + (1/3) * s[3]**2,

        (2/3) * s[0] * s[1] - (1/3) * s[0] * s[2] - (1/3) * s[1] * s[2] + 
        (1/3) * s[5]**2 + (1/3) * s[4]**2 - (2/3) * s[3]**2,
        
        -2 * s[2] * s[3] + 2 * s[5] * s[4],

        -2 * s[1] * s[4] + 2 * s[3] * s[5],

        -2 * s[0] * s[5] + 2 * s[3] * s[4]
    ])
    
    return dJ2_dsigma, dJ3_dsigma 