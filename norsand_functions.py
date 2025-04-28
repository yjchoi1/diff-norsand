import numpy as np
from typing import Tuple, List, Union

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
    s = sigma.copy()
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

def findM(theta: float, M_tc: float) -> float:
    """
    Compute M based on Lode angle and M_tc
    
    Args:
        theta: Lode angle
        M_tc: Critical state stress ratio in triaxial compression
        
    Returns:
        M_: Critical state stress ratio at given Lode angle
    """
    g_theta = 1 - (M_tc / (3 + M_tc)) * np.cos((3 * theta) / 2 + np.pi / 4)
    M_ = M_tc * g_theta
    return M_

def findM_i(M: float, M_tc: float, chi_i: float, psi_i: float, N: float) -> float:
    """
    Compute image stress ratio M_i
    
    Args:
        M: Critical state stress ratio
        M_tc: Critical state stress ratio in triaxial compression
        chi_i: Dilatancy parameter
        psi_i: State parameter at image state
        N: Material constant
        
    Returns:
        Mi_: Image stress ratio
    """
    Mi_ = M * (1 - chi_i * N * np.abs(psi_i) / M_tc)
    return Mi_

def findM_itc(N: float, chi_i: float, psi_i: float, M_tc: float) -> float:
    """
    Compute M_itc
    
    Args:
        N: Material constant
        chi_i: Dilatancy parameter
        psi_i: State parameter at image state
        M_tc: Critical state stress ratio in triaxial compression
        
    Returns:
        Mitc: M_itc parameter
    """
    Mitc = M_tc - N * chi_i * np.abs(psi_i)
    return Mitc

def findchi_i(M_tc: float, Chi_tc: float, Lambda: float) -> float:
    """
    Compute chi_i parameter
    
    Args:
        M_tc: Critical state stress ratio in triaxial compression
        Chi_tc: Chi parameter in triaxial compression
        Lambda: Material constant
        
    Returns:
        chii: Chi_i parameter
    """
    chii = Chi_tc / (1 - Lambda * Chi_tc / M_tc)
    return chii

def findpsipsii(Gamma: float, Lambda: float, p: float, p_i: float, e: float) -> Tuple[float, float]:
    """
    Compute psi and psi_i parameters
    
    Args:
        Gamma: Material constant
        Lambda: Material constant
        p: Mean pressure
        p_i: Image mean pressure
        e: Void ratio
        
    Returns:
        psi: State parameter
        psi_i: State parameter at image state
    """
    e_c = Gamma - Lambda * np.log(p)
    psi = e - e_c
    psi_i = psi + Lambda * np.log(p_i / p)
    return psi, psi_i

def findp_imax(chi_i: float, psi_i: float, p: float, M_itc: float) -> float:
    """
    Compute p_imax parameter
    
    Args:
        chi_i: Dilatancy parameter
        psi_i: State parameter at image state
        p: Mean pressure
        M_itc: M_itc parameter
        
    Returns:
        p_imax: Maximum image pressure
    """
    D_min = chi_i * psi_i
    p_imax = p * np.exp(-D_min / M_itc)
    return p_imax

def findF(p: float, q: float, M_i: float, p_i: float) -> float:
    """
    Compute yield surface value F
    
    Args:
        p: Mean pressure
        q: von Mises stress
        M_i: Image stress ratio
        p_i: Image mean pressure
        
    Returns:
        F_: Yield surface value
    """
    F_ = q / p - M_i * (1 - np.log(p / p_i))
    return F_

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

def finddFdsigma(sigma_vec: np.ndarray, dfdsig_params: np.ndarray) -> np.ndarray:
    """
    Compute derivative of yield function with respect to stress
    
    Args:
        sigma_vec: Stress vector in Voigt notation
        dfdsig_params: Parameters for computing dF/dsigma
        
    Returns:
        dfdsig: Derivative of yield function with respect to stress
    """
    # Unpack parameters
    theta = dfdsig_params[0]
    N = dfdsig_params[1]
    M_i = dfdsig_params[2]
    M_tc = dfdsig_params[3]
    p_i = dfdsig_params[4]
    chi_i = dfdsig_params[5]
    psi_i = dfdsig_params[6]

    # Find p,q from sigma vector
    p, q = stress_decomp(sigma_vec)

    # Compute deviatoric stress tensor, J2, J3 from stress tensor
    s, J2, J3 = find_sJ2J3(sigma_vec)

    # Compute dF/dp
    dfdp = M_i - q/p

    # Compute dF/dtheta
    dfdtheta = -(q/M_i) * (3/2) * (1 - (N * chi_i * np.abs(psi_i) / M_tc)) * (M_tc**2/(3 + M_tc)) * np.sin(3*theta/2 + np.pi/4)

    # Compute dp/dsigma
    dpdsigma = (1 / 3) * np.array([1, 1, 1, 0, 0, 0])

    # Compute dq/dsigma
    if q == 0:
        dqdsigma = 0
    else:
        dqdsigma = (3 / (2 * q)) * s

    # Compute dtheta/dsigma
    # Compute s^2 in Voigt notation
    s_squared = np.array([
        s[0]**2 + s[5]**2 + s[4]**2,  # s11^2
        s[1]**2 + s[5]**2 + s[3]**2,  # s22^2
        s[2]**2 + s[3]**2 + s[4]**2,  # s33^2
        s[5]*(s[0] + s[1]),           # s12^2
        s[4]*(s[0] + s[2]),           # s13^2
        s[3]*(s[1] + s[2]),           # s23^2
    ])
    
    tr_s_squared = np.sum(s_squared[0:3])
    if q < 1e-6 or np.cos(3*theta) < 1e-6:
        dthetadsigma = 0
    else:
        dthetadsigma = (np.sqrt(3) / (2*np.cos(3*theta)*np.real(np.sqrt(J2**3)))) * (
            s_squared - (1/3)*tr_s_squared*np.array([1, 1, 1, 0, 0, 0]) - (3/2)*(J3/J2)*s)

    # Use terms to find dF/dsigma using chain rule
    dfdsig = dfdp * dpdsigma + 1 * dqdsigma + dfdtheta * dthetadsigma
    return dfdsig

def finddFdepsilon_p(sigma_vec: np.ndarray, dfdep_params: np.ndarray, dfdsigma: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute derivative of yield function with respect to plastic strain
    
    Args:
        sigma_vec: Stress vector in Voigt notation
        dfdep_params: Parameters for computing dF/depsilon_p
        dfdsigma: Derivative of yield function with respect to stress
        
    Returns:
        dfdep_dfdsig: dF/depsilon_p : dF/dsigma
        dpideps_pd: dpi/depsilon_d^p
        dfdpi: dF/dpi
        dfdq_: dF/dq
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

    # Third term
    if psi_i/np.abs(psi_i) > 0:
        psi_term = 1
    elif psi_i/np.abs(psi_i) < 0:
        psi_term = -1
    else:
        psi_term = 0

    dMidpi = -(M / M_tc) * N * chi_i * psi_term * lamb / p_i

    # Combine terms
    dfdpi = parfparpi + dfdMi * dMidpi

    # Compute dpi/depsilon_d^p
    dpideps_pd = H * (M_i / M_itc) * (p_imax - p_i) * (p / p_i)

    # Compute df/dsigma : depsilon_d^p/depsilon^p, which is the deviatoric
    # projection of df/dsigma
    _, dfdq_ = vol_dev(sigma_vec, dfdsigma)  # Deviatoric projection of dfdsigma

    # Use terms to find dF/dsigma : dF/depsilon^p 
    dfdep_dfdsig = dfdpi * dpideps_pd * dfdq_
    
    return dfdep_dfdsig, dpideps_pd, dfdpi, dfdq_

def findC_p(C: np.ndarray, dfdsigma: np.ndarray, dfdepsilon_dfdsigma: float) -> np.ndarray:
    """
    Compute elasto-plastic tangent stiffness modulus
    
    Args:
        C: Elastic tangent operator
        dfdsigma: Derivative of yield function with respect to stress
        dfdepsilon_dfdsigma: dF/depsilon_p : dF/dsigma
        
    Returns:
        Cp: Elasto-plastic tangent stiffness modulus
    """
    # Compute dF/dsigma ^ T : C : dF/dsigma
    dfdsig_C_dfdsig = (dfdsigma @ C) @ dfdsigma

    denom = dfdsig_C_dfdsig - dfdepsilon_dfdsigma
    
    Cp = C - np.outer((C @ dfdsigma), (C @ dfdsigma)) / denom
    return Cp

def find_dlambda_p(C: np.ndarray, dfdsigma: np.ndarray, dfdepsilon_dfdsigma: float, depsilon: np.ndarray) -> float:
    """
    Compute change in plastic multiplier
    
    Args:
        C: Elastic tangent operator
        dfdsigma: Derivative of yield function with respect to stress
        dfdepsilon_dfdsigma: dF/depsilon_p : dF/dsigma
        depsilon: Strain increment
        
    Returns:
        dlambda_p: Change in plastic multiplier
    """
    term1 = np.dot(dfdsigma, C @ depsilon)
    term2 = np.dot(dfdsigma, C @ dfdsigma) - dfdepsilon_dfdsigma

    dlambda_p = term1 / term2
    return dlambda_p

def findp_ipsi_iM_i(N: float, chi_i: float, lamb: float, M_tc: float, psi: float, 
                    p: float, q: float, M: float) -> Tuple[float, float, float]:
    """
    Find p_i, psi_i and M_i
    
    Args:
        N: Material constant
        chi_i: Dilatancy parameter
        lamb: Lambda parameter
        M_tc: Critical state stress ratio in triaxial compression
        psi: State parameter
        p: Mean pressure
        q: von Mises stress
        M: Critical state stress ratio
        
    Returns:
        p_i: Image mean pressure
        psii: State parameter at image state
        Mi: Image stress ratio
    """
    # Set up terms to be used in four quadratic options
    a = N * chi_i * lamb / M_tc
    psi_term = N * chi_i * psi / M_tc
    b1 = a - 1 + psi_term
    b2 = a + 1 + psi_term
    c1 = psi_term + (q / p) / M - 1
    c2 = psi_term - (q / p) / M + 1
    
    # Find roots to quadratic equations
    x1a = (-b1 + np.sqrt(b1**2 - 4*a*c1)) / (2*a)
    x1b = (-b1 - np.sqrt(b1**2 - 4*a*c1)) / (2*a)
    x2a = (-b2 + np.sqrt(b2**2 - 4*a*c2)) / (2*a)
    x2b = (-b2 - np.sqrt(b2**2 - 4*a*c2)) / (2*a)
    
    # Initialize return values with defaults
    p_i = None
    psii = None
    Mi = None
    
    # Loop through each root
    x_list = [x1a, x1b, x2a, x2b]
    for x_test in x_list:
        psii_test = x_test * lamb + psi
        pi_test = p * np.exp(x_test)
        Mi_test = M * (1 - N * chi_i * np.abs(lamb*x_test + psi) / M_tc)

        # Check against yield function which runs through current stress state
        if psii_test > 0:
            F = a * x_test**2 + (a - 1 + psi_term) * x_test + (psi_term + (q / p) / M - 1)
            if np.abs(F) <= 1e-6 and Mi_test < M and Mi_test > 0.1 * M:
                psii = psii_test
                p_i = pi_test
                Mi = Mi_test
        elif psii_test < 0:
            F = a * x_test**2 + (a + 1 + psi_term) * x_test + (psi_term - (q / p) / M + 1)
            if np.abs(F) <= 1e-6 and Mi_test < M and Mi_test > 0.1 * M:
                psii = psii_test
                p_i = pi_test
                Mi = Mi_test
    
    return p_i, psii, Mi 

def pegasus(alpha0: float, alpha1: float, FTOL: float, dsig_trial: np.ndarray, 
            sigma_vec: np.ndarray, M_i: float, p_i: float, flag: int = 0, 
            F1_ep_unl: float = 0, MAXITS: int = 10) -> float:
    """
    Pegasus algorithm for elastic to plastic transition
    
    Args:
        alpha0: Initial alpha value
        alpha1: Final alpha value
        FTOL: Tolerance for yield function
        dsig_trial: Trial stress increment
        sigma_vec: Initial stress vector
        M_i: Image stress ratio
        p_i: Image mean pressure
        flag: Flag for elastic unloading
        F1_ep_unl: Yield function value for elastic unloading
        MAXITS: Maximum number of iterations
        
    Returns:
        alpha_: Optimal alpha value
    """
    # Compute yield surfaces F0 and F1
    sigma0 = sigma_vec + alpha0 * dsig_trial
    p0, q0 = stress_decomp(sigma0)
    F0 = findF(p0, q0, M_i, p_i)
    
    # Elastic to plastic
    if flag == 0:
        sigma1 = sigma_vec + alpha1 * dsig_trial
        p1, q1 = stress_decomp(sigma1)
        F1 = findF(p1, q1, M_i, p_i)
    # Elastic unloading involved
    else:
        F1 = F1_ep_unl

    # Begin loop, iterating alpha values to find correct alpha
    for i in range(1, MAXITS + 1):
        alpha = alpha1 - F1 * (alpha1 - alpha0) / (F1 - F0)
        sigmaprime = sigma_vec + alpha * dsig_trial
        pprime, qprime = stress_decomp(sigmaprime)
        Fprime = findF(pprime, qprime, M_i, p_i)

        # Exit condition
        if abs(Fprime) < FTOL:
            return alpha
        elif Fprime * F1 < 0:
            alpha1 = alpha0
            F1 = F0
        else:
            F1 = F1 * F0 / (F0 + Fprime)
        
        alpha0 = alpha
        F0 = Fprime
    
    # If we reach here without convergence, return the last alpha
    return alpha

def pegasus_ep_unl(Nsub: int, alpha0: float, alpha1: float, FTOL: float, 
                  dsig_trial: np.ndarray, sigma_vec: np.ndarray, 
                  M_i: float, p_i: float) -> float:
    """
    Pegasus algorithm for elasto-plastic unloading
    
    Args:
        Nsub: Number of subdivisions
        alpha0: Initial alpha value
        alpha1: Final alpha value
        FTOL: Tolerance for yield function
        dsig_trial: Trial stress increment
        sigma_vec: Initial stress vector
        M_i: Image stress ratio
        p_i: Image mean pressure
        
    Returns:
        alpha_: Optimal alpha value
    """
    sigma0 = sigma_vec + alpha0 * dsig_trial
    p0, q0 = stress_decomp(sigma0)
    F0 = findF(p0, q0, M_i, p_i)
    Fsave = F0

    dalpha = (alpha1 - alpha0) / Nsub

    while True:
        alpha = alpha0 + dalpha
        sigmaprime = sigma_vec + alpha * dsig_trial
        pprime, qprime = stress_decomp(sigmaprime)
        Fprime = findF(pprime, qprime, M_i, p_i)

        # Exit condition
        if Fprime > FTOL:
            alpha1 = alpha
            if F0 < -FTOL:
                F1 = Fprime
                alpha_ = pegasus(alpha0, alpha1, FTOL, dsig_trial, sigma_vec, M_i, p_i, 1, F1)
                return alpha_
            else:
                alpha0 = 0
                F0 = Fsave
                dalpha = (alpha - alpha0) / Nsub
        else:
            alpha0 = alpha
            F0 = Fsave

def stressCorrection(params: np.ndarray, F: float, sigma_vec: np.ndarray, 
                     p_i: float, FTOL: float, MAXITS: int) -> Tuple[np.ndarray, float]:
    """
    Stress correction algorithm
    
    Args:
        params: Material parameters
        F: Yield function value
        sigma_vec: Stress vector
        p_i: Image mean pressure
        FTOL: Tolerance for yield function
        MAXITS: Maximum number of iterations
        
    Returns:
        sigma_corr: Corrected stress vector
        p_i_corr: Corrected image mean pressure
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

    # Find initial yield surface F0
    sigma_0 = sigma_vec.copy()
    p_i0 = p_i

    # Begin loop
    for i in range(1, MAXITS + 1):
        p0, q0 = stress_decomp(sigma_0)
        theta_0 = lode_angle(sigma_0)
        M0 = findM(theta_0, M_tc)
        psi0, psi_i0 = findpsipsii(Gamma, Lambda, p0, p_i0, e)
        M_i0 = findM_i(M0, M_tc, chi_i, psi_i0, N)
        F0 = findF(p0, q0, M_i0, p_i0)

        if i == 1:
            F0 = F
        
        # Values for cap and maximum yield surface
        M_itc0 = findM_itc(N, chi_i, psi_i0, M_tc)
        p_imax0 = findp_imax(chi_i, psi_i0, p0, M_itc0)
        
        # Compute C_e (to compute denominator)
        C_e = findCe(G_max, p0, p_ref, m, nu)
    
        # dF/dsigma
        dfdsig_params0 = np.array([theta_0, N, M_i0, M_tc, p_i0, chi_i, psi_i0])
        dfdsigma0 = finddFdsigma(sigma_0, dfdsig_params0)
        
        # dF/depsilonp
        dfdep_params0 = np.array([H0, Hy, psi0, N, M0, M_i0, M_tc, M_itc0, p_i0, p_imax0, chi_i, psi_i0, Lambda])
        dfdepsilon0_dfdsigma0, dpi_depspd0, dfdpi0, dfdq_0 = finddFdepsilon_p(sigma_0, dfdep_params0, dfdsigma0)
    
        # Compute denominator
        denom = -dfdepsilon0_dfdsigma0 + np.dot(dfdsigma0, C_e @ dfdsigma0)
    
        # Compute del_lambda_p
        del_lambda = F0 / denom
        
        # Compute df/dpi
        # First term
        parfparpi = -p0 * M_i0 / p_i0
    
        # Second term
        dfdMi = -q0 / M_i0
    
        # Third term
        if psi_i0 != 0:
            psi_sign = psi_i0 / np.abs(psi_i0)
        else:
            psi_sign = 0
        dMidpi = -(M0 / M_tc) * N * chi_i * psi_sign * Lambda / p_i0
    
        # Combine terms
        dfdpi = parfparpi + dfdMi * dMidpi
    
        # Compute B0 = A0 / (df/dH) ~ A0 / (df/dpi)
        B0 = dfdepsilon0_dfdsigma0 / dfdpi
        
        # Update stresses, PIV
        sigma_1 = sigma_0  # Don't modify sigma directly in this implementation
        p_i1 = p_i0 + 20 * del_lambda * B0

        # Compute new yield surface
        theta_1 = lode_angle(sigma_1)
        M1 = findM(theta_1, M_tc)
        p1, q1 = stress_decomp(sigma_1)
        _, psi_i1 = findpsipsii(Gamma, Lambda, p1, p_i1, e)
        M_i1 = findM_i(M1, M_tc, chi_i, psi_i1, N)
        F1 = findF(p1, q1, M_i1, p_i1)
    
        # Check if stresses land on yield surface (return condition)
        if abs(F1) <= FTOL or i == MAXITS:
            return sigma_1, p_i1
        
        # Check for non-convergence of current scheme, correct parameters if needed
        if abs(F1) > abs(F0):
            del_lambda = F0 / np.dot(dfdsigma0, dfdsigma0)
            sigma_1 = sigma_0  # Don't modify sigma directly in this implementation
            p_i1 = p_i0
            theta_1 = lode_angle(sigma_1)
            M1 = findM(theta_1, M_tc)
            p1, q1 = stress_decomp(sigma_1)
            _, psi_i1 = findpsipsii(Gamma, Lambda, p1, p_i1, e)
            M_i1 = findM_i(M1, M_tc, chi_i, psi_i1, N)
            F1 = findF(p1, q1, M_i1, p_i1)
            if abs(F1) <= FTOL or i == MAXITS:
                return sigma_1, p_i1

        # Update stresses, PIV, epsilon_p for next iteration if needed
        sigma_0 = sigma_1.copy()
        p_i0 = p_i1

    # If we reach here without convergence, return the last values
    return sigma_0, p_i0

def ME(params: np.ndarray, sigma_vec: np.ndarray, depsilon: np.ndarray, 
       p_i: float, STOL: float, EPS: float, FTOL: float) -> Tuple[np.ndarray, float]:
    """
    Modified Euler with sub-stepping for error control
    
    Args:
        params: Material parameters
        sigma_vec: Initial stress vector
        depsilon: Strain increment
        p_i: Initial image mean pressure
        STOL: Stress tolerance
        EPS: Strain tolerance
        FTOL: Yield function tolerance
        
    Returns:
        sigma_ME: Final stress vector
        p_i_ME: Final image mean pressure
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

    # Set pseudo-time increment
    T = 0.0
    dT = 1.0

    # Initiate stresses
    sigma0 = sigma_vec.copy()
    p_i0 = p_i

    # Initiate void ratio
    e0 = e
    
    steps = 0
    flag_subst = 0
    
    # Begin looping until T reaches 1
    while T <= 1:
        flag_tol = 0  # Changes to 1 when STOL reached
        while flag_tol == 0:
            steps += 1
    
            # Compute depsilon for substep increment n
            depsilon_n = dT * depsilon
    
            # Forward Euler (predictor) routine -------------------------------
            # Compute p
            p0, _ = stress_decomp(sigma0)
    
            # Compute theta, M, psi_i, M_i
            theta0 = lode_angle(sigma0)
            M0 = findM(theta0, M_tc)
            
            psi0, psi_i0 = findpsipsii(Gamma, Lambda, p0, p_i0, e0)
            M_i0 = findM_i(M0, M_tc, chi_i, psi_i0, N)
    
            # Values for cap and maximum yield surface
            M_itc0 = findM_itc(N, chi_i, psi_i0, M_tc)
            p_imax0 = findp_imax(chi_i, psi_i0, p0, M_itc0)
    
            # Compute components needed for C_p, dlambda_p
            # dF/dsigma
            dfdsig_params0 = np.array([theta0, N, M_i0, M_tc, p_i0, chi_i, psi_i0])
            dfdsigma0 = finddFdsigma(sigma0, dfdsig_params0)
            
            # dF/depsilonp
            dfdep_params0 = np.array([H0, Hy, psi0, N, M0, M_i0, M_tc, M_itc0, p_i0, p_imax0, chi_i, psi_i0, Lambda])
            dfdepsilon0_dfdsigma0, dpi_depspd0, dfdpi0, dfdq_0 = finddFdepsilon_p(sigma0, dfdep_params0, dfdsigma0)
    
            # Compute C_p, dlambda_p for this substep
            C_e0 = findCe(G_max, p0, p_ref, m, nu)
            C_p0 = findC_p(C_e0, dfdsigma0, dfdepsilon0_dfdsigma0)
            dlambda_p0 = find_dlambda_p(C_e0, dfdsigma0, dfdepsilon0_dfdsigma0, depsilon_n)
    
            # Compute dsigma, depsilon_p, dpi for FE step
            dsigma0 = C_p0 @ depsilon_n
            depsilon_p0 = dlambda_p0 * dfdsigma0
            _, depsilon_qp0 = vol_dev(sigma0, depsilon_p0)
            dp_i0 = depsilon_qp0 * dpi_depspd0
    
            # Compute new sigma1, p_i1, epsilon_p1 for FE step
            sigma1 = sigma0 + dsigma0
            p_i1 = p_i0 + dp_i0
    
            # Update void ratio
            deps_v = np.sum(depsilon_n[0:3])
            de = (1 + e0) * deps_v
            e1 = e0 - de
    
            # Modified Euler (predictor-corrector) routine --------------------
            
            # Updated values corresponding to FE step
            # Compute p
            p1, _ = stress_decomp(sigma1)
    
            # Compute theta, M, psi_i, M_i
            theta1 = lode_angle(sigma1)
            M1 = findM(theta1, M_tc)
            
            psi1, psi_i1 = findpsipsii(Gamma, Lambda, p1, p_i1, e1)
            M_i1 = findM_i(M1, M_tc, chi_i, psi_i1, N)
    
            # Values for cap and maximum yield surface
            M_itc1 = findM_itc(N, chi_i, psi_i1, M_tc)
            p_imax1 = findp_imax(chi_i, psi_i1, p1, M_itc1)
    
            # Compute components needed for C_p, dlambda_p
            # dF/dsigma
            dfdsig_params1 = np.array([theta1, N, M_i1, M_tc, p_i1, chi_i, psi_i1])
            dfdsigma1 = finddFdsigma(sigma1, dfdsig_params1)
            
            # dF/depsilonp
            dfdep_params1 = np.array([H0, Hy, psi1, N, M1, M_i1, M_tc, M_itc1, p_i1, p_imax1, chi_i, psi_i1, Lambda])
            dfdepsilon1_dfdsigma1, dpi_depspd1, dfdpi1, dfdq_1 = finddFdepsilon_p(sigma1, dfdep_params1, dfdsigma1)
    
            # Compute C_p, dlambda_p for this substep
            C_e1 = findCe(G_max, p1, p_ref, m, nu)
            C_p1 = findC_p(C_e1, dfdsigma1, dfdepsilon1_dfdsigma1)
            dlambda_p1 = find_dlambda_p(C_e1, dfdsigma1, dfdepsilon1_dfdsigma1, depsilon_n)
    
            # Compute dsigma, depsilon_p, dpi for ME step
            dsigma1 = C_p1 @ depsilon_n
            depsilon_p1 = dlambda_p1 * dfdsigma1
            _, depsilon_qp1 = vol_dev(sigma1, depsilon_p1)
            dp_i1 = depsilon_qp1 * dpi_depspd1
    
            # Compute new sigma2, p_i2, epsilon_p2 for ME step
            sigma2 = sigma0 + (1/2) * (dsigma0 + dsigma1)
            p_i2 = p_i0 + (1/2) * (dp_i0 + dp_i1)

            # Update void ratio
            e2 = e1
    
            # Convert sigma2, dsigma0, dsigma1 to 3x3 tensors for proper norm computation
            # In Python, we can use voigt_norm for simpler computation
            
            # Compute errors for stress, PIV
            Error_sigma = 0.5 * voigt_norm(dsigma1 - dsigma0) / voigt_norm(sigma2)
            
            # Avoid division by zero for Error_pi
            if np.abs(p_i2) < 1e-10:
                Error_pi = 0
            else:
                Error_pi = 0.5 * np.abs(dp_i1 - dp_i0) / np.abs(p_i2)
    
            # Compute relative error
            Rn = max(Error_sigma, Error_pi)
            Rn = max(Rn, EPS)
    
            # Update stresses, epsilon_p, psi if successful
            if np.abs(Rn) <= STOL:
                flag_tol = 1
                # Compute values needed for new yield surface
                p2, q2 = stress_decomp(sigma2)
                theta2 = lode_angle(sigma2)
                M2 = findM(theta2, M_tc)
                _, psi_i2 = findpsipsii(Gamma, Lambda, p2, p_i2, e2)
                M_i2 = findM_i(M2, M_tc, chi_i, psi_i2, N)
        
                # Compute new yield surface
                F2 = findF(p2, q2, M_i2, p_i2)
                
                # Check if stresses are on new ys, correct if not
                if np.abs(F2) > FTOL:
                    sigma2, p_i2 = stressCorrection(params, F2, sigma2, p_i2, FTOL, 10)
    
                # Update corrected sigma, p_i
                sigma0 = sigma2.copy()
                p_i0 = p_i2
                
                # Update void ratio for next step
                deps_v = np.sum(depsilon_n[0:3])
                de = (1 + e) * deps_v
                e0 = e0 - de

                if T == 1:
                    return sigma2, p_i2
            else:
                # Update dT parameters
                q = max(0.9 * np.real(np.sqrt(STOL / Rn)), 0.1)
                q = min(q, 1)
                dT = max(q*dT, 1e-6)
                dT = min(dT, 1 - T)
                flag_subst = 0
                
        if flag_subst == 0:
            q = min(0.9*(np.sqrt(STOL/Rn)), 1.0)
        elif flag_subst == 1:
            q = min(0.9*(np.sqrt(STOL/Rn)), 1.1)
            
        q = max(q, 0.1)
        # Apply constraints to dT
        dT = max(q*dT, 1e-6)
        dT = min(dT, 1 - T)
        # Update T
        T = T + dT
        flag_subst = 1
        
    # If we reach here, return the last values
    return sigma0, p_i0 