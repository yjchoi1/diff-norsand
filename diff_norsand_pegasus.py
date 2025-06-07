import torch
from diff_norsand_functions import findF, stress_decomp, findM, findpsipsii, findM_i, findM_itc, findp_imax, finddFdsigma, finddFdepsilon_p, find_dlambda_p, findC_p
from diff_utils import lode_angle, findCe, vol_dev, voigt_norm


def pegasus(
    alpha0: torch.Tensor,
    alpha1: torch.Tensor, 
    FTOL: torch.Tensor,
    dsig_trial: torch.Tensor,
    sigma_vec: torch.Tensor,
    M_i: torch.Tensor,
    p_i: torch.Tensor,
    flag: int = 0,
    F1_ep_unl: torch.Tensor = None,
    MAXITS: int = 20
) -> torch.Tensor:
    """
    Differentiable pegasus algorithm for elastic to plastic transition
    
    Args:
        alpha0: Initial alpha value
        alpha1: Final alpha value
        FTOL: Tolerance for yield function
        dsig_trial: Trial stress increment
        sigma_vec: Initial stress vector
        M_i: Image stress ratio
        p_i: Image mean pressure
        flag: Flag for different modes (0 for elastic-plastic, 1 for ep_unl)
        F1_ep_unl: F1 value for ep_unl mode
        MAXITS: Maximum number of iterations
        
    Returns:
        alpha_: Optimal alpha value
    """
    eps = 1e-10  # Small epsilon to prevent division by zero
    sigmoid_slope = 10  # sigmoid slope steepness
    result_check_tol = 1e-5  # Tolerance for result check
    
    # Compute initial yield surfaces F0 and F1
    sigma0 = sigma_vec + alpha0 * dsig_trial
    p0, q0 = stress_decomp(sigma0)
    F0 = findF(p0, q0, M_i, p_i)
    
    sigma1 = sigma_vec + alpha1 * dsig_trial
    p1, q1 = stress_decomp(sigma1)
    F1 = findF(p1, q1, M_i, p_i)
    
    # Use fixed number of iterations for differentiability
    # Start with the initial bracket
    alpha_low = alpha0
    alpha_high = alpha1
    F_low = F0
    F_high = F1
    
    # Simple bisection-like approach with pegasus updates
    for i in range(MAXITS):
        # Pegasus formula: alpha = alpha1 - F1 * (alpha1 - alpha0) / (F1 - F0)
        denominator = F_high - F_low
        # Use a safe division to avoid NaN
        safe_denom = torch.where(torch.abs(denominator) < eps, 
                               torch.sign(denominator) * eps + eps, denominator)
        
        alpha_new = alpha_high - F_high * (alpha_high - alpha_low) / safe_denom
        
        # Compute new function value
        sigma_new = sigma_vec + alpha_new * dsig_trial
        p_new, q_new = stress_decomp(sigma_new)
        F_new = findF(p_new, q_new, M_i, p_i)
        
        # Update bounds based on sign of F_new
        # This is a differentiable approximation of the conditional logic
        sign_change = F_new * F_high
        
        # Use sigmoid-like smooth functions instead of hard conditionals
        # This helps maintain gradient flow
        update_weight = torch.sigmoid(-sigmoid_slope * sign_change)  # Close to 1 when sign_change < 0, close to 0 otherwise
        
        # Smooth update of the bounds
        alpha_low = (1 - update_weight) * alpha_low + update_weight * alpha_new
        alpha_high = (1 - update_weight) * alpha_high + update_weight * alpha_low
        F_low = (1 - update_weight) * F_low + update_weight * F_new  
        F_high = (1 - update_weight) * F_high + update_weight * F_low
        
        # Additional simple update that doesn't use conditionals
        alpha_low = alpha_new
        F_low = F_new
    
    # Check if final F value is close enough to zero
    final_sigma = sigma_vec + alpha_new * dsig_trial
    final_p, final_q = stress_decomp(final_sigma)
    final_F = findF(final_p, final_q, M_i, p_i)
    
    if torch.any(torch.abs(final_F) > result_check_tol):
        raise ValueError(
            f"Pegasus algorithm did not converge to yield surface. Final F value: {final_F}, tolerance: {result_check_tol}")
    
    # Return the last computed alpha
    return alpha_new
