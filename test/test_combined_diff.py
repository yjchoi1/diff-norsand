import torch
import pytest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diff_norsand_functions import findM, findM_i, findM_itc, findchi_i, findpsipsii, findp_imax, findF, finddFdsigma, findp_ipsi_iM_i, findC_p
from diff_utils import stress_decomp, find_sJ2J3, lode_angle
from norsand_py.norsand_functions import findp_ipsi_iM_i as findp_ipsi_iM_i_np


def finddFdsigma():
    """
    Compute the derivative of the yield function (F) with respect to the stress tensor.
    Computation flow:
    - Define material properties:
        - Gamma, Lambda, M_tc, N, H0, Hy, chi_tc, Ir, nu, Psi_0, G_max, p_ref, m
    - Find current material state:
        - At first computation step, 
            - theta = lode_angle(sigma_vec)
            - M = findM(theta, M_tc)
            - p, q = stress_decomp(sigma_vec)
            - psi = Psi_0
            - e_c = Gamma - Lambda * np.log(p)
            - e = psi + e_c
            - p_i, psi_i, M_i = findp_ipsi_iM_i(N, chi_i, Lambda, M_tc, psi, p, q, M)
        - For subsequent computation steps, 
            - deps_v = np.sum(deps[0:3])
            - de = (1 + e) * deps_v
            - psi, psi_i = findpsipsii(Gamma, Lambda, p, p_i, e)
            - M_i = findM_i(M, M_tc, chi_i, psi_i, N)
            
        
    - F = f(p, q, M_i, p_i)
        - M_i = f(M, chi_i, psi_i, N, M_tc)
            - M = f(M_tc, theta)
            - chi_i = f(M_tc, chi_tc, M_tc, lambda)
            - psi_i = f(psi, lambda, pi)
        - p_i
            - For the initial state, we get p_i from `findp_ipsi_iM_i`
                - findp_ipsi_iM_i is a function of (N, chi_i, Lambda, M_tc, psi, p, q, M)
            - For the subsequent computation steps, we update p_i from hardening law.
                - p_i is the output of `ME` function.
        
    """
    pass

def finddFdepsilon_p():
    """
    Compute the derivative of the yield function (F) with respect to the plastic strain.
    """
    pass

def findC_p():
    """
    Compute the elasto-plastic tangent stiffness modulus.
    Computation flow:
        - C_p = f(C_e, dfdsig, dfdep_dfdsig)
            - C_e = f(G_max, p, p_ref, m, nu)
            - dfdsig = f(...)
            - dfdep_dfdsig = f(...)

    """
    pass


if __name__ == "__main__":
    finddFdsigma()
    # findC_p()





