from utils import stress_decomp, find_sJ2J3, lode_angle
from norsand_functions import findM, findM_i, findF
import numpy as np

def test_imports():
    # Test utility functions
    sigma = np.array([100.0, 50.0, 50.0, 0.0, 0.0, 0.0])
    p, q = stress_decomp(sigma)
    print(f"stress_decomp: p = {p}, q = {q}")
    
    s, J2, J3 = find_sJ2J3(sigma)
    print(f"find_sJ2J3: s = {s}, J2 = {J2}, J3 = {J3}")
    
    theta = lode_angle(sigma)
    print(f"lode_angle: theta = {theta}")
    
    # Test NorSand functions
    M_tc = 1.2
    M = findM(theta, M_tc)
    print(f"findM: M = {M}")
    
    chi_i = 3.5
    psi_i = -0.05
    N = 0.3
    M_i = findM_i(M, M_tc, chi_i, psi_i, N)
    print(f"findM_i: M_i = {M_i}")
    
    p_i = 200.0
    F = findF(p, q, M_i, p_i)
    print(f"findF: F = {F}")
    
    print("All imports tested successfully!")

if __name__ == "__main__":
    test_imports() 