import numpy as np
import matplotlib.pyplot as plt
from norsand_functions import *

def main():
    """
    Main function to run the NorSand model simulation
    """
    # Choose type of test: 1 for undrained tx, 2 for drained tx, 3 for plane strain undrained
    soiltest = 2
    
    # List material properties based on test type
    if soiltest == 1 or soiltest == 2:  # Triaxial
        Gamma = 0.9
        Lambda = 0.02
        M_tc = 1.20
        N = 0.30
        H0 = 300
        Hy = 50
        Chi_tc = 3.5
        Ir = 300
        nu = 0.15
        Psi_0 = 0.05
        G_max = 40000
        p_ref = 100
        m = 0
    elif soiltest == 3:  # Plane strain
        Gamma = 1.0
        Lambda = 0.03
        M_tc = 1.40
        N = 0.40
        H0 = 75.0
        Hy = 375.0
        Chi_tc = 4.5
        Ir = 400
        nu = 0.15
        Psi_0 = -0.15
        G_max = 40000
        p_ref = 100
        m = 0
    
    # Set up Bardet drivers based on type of test
    if soiltest == 1:  # Undrained triaxial test
        # For stresses
        S = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ])
        
        # For strains
        E = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
        ])
    elif soiltest == 2:  # Drained triaxial test
        # For stresses
        S = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ])
        
        # For strains
        E = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
        ])
    elif soiltest == 3:  # Plain strain
        # For stresses
        S = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        # For strains
        E = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0]
        ])
    
    # Loading vector
    V = np.array([0, 0, 0, 0, 0, 1e-5])
    
    # Initiate stress, strain vectors
    sigma_vec = np.array([100.6711409, 99.66442953, 99.66442953, 0, 0, 0])
    epsilon_vec = np.zeros(6)
    epsilon_p_vec = np.zeros(6)
    
    # Initiate p, q, e lists for storing results
    max_steps = 10000
    p_list = np.zeros(max_steps)
    q_list = np.zeros(max_steps)
    e_list = np.zeros(max_steps)
    eps1_list = np.zeros(max_steps)
    eps2_list = np.zeros(max_steps)
    eps3_list = np.zeros(max_steps)
    eps_vol_list = np.zeros(max_steps)
    s1_list = np.zeros(max_steps)
    s2_list = np.zeros(max_steps)
    s3_list = np.zeros(max_steps)
    p_ilist = np.zeros(max_steps)
    dsigma1_list = np.zeros(max_steps)
    dsigma2_list = np.zeros(max_steps)
    dsigma3_list = np.zeros(max_steps)
    theta_list = np.zeros(max_steps)
    dfdsigma_list = np.zeros(max_steps)
    D_list = np.zeros(max_steps)
    eta_list = np.zeros(max_steps)
    M_list = np.zeros(max_steps)
    Mi_list = np.zeros(max_steps)
    
    # Chi_i, M does not change, Lode angle reflects pure triaxial compression
    chi_i = findchi_i(M_tc, Chi_tc, Lambda)
    
    # Begin loop
    for i in range(max_steps):
        # Cap shear stresses
        for j in range(3, 6):
            if sigma_vec[j] < 0.1:
                sigma_vec[j] = 0
        
        theta = lode_angle(sigma_vec)
        M = findM(theta, M_tc)
        
        # Find p, q at current step
        p, q = stress_decomp(sigma_vec)
        
        # Find void ratio, psi, and initiate p_i psi_i M_i
        if i == 0:
            psi = Psi_0
            e_c = Gamma - Lambda * np.log(p)
            e = psi + e_c
            p_i, psi_i, M_i = findp_ipsi_iM_i(N, chi_i, Lambda, M_tc, psi, p, q, M)
        else:
            deps_v = np.sum(deps[0:3])
            de = (1 + e) * deps_v
            e = e - de
            psi, psi_i = findpsipsii(Gamma, Lambda, p, p_i, e)
            M_i = findM_i(M, M_tc, chi_i, psi_i, N)
        
        # Store values
        p_list[i] = p
        q_list[i] = q
        e_list[i] = e
        eps_vol_list[i] = np.sum(epsilon_vec[0:3])
        eps1_list[i] = epsilon_vec[0]
        eps2_list[i] = epsilon_vec[1]
        eps3_list[i] = epsilon_vec[2]
        s1_list[i] = sigma_vec[0]
        s2_list[i] = sigma_vec[1]
        s3_list[i] = sigma_vec[2]
        p_ilist[i] = p_i
        theta_list[i] = theta
        D_list[i] = M_i - q/p
        eta_list[i] = q/p
        M_list[i] = M
        Mi_list[i] = M_i
        
        # Values for cap and maximum yield surface
        M_itc = findM_itc(N, chi_i, psi_i, M_tc)
        p_imax = findp_imax(chi_i, psi_i, p, M_itc)
        
        # Cap p, q
        p_cap = p_i * np.exp(chi_i * psi_i / M_itc)
        q_cap = p_cap * M_i * (1 - np.log(p_cap / p_i))
        
        # Find dF/dsigma
        dfdsig_params = np.array([theta, N, M_i, M_tc, p_i, chi_i, psi_i])
        dfdsig = finddFdsigma(sigma_vec, dfdsig_params)
        dfdsigma_list[i] = np.linalg.norm(dfdsig)
        
        # Find dF/depsilon
        dfdep_params = np.array([H0, Hy, psi, N, M, M_i, M_tc, M_itc, p_i, p_imax, chi_i, psi_i, Lambda])
        dfdep_dfdsig, dpi_depspd, dfdpi, dfdq_ = finddFdepsilon_p(sigma_vec, dfdep_params, dfdsig)
        
        # Find elastic tangent Ce
        C_e = findCe(G_max, p, p_ref, m, nu)
        
        # Compute Cp
        C_p = findC_p(C_e, dfdsig, dfdep_dfdsig)
        
        # Obtain strain increment
        if i < 5:
            deps = np.linalg.solve(S @ C_e + E, V)
        else:
            deps = np.linalg.solve(S @ C_p + E, V)
        
        # Check for transition from elastic to plastic
        
        # Obtain trial stress increment and trial stress
        dsig_tr = C_e @ deps
        sigma_tr = sigma_vec + dsig_tr
        p_tr, q_tr = stress_decomp(sigma_tr)
        F_tr = findF(p_tr, q_tr, M_i, p_i)
        
        # Yield surface value for current state of stress sigma_n
        F0 = findF(p, q, M_i, p_i)
        
        # Set FTOL
        FTOL = 1e-5
        
        if F_tr <= FTOL:  # Purely elastic step
            sigma_vec = sigma_tr
            epsilon_vec = epsilon_vec + deps
            continue
        
        if F0 < -FTOL and F_tr > FTOL:
            alpha = pegasus(0, 1, FTOL, dsig_tr, sigma_vec, M_i, p_i, 0, 0, 10)
        elif abs(F0) < FTOL and F_tr > FTOL:
            cos_theta = np.dot(dfdsig, dsig_tr) / (voigt_norm(dfdsig) * voigt_norm(dsig_tr))
            if cos_theta >= -10e-6:
                alpha = 0
            else:
                print('el_p unloading')
                break
        else:
            print('Illegal stress state')
            break
        
        # Update stresses from elastic step
        sigma_vec = sigma_vec + alpha * C_e @ deps
        deps_flow = (1 - alpha) * deps
        
        # Update stress, strain vectors with substepping algorithm
        params = np.array([Lambda, M_tc, N, H0, Hy, Ir, nu, chi_i, psi, Gamma, e, G_max, p_ref, m])
        sigma_vec_next, p_i = ME(params, sigma_vec, deps_flow, p_i, 1, 1e-16, 1e-6)
        
        # Debugging
        dsigma1 = sigma_vec_next[0] - sigma_vec[0]
        dsigma2 = sigma_vec_next[1] - sigma_vec[1]
        dsigma3 = sigma_vec_next[2] - sigma_vec[2]
        if i < max_steps - 1:
            dsigma1_list[i+1] = dsigma1
            dsigma2_list[i+1] = dsigma2
            dsigma3_list[i+1] = dsigma3
        
        sigma_vec = sigma_vec_next
        epsilon_vec = epsilon_vec + deps
        
        # Stop if we've reached the end of meaningful simulation
        if i > 100 and np.abs(q_list[i] - q_list[i-1]) < 1e-10:
            max_steps = i + 1
            break
    
    # Truncate arrays to actual size used
    p_list = p_list[:max_steps]
    q_list = q_list[:max_steps]
    e_list = e_list[:max_steps]
    eps1_list = eps1_list[:max_steps]
    eps2_list = eps2_list[:max_steps]
    eps3_list = eps3_list[:max_steps]
    eps_vol_list = eps_vol_list[:max_steps]
    s1_list = s1_list[:max_steps]
    s2_list = s2_list[:max_steps]
    s3_list = s3_list[:max_steps]
    p_ilist = p_ilist[:max_steps]
    dsigma1_list = dsigma1_list[:max_steps]
    dsigma2_list = dsigma2_list[:max_steps]
    dsigma3_list = dsigma3_list[:max_steps]
    theta_list = theta_list[:max_steps]
    dfdsigma_list = dfdsigma_list[:max_steps]
    D_list = D_list[:max_steps]
    eta_list = eta_list[:max_steps]
    M_list = M_list[:max_steps]
    Mi_list = Mi_list[:max_steps]
    
    # Plot results
    
    # p-q stress path
    plt.figure(figsize=(10, 8))
    plt.plot(p_list, q_list, color='blue', linewidth=2, label='Stress path (ME)')
    plt.xlabel('p (kPa)', fontsize=20)
    plt.ylabel('q (kPa)', fontsize=20)
    plt.title('Stress path in p-q space', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig('p_q_stress_path.png')
    plt.close()
    
    # q-epsilon_v stress-strain curve
    plt.figure(figsize=(10, 8))
    plt.plot(eps1_list, q_list, color='blue', linewidth=2, label='q-path (ME)')
    plt.xlabel('ε_A', fontsize=20)
    plt.ylabel('q (kPa)', fontsize=20)
    plt.title('q-ε_A Curve', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig('q_eps_A_curve.png')
    plt.close()
    
    # p_i-eps1 curve
    plt.figure(figsize=(10, 8))
    plt.plot(eps1_list, p_ilist, color='blue', linewidth=2, label='p_i-path (ME)')
    plt.xlabel('ε_A', fontsize=20)
    plt.ylabel('p_i (kPa)', fontsize=20)
    plt.title('p_i  ε_A Curve', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig('p_i_eps_A_curve.png')
    plt.close()
    
    # eps1-dsigma curve
    plt.figure(figsize=(10, 8))
    plt.plot(eps1_list, dsigma1_list, linewidth=2, label='dsigma1-path (ME)')
    plt.plot(eps1_list, dsigma2_list, linewidth=2, label='dsigma2-path (ME)')
    plt.plot(eps1_list, dsigma3_list, linewidth=2, label='dsigma3-path (ME)')
    plt.xlabel('ε_A', fontsize=20)
    plt.ylabel('dsigma1 (kPa)', fontsize=20)
    plt.title('dsigma1-ε_A Curve', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig('dsigma_eps_A_curve.png')
    plt.close()
    
    # strains curve
    plt.figure(figsize=(10, 8))
    plt.plot(eps1_list, eps_vol_list, color='blue', linewidth=2, label='Strains (ME)')
    plt.xlabel('ε_1', fontsize=20)
    plt.ylabel('ε_vol', fontsize=20)
    plt.title('Strains', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig('strains_curve.png')
    plt.close()
    
    # Dp-eta curve
    plt.figure(figsize=(10, 8))
    plt.plot(D_list, eta_list, color='blue', linewidth=2, label='ME')
    plt.xlabel('D^p', fontsize=20)
    plt.ylabel('η', fontsize=20)
    plt.title('D^p vs. η', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig('Dp_eta_curve.png')
    plt.close()
    
    # Additional plane-strain plots if applicable
    if soiltest == 3:
        plt.figure(figsize=(10, 8))
        plt.plot(eps1_list, s1_list, color='blue', linewidth=2, label='σ_1 (ME)')
        plt.plot(eps1_list, s2_list, color='red', linewidth=2, label='σ_2 (ME)')
        plt.xlabel('ε_1', fontsize=20)
        plt.ylabel('σ_1, σ_2', fontsize=20)
        plt.title('Stresses vs. ε_1', fontsize=20)
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.legend()
        plt.savefig('plane_strain_stresses.png')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        plt.plot(eps1_list, eps_vol_list, color='blue', linewidth=2, label='Strains (ME)')
        plt.xlabel('ε_1', fontsize=20)
        plt.ylabel('ε_vol', fontsize=20)
        plt.title('Strains', fontsize=20)
        plt.grid(True)
        plt.legend()
        plt.savefig('plane_strain_strains.png')
        plt.close()

if __name__ == "__main__":
    main()
