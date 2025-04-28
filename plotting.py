import matplotlib.pyplot as plt
import os

def plot_pq_stress_path(results, save_dir='.'):
    """Plot p-q stress path"""
    plt.figure(figsize=(10, 8))
    plt.plot(results['p_list'], results['q_list'], color='blue', linewidth=2, label='Stress path (ME)')
    plt.xlabel('p (kPa)', fontsize=20)
    plt.ylabel('q (kPa)', fontsize=20)
    plt.title('Stress path in p-q space', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'p_q_stress_path.png'))
    plt.close()

def plot_q_eps_curve(results, save_dir='.'):
    """Plot q-epsilon_v stress-strain curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(results['eps1_list'], results['q_list'], color='blue', linewidth=2, label='q-path (ME)')
    plt.xlabel('ε_A', fontsize=20)
    plt.ylabel('q (kPa)', fontsize=20)
    plt.title('q-ε_A Curve', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'q_eps_A_curve.png'))
    plt.close()

def plot_pi_eps_curve(results, save_dir='.'):
    """Plot p_i-eps1 curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(results['eps1_list'], results['p_ilist'], color='blue', linewidth=2, label='p_i-path (ME)')
    plt.xlabel('ε_A', fontsize=20)
    plt.ylabel('p_i (kPa)', fontsize=20)
    plt.title('p_i  ε_A Curve', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'p_i_eps_A_curve.png'))
    plt.close()

def plot_dsigma_eps_curve(results, save_dir='.'):
    """Plot eps1-dsigma curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(results['eps1_list'], results['dsigma1_list'], linewidth=2, label='dsigma1-path (ME)')
    plt.plot(results['eps1_list'], results['dsigma2_list'], linewidth=2, label='dsigma2-path (ME)')
    plt.plot(results['eps1_list'], results['dsigma3_list'], linewidth=2, label='dsigma3-path (ME)')
    plt.xlabel('ε_A', fontsize=20)
    plt.ylabel('dsigma1 (kPa)', fontsize=20)
    plt.title('dsigma1-ε_A Curve', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'dsigma_eps_A_curve.png'))
    plt.close()

def plot_strains_curve(results, save_dir='.'):
    """Plot strains curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(results['eps1_list'], results['eps_vol_list'], color='blue', linewidth=2, label='Strains (ME)')
    plt.xlabel('ε_1', fontsize=20)
    plt.ylabel('ε_vol', fontsize=20)
    plt.title('Strains', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'strains_curve.png'))
    plt.close()

def plot_Dp_eta_curve(results, save_dir='.'):
    """Plot Dp-eta curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(results['D_list'], results['eta_list'], color='blue', linewidth=2, label='ME')
    plt.xlabel('D^p', fontsize=20)
    plt.ylabel('η', fontsize=20)
    plt.title('D^p vs. η', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'Dp_eta_curve.png'))
    plt.close()

def plot_plane_strain_stresses(results, save_dir='.'):
    """Plot plane strain stresses"""
    plt.figure(figsize=(10, 8))
    plt.plot(results['eps1_list'], results['s1_list'], color='blue', linewidth=2, label='σ_1 (ME)')
    plt.plot(results['eps1_list'], results['s2_list'], color='red', linewidth=2, label='σ_2 (ME)')
    plt.xlabel('ε_1', fontsize=20)
    plt.ylabel('σ_1, σ_2', fontsize=20)
    plt.title('Stresses vs. ε_1', fontsize=20)
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'plane_strain_stresses.png'))
    plt.close()

def plot_plane_strain_strains(results, save_dir='.'):
    """Plot plane strain strains"""
    plt.figure(figsize=(10, 8))
    plt.plot(results['eps1_list'], results['eps_vol_list'], color='blue', linewidth=2, label='Strains (ME)')
    plt.xlabel('ε_1', fontsize=20)
    plt.ylabel('ε_vol', fontsize=20)
    plt.title('Strains', fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'plane_strain_strains.png'))
    plt.close()

def plot_all_results(results, save_dir='.'):
    """Plot all results"""
    plot_pq_stress_path(results, save_dir)
    plot_q_eps_curve(results, save_dir)
    plot_pi_eps_curve(results, save_dir)
    plot_dsigma_eps_curve(results, save_dir)
    plot_strains_curve(results, save_dir)
    plot_Dp_eta_curve(results, save_dir)
    
    if results['soiltest'] == 3:
        plot_plane_strain_stresses(results, save_dir)
        plot_plane_strain_strains(results, save_dir) 