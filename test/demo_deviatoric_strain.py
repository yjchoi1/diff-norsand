import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from diff_utils import vol_dev, vol_dev_stress_weighted

def main():
    print("Demonstration: Standard vs Stress-Weighted Deviatoric Strain")
    print("=" * 65)
    
    # Test case 1: Same strain, different stress states
    print("\nTest 1: Same strain, different stress states")
    print("-" * 45)
    
    # Fixed strain
    epsilon = torch.tensor([0.02, 0.01, 0.005, 0.008, 0.004, 0.002])
    
    # Different stress states
    sigma1 = torch.tensor([100.0, 50.0, 25.0, 30.0, 15.0, 10.0])  # General stress
    sigma2 = torch.tensor([60.0, 60.0, 60.0, 30.0, 15.0, 10.0])   # More hydrostatic
    
    # Standard deviatoric strain (should be same for both)
    _, e_q_std1 = vol_dev(sigma1, epsilon)
    _, e_q_std2 = vol_dev(sigma2, epsilon)
    
    # Stress-weighted deviatoric strain (should be different)
    _, e_q_weighted1 = vol_dev_stress_weighted(sigma1, epsilon)
    _, e_q_weighted2 = vol_dev_stress_weighted(sigma2, epsilon)
    
    print(f"Same strain: {epsilon}")
    print(f"Stress 1:    {sigma1}")
    print(f"Stress 2:    {sigma2}")
    print()
    print(f"Standard deviatoric strain:")
    print(f"  Stress 1: {e_q_std1:.6f}")
    print(f"  Stress 2: {e_q_std2:.6f}")
    print(f"  Difference: {abs(e_q_std1 - e_q_std2):.6f} (should be ~0)")
    print()
    print(f"Stress-weighted deviatoric strain:")
    print(f"  Stress 1: {e_q_weighted1:.6f}")
    print(f"  Stress 2: {e_q_weighted2:.6f}")
    print(f"  Difference: {abs(e_q_weighted1 - e_q_weighted2):.6f} (should be >0)")
    
    # Test case 2: Pure hydrostatic vs pure shear
    print("\n\nTest 2: Pure hydrostatic vs pure shear strain")
    print("-" * 50)
    
    # Pure hydrostatic strain
    eps_hydro = torch.tensor([0.01, 0.01, 0.01, 0.0, 0.0, 0.0])
    sigma_hydro = torch.tensor([100.0, 100.0, 100.0, 0.0, 0.0, 0.0])
    
    # Pure shear strain  
    eps_shear = torch.tensor([0.005, -0.005, 0.0, 0.01, 0.0, 0.0])
    sigma_shear = torch.tensor([0.0, 0.0, 0.0, 50.0, 0.0, 0.0])
    
    e_v_hydro, e_q_std_hydro = vol_dev(sigma_hydro, eps_hydro)
    _, e_q_weighted_hydro = vol_dev_stress_weighted(sigma_hydro, eps_hydro)
    
    e_v_shear, e_q_std_shear = vol_dev(sigma_shear, eps_shear)
    _, e_q_weighted_shear = vol_dev_stress_weighted(sigma_shear, eps_shear)
    
    print("Hydrostatic case:")
    print(f"  Volumetric strain: {e_v_hydro:.6f}")
    print(f"  Standard deviatoric: {e_q_std_hydro:.6f}")
    print(f"  Stress-weighted: {e_q_weighted_hydro:.6f}")
    print()
    print("Pure shear case:")
    print(f"  Volumetric strain: {e_v_shear:.6f}")
    print(f"  Standard deviatoric: {e_q_std_shear:.6f}")
    print(f"  Stress-weighted: {e_q_weighted_shear:.6f}")
    
    print("\n\nSummary:")
    print("- Standard deviatoric strain depends only on strain tensor")
    print("- Stress-weighted deviatoric strain depends on both stress and strain")
    print("- Stress-weighted version represents strain increment in stress direction")
    print("- Both are differentiable and useful for different applications")

if __name__ == "__main__":
    main() 