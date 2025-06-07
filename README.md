# Diff NorSand - Differentiable NorSand Constitutive Model

[![Version](https://img.shields.io/badge/version-0.1--dev-orange.svg)](https://github.com/yjchoi1/diff-norsand)
[![Tests](https://github.com/yjchoi1/diff-norsand/workflows/Tests/badge.svg)](https://github.com/yjchoi1/diff-norsand/actions)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A differentiable implementation of the NorSand constitutive model for sand using PyTorch, enabling automatic differentiation and gradient-based optimization for geotechnical applications.

## Overview

The Diff NorSand project provides both differentiable (PyTorch) and traditional (NumPy) implementations of the NorSand constitutive model. The differentiable version aims to enables:

- **Parameter Calibration**: Gradient-based optimization for model parameters
- **Inverse Problems**: Back-analysis of experimental data  
- **Sensitivity Analysis**: Automated computation of parameter sensitivities
- **Machine Learning Integration**: Seamless integration with neural networks

## Test Status

| Test Module | Description |
|-------------|-------------|
| **test_combined_diff.py** | Combined differential operations |
| **test_dff_modified_euler.py** | Modified Euler integration method |
| **test_diff_norsand_fn.py** | Core NorSand constitutive functions |
| **test_diff_pegasus.py** | Pegasus root finding algorithm |
| **test_diff_stress_correction.py** | Stress correction algorithms |
| **test_diff_utils.py** | Differential utility functions |

*Click the Tests badge above to see detailed test results and individual test status.*

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd diff-norsand

# Install dependencies
pip install torch numpy matplotlib
```
## Documentation

Comprehensive documentation is on updated available with:

- **API Reference**: Detailed function documentation
- **Mathematical Background**: Complete theoretical framework
- **Getting Started Guide**: Step-by-step introduction

Visit [https://yjchoi1.github.io/diff-norsand/](https://yjchoi1.github.io/diff-norsand/)

### Building Documentation

```bash
cd docs
make html
# Open build/html/index.html in your browser
```
The documentation includes:

- **Getting Started**: Installation and basic usage
- **API Reference**: Complete function documentation for all modules
- **Development Guide**: Contributing guidelines and development setup

## Development Status

This is a development version intended for research and evaluation. The API may change in future versions based on user feedback. Contributions are welcome.

## License

N/A


## Inspiration
[NorSand Documentation](https://srinivas12viv.github.io/NorSand/)
Sean, J. (2025). NORSAND Implementation in Additive and Multiplicative Elastoplasticity. Master thesis, Georgia Institute of Technology.