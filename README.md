# Diff NorSand - Differentiable NorSand Constitutive Model

[![Version](https://img.shields.io/badge/version-0.1--dev-orange.svg)](https://github.com/your-repo/diff-norsand)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A differentiable implementation of the NorSand constitutive model for sand using PyTorch, enabling automatic differentiation and gradient-based optimization for geotechnical applications.

## Overview

The Diff NorSand project provides both differentiable (PyTorch) and traditional (NumPy) implementations of the NorSand constitutive model. The differentiable version enables:

- **Parameter Calibration**: Gradient-based optimization for model parameters
- **Inverse Problems**: Back-analysis of experimental data  
- **Sensitivity Analysis**: Automated computation of parameter sensitivities
- **Machine Learning Integration**: Seamless integration with neural networks

## Key Features to implement

- 📊 **Complete Differentiable NorSand Model**: Ensures differentiability for stress updating, drift correction, and substepping.
- 🧮 **Automatic Differentiation**: Full PyTorch integration for gradients
- 📚 **Documentation**: Detailed API docs with mathematical background
- 🎯 **Parameter Optimization**: Built-in support for gradient-based calibration

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

This is a development version intended for research and evaluation. The API may change in future versions based on user feedback. 

## Contributing

Contributions are welcome! Please see the [Development Guide](docs/source/development.rst) for:

- Development setup
- Coding standards  
- Testing guidelines
- Pull request process

## License

N/A


## Acknowledgements