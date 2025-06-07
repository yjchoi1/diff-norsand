# Diff NorSand - Differentiable NorSand

![Version 0.1-dev](https://img.shields.io/badge/version-0.1--dev-orange.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

Welcome to the **Diff NorSand** documentation! This project implements a differentiable version of the NorSand constitutive model for sand using PyTorch, enabling automatic differentiation and gradient-based optimization for geotechnical applications.

## Overview

The Diff NorSand project provides both differentiable (PyTorch) and traditional (NumPy) implementations of the NorSand constitutive model. 
The differentiable version aims to enables:

* **Parameter Calibration**: Gradient-based optimization for model parameters
* **Inverse Problems**: Back-analysis of experimental data
* **Sensitivity Analysis**: Automated computation of parameter sensitivities
* **Machine Learning Integration**: Seamless integration with neural networks for data-driven constitutive modeling which enables improved model for specific applications while still partially preserving the physical consistency of the NorSand model.

## Key Features

* **Automatic Differentiation**: Full PyTorch integration for gradient computation
* **Dual Implementation**: Both differentiable (PyTorch) and traditional (NumPy) versions
* **Comprehensive Documentation**: Detailed API documentation with mathematical formulations

## Contents

```{toctree}
:maxdepth: 2
:caption: API Reference

modules/diff_norsand_functions
modules/diff_utils
modules/diff_norsand_stress_correction
modules/diff_norsand_pegasus
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started.md
mathematical_background.md
examples.md
```

```{toctree}
:maxdepth: 1
:caption: Developer Notes

development.md
changelog.md
```

The differentiable implementation enables:

* **Parameter Calibration**: Gradient-based optimization for model parameters
* **Inverse Problems**: Back-analysis of experimental data
* **Sensitivity Analysis**: Automated computation of parameter sensitivities
* **Machine Learning Integration**: Seamless integration with neural networks

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search` 