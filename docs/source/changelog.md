# Changelog

All notable changes to the Diff NorSand project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Initial development version
- Documentation setup with Sphinx

## [0.1-dev] - Development

### Added

* **Differentiable NorSand Implementation**
  
  * `diff_norsand_functions.py`: Core differentiable NorSand functions using PyTorch
  * `diff_utils.py`: Utility functions for stress/strain analysis with automatic differentiation
  * `diff_norsand_stress_correction.py`: Differentiable stress correction algorithms
  * `diff_norsand_pegasus.py`: Differentiable Pegasus method for root finding

* **Traditional NumPy Implementation**
  
  * `norsand_py/norsand_functions.py`: Classical NorSand implementation
  * `norsand_py/norsand_utils.py`: NumPy-based utility functions
  * `norsand_py/norsand_py.py`: Main interface for traditional implementation
  * `norsand_py/plotting.py`: Visualization functions

* **Documentation**
  
  * API reference for all modules
  * Getting started guide

* **Examples and Testing**
  
  * We don't have any examples yet.
  * Basic test framework setup

### TODOs

### Notes

This is a development version intended for research and evaluation. The API may change
in future versions based on user feedback and requirements.

For the latest updates and development progress, see the project repository. 