diff_norsand_functions module
===============================

.. automodule:: diff_norsand_functions
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``diff_norsand_functions`` module contains the core differentiable implementation of the NorSand constitutive model using PyTorch. 
This module provides automatic differentiation capabilities for all NorSand-related computations, enabling gradient-based optimization and parameter calibration.

Notes
-----

* Special attention is paid to cases where absolute values might cause gradient discontinuities 