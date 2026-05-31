# tencyl

`tencyl` is a PyTorch based toolkit for modelling the scattering of cylindrical structures including
clusters of cylinders, clusters embedded inside larger cylinders, and one-dimensional periodic cylinders lattices.

## Requirements

The package expects:

- Python 3
- PyTorch
- SciPy
- NumPy
- pvlib, for solar-spectrum utilities
- Matplotlib, for geometry plots

Install these in your environment before importing the package.

## Package Layout

- `fibre.py`: `Fibre` class to hold matrices for cylinder multiple scattering.
- `scat.py`: calculates cylindrical scattering coefficients.
- `func.py`: Bessel/Hankel helper functions and Shanks-accelerated lattice sums.
- `metrics.py`: cross sections, external fields, lattice reflection/transmission, and
  solar quadrature helpers.
- `geo.py`: geometry construction, previewing, and random structure generation.
- `opt.py`: optimization helpers, geometry penalties, and other utilities.

## Basic Usage

There are a number of python notebooks in the examples folder that explain how to use the module

## Todo

Add plotting of internal scattering




