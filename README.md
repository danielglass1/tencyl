# tencyl

`tencyl` is a small PyTorch-based toolkit for modelling cylindrical scattering
from fibres, embedded cylinders, and one-dimensional periodic fibre lattices.

The code represents fields with cylindrical harmonics. A `Fibre` stores the
single-fibre scattering matrix, can be updated with embedded cylinders, and can
then be used to compute cross sections, external fields, or lattice reflection and
transmission.

## Requirements

The package expects:

- Python 3
- PyTorch
- SciPy
- NumPy
- pvlib, for solar-spectrum utilities
- Matplotlib, only for geometry preview plots

Install these in your own environment before importing the package.

## Package Layout

- `fibre.py`: `Fibre` class and embedded-cylinder multiple scattering.
- `scat.py`: cylindrical-interface scattering coefficients.
- `func.py`: Bessel/Hankel helpers and Shanks-accelerated lattice sums.
- `rt.py`: cross sections, external fields, lattice reflection/transmission, and
  solar quadrature helpers.
- `geo.py`: geometry construction, previewing, and random structure generation.
- `opt.py`: optimization helpers, geometry penalties, and small conversion utilities.

## Basic Usage

```python
import torch
from tencyl.fibre import Fibre
from tencyl.geo import build_cyl_matrix
from tencyl.rt import cross_sections, external_field, lattice_rt

k0 = torch.tensor(2 * torch.pi / 0.55)
phi = torch.tensor(torch.pi / 2)
fibre_n = torch.tensor(1.45 + 0j)
fibre_radius = torch.tensor(3.0)

fibre = Fibre(k0, phi, fibre_n, fibre_radius)

pos_x = torch.tensor([0.4, -0.5])
pos_y = torch.tensor([0.2, 0.3])
rad = torch.tensor([0.08, 0.10])
cyl_n = torch.tensor(1.0 + 0j)

cylinders = build_cyl_matrix(pos_x, pos_y, rad, cyl_n)
fibre.add_cylinders(cylinders)
```

The cylinder matrix has columns:

```text
x, y, radius, refractive_index, truncation
```

If the truncation column is `0`, the code uses the Wiscombe-style truncation
estimate for that cylinder.

## Cross Sections

```python
theta = torch.linspace(0.1, 1.4, 100)
delta = torch.tensor(0.0)

scattering, extinction = cross_sections(fibre, theta, delta)
```

`theta` and `delta` broadcast together, and the returned tensors have the same
broadcasted shape.

## External Field

```python
E, K = external_field(
    fibre,
    theta=torch.tensor(0.4),
    delta=torch.tensor(0.0),
    xrange=(-6.0, 6.0),
    yrange=(-6.0, 6.0),
    grid_pts=300,
)
```

`E` and `K` are complex 2D tensors for the total exterior field. The source field
is evaluated as a direct plane wave, while the scattered field is reconstructed
from outgoing Hankel functions and the fibre scattering matrix. Values inside the
fibre are returned as complex `nan`.

## Periodic Lattice Reflection and Transmission

```python
period = torch.tensor(8.0)
theta = torch.linspace(0.1, 1.4, 100)
delta = torch.tensor(0.0)

R, T = lattice_rt(fibre, period, theta, delta)
```

`lattice_rt` returns total reflected and transmitted power summed over open
diffraction orders. The lattice sum uses Shanks acceleration and a small imaginary
regularization by default.

## Optimization Helpers

```python
from tencyl.opt import combine_inputs, separate_inputs, get_penalty

var = combine_inputs(period, pos_x, pos_y, rad)
period, pos_x, pos_y, rad = separate_inputs(var, num_cyls=2, fibre_a=fibre_radius)

penalty = get_penalty(period, pos_x, pos_y, rad, fibre_radius)
```

`seperate_inputs` is still available as a compatibility alias for older scripts,
but new code should use `separate_inputs`.

## Notes

- Tensors are generally kept on the device of the input/fibre tensors.
- Bessel functions are evaluated through SciPy wrappers, so full GPU execution is
  not expected for those calls.
- `plane_wave_coeffs` returns cylindrical-harmonic input coefficients ordered as
  all `E` coefficients followed by all `K` coefficients.
