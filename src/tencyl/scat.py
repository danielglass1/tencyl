import torch

from .func import jv_seq, yv_seq

TRUNCATION_SCALE = 1.0

def set_global_truncation(scale: float = 1.0):
    """Set a global multiplier for Wiscombe-style truncation estimates.

    Args:
        scale: Multiplicative factor applied to future truncation estimates.

    Returns:
        None
    """
    global TRUNCATION_SCALE
    TRUNCATION_SCALE = float(scale)


def get_k_perp(n, k0, k_z):
    """Compute the wavevector component perpendicular to the cylinder length.

    Args:
        n: Refractive index.
        k0: Vacuum wavenumber.
        k_z: Wavevector component parallel to the cylinder length.

    Returns:
        torch.Tensor: Perpendicular wavevector component.
    """
    k = n * k0
    return torch.sqrt(k * k - k_z * k_z)


def wiscombe_trunc(size_parameter: torch.Tensor) -> torch.Tensor:
    """Estimate a conservative cylindrical harmonic truncation order.

    Args:
        size_parameter: Complex size parameter, normally k_perp * radius.

    Returns:
        torch.Tensor: Integer-like truncation limit after applying TRUNCATION_SCALE.
    """
    size_real = size_parameter.to(torch.complex128).real
    wiscombe = size_real + 4 * size_real ** (1 / 2) + 2
    return TRUNCATION_SCALE * torch.ceil(wiscombe).to(torch.int)


def _bessel_deriv_centered(values):
    """Compute Bessel derivatives from neighboring orders.

    Args:
        values: Tensor containing orders from -1 through trunc + 1 on axis 0.

    Returns:
        torch.Tensor: Centered derivative values for orders 0 through trunc.
    """
    return 0.5 * (values[:-2] - values[2:])


def _add_negative_orders(coeffs):
    """Mirror non-negative scattering coefficients into negative harmonic orders.

    Args:
        coeffs: Tensor shaped as four polarization coefficient rows followed by
        order and optional batch dimensions.

    Returns:
        torch.Tensor: Coefficients ordered from -trunc through +trunc.
    """
    coeffs_neg = torch.flip(coeffs[:, 1:, ...], dims=[1])
    coeffs_neg[1:3].neg_()
    return torch.cat([coeffs_neg, coeffs], dim=1)


def coeffs(
    k_0: torch.Tensor,
    k_z: torch.Tensor,
    n_o: torch.Tensor,
    n_i: torch.Tensor,
    a: torch.Tensor,
    trunc: int,
):
    """Compute scattering coefficients at a cylindrical refractive-index boundary.

    Args:
        k_0: Vacuum wavenumber.
        k_z: Component of wavevector along the cylinder length.
        n_o: Exterior refractive index.
        n_i: Interior refractive index.
        a: Cylinder boundary radius.
        trunc: Maximum non-negative harmonic order.

    Returns:
        tuple: R_out, R_in, T_out, and T_in tensors. Each has four polarization
        rows ordered EE, EK, KE, KK and harmonic orders from -trunc to +trunc.
    """
    k_perp_o = get_k_perp(n_o, k_0, k_z)
    k_perp_i = get_k_perp(n_i, k_0, k_z)
    trunc = int(torch.as_tensor(trunc).detach().cpu().item())

    ka_combined = torch.stack([k_perp_o, k_perp_i]) * a

    j_combined = jv_seq(trunc + 1, ka_combined)
    h_combined = j_combined + 1j * yv_seq(trunc + 1, ka_combined)

    j_o = j_combined[:, 0]
    j_i = j_combined[:, 1]
    h_o = h_combined[:, 0]
    h_i = h_combined[:, 1]

    j_o = torch.cat((-j_o[1].unsqueeze(0), j_o))
    j_i = torch.cat((-j_i[1].unsqueeze(0), j_i))
    h_o = torch.cat((-h_o[1].unsqueeze(0), h_o))
    h_i = torch.cat((-h_i[1].unsqueeze(0), h_i))

    hp_o = _bessel_deriv_centered(h_o)
    hp_i = _bessel_deriv_centered(h_i)
    jp_o = _bessel_deriv_centered(j_o)
    jp_i = _bessel_deriv_centered(j_i)

    j_o = j_o[1:-1]
    j_i = j_i[1:-1]
    h_o = h_o[1:-1]
    h_i = h_i[1:-1]
    order_vec = torch.arange(0, trunc + 1, device=ka_combined.device)

    x_jj_e = k_perp_o * n_i**2 * jp_i * j_o - k_perp_i * n_o**2 * j_i * jp_o
    x_jj_m = k_perp_o * jp_i * j_o - k_perp_i * j_i * jp_o
    x_hh_e = k_perp_o * n_i**2 * hp_i * h_o - k_perp_i * n_o**2 * h_i * hp_o
    x_hh_m = k_perp_o * hp_i * h_o - k_perp_i * h_i * hp_o
    x_jh_e = k_perp_o * n_i**2 * jp_i * h_o - k_perp_i * n_o**2 * j_i * hp_o
    x_jh_m = k_perp_o * jp_i * h_o - k_perp_i * j_i * hp_o

    c = (k_perp_i**2 - k_perp_o**2) * (k_z * 1j * order_vec) / (
        a * k_0 * k_perp_i * k_perp_o
    )
    delta = (c * j_i * h_o) ** 2 + x_jh_e * x_jh_m
    denominator = 1 / delta
    w_o = (2j) / (torch.pi * k_perp_o * a)
    w_i = (2j) / (torch.pi * k_perp_i * a)

    r_out_diag = -(c * j_i) ** 2 * h_o * j_o
    r_out_ee = denominator * (r_out_diag - x_jh_m * x_jj_e)
    r_out_kk = denominator * (r_out_diag - x_jh_e * x_jj_m)
    r_out_cross = denominator * k_perp_i * c * w_o * j_i**2
    r_out_ek = -r_out_cross
    r_out_ke = r_out_cross * n_o**2
    r_out = _add_negative_orders(torch.stack([r_out_ee, r_out_ek, r_out_ke, r_out_kk]))

    t_in_diag = -denominator * k_perp_i * w_o
    t_in_ee = t_in_diag * n_o**2 * x_jh_m
    t_in_kk = t_in_diag * x_jh_e
    t_in_cross = t_in_diag * c * j_i * h_o
    t_in_ek = t_in_cross
    t_in_ke = -t_in_cross * n_o**2

    r_in_diag = -(c * h_o) ** 2 * h_i * j_i
    r_in_ee = denominator * (r_in_diag - x_jh_m * x_hh_e)
    r_in_kk = denominator * (r_in_diag - x_jh_e * x_hh_m)
    r_in_cross = denominator * k_perp_o * c * w_i * h_o**2
    r_in_ek = -r_in_cross
    r_in_ke = r_in_cross * n_i**2

    t_out_diag = -denominator * k_perp_o * w_i
    t_out_ee = t_out_diag * n_i**2 * x_jh_m
    t_out_kk = t_out_diag * x_jh_e
    t_out_cross = t_out_diag * c * j_i * h_o
    t_out_ek = t_out_cross
    t_out_ke = -t_out_cross * n_i**2

    r_in = torch.stack([r_in_ee, r_in_ek, r_in_ke, r_in_kk])
    t_out = torch.stack([t_out_ee, t_out_ek, t_out_ke, t_out_kk])
    t_in = torch.stack([t_in_ee, t_in_ek, t_in_ke, t_in_kk])

    return (
        r_out,
        _add_negative_orders(r_in),
        _add_negative_orders(t_out),
        _add_negative_orders(t_in),
    )


def R_out_batched_radii(
    k_0: torch.Tensor,
    k_z: torch.Tensor,
    n_o: torch.Tensor,
    n_i: torch.Tensor,
    a: torch.Tensor,
    trunc: int,
) -> torch.Tensor:
    """Compute exterior reflection coefficients for many cylinder radii at once.

    Args:
        k_0: Vacuum wavenumber.
        k_z: Component of wavevector along the cylinder length.
        n_o: Exterior refractive index shared by all cylinders.
        n_i: Interior refractive indices for each cylinder.
        a: Cylinder radii.
        trunc: Maximum non-negative harmonic order.

    Returns:
        torch.Tensor: R_out coefficients with shape [4, 2 * trunc + 1, num_cyls].
    """
    k_perp_o = get_k_perp(n_o, k_0, k_z)
    k_perp_i = get_k_perp(n_i, k_0, k_z)
    trunc = int(torch.as_tensor(trunc).detach().cpu().item())

    a = torch.as_tensor(a, dtype=torch.float64, device=k_0.device)
    n_i = torch.as_tensor(n_i, dtype=torch.complex128, device=k_0.device)
    k_perp_i = torch.as_tensor(k_perp_i, dtype=torch.complex128, device=k_0.device)

    ka_combined = torch.stack([k_perp_o * a, k_perp_i * a], dim=0)
    j_combined = jv_seq(trunc + 1, ka_combined)

    j_o = j_combined[:, 0, :]
    j_i = j_combined[:, 1, :]
    h_o = j_o + 1j * yv_seq(trunc + 1, k_perp_o * a)

    j_o = torch.cat((-j_o[1:2, :], j_o), dim=0)
    j_i = torch.cat((-j_i[1:2, :], j_i), dim=0)
    h_o = torch.cat((-h_o[1:2, :], h_o), dim=0)

    hp_o = _bessel_deriv_centered(h_o)
    jp_o = _bessel_deriv_centered(j_o)
    jp_i = _bessel_deriv_centered(j_i)

    j_o = j_o[1:-1, :]
    j_i = j_i[1:-1, :]
    h_o = h_o[1:-1, :]
    order_vec = torch.arange(0, trunc + 1, device=ka_combined.device)[:, None]

    k_perp_i_b = k_perp_i[None, :]
    n_i_b = n_i[None, :]
    a_b = a[None, :]

    x_jj_e = k_perp_o * n_i_b**2 * jp_i * j_o - k_perp_i_b * n_o**2 * j_i * jp_o
    x_jj_m = k_perp_o * jp_i * j_o - k_perp_i_b * j_i * jp_o
    x_jh_e = k_perp_o * n_i_b**2 * jp_i * h_o - k_perp_i_b * n_o**2 * j_i * hp_o
    x_jh_m = k_perp_o * jp_i * h_o - k_perp_i_b * j_i * hp_o

    c = (k_perp_i_b**2 - k_perp_o**2) * (k_z * 1j * order_vec) / (
        a_b * k_0 * k_perp_i_b * k_perp_o
    )
    delta = (c * j_i * h_o) ** 2 + x_jh_e * x_jh_m
    denominator = 1 / delta
    w_o = (2j) / (torch.pi * k_perp_o * a_b)

    r_out_diag = -(c * j_i) ** 2 * h_o * j_o
    r_out_ee = denominator * (r_out_diag - x_jh_m * x_jj_e)
    r_out_kk = denominator * (r_out_diag - x_jh_e * x_jj_m)
    r_out_cross = denominator * k_perp_i_b * c * w_o * j_i**2
    r_out_ek = -r_out_cross
    r_out_ke = r_out_cross * n_o**2

    return _add_negative_orders(torch.stack([r_out_ee, r_out_ek, r_out_ke, r_out_kk]))

def T_in_batched_radii(
    k_0: torch.Tensor,
    k_z: torch.Tensor,
    n_o: torch.Tensor,
    n_i: torch.Tensor,
    a: torch.Tensor,
    trunc: int,
) -> torch.Tensor:
    """Compute inward transmission coefficients for many cylinder radii at once.

    Args:
        k_0: Vacuum wavenumber.
        k_z: Component of wavevector along the cylinder length.
        n_o: Exterior refractive index shared by all cylinders.
        n_i: Interior refractive indices for each cylinder.
        a: Cylinder radii.
        trunc: Maximum non-negative harmonic order.

    Returns:
        torch.Tensor: T_in coefficients with shape [4, 2 * trunc + 1, num_cyls].
    """
    k_perp_o = get_k_perp(n_o, k_0, k_z)
    k_perp_i = get_k_perp(n_i, k_0, k_z)
    trunc = int(torch.as_tensor(trunc).detach().cpu().item())

    a = torch.as_tensor(a, dtype=torch.float64, device=k_0.device)
    n_i = torch.as_tensor(n_i, dtype=torch.complex128, device=k_0.device)
    k_perp_i = torch.as_tensor(k_perp_i, dtype=torch.complex128, device=k_0.device)

    ka_combined = torch.stack([k_perp_o * a, k_perp_i * a], dim=0)
    j_combined = jv_seq(trunc + 1, ka_combined)

    j_o = j_combined[:, 0, :]
    j_i = j_combined[:, 1, :]
    h_o = j_o + 1j * yv_seq(trunc + 1, k_perp_o * a)

    j_o = torch.cat((-j_o[1:2, :], j_o), dim=0)
    j_i = torch.cat((-j_i[1:2, :], j_i), dim=0)
    h_o = torch.cat((-h_o[1:2, :], h_o), dim=0)

    hp_o = _bessel_deriv_centered(h_o)
    jp_i = _bessel_deriv_centered(j_i)

    j_i = j_i[1:-1, :]
    h_o = h_o[1:-1, :]
    
    order_vec = torch.arange(0, trunc + 1, device=ka_combined.device)[:, None]

    k_perp_i_b = k_perp_i[None, :]
    n_i_b = n_i[None, :]
    a_b = a[None, :]

    x_jh_e = k_perp_o * n_i_b**2 * jp_i * h_o - k_perp_i_b * n_o**2 * j_i * hp_o
    x_jh_m = k_perp_o * jp_i * h_o - k_perp_i_b * j_i * hp_o

    c = (k_perp_i_b**2 - k_perp_o**2) * (k_z * 1j * order_vec) / (
        a_b * k_0 * k_perp_i_b * k_perp_o
    )
    delta = (c * j_i * h_o) ** 2 + x_jh_e * x_jh_m
    denominator = 1 / delta
    w_o = (2j) / (torch.pi * k_perp_o * a_b)

    t_in_diag = -denominator * k_perp_i_b * w_o
    t_in_ee = t_in_diag * n_o**2 * x_jh_m
    t_in_kk = t_in_diag * x_jh_e
    t_in_cross = t_in_diag * c * j_i * h_o
    t_in_ek = t_in_cross
    t_in_ke = -t_in_cross * n_o**2

    return _add_negative_orders(torch.stack([t_in_ee, t_in_ek, t_in_ke, t_in_kk]))
