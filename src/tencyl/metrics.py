import math

import numpy as np
import pvlib
import torch

from . import func


def plane_wave_coeffs(fibre_trunc, theta, delta, phi, complex_dtype=torch.complex128):
    """Build cylindrical-harmonic input coefficients for an incident plane wave.

    Args:
        fibre_trunc: Maximum fibre harmonic order.
        theta: Incident propagation angle tensor, flattened or broadcastable with delta.
        delta: Incident polarization mixing angle tensor.
        phi: Angle relative to the fibre length.
        complex_dtype: Complex dtype for the returned coefficients.

    Returns:
        torch.Tensor: Input coefficient matrix with E coefficients followed by K
        coefficients along the final dimension.
    """
    theta, delta = torch.broadcast_tensors(theta, delta)
    theta = theta.reshape(-1)
    delta = delta.reshape(-1)

    real_dtype = theta.dtype
    device = theta.device
    phi = torch.as_tensor(phi, dtype=real_dtype, device=device)

    order_vec = torch.arange(
        -fibre_trunc,
        fibre_trunc + 1,
        device=device,
        dtype=real_dtype,
    )
    phase = torch.exp(
        1j * order_vec[None, :] * (torch.pi / 2 - theta[:, None])
    )

    shared = torch.sin(phi) * phase
    coeffs_e = shared * torch.cos(delta)[:, None]
    coeffs_k = shared * torch.sin(delta)[:, None]
    return torch.cat((coeffs_e, coeffs_k), dim=1).to(complex_dtype)

def forward(fibre):
    def cor(S):
        N, _ = S.shape
        L = 2*N - 1 
        F = torch.fft.fft2(S, s=(L, L))
        C = torch.fft.ifft2(F.conj() * F)  
        C = torch.fft.fftshift(C, dim=(0, 1))
        idx = torch.arange(L)
        diag = C[idx, idx]
        d = torch.arange(-(N-1), N)
        w = torch.sinc(d / 2)
        return (w * diag).real.sum()
    
    matrix=fibre.R_out_complete
    terms=int(len(matrix)/2)
    S_0_EE=matrix[:terms,:terms]
    S_0_EK=matrix[:terms,terms:]
    S_0_KE=matrix[terms:,:terms]
    S_0_KK=matrix[terms:,terms:]
    
    forward_sigma=(cor(S_0_EE)+cor(S_0_EK)+cor(S_0_KE)+cor(S_0_KK)+2*(S_0_EE+S_0_KK).trace().real)/fibre.k_perp_0.real
    return forward_sigma

def cross_sections(fibre, theta, delta):
    """Compute fibre scattering and extinction cross sections.

    Args:
        fibre: instance of Fibre class.
        theta: Incident propagation angle(s), measured from the x-axis.
        delta: Incident polarization mixing angle(s); broadcasts with theta.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Scattering and extinction cross sections
        with the same broadcasted shape as theta and delta.
    """
    S = fibre.R_out_complete
    device = S.device
    real_dtype = fibre.k_perp_0.real.dtype
    complex_dtype = S.dtype if S.is_complex() else torch.complex128

    theta = torch.as_tensor(theta, dtype=real_dtype, device=device)
    delta = torch.as_tensor(delta, dtype=real_dtype, device=device)
    theta, delta = torch.broadcast_tensors(theta, delta)
    theta_shape = theta.shape

    theta_flat = theta.reshape(-1)
    delta_flat = delta.reshape(-1)
    phi = fibre.phi.to(device=device) if torch.is_tensor(fibre.phi) else fibre.phi

    A = plane_wave_coeffs(
        fibre.trunc,
        theta_flat,
        delta_flat,
        phi,
        complex_dtype,
    )
    B = torch.einsum("ab,nb->na", S.to(complex_dtype), A)

    k_perp = fibre.k_perp_0.real.to(device=device, dtype=real_dtype)
    p_inc = torch.sin(torch.as_tensor(phi, dtype=real_dtype, device=device)) ** 2
    scale = 4 / (k_perp * p_inc)

    scattering = scale * torch.sum(B.real**2 + B.imag**2, dim=1)
    extinction = -scale * torch.sum((A.conj() * B).real, dim=1)
    return scattering.reshape(theta_shape), extinction.reshape(theta_shape)


def lattice_rt(fibre, period, inc_theta, inc_delta, findT=True, lattice_imag=1e-8):
    """Compute reflected and transmitted power for a 1D periodic fibre lattice.

    The incident angles `inc_theta` and polarisation mixing angles `inc_delta` may
    be scalars or tensors. They are broadcast to a common shape, flattened into a
    single angle batch, and evaluated together. The lattice sums are computed for
    all incident k_x values in one batched Shanks call, and the multiple-scattering
    linear systems are assembled into a batched matrix solve.

    Args:
        fibre: instance of Fibre class.
        period: Period / center-to-center separation of the fibre lattice.
        inc_theta: Incident propagation angle(s), measured from the lattice x-axis.
        inc_delta: Incident polarization mixing angle(s); broadcasts with inc_theta.
        findT: If True, compute transmission; otherwise return a zero T tensor.
        lattice_imag: Small imaginary regularization added to k_perp_0 in the
            lattice sum to avoid singular behavior at anomalies.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Reflected and transmitted power tensors
        with the same broadcasted shape as inc_theta and inc_delta.
    """
    eps = 1e-100

    S = fibre.R_out_complete
    device = S.device
    real_dtype = fibre.k_perp_0.real.dtype
    complex_dtype = S.dtype if S.is_complex() else torch.complex128

    period = torch.as_tensor(period, dtype=real_dtype, device=device)
    inc_theta = torch.as_tensor(inc_theta, dtype=real_dtype, device=device)
    inc_delta = torch.as_tensor(inc_delta, dtype=real_dtype, device=device)
    inc_theta, inc_delta = torch.broadcast_tensors(inc_theta, inc_delta)
    theta_shape = inc_theta.shape

    theta = inc_theta.reshape(-1)
    delta = inc_delta.reshape(-1)
    ntheta = theta.numel()

    fibre_trunc = fibre.trunc
    terms = 2 * fibre_trunc + 1
    lattice_trunc = 2 * fibre_trunc

    jacket_vec = torch.arange(
        -fibre_trunc,
        fibre_trunc + 1,
        device=device,
        dtype=real_dtype,
    )

    p = torch.arange(-fibre_trunc, fibre_trunc + 1, device=device)[:, None]
    q = torch.arange(-fibre_trunc, fibre_trunc + 1, device=device)[None, :]
    matrix_indices = (q - p) + lattice_trunc

    eye = torch.eye(2 * terms, dtype=complex_dtype, device=device)

    k_perp_0 = fibre.k_perp_0.real.to(device=device, dtype=real_dtype)
    phi = fibre.phi.to(device=device) if torch.is_tensor(fibre.phi) else fibre.phi

    k_x_0 = k_perp_0 * torch.cos(theta)
    k_y_0 = k_perp_0 * torch.sin(theta)

    W_vec = func.shanks_seq(
        lattice_trunc,
        k_perp_0.to(complex_dtype) + 1j * lattice_imag,
        k_x_0,
        period,
        100,
    )

    W_vec = W_vec.to(device=device, dtype=complex_dtype)

    if W_vec.ndim == 1:
        W_vec = W_vec[None, :]

    if W_vec.shape[0] != ntheta and W_vec.shape[-1] == ntheta:
        W_vec = W_vec.T

    expected_w = 2 * lattice_trunc + 1
    if W_vec.shape != (ntheta, expected_w):
        raise ValueError(
            f"Expected shanks_seq to return shape {(ntheta, expected_w)}, "
            f"but got {tuple(W_vec.shape)}"
        )

    W_matrix_block = W_vec[:, matrix_indices]

    SW_left = torch.einsum(
        "ab,nbc->nac",
        S[:, :terms].to(complex_dtype),
        W_matrix_block,
    )

    SW_right = torch.einsum(
        "ab,nbc->nac",
        S[:, terms:].to(complex_dtype),
        W_matrix_block,
    )

    SW = torch.empty(
        ntheta, 2 * terms, 2 * terms,
        dtype=complex_dtype,
        device=device,
    )

    SW[:, :, :terms] = SW_left
    SW[:, :, terms:] = SW_right

    M = eye[None, :, :] - SW

    A = plane_wave_coeffs(fibre_trunc, theta, delta, phi, complex_dtype)
    RHS = torch.einsum("ab,nb->na", S.to(complex_dtype), A)
    B = torch.linalg.solve(M, RHS[..., None]).squeeze(-1)

    coeffsE, coeffsK = torch.split(B, terms, dim=1)

    cn_max = torch.floor(
        (k_perp_0 - k_x_0) * period.detach() / (2 * torch.pi)
    ).to(torch.int64)

    cn_min = torch.ceil(
        (-k_perp_0 - k_x_0) * period.detach() / (2 * torch.pi)
    ).to(torch.int64)

    cn_global_min = cn_min.min()
    cn_global_max = cn_max.max()

    cn_vec = torch.arange(
        cn_global_min,
        cn_global_max + 1,
        device=device,
        dtype=torch.int64,
    )

    valid = (cn_vec[None, :] >= cn_min[:, None]) & (
        cn_vec[None, :] <= cn_max[:, None]
    )

    cn_real = cn_vec.to(real_dtype)
    kx_cn = k_x_0[:, None] + (2 * torch.pi * cn_real[None, :]) / period

    ky_sq = k_perp_0**2 - kx_cn**2
    ky_cn = torch.sqrt(torch.clamp(ky_sq, min=0.0) + eps)

    arg_k_cn = torch.atan2(ky_cn + eps, kx_cn + eps)
    shared_term = 2 / (period * ky_cn[:, :, None])

    Rnm_shared = shared_term * torch.exp(
        1j * jacket_vec[None, None, :] *
        (-arg_k_cn[:, :, None] - torch.pi / 2)
    )

    Rn_E = torch.sum(Rnm_shared * coeffsE[:, None, :], dim=2)
    Rn_K = torch.sum(Rnm_shared * coeffsK[:, None, :], dim=2)

    P_inc = torch.sin(phi) ** 2

    abs_Rn = (
        (
            Rn_E.real**2 + Rn_E.imag**2
            + Rn_K.real**2 + Rn_K.imag**2
        )
        * ky_cn
        / (P_inc * k_y_0[:, None])
    )

    abs_Rn = torch.where(valid, abs_Rn, torch.zeros_like(abs_Rn))
    R_total = torch.sum(abs_Rn, dim=1)

    T_total = torch.zeros_like(R_total)

    if findT:
        Tnm_shared = shared_term * torch.exp(
            1j * jacket_vec[None, None, :] *
            (arg_k_cn[:, :, None] - torch.pi / 2)
        )

        Tn_E = torch.sum(Tnm_shared * coeffsE[:, None, :], dim=2)
        Tn_K = torch.sum(Tnm_shared * coeffsK[:, None, :], dim=2)

        zero_order = ((cn_vec[None, :] == 0) & valid).to(real_dtype)

        Tn_E = Tn_E + zero_order * torch.cos(delta)[:, None] * torch.sin(phi)
        Tn_K = Tn_K + zero_order * torch.sin(delta)[:, None] * torch.sin(phi)

        abs_Tn = (
            (
                Tn_E.real**2 + Tn_E.imag**2
                + Tn_K.real**2 + Tn_K.imag**2
            )
            * ky_cn
            / (P_inc * k_y_0[:, None])
        )

        abs_Tn = torch.where(valid, abs_Tn, torch.zeros_like(abs_Tn))
        T_total = torch.sum(abs_Tn, dim=1)

    return R_total.reshape(theta_shape), T_total.reshape(theta_shape)


def lattice_rt_broadband(fibres, period, theta, delta, findT=True, lattice_imag=1e-8):
    """Evaluate lattice reflection/transmission across many wavelength samples.

    Args:
        fibres: Sequence of Fibre objects, usually one per k0 value.
        period: Lattice period shared across the sequence.
        theta: Tensor or sequence of incident angles.
        delta: Tensor or sequence of polarization mixing angles.
        findT: If True, compute transmission as well as reflection.
        lattice_imag: Imaginary regularization passed through to lattice_rt.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: R and T arrays shaped [num_k0, num_theta].
    """
    theta_pts = len(theta)
    k0_pts = len(fibres)

    spec_R = torch.empty(k0_pts, theta_pts)
    spec_T = torch.empty(k0_pts, theta_pts)
    for i in range(k0_pts):
        spec_R[i, :], spec_T[i, :] = lattice_rt(
            fibres[i],
            period,
            theta,
            delta,
            findT,
            lattice_imag,
        )
    return spec_R, spec_T


def solar_weight(k0_vals, irr_vals, spectral_quantity):
    """Compute the irradiance-weighted average of a spectral quantity.

    Args:
        k0_vals: Wavenumber samples.
        irr_vals: Solar irradiance values at k0_vals.
        spectral_quantity: Quantity to average, such as reflectivity or transmission.

    Returns:
        torch.Tensor: Irradiance-weighted scalar average.
    """
    numerator = torch.trapezoid(irr_vals * spectral_quantity, x=k0_vals)
    denominator = torch.trapezoid(irr_vals, x=k0_vals)
    return numerator / denominator


def solar_irr(k0_vals):
    """Convert SI wavenumbers to reference solar spectral irradiance.

    The pvlib reference spectrum is tabulated by wavelength, so this function
    interpolates in wavelength and applies the Jacobian back to wavenumber.

    Args:
        k0_vals: SI wavenumber tensor.

    Returns:
        torch.Tensor: Spectral irradiance in W m^-1 on the same device as k0_vals.
    """
    device = k0_vals.device
    dtype = k0_vals.dtype
    k0_vals_np = k0_vals.detach().cpu().numpy()
    lam_nm = (2 * np.pi / k0_vals_np) * 1e9

    df_standard = pvlib.spectrum.get_reference_spectra()
    ref_wl = df_standard.index.values
    ref_irr = df_standard["global"].values

    jacobian = 2 * np.pi / (k0_vals_np**2) * 1e9
    irr_vals = jacobian * np.interp(lam_nm, ref_wl, ref_irr, left=0.0, right=0.0)

    return torch.as_tensor(irr_vals, dtype=dtype, device=device)


def solar_k0_quadrature(k0_pts, k0_min, k0_max, k0_pts_dense_grid=200000):
    """Generate solar-weighted wavenumber nodes and quadrature weights.

    Args:
        k0_pts: Number of quadrature points to generate.
        k0_min: Lower wavenumber bound.
        k0_max: Upper wavenumber bound.
        k0_pts_dense_grid: Resolution of the internal CDF integration grid.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Sampled wavenumber nodes and associated
        weights for integrating functions under the solar spectrum.

    Raises:
        ValueError: If the solar irradiance in the requested range is zero.
    """
    k0_vals = torch.linspace(k0_min, k0_max, k0_pts_dense_grid)
    x = k0_vals.detach().cpu().numpy()
    y = solar_irr(k0_vals).clamp_min(0).detach().cpu().numpy()

    area = np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))]
    total = area[-1]

    if total <= 0:
        raise ValueError("Solar spectrum has zero irradiance in this wavelength range.")

    area_u, idx = np.unique(area, return_index=True)
    x_u = x[idx]

    nodes, weights = np.polynomial.legendre.leggauss(k0_pts)

    cdf_nodes = 0.5 * (nodes + 1.0) * total
    k0_weights = 0.5 * weights

    k0_vals = np.interp(cdf_nodes, area_u, x_u)

    return torch.as_tensor(k0_vals), torch.as_tensor(k0_weights)


def angular_quadrature(theta_min, theta_max, pts):
    """Compute Gauss-Legendre quadrature nodes and weights over an angle interval.

    Args:
        theta_min: Lower bound of the integration interval.
        theta_max: Upper bound of the integration interval.
        pts: The number of quadrature points to generate.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Angle nodes and interval-scaled weights.
    """
    nodes, weights = np.polynomial.legendre.leggauss(pts)
    nodes = torch.from_numpy(nodes).to(torch.float64)
    weights = torch.from_numpy(weights).to(torch.float64)
    theta = 0.5 * (theta_max - theta_min) * nodes + 0.5 * (
        theta_max + theta_min
    )
    adjusted_weights = 0.5 * (theta_max - theta_min) * weights
    return theta, adjusted_weights


def dist_2_anomaly(k_perp, k_x, period, buffer=5):
    """Calculate distance in period to the nearest Wood anomaly.

    Args:
        k_perp: Perpendicular wavevector.
        k_x: Wavevector component parallel to the lattice axis.
        period: Lattice period.
        buffer: Extra diffraction orders to include beyond the estimated bound.

    Returns:
        torch.Tensor: Distance to the closest anomaly in reciprocal-space mismatch.
    """
    if torch.is_tensor(period):
        device = period.device
    elif torch.is_tensor(k_x):
        device = k_x.device
    elif torch.is_tensor(k_perp):
        device = k_perp.device
    else:
        device = torch.device("cpu")

    period = torch.as_tensor(period, dtype=torch.float64, device=device)
    k_x = torch.as_tensor(k_x, dtype=torch.float64, device=device)
    k_perp = torch.as_tensor(k_perp, device=device).real

    G = 2 * math.pi / period
    m_bound = torch.ceil((k_perp + torch.abs(k_x)) / G)
    m_max = m_bound + buffer
    m_max_int = int(m_max.detach().cpu().item())
    m = torch.arange(-m_max_int, m_max_int + 1, dtype=torch.float64, device=device)
    k_xm = k_x + m * G
    distances = torch.abs(torch.abs(k_xm) - k_perp)
    min_dist, _ = torch.min(distances, dim=0)
    return min_dist


def external_field(fibre, theta, delta, xrange, yrange, grid_pts=300):
    """Return the exterior source-plus-scattered E and K fields on a 2D grid.

    Args:
        fibre: instance of the Fibre class
        theta: Incident propagation angle. This helper expects one angle.
        delta: Incident polarization mixing angle. This helper expects one value.
        xrange: Either (xmin, xmax) or a one-dimensional tensor of x coordinates.
        yrange: Either (ymin, ymax) or a one-dimensional tensor of y coordinates.
        grid_pts: Number of points per axis when xrange or yrange is a two-value
            interval.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Complex 2D tensors for total E and K.
    """
    S = fibre.R_out_complete
    device = S.device
    real_dtype = fibre.k_perp_0.real.dtype
    complex_dtype = S.dtype if S.is_complex() else torch.complex128

    theta = torch.as_tensor(theta, dtype=real_dtype, device=device)
    delta = torch.as_tensor(delta, dtype=real_dtype, device=device)
    theta, delta = torch.broadcast_tensors(theta, delta)
    if theta.numel() != 1:
        raise ValueError("external_field expects a single theta/delta pair.")

    phi = fibre.phi.to(device=device) if torch.is_tensor(fibre.phi) else fibre.phi
    A = plane_wave_coeffs(
        fibre.trunc,
        theta.reshape(-1),
        delta.reshape(-1),
        phi,
        complex_dtype,
    )
    B = torch.einsum("ab,nb->na", S.to(complex_dtype), A)

    x = _field_axis(xrange, grid_pts, real_dtype, device)
    y = _field_axis(yrange, grid_pts, real_dtype, device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    theta = theta.reshape(())
    delta = delta.reshape(())
    phi = torch.as_tensor(phi, dtype=real_dtype, device=device)
    k_perp = fibre.k_perp_0.to(device=device, dtype=complex_dtype)
    radius = torch.sqrt(xx**2 + yy**2)
    outside = radius >= fibre.a.to(device=device, dtype=real_dtype)

    phase = torch.exp(1j * k_perp * (xx * torch.cos(theta) + yy * torch.sin(theta)))
    source_e = torch.sin(phi) * torch.cos(delta) * phase
    source_k = torch.sin(phi) * torch.sin(delta) * phase

    terms = 2 * fibre.trunc + 1
    b_e, b_k = torch.split(B[0], terms)
    basis_h = _external_hankel_basis(
        fibre.trunc,
        k_perp,
        radius,
        torch.atan2(yy, xx),
        outside,
    )

    scatter_e = torch.sum(b_e[:, None, None] * basis_h, dim=0)
    scatter_k = torch.sum(b_k[:, None, None] * basis_h, dim=0)
    complex_nan = complex(float("nan"), float("nan"))
    E = torch.where(
        outside,
        source_e + scatter_e,
        torch.full_like(source_e, complex_nan),
    )
    K = torch.where(
        outside,
        source_k + scatter_k,
        torch.full_like(source_k, complex_nan),
    )
    return E, K

def _field_axis(axis_range, grid_pts, dtype, device):
    """Build one plot axis from either an interval or explicit coordinates.

    Args:
        axis_range: Two-value interval or one-dimensional coordinate tensor.
        grid_pts: Number of samples when axis_range is an interval.
        dtype: Real dtype for the output tensor.
        device: Device for the output tensor.

    Returns:
        torch.Tensor: One-dimensional coordinate tensor.
    """
    axis = torch.as_tensor(axis_range, dtype=dtype, device=device)
    if axis.ndim != 1:
        raise ValueError("xrange and yrange must be one-dimensional.")
    if axis.numel() == 2:
        return torch.linspace(axis[0], axis[1], grid_pts, dtype=dtype, device=device)
    return axis

def _external_hankel_basis(trunc, k_perp, radius, angle, outside):
    """Build H_m^(1) angular basis tensors for exterior scattered-field plots.

    Args:
        trunc: Maximum cylindrical harmonic order.
        k_perp: Exterior perpendicular wavenumber.
        radius: Grid radius tensor.
        angle: Grid azimuth tensor.
        outside: Boolean mask selecting exterior grid points.

    Returns:
        torch.Tensor: H basis tensor ordered from -trunc to +trunc on axis 0.
    """
    safe_radius = torch.where(outside, radius.clamp_min(1e-300), radius.new_ones(()))
    arg = k_perp * safe_radius
    h_nonneg = func.h1v_seq(trunc, arg)

    signs = (1 - 2 * (torch.arange(1, trunc + 1, device=radius.device) % 2))
    signs = signs[:, None, None]
    h_full = torch.cat([torch.flip(h_nonneg[1:] * signs, [0]), h_nonneg], dim=0)

    orders = torch.arange(-trunc, trunc + 1, dtype=radius.dtype, device=radius.device)
    angular = torch.exp(1j * orders[:, None, None] * angle[None, :, :])
    basis_h = h_full * angular

    mask = outside[None, :, :]
    return basis_h * mask