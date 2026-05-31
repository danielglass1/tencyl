import scipy.special as sp
import torch
from torch.autograd import Function


class BesselJv(Function):
    """Autograd wrapper for scipy.special.jv with gradients in the argument only."""

    @staticmethod
    def forward(ctx, n, x):
        """Evaluate J_n(x) through SciPy.

        Args:
            ctx: Autograd context used to store x and n for backward.
            n: Bessel order tensor; gradients are not propagated through this.
            x: Real or complex argument tensor.

        Returns:
            torch.Tensor: Bessel J values with the same device as x.
        """
        ctx.save_for_backward(x)
        ctx.n = n
        n_np = n.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        res = sp.jv(n_np, x_np)
        return torch.from_numpy(res).to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        """Backpropagate through x using dJ_n/dx = 0.5 * (J_{n-1} - J_{n+1}).

        Args:
            ctx: Autograd context containing the saved forward inputs.
            grad_output: Upstream gradient from PyTorch.

        Returns:
            tuple: None for n and the gradient contribution for x.
        """
        (x,) = ctx.saved_tensors
        n = ctx.n
        n_np = n.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        deriv = 0.5 * (sp.jv(n_np - 1, x_np) - sp.jv(n_np + 1, x_np))
        return None, grad_output * torch.conj(torch.from_numpy(deriv).to(x.device))


class BesselYv(Function):
    """Autograd wrapper for scipy.special.yv with gradients in the argument only."""

    @staticmethod
    def forward(ctx, n, x):
        """Evaluate Y_n(x) through SciPy.

        Args:
            ctx: Autograd context used to store x and n for backward.
            n: Bessel order tensor; gradients are not propagated through this.
            x: Real or complex argument tensor.

        Returns:
            torch.Tensor: Bessel Y values with the same device as x.
        """
        ctx.save_for_backward(x)
        ctx.n = n
        n_np = n.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        res = sp.yv(n_np, x_np)
        return torch.from_numpy(res).to(x.device)

    @staticmethod
    def backward(ctx, grad_output):
        """Backpropagate through x using dY_n/dx = 0.5 * (Y_{n-1} - Y_{n+1}).

        Args:
            ctx: Autograd context containing the saved forward inputs.
            grad_output: Upstream gradient from PyTorch.

        Returns:
            tuple: None for n and the gradient contribution for x.
        """
        (x,) = ctx.saved_tensors
        n = ctx.n
        n_np = n.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        deriv = 0.5 * (sp.yv(n_np - 1, x_np) - sp.yv(n_np + 1, x_np))
        return None, grad_output * torch.conj(torch.from_numpy(deriv).to(x.device))


def jv_seq(trunc_max, z):
    """Compute J_n(z) for all orders n from 0 to trunc_max.

    Args:
        trunc_max: Largest non-negative Bessel order to evaluate.
        z: Argument tensor; any shape is allowed.

    Returns:
        torch.Tensor: Values with order as the first dimension and z's shape after it.
    """
    n_vec = torch.arange(0, trunc_max + 1, device=z.device)
    n_vec = n_vec.view(-1, *(1,) * z.ndim)
    return BesselJv.apply(n_vec, z)


def yv_seq(trunc_max, z):
    """Compute Y_n(z) for all orders n from 0 to trunc_max.

    Args:
        trunc_max: Largest non-negative Bessel order to evaluate.
        z: Argument tensor; any shape is allowed.

    Returns:
        torch.Tensor: Values with order as the first dimension and z's shape after it.
    """
    n_vec = torch.arange(0, trunc_max + 1, device=z.device)
    n_vec = n_vec.view(-1, *(1,) * z.ndim)
    return BesselYv.apply(n_vec, z)


def h1v_seq(trunc_max, z):
    """Compute Hankel H_n^(1)(z) for all orders n from 0 to trunc_max.

    Args:
        trunc_max: Largest non-negative Hankel order to evaluate.
        z: Argument tensor; any shape is allowed.

    Returns:
        torch.Tensor: Values of J_n(z) + iY_n(z), ordered by n along axis 0.
    """
    return jv_seq(trunc_max, z) + 1j * yv_seq(trunc_max, z)


_SHANKS_STATIC_CACHE = {}


def _device_cache_key(device: torch.device) -> tuple[str, int]:
    """Convert a torch device into a hashable cache key.

    Args:
        device: Device whose type and index should identify cached tensors.

    Returns:
        tuple[str, int]: Device type and index, with -1 representing the default index.
    """
    return device.type, -1 if device.index is None else device.index


def _get_shanks_static(trunc_max: int, num_terms: int, device: torch.device):
    """Fetch or create static Shanks helper tensors for one device and size.

    Args:
        trunc_max: Maximum harmonic order in the non-negative half of the sum.
        num_terms: Number of positive lattice terms in the real-space series.
        device: Device where the cached tensors should live.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Positive lattice indices and parity signs.
    """
    key = (_device_cache_key(device), trunc_max, num_terms)
    cached = _SHANKS_STATIC_CACHE.get(key)
    if cached is not None:
        return cached

    p = torch.arange(1, num_terms + 1, dtype=torch.float64, device=device)
    signs = (1 - 2 * (torch.arange(trunc_max + 1, device=device) & 1)).unsqueeze(1)
    cached = p, signs
    _SHANKS_STATIC_CACHE[key] = cached
    return cached


def apply_shanks(terms, signs, tol):
    """Apply the epsilon/Shanks transform and mirror positive orders to negatives.

    Args:
        terms: Series terms with lattice index on the final dimension.
        signs: Harmonic parity signs used to recover negative orders.
        tol: Minimum denominator magnitude in the recurrence.

    Returns:
        torch.Tensor: Accelerated coefficients ordered from -trunc_max to +trunc_max.
    """
    partial_sums = torch.cumsum(terms, dim=-1)
    eps_prev = torch.zeros_like(partial_sums)
    eps_curr = partial_sums
    w_pos = eps_curr[..., 0]

    for i in range(1, partial_sums.shape[-1]):
        diff = eps_curr[..., 1:] - eps_curr[..., :-1]
        diff = diff.masked_fill(torch.abs(diff) < tol, tol)
        eps_next = eps_prev[..., 1:] + 1.0 / diff
        eps_prev = eps_curr[..., :-1]
        eps_curr = eps_next
        if i % 2 == 0:
            w_pos = eps_curr[..., 0]

    w_neg = torch.flip(signs.squeeze(-1) * w_pos, dims=[-1])[..., :-1]
    return torch.cat((w_neg, w_pos), dim=-1)


class ShanksLatticeSumPeriod(torch.autograd.Function):
    """Shanks-accelerated lattice sum with a custom gradient for period only."""

    @staticmethod
    def forward(ctx, trunc_max, k_perp, k_x, period, num_terms, tol):
        """Evaluate quasi-periodic lattice coefficients.

        Args:
            ctx: Autograd context used to save tensors for the custom backward pass.
            trunc_max: Maximum harmonic order for the returned coefficients.
            k_perp: Perpendicular background wavenumber.
            k_x: Bloch/Floquet component along the lattice direction; may be batched.
            period: Lattice period.
            num_terms: Number of real-space terms before Shanks acceleration.
            tol: Minimum denominator magnitude in the Shanks recurrence.

        Returns:
            torch.Tensor: Coefficients ordered from -trunc_max to +trunc_max.
        """
        p, signs = _get_shanks_static(trunc_max, num_terms, k_perp.device)
        arg = k_perp * p * period

        h_all = h1v_seq(trunc_max + 1, arg)
        hankel_term = h_all[:-1]
        is_batched = k_x.ndim > 0

        if is_batched:
            phase_arg = k_x[:, None] * p[None, :] * period
            phase_plus = torch.exp(1j * phase_arg).unsqueeze(1)
            phase_minus = torch.exp(-1j * phase_arg).unsqueeze(1)
            signs_used = signs.unsqueeze(0)
            terms = hankel_term.unsqueeze(0) * (phase_minus + signs_used * phase_plus)
        else:
            phase_arg = k_x * p * period
            phase_plus = torch.exp(1j * phase_arg)
            phase_minus = torch.exp(-1j * phase_arg)
            signs_used = signs
            terms = hankel_term * (phase_minus + signs_used * phase_plus)

        ctx.save_for_backward(
            k_perp,
            k_x,
            period,
            p,
            signs,
            phase_plus,
            phase_minus,
            h_all,
        )
        ctx.tol = tol
        ctx.is_batched = is_batched

        return apply_shanks(terms, signs, tol)

    @staticmethod
    def backward(ctx, grad_output):
        """Backpropagate through the lattice period.

        Args:
            ctx: Autograd context containing saved forward tensors.
            grad_output: Upstream gradient from PyTorch.

        Returns:
            tuple: Gradients for forward inputs; only period is currently populated.
        """
        k_perp, k_x, period, p, signs, phase_plus, phase_minus, h_all = ctx.saved_tensors
        tol = ctx.tol
        is_batched = ctx.is_batched

        grad_period = None

        if ctx.needs_input_grad[3]:
            hankel_term = h_all[:-1]
            h_plus = h_all[1:]
            h_minus = torch.cat([-1 * h_all[1:2], h_all[:-2]], dim=0)

            dh_darg = 0.5 * (h_minus - h_plus)

            if is_batched:
                dh_darg = dh_darg.unsqueeze(0)
                hankel_term = hankel_term.unsqueeze(0)
                signs_used = signs.unsqueeze(0)
                kx_p = k_x[:, None, None] * p[None, None, :]
            else:
                signs_used = signs
                kx_p = k_x * p

            term1 = (k_perp * p) * dh_darg * (phase_minus + signs_used * phase_plus)
            term2 = hankel_term * (-1j * kx_p) * (phase_minus - signs_used * phase_plus)

            terms_grad_period = term1 + term2
            res_grad_period = apply_shanks(terms_grad_period, signs, tol)
            grad_period = torch.sum(
                grad_output * res_grad_period.conj()
            ).real.view_as(period)

        return None, None, None, grad_period, None, None


def shanks_seq(trunc_max, k_perp, k_x, period, num_terms=100, tol=1e-60):
    """Compute Shanks-accelerated quasi-periodic lattice-sum coefficients.

    Args:
        trunc_max: Maximum cylindrical harmonic order in the lattice sum.
        k_perp: Transverse wavenumber, often with a small positive imaginary part.
        k_x: Bloch/Floquet wavevector component along the lattice; scalar or batched.
        period: Lattice spacing.
        num_terms: Number of real-space lattice terms before Shanks acceleration.
        tol: Small cutoff used to avoid division by near-zero recurrence differences.

    Returns:
        torch.Tensor: Coefficients ordered from negative to positive harmonic index.
    """
    return ShanksLatticeSumPeriod.apply(trunc_max, k_perp, k_x, period, num_terms, tol)
