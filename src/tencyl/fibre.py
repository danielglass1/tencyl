import torch

from . import func, scat

class Fibre:
    """Scattering model for a cylindrical fibre, optionally with embedded cylinders."""

    def __init__(self, k0, phi, n, a, trunc=0):
        """Evaluate and store the scattering matrices of a single fibre.

        Args:
            k_0: Vacuum wavenumber.
            phi: Angle relative to the fibre length.
            n: Fibre refractive index.
            a: Fibre radius.
            trunc: Harmonic truncation; if 0, use the Wiscombe-style estimate.

        Returns:
            None
        """
        self.k_0 = torch.as_tensor(k0, dtype=torch.float64)
        device = self.k_0.device
        self.phi = torch.as_tensor(phi, dtype=torch.float64, device=device)
        self.k_z = self.k_0 * torch.cos(self.phi)
        self.n_0 = torch.tensor(1 + 0j, dtype=torch.complex128, device=device)
        self.k_perp_0 = scat.get_k_perp(self.n_0, self.k_0, self.k_z)

        self.n = torch.as_tensor(n, dtype=torch.complex128, device=device)
        self.a = torch.as_tensor(a, dtype=torch.float64, device=device)

        if trunc == 0:
            self.trunc = scat.wiscombe_trunc(self.k_perp_0 * self.a).to(torch.int64)
        else:
            self.trunc = torch.as_tensor(trunc, dtype=torch.int, device=device)

        R_out, R_in, T_out, T_in = scat.coeffs(
            self.k_0, self.k_z, self.n_0, self.n, self.a, self.trunc
        )
        self.R_out_coeffs = R_out
        self.R_in_coeffs = R_in
        self.T_out_coeffs = T_out
        self.T_in_coeffs = T_in

        self.R_out = _scat_coeffs2matrix(R_out)
        self.R_in = _scat_coeffs2matrix(R_in)
        self.T_out = _scat_coeffs2matrix(T_out)
        self.T_in = _scat_coeffs2matrix(T_in)

        self.R_out_complete = self.R_out

    def add_cylinders(self, cylinders):
        """Solve embedded-cylinder scattering and update the fibre scattering matrix.

        Args:
            cylinders: Complex tensor with columns
                [x, y, radius, refractive_index, trunc]. If trunc is 0, Wiscombe
                truncation is used for that cylinder.

        Returns:
            self
        """
        device = self.k_0.device
        if torch.is_tensor(cylinders):
            cylinders = cylinders.to(dtype=torch.complex128, device=device)
        else:
            cylinders = torch.as_tensor(cylinders, dtype=torch.complex128, device=device)

        # Fibre properties
        self.fibre_k_perp = scat.get_k_perp(self.n, self.k_0, self.k_z)

        # Cylinder properties
        self.cylinders = cylinders
        self.cyl_pos = cylinders[:, :2].real
        self.cyl_a = cylinders[:, 2].real
        self.cyl_n = cylinders[:, 3]
        self.cyl_count = cylinders.shape[0]
        self.cyl_k_perp = scat.get_k_perp(self.cyl_n, self.k_0, self.k_z)

        cyl_trunc_base = cylinders[:, 4].real
        w_trunc = scat.wiscombe_trunc(self.fibre_k_perp * self.cyl_a)
        self.cyl_trunc = (cyl_trunc_base + w_trunc * (cyl_trunc_base == 0)).to(
            torch.int64)

        self.cyl_terms = torch.tensor(
            [2 * v + 1 for v in self.cyl_trunc],
            device=device,
        )
        self.cyl_terms_total = self.cyl_terms.sum().item()

        # Geometry, using complex numbers for cleaner modulus/argument math.
        pos_c = self.cyl_pos[:, 0] + 1j * self.cyl_pos[:, 1]
        cyl_mod, cyl_arg = pos_c.abs(), pos_c.angle()

        pos_diff = pos_c[None, :] - pos_c[:, None]
        cyl_mod_diff, cyl_arg_diff = pos_diff.abs(), pos_diff.angle()

        # Indices
        max_cyl = (self.cyl_trunc).max()
        max_order = 2 * max_cyl
        d_idx = torch.repeat_interleave(
            torch.arange(self.cyl_count, device=device),
            self.cyl_terms).to(torch.int64)
        v_idx = torch.cat(
            [torch.arange(-t, t + 1, device=device) for t in self.cyl_trunc]
        ).to(torch.int64)
        f_vec = torch.arange(
            -self.trunc,
            self.trunc + 1,
            device=device,
        )[:, None]

        self.max_cyl = max_cyl
        self.max_order = max_order
        self.d_idx = d_idx
        self.v_idx = v_idx

        # Cylinder <-> centre translation matrices: L_oc, L_co.
        L_co_max = (self.trunc + max_cyl).to(torch.int64)
        J_part = func.jv_seq(L_co_max, self.fibre_k_perp * cyl_mod)
        signs2d = (1 - 2 * (torch.arange(1, L_co_max + 1, device=device) % 2))[
            :, None
        ]
        J_pre = torch.cat([torch.flip(J_part[1:] * signs2d, [0]), J_part], dim=0)

        order = (f_vec - v_idx[None, :]).to(torch.int64)
        J_val = J_pre[order + L_co_max, d_idx]
        phase = order * cyl_arg[d_idx][None, :]

        self.L_oc_top_left = J_val * torch.exp(-1j * phase)
        self.L_co_top_left = (J_val * torch.exp(1j * phase)).T.contiguous()

        # Cylinder-to-cylinder translation matrix: L_cc.
        args = (self.fibre_k_perp * cyl_mod_diff).clone().fill_diagonal_(10)
        h_part = func.h1v_seq(max_order, args)
        signs3d = (1 - 2 * (torch.arange(1, max_order + 1, device=device) % 2))[
            :, None, None
        ]
        self.h_precalc = torch.cat(
            [torch.flip(h_part[1:] * signs3d, [0]), h_part],
            dim=0,
        )

        orders = torch.arange(-max_order, max_order + 1, device=device)
        L_cc_pre = self.h_precalc * torch.exp(
            -1j * orders[:, None, None] * cyl_arg_diff[None, :, :]
        )

        o_mat = v_idx[:, None] - v_idx[None, :] + max_order
        mask = d_idx[:, None] != d_idx[None, :]
        self.L_cc_top_left = L_cc_pre[o_mat, d_idx[:, None], d_idx[None, :]] * mask

        # Scattering matrices and linear solve.
        R_out_all = scat.R_out_batched_radii(
            self.k_0, self.k_z, self.n, self.cyl_n, self.cyl_a, max_cyl
        )
        self.c_R_out_coeffs = R_out_all[:, max_cyl + v_idx, d_idx]

        R_in, T_in, T_out = self.R_in_coeffs, self.T_in_coeffs, self.T_out_coeffs
        L_cc, L_co, L_oc = self.L_cc_top_left, self.L_co_top_left, self.L_oc_top_left

        M_inner = (
            L_cc + _sandwich(L_co, R_in[0], L_oc),
            _sandwich(L_co, R_in[1], L_oc),
            _sandwich(L_co, R_in[2], L_oc),
            L_cc + _sandwich(L_co, R_in[3], L_oc),
        )
        self.M = _block2x2(*_apply_diag(self.c_R_out_coeffs, *M_inner))

        S_inner = (
            L_co * T_in[0][None, :],
            L_co * T_in[1][None, :],
            L_co * T_in[2][None, :],
            L_co * T_in[3][None, :],
        )
        self.solve_input = _block2x2(*_apply_diag(self.c_R_out_coeffs, *S_inner))

        I = torch.eye(2 * self.cyl_terms_total, dtype=torch.complex128, device=device)
        self.cyl_S_matrix = torch.linalg.solve(I - self.M, self.solve_input)

        # Back substitution onto the fibre basis.
        B_top, B_bot = torch.split(self.cyl_S_matrix, self.cyl_terms_total, dim=0)
        oc_top, oc_bot = torch.split(
            L_oc @ torch.cat([B_top, B_bot], dim=1),
            self.cyl_S_matrix.shape[1],
            dim=1,
        )

        f_S_top = T_out[0][:, None] * oc_top + T_out[1][:, None] * oc_bot
        f_S_bot = T_out[2][:, None] * oc_top + T_out[3][:, None] * oc_bot
        self.R_out_complete = torch.cat([f_S_top, f_S_bot], dim=0) + self.R_out

        return self

    def _dup_block(self, block):
        """Duplicate a scalar-polarization block into the two-polarization layout.

        Args:
            block: Matrix for one polarization channel.

        Returns:
            torch.Tensor: Block-diagonal matrix with the input copied into both
            polarization channels.
        """
        rows, cols = block.shape
        out = block.new_zeros((2 * rows, 2 * cols))
        out[:rows, :cols] = out[rows:, cols:] = block
        return out

    @property
    def L_oc(self):
        """Return the full origin-to-cylinder translation matrix.

        Returns:
            torch.Tensor: Two-polarization version of L_oc_top_left.
        """
        return self._dup_block(self.L_oc_top_left)

    @property
    def L_co(self):
        """Return the full cylinder-to-origin translation matrix.

        Returns:
            torch.Tensor: Two-polarization version of L_co_top_left.
        """
        return self._dup_block(self.L_co_top_left)

    @property
    def L_cc(self):
        """Return the full cylinder-to-cylinder translation matrix.

        Returns:
            torch.Tensor: Two-polarization version of L_cc_top_left.
        """
        return self._dup_block(self.L_cc_top_left)

    @property
    def c_R_out(self):
        """Return embedded-cylinder reflection coefficients as a block matrix.

        Returns:
            torch.Tensor: Dense two-polarization reflection matrix for cylinders.
        """
        c = self.c_R_out_coeffs
        t = c.shape[1]
        out = c.new_zeros((2 * t, 2 * t))
        out[:t, :t], out[:t, t:] = torch.diag(c[0]), torch.diag(c[1])
        out[t:, :t], out[t:, t:] = torch.diag(c[2]), torch.diag(c[3])
        return out


def _scat_coeffs2matrix(C):
    """Convert four diagonal coefficient vectors into a dense 2x2 block matrix.

    Args:
        C: Tensor with rows [EE, EK, KE, KK] and one column per harmonic order.

    Returns:
        torch.Tensor: Dense scattering matrix with polarization block structure.
    """
    M = C.shape[1]
    matrix = torch.zeros((2 * M, 2 * M), dtype=C.dtype, device=C.device)
    matrix[:M, :M] = torch.diag(C[0])
    matrix[:M, M:] = torch.diag(C[1])
    matrix[M:, :M] = torch.diag(C[2])
    matrix[M:, M:] = torch.diag(C[3])
    return matrix


def _block2x2(tl, tr, bl, br):
    """Assemble four tensors into one 2x2 block matrix.

    Args:
        tl: Top-left block.
        tr: Top-right block.
        bl: Bottom-left block.
        br: Bottom-right block.

    Returns:
        torch.Tensor: Concatenated block matrix.
    """
    return torch.cat(
        [torch.cat([tl, tr], dim=1), torch.cat([bl, br], dim=1)],
        dim=0,
    )


def _apply_diag(coeffs, tl, tr, bl, br):
    """Left-apply diagonal scattering coefficients to four matrix blocks.

    Args:
        coeffs: Tensor with coefficient rows [EE, EK, KE, KK].
        tl: Top-left block.
        tr: Top-right block.
        bl: Bottom-left block.
        br: Bottom-right block.

    Returns:
        tuple: Transformed top-left, top-right, bottom-left, and bottom-right blocks.
    """
    c0, c1, c2, c3 = [c[:, None] for c in coeffs]
    return (
        c0 * tl + c1 * bl,
        c0 * tr + c1 * br,
        c2 * tl + c3 * bl,
        c2 * tr + c3 * br,
    )


def _sandwich(left, coeffs, right):
    """Compute left @ diag(coeffs) @ right without materializing the diagonal.

    Args:
        left: Left dense matrix.
        coeffs: Diagonal coefficients.
        right: Right dense matrix.

    Returns:
        torch.Tensor: Product left @ diag(coeffs) @ right.
    """
    return (left * coeffs[None, :]) @ right
