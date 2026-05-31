import torch

def get_penalty(period, pos_x, pos_y, rad, container_radius, beta=25):
    """Compute a smooth penalty for invalid lattice and cylinder geometry.

    Args:
        period: Lattice period; penalized if it is too small for the container.
        pos_x: Tensor of cylinder x positions.
        pos_y: Tensor of cylinder y positions.
        rad: Tensor of cylinder radii.
        container_radius: Radius of the containing fibre.
        beta: Softplus sharpness; higher values approximate a hard constraint.

    Returns:
        torch.Tensor: Scalar penalty for lattice overlap, boundary escape, and
        cylinder-cylinder overlap.
    """
    num_cyls = pos_x.shape[0]
    lat_pen = num_cyls * torch.nn.functional.softplus(
        (2 * container_radius - period) / container_radius,
        beta=beta,
    )

    pos_r = torch.sqrt(pos_x**2 + pos_y**2)
    fibre_pen = torch.sum(
        torch.nn.functional.softplus(
            ((pos_r + rad) / container_radius) - 1,
            beta=beta,
        )
    )

    i, j = torch.triu_indices(num_cyls, num_cyls, offset=1)
    if i.numel() == 0:
        hole_pen = pos_x.new_zeros(())
    else:
        pos_diff_sq = (pos_x[i] - pos_x[j]) ** 2 + (pos_y[i] - pos_y[j]) ** 2
        rad_sum_sq = (rad[i] + rad[j]) ** 2
        hole_pen = torch.sum(
            torch.nn.functional.softplus(1 - (pos_diff_sq / rad_sum_sq), beta=beta)
        )

    return lat_pen + fibre_pen + hole_pen


def separate_inputs(var_input, num_cyls, fibre_a, beta=25):
    """Split a flat optimization tensor into period, positions, and positive radii.

    Args:
        var_input: One-dimensional optimization tensor.
        num_cyls: Number of cylinders represented in var_input.
        fibre_a: Fibre radius used to scale the softplus radius transform.
        beta: Softplus sharpness for enforcing positive radii.

    Returns:
        tuple: period, pos_x, pos_y, and rad tensors.
    """
    period = var_input[0]
    pos_x, pos_y, raw_rad = torch.split(var_input[1:], num_cyls)

    rad_scaled = raw_rad / fibre_a
    rad = torch.nn.functional.softplus(rad_scaled, beta=beta) * fibre_a
    return period, pos_x, pos_y, rad


def combine_inputs(period, pos_x, pos_y, rad):
    """Concatenate structural optimization parameters into a flat tensor.

    Args:
        period: Scalar lattice period.
        pos_x: Tensor of cylinder x coordinates.
        pos_y: Tensor of cylinder y coordinates.
        rad: Tensor of cylinder radii.

    Returns:
        torch.Tensor: One-dimensional tensor containing period, positions, and radii.
    """
    period = torch.as_tensor(period, dtype=pos_x.dtype, device=pos_x.device).reshape(1)
    return torch.cat([period, pos_x, pos_y, rad])


def get_max_error(true_value: torch.Tensor, test_value: torch.Tensor, sep_real_imag=True):
    """Compute the maximum relative error between two tensors.

    Args:
        true_value: Reference tensor.
        test_value: Tensor being compared to true_value.
        sep_real_imag: If True, compare real and imaginary parts separately.

    Returns:
        torch.Tensor: Scalar maximum relative error, or zero when all reference
        entries used for comparison are exactly zero.
    """
    true_value = true_value.to(torch.complex128)
    test_value = test_value.to(torch.complex128)

    if sep_real_imag:
        true_real = true_value.real
        test_real = test_value.real
        real_mask = true_real != 0
        rel_parts = []

        if torch.any(real_mask):
            rel_parts.append(
                (test_real[real_mask] - true_real[real_mask]).abs()
                / true_real[real_mask].abs()
            )

        true_imag = true_value.imag
        test_imag = test_value.imag
        imag_mask = true_imag != 0

        if torch.any(imag_mask):
            rel_parts.append(
                (test_imag[imag_mask] - true_imag[imag_mask]).abs()
                / true_imag[imag_mask].abs()
            )

        if not rel_parts:
            return true_value.real.new_zeros(())

        return torch.max(torch.stack([part.max() for part in rel_parts]))

    norm = torch.abs(true_value)
    mask = norm != 0
    if not torch.any(mask):
        return true_value.real.new_zeros(())

    rel_diff = torch.abs(true_value - test_value)[mask] / norm[mask]
    return rel_diff.max()


def ten2str(num):
    """Convert a tensor-like number into a filename-safe string.

    Args:
        num: Tensor or scalar numeric value.

    Returns:
        str: String with decimal points replaced by p and minus signs by m.
    """
    value = torch.as_tensor(num).detach().cpu().item()
    return str(value).replace(".", "p").replace("-", "m")


def str2ten(num):
    """Convert a ten2str filename token back into a float tensor.

    Args:
        num: String produced by ten2str.

    Returns:
        torch.Tensor: Float tensor represented by the encoded string.
    """
    return torch.tensor(float(num.replace("p", ".").replace("m", "-")))
