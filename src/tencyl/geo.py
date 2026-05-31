import math
import torch

def to_1d_tensor(x, *, dtype=torch.float64, device=None, name="input"):
    x = torch.as_tensor(x, dtype=dtype, device=device)

    if x.ndim == 0:
        x = x[None]          # tensor(1) -> tensor([1])

    if x.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D, got shape {tuple(x.shape)}")

    return x

def build_cyl_matrix(cyls_x, cyls_y, cyls_a, cyls_n, cyls_trunc=0):
    """Construct a cylinder table for Fibre.add_cylinders.

    Args:
        cyls_x: Tensor of x coordinates.
        cyls_y: Tensor of y coordinates.
        cyls_a: Tensor of radii.
        cyls_n: Refractive index of the cylinders; scalar or tensor.

    Returns:
        torch.Tensor: Complex matrix with columns [x, y, radius, n, trunc].
    """

    cyls_x = to_1d_tensor(cyls_x, name="cyls_x")
    cyls_y = to_1d_tensor(cyls_y, name="cyls_y")
    cyls_a = to_1d_tensor(cyls_a, name="cyls_a")
    cyls_n = to_1d_tensor(cyls_n, dtype=torch.complex128, name="cyls_n")
    
    if not isinstance(cyls_n, torch.Tensor):
        cyls_n = torch.tensor(cyls_n, dtype=torch.complex128, device=cyls_x.device)
    else:
        cyls_n = cyls_n.to(device=cyls_x.device, dtype=torch.complex128)

    matrix = torch.empty(
        (cyls_x.shape[0], 5),
        dtype=torch.complex128,
        device=cyls_x.device,
    )
    matrix[:, 0] = cyls_x
    matrix[:, 1] = cyls_y
    matrix[:, 2] = cyls_a
    matrix[:, 3] = cyls_n
    matrix[:, 4] = cyls_trunc
    return matrix


def preview_structure(
    cyls_x,
    cyls_y,
    cyls_a,
    fibre_a,
    title=None,
):
    """Plot a 2D cross-section of the fibre and internal cylinders.

    Args:
        cyls_x: Tensor/list of x coordinates.
        cyls_y: Tensor/list of y coordinates.
        cyls_a: Tensor/list of cylinder radii.
        fibre_a: Radius of the outer fibre. If None, no outer fibre is drawn.
        title: Optional plot title.

    Returns:
        fig, ax: The matplotlib Figure and Axes objects.
    """
    import matplotlib.pyplot as plt

    cyls_x = to_1d_tensor(cyls_x, name="cyls_x")
    cyls_y = to_1d_tensor(cyls_y, name="cyls_y")
    cyls_a = to_1d_tensor(cyls_a, name="cyls_a")
    cyl_count = len(cyls_a)

    fig, ax = plt.subplots()

    for i in range(cyl_count):
        circle = plt.Circle(
            (float(cyls_x[i]), float(cyls_y[i])),
            float(cyls_a[i]),
            edgecolor="blue",
            facecolor="none",
            linewidth=0.5,
        )
        ax.add_patch(circle)

    if fibre_a is not None:
        fibre = plt.Circle(
            (0, 0),
            float(fibre_a),
            edgecolor="blue",
            facecolor="none",
            linewidth=0.5,
        )
        ax.add_patch(fibre)

        lim = float(fibre_a) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)

    return fig, ax


def gen_structure(
    center,
    spread,
    container_radius,
    pack_dist=1e-3,
    max_attempts=3000,
    min_rad=0.03,
):
    """Randomly pack a circular container with smaller non-overlapping circles.

    Args:
        center: Desired circle radius with the maximum sampling frequency.
        spread: Spread of generated circle radii away from center.
        container_radius: Radius of the larger containing circle.
        pack_dist: Minimum surface distance between packed circles.
        max_attempts: Maximum number of placement attempts.
        min_rad: Minimum normalized radius; smaller sampled circles are ignored.

    Returns:
        tuple: cyls_x, cyls_y, cyls_a, and cyls_count for successfully packed circles.
    """
    norm_center = center / container_radius
    norm_pack_dist = pack_dist / container_radius

    spread = torch.clamp(torch.as_tensor(spread), min=1e-200)
    alpha = norm_center / spread
    beta = (1 - norm_center) / spread
    dist = torch.distributions.Beta(alpha, beta)

    placed_x = torch.zeros(max_attempts, dtype=torch.float64)
    placed_y = torch.zeros(max_attempts, dtype=torch.float64)
    placed_r = torch.zeros(max_attempts, dtype=torch.float64)

    final_count = 0
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        rad = dist.sample()
        R_bound = 1.0 - rad - norm_pack_dist
        if rad <= min_rad or R_bound <= 0:
            continue

        test_theta = 2 * math.pi * torch.rand(1)
        test_r = R_bound * torch.sqrt(torch.rand(1))
        tx = test_r * torch.cos(test_theta)
        ty = test_r * torch.sin(test_theta)

        if final_count == 0:
            placed_x[0], placed_y[0], placed_r[0] = tx, ty, rad
            final_count = 1
            continue

        dx = placed_x[:final_count] - tx
        dy = placed_y[:final_count] - ty
        dist_sq = dx**2 + dy**2

        min_dist_req = placed_r[:final_count] + rad + norm_pack_dist
        min_dist_sq_req = min_dist_req * min_dist_req

        if torch.all(dist_sq >= min_dist_sq_req):
            placed_x[final_count], placed_y[final_count], placed_r[final_count] = tx, ty, rad
            final_count += 1

    final_x = placed_x[:final_count] * container_radius
    final_y = placed_y[:final_count] * container_radius
    final_a = placed_r[:final_count] * container_radius

    return final_x, final_y, final_a, final_count
