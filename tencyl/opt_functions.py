import torch
import matplotlib.pyplot as plt
def dist2anomaly(k_perp, k_x, cyl_sep, buffer=5):
    """
    Dynamically calculates distance to Wood's anomaly based on the background k-vector.
    """
    k_perp=k_perp.real
    G = 2 * torch.math.pi / cyl_sep
    m_bound = torch.ceil(((k_perp + torch.abs(k_x)) / G))
    m_max = m_bound + buffer
    m = torch.arange(-m_max, m_max + 1, dtype=torch.float64)
    k_xm = k_x + m * G
    distances = torch.abs(torch.abs(k_xm) - k_perp)
    min_dist, min_idx = torch.min(distances, dim=0)
    # critical_m = m[min_idx].item()
    return min_dist

def get_k_perp(n,k_0,k_z):
            """
            Determines the component wave vector perpendicular to the cylinder's length
            Args:
                n: refractive index
                k_0: incident wavevector in vacuum
                k_z: wavevector component parallel to cylinder's length
            Returns:
                k_perp: perpendicular wave vector
            """
            k=n*k_0
            return torch.sqrt(k*k-k_z*k_z)

def wiscombe_trunc(size_parameter: torch.Tensor) -> torch.Tensor:
    """
    Computes a suitable Mie series truncation, per Wiscombe (1980) https://doi.org/10.1364/AO.19.001505.
    Args:
        z: Complex size parameter, Argument of Bessel Functions.
    Returns:
        n_max: Integer truncation limit.
    """
    s=size_parameter.to(torch.complex128)
    s=s.real
    wiscombe=(s + 4 * s ** (1/3) + 2)
    return torch.ceil(wiscombe).to(torch.int)

def get_max_error(true_value: torch.Tensor,test_value: torch.Tensor,sep_real_imag=True):
    """
    Computes the maximum relative error of the test tensor against the true tensor.

    Args:
        true_value: correct tensor
        test_value: test tensor shape of correct tensor
        sep_real_imag: If True will calculate the relative error for real and imaginary components seperately
    Returns:
        maximum relative error
    """
    true_value.to(torch.complex128)
    test_value.to(torch.complex128)

    if sep_real_imag:

        true_value_real = true_value.real
        test_value_real = test_value.real

        zeros_mask_real = ~(true_value_real==0)
        true_value_real = true_value_real[zeros_mask_real]
        test_value_real = test_value_real[zeros_mask_real]

        rel_diff_real=(test_value_real-true_value_real).abs()/true_value_real.abs()

        true_value_imag = true_value.imag
        test_value_imag = test_value.imag

        zeros_mask_imag = ~(true_value_imag==0)
        true_value_imag = true_value_imag[zeros_mask_imag]
        test_value_imag = test_value_imag[zeros_mask_imag]

        rel_diff_imag=(test_value_imag-true_value_imag).abs()/true_value_imag.abs()
        return torch.max(rel_diff_real.max(),rel_diff_imag.max())

    else:
        norm=torch.abs(true_value)
        mask=~(norm==0)
        diff=torch.abs(true_value-test_value)
        rel_diff=diff[mask]/norm[mask]
        return rel_diff.max()

def build_cyl_matrix(pos_x,pos_y,rad,cyls_n):
    matrix=torch.column_stack([
    pos_x, 
    pos_y, 
    rad, 
    torch.full_like(pos_x, cyls_n,dtype=torch.complex128), 
    torch.full_like(pos_x, 0)
    ])
    return matrix

def seperate_inputs(var_input,num_cyls,beta=25):
    softplus=torch.nn.Softplus(beta)
    cyl_sep=var_input[0]
    pos_x,pos_y,raw_rad=torch.split(var_input[1:],num_cyls)
    rad=softplus(raw_rad)
    return cyl_sep,pos_x,pos_y,rad


def combine_inputs(cyl_sep,pos_x,pos_y,rad):
    return torch.cat([cyl_sep,pos_x,pos_y,rad])
    
def preview_structure(pos_x,pos_y,rad,num_cyls,jacket_a,title,file_name):
        fig, ax = plt.subplots()
        for i in range(num_cyls):
            circle = plt.Circle((pos_x[i], pos_y[i]), rad[i], edgecolor='blue', facecolor='none', linewidth=0.5)
            ax.add_patch(circle)
        jacket = plt.Circle((0, 0), jacket_a, edgecolor='blue', facecolor='none', linewidth=0.5)
        ax.add_patch(jacket)
        ax.set_aspect('equal')
        lim=jacket_a*1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.title(title)
        plt.savefig(file_name)
        plt.close()

def get_penalty(penalty_weight,cyl_sep,pos_x,pos_y,rad,jacket_a,beta=25):
    softplus=torch.nn.Softplus(beta)
    # Lattice Sep Penalty
    lat_pen=softplus(2.01*jacket_a-cyl_sep)

    # Fibre Intersect
    pos_r=torch.sqrt(pos_x**2+pos_y**2)
    fibre_boundary=((pos_r+rad)/jacket_a)-1 
    fibre_pen=torch.sum(softplus(fibre_boundary))
    
    # Hole Intersect
    pos_x_diff=pos_x[None,:]-pos_x[:,None]    
    pos_y_diff=pos_y[None,:]-pos_y[:,None] 
    pos_diff_sq=pos_x_diff*pos_x_diff+pos_y_diff*pos_y_diff
    rad_total=rad[None,:]+rad[:,None]
    hole_pen_norm=1-((pos_diff_sq)/(rad_total*rad_total))
    hole_pen=softplus(hole_pen_norm) 
    hole_pen=torch.sum(torch.triu(hole_pen,diagonal=1))

    return penalty_weight*(fibre_pen+hole_pen+lat_pen)

def sample_radii(n, center, lo, hi, kappa): 
    p = (center - lo) / (hi - lo)
    alpha = p * kappa
    beta  = (1 - p) * kappa
    dist = torch.distributions.Beta(alpha, beta)
    y = dist.sample((n,))
    return lo + (hi - lo) * y  # in [lo, hi]

def pack_circles(max_circles, container_radius, radii_spread, cyl_sep_max,  max_attempts=2000):

    approx_center=torch.sqrt(0.5/torch.tensor(max_circles))*container_radius
    test_radii=sample_radii(max_circles,approx_center,0,container_radius,radii_spread)

    radii_sorted,_ = torch.sort(test_radii)
    placed_x=torch.zeros(max_circles)
    placed_y=torch.zeros(max_circles)
    placed_r=torch.zeros(max_circles)    
    for i in range(max_circles):
        for _ in range(max_attempts):
            R_bound=container_radius-radii_sorted[i]            
            test_theta = 2*torch.pi * torch.rand(1)          
            test_r = R_bound * torch.sqrt(torch.rand(1))
            test_x = test_r*torch.cos(test_theta)
            test_y = test_r*torch.sin(test_theta)

            if i==0:
                placed_x[0]=torch.abs(test_x)
                placed_y[0]=0
                placed_r[0]=radii_sorted[i]
                num_placed=1
                break

            dx=(placed_x[:i] - test_x)
            dy=(placed_y[:i] - test_y)
            distance_sq = dx * dx + dy * dy
            total_rad=placed_r[:i]+radii_sorted[i]
            total_rad_sq=total_rad*total_rad
            diff=distance_sq-total_rad_sq

            if torch.min(diff)>=0:
                placed_x[i]=test_x
                placed_y[i]=test_y
                placed_r[i]=radii_sorted[i]
                num_placed+=1
                break
    
    cyl_sep=(2+cyl_sep_max*torch.rand(1))*container_radius
    return cyl_sep,placed_x[:num_placed],placed_y[:num_placed],placed_r[:num_placed],num_placed

def ten2str(num):
    return str(num.item()).replace(".", "p").replace("-", "m")

def str2ten(num):
    return torch.tensor(float(num.replace("p", ".").replace("m", "-")))