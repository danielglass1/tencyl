import torch
from .bessel import jv, h1v
from .opt_functions import get_k_perp
def get_scat_coeffs(k_0: torch.Tensor,k_z: torch.Tensor,n_o: torch.Tensor, n_i: torch.Tensor,a: torch.Tensor,trunc: int,i_sources=True):
    """
    Determines the scattering coeffecients at a cylindrical interface of changing refractive index.

    Args:
        k_0: vacuum wavenumber
        k_z: component of wavevector along cylinder length
        n_o: external refractive index
        n_i: internal refractive index
        a: cylindrical boundary radius
        trunc: Maximum truncation required for coeffecients
        i_sources: If set to False function only return R_out
    Returns:
        Four Tensors are returned corresponding to:
        Reflection on the outside of the cylinder (R_out), 
        reflection on the cylinder's internal wall (R_in),
        transmission out of the cylinder (T_out) and
        transmission into the cylinder (T_in).
        Each tensor contains 4 vectors with length trunc+1
        for each polarisation and cross-polarisation: EE, EK, KE, KK.
    """
    k_perp_o=get_k_perp(n_o,k_0,k_z)
    k_perp_i=get_k_perp(n_i,k_0,k_z)
    
    ka_combined = torch.stack([k_perp_o, k_perp_i])*a

    J_combined = jv(trunc + 1, ka_combined)
    H_combined = h1v(trunc + 1, ka_combined)

    J_o = J_combined[:,0] 
    J_i = J_combined[:,1]
    H_o = H_combined[:,0]
    H_i = H_combined[:,1]

    J_o=torch.cat((-J_o[1].unsqueeze(0), J_o))
    J_i=torch.cat((-J_i[1].unsqueeze(0), J_i))
    H_o=torch.cat((-H_o[1].unsqueeze(0), H_o))
    H_i=torch.cat((-H_i[1].unsqueeze(0), H_i))

    # limit bessel function calls.
    def bessel_deriv(bessel_func):
        return 0.5*(bessel_func[:-2]-bessel_func[2:])

    HP_o=bessel_deriv(H_o)
    HP_i=bessel_deriv(H_i)
    JP_o=bessel_deriv(J_o)
    JP_i=bessel_deriv(J_i)

    J_o = J_o[1:-1]
    J_i = J_i[1:-1]
    H_o = H_o[1:-1]
    H_i = H_i[1:-1]
    order_vec=torch.arange(0,trunc+1)

    X_JJ_e=k_perp_o*n_i**2*JP_i*J_o - k_perp_i*n_o**2*J_i*JP_o
    X_JJ_m=k_perp_o       *JP_i*J_o - k_perp_i       *J_i*JP_o
    X_HH_e=k_perp_o*n_i**2*HP_i*H_o - k_perp_i*n_o**2*H_i*HP_o
    X_HH_m=k_perp_o       *HP_i*H_o - k_perp_i       *H_i*HP_o
    X_JH_e=k_perp_o*n_i**2*JP_i*H_o - k_perp_i*n_o**2*J_i*HP_o
    X_JH_m=k_perp_o       *JP_i*H_o - k_perp_i       *J_i*HP_o

    C=(k_perp_i**2-k_perp_o**2)*(k_z*1j*order_vec)/(a*k_0*k_perp_i*k_perp_o)
    delta=(C*J_i*H_o)**2+X_JH_e*X_JH_m
    denominator=1/(delta)
    W_o=(2j)/(torch.pi*k_perp_o*a)
    W_i=(2j)/(torch.pi*k_perp_i*a)

    R_out_diag=-(C*J_i)**2*H_o*J_o
    R_out_EE=denominator*(R_out_diag-X_JH_m*X_JJ_e)
    R_out_KK=denominator*(R_out_diag-X_JH_e*X_JJ_m)
    R_out_cross=denominator*k_perp_i*C*W_o*J_i**2
    R_out_EK=-R_out_cross
    R_out_KE=R_out_cross*n_o**2

    R_out = torch.stack([R_out_EE, R_out_EK, R_out_KE, R_out_KK])

    def add_neg_ord(C):
        C_neg = torch.flip(C[:, 1:], dims=[1])
        C_neg[1:3] = -C_neg[1:3]
        return torch.cat([C_neg, C], dim=1)
    
    R_out=add_neg_ord(R_out)
    
    if i_sources:
        T_in_diag=-denominator*k_perp_i*W_o
        T_in_EE=T_in_diag*n_o**2*X_JH_m
        T_in_KK=T_in_diag      *X_JH_e

        T_in_cross=T_in_diag*C*J_i*H_o
        T_in_EK=T_in_cross
        T_in_KE=-T_in_cross*n_o**2

 
        R_in_diag=-(C*H_o)**2*H_i*J_i
        R_in_EE=denominator*(R_in_diag-X_JH_m*X_HH_e)
        R_in_KK=denominator*(R_in_diag-X_JH_e*X_HH_m)

        R_in_cross=denominator*k_perp_o*C*W_i*H_o**2
        R_in_EK=-R_in_cross
        R_in_KE=R_in_cross*n_i**2

        T_out_diag=-denominator*k_perp_o*W_i
        T_out_EE=T_out_diag*n_i**2*X_JH_m
        T_out_KK=T_out_diag       *X_JH_e

        T_out_cross=T_out_diag*C*J_i*H_o
        T_out_EK=T_out_cross
        T_out_KE=-T_out_cross*n_i**2

        R_in  = torch.stack([R_in_EE,  R_in_EK,  R_in_KE,  R_in_KK])
        T_out = torch.stack([T_out_EE, T_out_EK, T_out_KE, T_out_KK])
        T_in = torch.stack([T_in_EE, T_in_EK, T_in_KE, T_in_KK])

        R_in=add_neg_ord(R_in)
        T_out=add_neg_ord(T_out)
        T_in=add_neg_ord(T_in)
        
        return R_out,R_in,T_out,T_in
    
    else:
        return R_out

    


