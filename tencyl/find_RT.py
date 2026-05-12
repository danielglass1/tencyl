from .shanks import shanks_vec
import torch
#hello hi
def find_RT(solver,cyl_sep,inc_theta,inc_delta,findT=True,lat_imag=1e-9): 
    S=solver.fibre_B_matrix
    fibre_trunc=solver.fibre_trunc
    jacket_vec=torch.arange(-fibre_trunc,fibre_trunc+1)
    lattice_trunc=2*fibre_trunc

    k_perp_0=solver.k_perp_0.real
    phi=solver.phi
    k_x_0=k_perp_0*torch.cos(inc_theta)
    k_y_0=k_perp_0*torch.sin(inc_theta)

    W_vec=shanks_vec(lattice_trunc,k_perp_0+lat_imag*1j,k_x_0,cyl_sep)

    p = torch.arange(-fibre_trunc,fibre_trunc+1)[:, None] # Column vector [0, 1, ..., nmax-1]
    q = torch.arange(-fibre_trunc,fibre_trunc+1)          # Row vector    [0, 1, ..., nmax-1]
    matrix_indices = (q-p) + lattice_trunc
    W_matrix_block = torch.as_tensor(W_vec[matrix_indices],dtype=torch.complex128)
    W=torch.block_diag(W_matrix_block,W_matrix_block)
    I=torch.eye(2*(fibre_trunc*2+1),dtype=torch.complex128)
    M=I-S@W     

    # REFLECTANCE
    #cutoff n
    cn_max=torch.floor((k_perp_0-k_x_0)*cyl_sep/(2*torch.pi)).to(torch.int64).item()
    cn_min=torch.ceil((-k_perp_0-k_x_0)*cyl_sep/(2*torch.pi)).to(torch.int64).item()

    cn_vec=torch.arange(cn_min,cn_max+1)

    kx_cn=k_x_0+(2*torch.pi*cn_vec)/cyl_sep
    ky_cn=torch.sqrt(k_perp_0**2-kx_cn**2)
    arg_k_cn=torch.atan2(ky_cn,kx_cn)

    #input coeffs
    V0=1
    Vshared=V0*torch.sin(phi)*torch.exp(1j*jacket_vec*(torch.pi/2-inc_theta))
    VE=Vshared*torch.cos(inc_delta)
    VK=Vshared*torch.sin(inc_delta)
    A=torch.cat((VE,VK))
    B=torch.linalg.solve(M,S@A)

    P_inc = torch.sin(phi)**2

    coeffsE,coeffsK=torch.split(B,(2*fibre_trunc+1))
    shared_term=2/(cyl_sep*ky_cn[:,None])
    Rnm_shared=shared_term*torch.exp(1j*jacket_vec[None,:]*(-arg_k_cn[:,None]-torch.pi/2))
    

    Rn_E=torch.sum(Rnm_shared*coeffsE[None,:],axis=1)
    Rn_K=torch.sum(Rnm_shared*coeffsK[None,:],axis=1)

    abs_Rn=(Rn_E.real**2+Rn_E.imag**2+Rn_K.real**2+Rn_K.imag**2)*ky_cn/(P_inc*k_y_0)
    R_total=torch.sum(abs_Rn)

    T_total=0
    if findT:
        Tnm_shared=shared_term*torch.exp(1j*jacket_vec[None,:]*(arg_k_cn[:,None]-torch.pi/2))

        Tn_E=torch.sum(Tnm_shared*coeffsE[None,:],axis=1)
        Tn_K=torch.sum(Tnm_shared*coeffsK[None,:],axis=1)

        idx_0 = torch.where(cn_vec == 0)[0]
        Tn_E[idx_0] += torch.cos(inc_delta)*torch.sin(phi)
        Tn_K[idx_0] += torch.sin(inc_delta)*torch.sin(phi)
        
        abs_Tn=(Tn_E.real**2+Tn_E.imag**2+Tn_K.real**2+Tn_K.imag**2)*ky_cn/(P_inc*k_y_0)
        T_total=torch.sum(abs_Tn)
    
    return R_total,T_total