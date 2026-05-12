import torch
from .get_matrices import get_scat_coeffs  #needs a dot cause in same folder?
from .bessel import jv, h1v
from .opt_functions import wiscombe_trunc,get_k_perp
import time
default_type=torch.float64
default_ctype=torch.complex128
torch.set_default_dtype(default_type)

class solver:
    def __init__(self,fibre,cylinders):
        eps=1e-200
        self.k_0=fibre.k_0
        self.k_z=fibre.k_z
        self.k_perp_0=fibre.k_perp_0
        self.phi=fibre.phi

        self.fibre_trunc=fibre.trunc
        self.fibre_terms=2*self.fibre_trunc+1
        self.fibre_k_perp=get_k_perp(fibre.n,self.k_0,self.k_z)


        #CYLINDER
        self.cyl_pos=cylinders[:,0:2].real
        self.cyl_a=cylinders[:,2].real
        self.cyl_n=cylinders[:,3]
        row,col=cylinders.shape
        self.cyl_count=row
        self.cyl_k_perp=get_k_perp(self.cyl_n,self.k_0,self.k_z)

        self.cyl_trunc=cylinders[:,4].real
        cyl_wiscombe=wiscombe_trunc(self.fibre_k_perp*self.cyl_a)
        self.cyl_trunc=(self.cyl_trunc+cyl_wiscombe*(self.cyl_trunc==0)).to(torch.int) #take any 0 truncations and add the wiscombe truncation

        self.cyl_terms=2*self.cyl_trunc+1
        self.cyl_terms_total=torch.sum(self.cyl_terms)

        cyl_arg=torch.atan2(self.cyl_pos[:,1]+eps,self.cyl_pos[:,0]+eps)
        cyl_mod=torch.linalg.norm(self.cyl_pos,dim=1)
        

        cyl_pos_diff=self.cyl_pos[None,:,:]-self.cyl_pos[:,None,:]  
        cyl_mod_diff=torch.linalg.norm(cyl_pos_diff,axis=2) #distance between any pair of cylinders
        cyl_arg_diff=torch.atan2(cyl_pos_diff[...,1]+eps,cyl_pos_diff[...,0]+eps) #arg between any pair of cylinders

        # CYLINDER <-> CENTER
        max_cyl_trunc = int(max(self.cyl_trunc))
        L_co_max_order = self.fibre_trunc + max_cyl_trunc
        L_co_min_order = -L_co_max_order
        L_co_ka = self.fibre_k_perp * torch.as_tensor(cyl_mod)

        L_co_J_part = jv(L_co_max_order, L_co_ka)
        L_co_J_pos = L_co_J_part[1:,:]
        L_co_trunc_pos = torch.arange(1, L_co_max_order + 1)
        L_co_signs = (1 - 2 * (L_co_trunc_pos & 1))[:,None]
        L_co_J_neg_flipped = torch.flip(L_co_J_pos * L_co_signs, dims=[0])
        
        L_co_J_precalc = torch.cat([L_co_J_neg_flipped, L_co_J_part], dim=0)

        fibre_trunc_vec = torch.arange(-self.fibre_trunc, self.fibre_trunc + 1)[:, None]
        L_oc_blocks = []
        L_co_blocks = []

        for i in range(self.cyl_count):
            cyl_trunc_vec = torch.arange(-self.cyl_trunc[i], self.cyl_trunc[i] + 1)[None, :]
            order = fibre_trunc_vec - cyl_trunc_vec
            
            order_idx = order - L_co_min_order
            J_val = L_co_J_precalc[order_idx, i]
            
            phase = order * cyl_arg[i]
            
            L_oc_blocks.append(J_val * torch.exp(-1j * phase))
            L_co_blocks.append(J_val.T * torch.exp(1j * phase.T))

        L_oc_top_left = torch.cat(L_oc_blocks, dim=1)
        self.L_oc = torch.zeros((2 * self.fibre_terms, 2 * self.cyl_terms_total), dtype=default_ctype)
        self.L_oc[:self.fibre_terms, :self.cyl_terms_total] = L_oc_top_left
        self.L_oc[self.fibre_terms:, self.cyl_terms_total:] = L_oc_top_left

        L_co_top_left = torch.cat(L_co_blocks, dim=0)
        self.L_co = torch.zeros((2 * self.cyl_terms_total, 2 * self.fibre_terms), dtype=default_ctype)
        self.L_co[:self.cyl_terms_total, :self.fibre_terms] = L_co_top_left
        self.L_co[self.cyl_terms_total:, self.fibre_terms:] = L_co_top_left

        # CYLINDER TO CYLINDER
        s=time.time()
        self.L_cc = torch.zeros((2*self.cyl_terms_total, 2*self.cyl_terms_total), dtype=default_ctype)

        max_order = 2 * self.cyl_trunc.max()
        orders = torch.arange(-max_order, max_order + 1, device=self.L_cc.device) # Ensures same device

        self.args = self.fibre_k_perp * cyl_mod_diff
        self.args.fill_diagonal_(10) 

        h_precalc_part = h1v(max_order, self.args)
        h_precalc_pos=h_precalc_part[1:,:,:]
        h_trunc_vec_pos=torch.arange(1,max_order+1)
        h_precalc_signs = (1 - 2 * (h_trunc_vec_pos & 1))[:,None,None]
        h_precalc_neg = torch.flip(h_precalc_pos * h_precalc_signs, dims=[0])
        self.h_precalc = torch.cat([h_precalc_neg, h_precalc_part], dim=0)

        phase = -1j * orders[:, None, None] * cyl_arg_diff[None, :, :]
        L_cc_combined_precalc = self.h_precalc * torch.exp(phase)

        print("L_cc hvals",time.time()-s)

        N = self.cyl_terms_total

        n_vals = 2 * self.cyl_trunc + 1
        d_idx = torch.repeat_interleave(torch.arange(self.cyl_count, device=self.L_cc.device), n_vals)
        v_idx = torch.cat([torch.arange(-t.item(), t.item() + 1, device=self.L_cc.device) for t in self.cyl_trunc])

        order_matrix = v_idx[:, None] - v_idx[None, :] + max_order
        d_matrix = d_idx[:, None]
        c_matrix = d_idx[None, :]

        L_cc_top_left = L_cc_combined_precalc[order_matrix, d_matrix, c_matrix]

        mask = d_matrix != c_matrix
        L_cc_top_left = torch.where(mask, L_cc_top_left, torch.zeros_like(L_cc_top_left))

        self.L_cc[:N, :N] = L_cc_top_left
        self.L_cc[N:, N:] = L_cc_top_left
        

        ### SCATTERING MATRICES
        self.c_R_out=torch.zeros((2*self.cyl_terms_total, 2*self.cyl_terms_total),dtype=default_ctype)
        offset=0
        for i in range(self.cyl_count):   #jacket?, trunc,    n_o,         n_i,     a,       k_z,    k_0):
            R_out=get_scat_coeffs(self.k_0,self.k_z,fibre.n, self.cyl_n[i], self.cyl_a[i],self.cyl_trunc[i],False)
            ct=self.cyl_terms[i]
            self.c_R_out.diagonal(0)[offset:offset+ct] = R_out[0,:]
            self.c_R_out[:self.cyl_terms_total, self.cyl_terms_total:].diagonal(0)[offset:offset+ct] = R_out[1,:]
            self.c_R_out[self.cyl_terms_total:,:self.cyl_terms_total].diagonal(0)[offset:offset+ct] = R_out[2,:]
            self.c_R_out.diagonal(0)[offset+self.cyl_terms_total:offset+self.cyl_terms_total+ct] = R_out[3,:]

            offset+=self.cyl_terms[i]

        s=time.time()
        I=torch.eye((2*self.cyl_terms_total),dtype=default_ctype)
        self.j_R_in=fibre.R_in
        self.j_R_out=fibre.R_out
        # solve
        self.M=self.c_R_out @ (self.L_cc + self.L_co @ fibre.R_in @ self.L_oc)
        solve_input=self.c_R_out @ self.L_co @ fibre.T_in
    
        
        B_LU,B_pivots=torch.linalg.lu_factor(I-self.M)
        self.cyl_B_matrix=torch.linalg.lu_solve(B_LU,B_pivots,solve_input)
        

        self.fibre_B_matrix=fibre.T_out @ self.L_oc @ self.cyl_B_matrix + fibre.R_out
        print("mat mult",time.time()-s)






        

        

    

       
        
        