import torch
from .get_matrices import get_scat_coeffs  
from .opt_functions import wiscombe_trunc,get_k_perp
import time
default_type=torch.float64
default_ctype=torch.complex128
torch.set_default_dtype(default_type)

class fibre:
    def __init__(self, k_0, phi, n, a, trunc=0):
        """
        Evaluates and stores the scattering matrices of a single fibre.

        Args:
            k_0: vacuum wavenumber 
            phi: angle relative to the fiber length
            fibre_n: fibre refractive index
            fibre_a: fibre radius
            fibre_trunc: fibre truncation. If set to 0, employs Wiscombe's (1980) truncation.
        """
        self.k_0=k_0
        self.phi=phi
        self.k_z=k_0*torch.cos(self.phi)                     
        self.n_0=torch.tensor(1+0j,dtype=torch.complex128)
        self.k_perp_0=get_k_perp(self.n_0,self.k_0,self.k_z)

        self.n=n
        self.a=a

        if trunc==0:
            self.trunc=wiscombe_trunc(self.k_perp_0*self.a)
        else:
            self.trunc=torch.tensor(trunc)
            
        R_out,R_in,T_out,T_in=get_scat_coeffs(self.k_0,self.k_z,self.n_0,self.n,self.a,self.trunc)

        def scat_coeffs2matrix(C):
            M = C.shape[1]
            matrix = torch.zeros((2 * M, 2 * M), dtype=C.dtype)
            matrix[:M, :M] = torch.diag(C[0])
            matrix[:M, M:] = torch.diag(C[1])
            matrix[M:, :M] = torch.diag(C[2])
            matrix[M:, M:] = torch.diag(C[3])
            return matrix
        
        self.R_out=scat_coeffs2matrix(R_out)
        self.R_in=scat_coeffs2matrix(R_in)
        self.T_out=scat_coeffs2matrix(T_out)
        self.T_in=scat_coeffs2matrix(T_in)

    
    
    
