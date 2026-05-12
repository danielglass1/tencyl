import torch
import torch_bessel

K0=torch_bessel.ops.modified_bessel_k0
K1=torch_bessel.ops.modified_bessel_k1

def h1v(trunc_max, z):
    H_out = torch.empty((trunc_max + 1,) + z.shape, dtype=z.dtype, device=z.device)
    minus_iz = -1j * z
    two_over_pi = 2.0 / torch.math.pi
    
    H_out[0] = -1j * two_over_pi * K0(minus_iz)
    
    if trunc_max > 0:
        H_out[1] = -two_over_pi * K1(minus_iz)
        two_z_inv = 2.0 / z
        for n in range(1, trunc_max):
            H_out[n+1] = (n * two_z_inv) * H_out[n] - H_out[n-1]
            
    return H_out

def jv(trunc_max, z):
    J_out = torch.empty((trunc_max + 1,) + z.shape, dtype=z.dtype, device=z.device)
    two_z_inv = 2.0 / z
    minus_iz = -1j * z
    
    h0 = (-2j / torch.math.pi) * K0(minus_iz)
    h1 = (-2.0 / torch.math.pi) * K1(minus_iz)
    
    start_order = trunc_max + 30
    r = torch.zeros_like(z)
    
    R_len = max(trunc_max, 1)
    R = torch.empty((R_len,) + z.shape, dtype=z.dtype, device=z.device)

    for v in range(start_order, R_len, -1):
        r = torch.reciprocal(v * two_z_inv - r)

    for v in range(R_len, 0, -1):
        r = torch.reciprocal(v * two_z_inv - r)
        R[v - 1] = r

    j0 = (2j / (torch.math.pi * z)) / (R[0] * h0 - h1)
    
    J_out[0] = j0
    J_out[1:] = torch.cumprod(R[:trunc_max], dim=0) * j0
    
    return J_out

# Stabler Scipy Functions

# from torch.autograd import Function
# import scipy.special as sp
# class BesselJv(Function):
#     @staticmethod
#     def forward(ctx, n, x):
#         ctx.save_for_backward(x)
#         ctx.n = n
#         n_np, x_np = n.cpu().numpy(), x.detach().cpu().numpy()
#         res = sp.jv(n_np, x_np)
#         return torch.from_numpy(res).to(x.device)

#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#         n = ctx.n
#         n_np, x_np = n.cpu().numpy(), x.detach().cpu().numpy()
#         # Derivative: $\frac{d}{dz} J_\nu(z) = \frac{1}{2}(J_{\nu-1}(z) - J_{\nu+1}(z))$
#         dj = 0.5 * (sp.jv(n_np - 1, x_np) - sp.jv(n_np + 1, x_np))
#         return None, grad_output * torch.conj(torch.from_numpy(dj).to(x.device))
    
# class BesselYv(Function):
#     @staticmethod
#     def forward(ctx, n, x):
#         ctx.save_for_backward(x)
#         ctx.n = n
#         n_np, x_np = n.cpu().numpy(), x.detach().cpu().numpy()
#         res = sp.yv(n_np, x_np)
#         return torch.from_numpy(res).to(x.device)

#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#         n = ctx.n
#         n_np, x_np = n.cpu().numpy(), x.detach().cpu().numpy()
#         dy = 0.5 * (sp.yv(n_np - 1, x_np) - sp.yv(n_np + 1, x_np))
#         return None, grad_output * torch.conj(torch.from_numpy(dy).to(x.device))
    

# def jv(trunc_max, z):
#     n_vec=torch.arange(0,trunc_max+1)
#     n_vec=n_vec.view(-1, *(1,) * z.ndim)
#     return BesselJv.apply(n_vec, z)

# def yv(trunc_max, z):
#     n_vec=torch.arange(0,trunc_max+1)
#     n_vec=n_vec.view(-1, *(1,) * z.ndim)
#     return BesselYv.apply(n_vec, z)

# def h1v(trunc_max, z):
#     return jv(trunc_max, z) + 1j * yv(trunc_max, z)