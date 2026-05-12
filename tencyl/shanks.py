import torch
from .bessel import h1v

#should definitely try the twersky formulation 
# file:///Users/danielglass/Downloads/A_1004377501747.pdf

def shanks_vec(trunc_max, k_perp, k_x, cyl_sep, num_terms=100, tol=1e-60):
    p = torch.arange(1, num_terms + 1, dtype=torch.float64)
    
    arg = k_perp * p * cyl_sep
    hankel_term = h1v(trunc_max, arg)
    
    phase_plus = torch.exp(1j * k_x * p * cyl_sep)
    phase_minus = torch.exp(-1j * k_x * p * cyl_sep)
    
    signs = (1 - 2 * (torch.arange(trunc_max + 1) & 1)).unsqueeze(1)
    terms = hankel_term * (phase_minus + signs * phase_plus)

    S = torch.cumsum(terms, dim=1)

    T, n_elements = S.shape
    eps = torch.empty((T, n_elements + 1, n_elements), dtype=torch.complex128)
    eps[:, 0, :] = torch.zeros_like(S)
    eps[:, 1, :] = S

    for i in range(1, n_elements):
        diff = eps[:, i, 1:n_elements-i+1] - eps[:, i, :n_elements-i]
        mask = torch.abs(diff) < tol
        diff = diff.masked_fill(mask, tol)
        eps[:, i + 1, :n_elements-i] = eps[:, i - 1, 1:n_elements-i+1] + 1.0 / diff

    last_col = n_elements if n_elements % 2 != 0 else n_elements - 1
    W_pos=eps[:, last_col, 0]
    W_neg=torch.flip(signs.squeeze(1)*W_pos,dims=[0])[:-1]
    return torch.cat((W_neg,W_pos))