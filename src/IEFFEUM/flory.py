import torch
import math
### this was used to generate the ensemble for the two-state model during training.

def generate_two_state_ensemble(positions, dG, num_bins, min_bin, max_bin,):
    # positions: shape [B, N_res, 3]
    # dG: shape [B, 1]
    # return: shape [B, N_res, N_res, num_bins]
    lower_breaks = torch.linspace(min_bin, max_bin, num_bins, device=positions.device)
    upper_breaks = torch.cat([lower_breaks[1:], torch.tensor([1e4], dtype=torch.float32, device=positions.device)], dim=-1)
    
    def _generate_U_flory(N_res):
        U = torch.zeros((N_res, N_res, num_bins), device=positions.device)
        # Create a tensor for the index differences
        index_diff = torch.arange(N_res, device=positions.device).unsqueeze(0) - torch.arange(N_res, device=positions.device).unsqueeze(1)
        unfolded_dist = math.sqrt(6) * 1.927 * (index_diff.abs() ** 0.588)
        # Calculate absolute differences for all combinations at once
        abs_diff = torch.abs(lower_breaks - unfolded_dist.unsqueeze(2))
        # Find the closest index for each pair (i, j)
        closest_indices = torch.argmin(abs_diff, dim=2)
        # Set the corresponding positions in U to 1
        U.scatter_(2, closest_indices.unsqueeze(2), 1)
        return U.unsqueeze(0)
    
    def _squared_difference(x, y):
        return torch.square(x - y)
    
    B, N_res, _ = positions.shape
    coeff = torch.exp(dG / 0.58).view(B,1,1,1)
    U_onehot = _generate_U_flory(N_res).repeat(B,1,1,1)
    
    dist2 = torch.sqrt(torch.sum(
        _squared_difference(
            positions.unsqueeze(2),  # Adding an extra dimension for broadcasting
            positions.unsqueeze(1)),
        dim=-1, keepdim=True))
    
    d_onehot = ((dist2 > lower_breaks).float() *
            (dist2 < upper_breaks).float())
    
    ensemble = (U_onehot + coeff * d_onehot) / (1 + coeff)
    
    # After ensemble calculation
    diagonal_indices = torch.arange(N_res, device=positions.device)
    ensemble[:, diagonal_indices, diagonal_indices, 0] = 1.0
    ensemble[:, diagonal_indices, diagonal_indices, 1:] = 0.0
    
    return d_onehot, ensemble