import torch
import math
import argparse
import os
import torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from IEFFEUM import model

class SimpleBatchDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob(os.path.join(data_dir, "*.pt"))
        print(f"Loaded {len(self.files)} training files.")
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    def __len__(self):
        return len(self.files)

    def _get_indices(self, seq):
        indices = [self.aa_to_idx.get(aa) for aa in seq] 
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        pt = torch.load(self.files[idx])
        seq_indices = self._get_indices(pt['seq']) 
        
        return (
            pt['name'], 
            pt['prott5'],    # [L, 1024]
            pt['esm_if1'],   # [L, 512]
            pt['CA'],        # [L, 3]
            pt['dG'],        # [1]
            seq_indices      # [L]
        )
def collate_fn(batch):
    names, prott5s, esm_if1s, CAs, dGs, seq_indices_list = zip(*batch)
    
    max_len = max([p.shape[0] for p in prott5s])
    batch_size = len(batch)

    def pad(t, length, fill_value=0):
        out = torch.full((length, *t.shape[1:]), fill_value, dtype=t.dtype)
        out[:t.shape[0]] = t
        return out

    padded_prott5 = torch.stack([pad(p, max_len, 0.0) for p in prott5s])
    padded_esm_if1 = torch.stack([pad(e, max_len, 0.0) for e in esm_if1s])
    padded_CA = torch.stack([pad(c, max_len, 0.0) for c in CAs]) 
    padded_seq_indices = torch.stack([pad(s, max_len, -100) for s in seq_indices_list])
    
    dGs = torch.stack(dGs)

    mask_2d = torch.zeros((batch_size, max_len, max_len), dtype=torch.bool)
    mask_1d = torch.zeros((batch_size, 1, max_len), dtype=torch.bool)

    for i, p in enumerate(prott5s):
        l = p.shape[0]
        mask_2d[i, :l, :l] = True
        mask_1d[i, :, :l] = True

    return padded_prott5, padded_esm_if1, padded_CA, dGs, padded_seq_indices, mask_1d, mask_2d

def generate_two_state_ensemble(positions, dG, num_bins, min_bin, max_bin,):
    lower_breaks = torch.linspace(min_bin, max_bin, num_bins, device=positions.device)
    upper_breaks = torch.cat([lower_breaks[1:], torch.tensor([1e4], dtype=torch.float32, device=positions.device)], dim=-1)
    
    def _generate_U_flory(N_res):
        U = torch.zeros((N_res, N_res, num_bins), device=positions.device)
        index_diff = torch.arange(N_res, device=positions.device).unsqueeze(0) - torch.arange(N_res, device=positions.device).unsqueeze(1)
        unfolded_dist = math.sqrt(6) * 1.927 * (index_diff.abs() ** 0.598)
        abs_diff = torch.abs(lower_breaks - unfolded_dist.unsqueeze(2))
        closest_indices = torch.argmin(abs_diff, dim=2)
        U.scatter_(2, closest_indices.unsqueeze(2), 1)
        return U.unsqueeze(0)
    
    def _squared_difference(x, y):
        return torch.square(x - y)
    
    B, N_res, _ = positions.shape
    coeff = torch.exp(dG / 0.58).view(B,1,1,1)
    U_onehot = _generate_U_flory(N_res).repeat(B,1,1,1)
    
    dist2 = torch.sqrt(torch.sum(
        _squared_difference(
            positions.unsqueeze(2),
            positions.unsqueeze(1)),
        dim=-1, keepdim=True))
    
    d_onehot = ((dist2 > lower_breaks).float() *
            (dist2 < upper_breaks).float())
    
    ensemble = (U_onehot + coeff * d_onehot) / (1 + coeff)
    
    diagonal_indices = torch.arange(N_res, device=positions.device)
    ensemble[:, diagonal_indices, diagonal_indices, 0] = 1.0
    ensemble[:, diagonal_indices, diagonal_indices, 1:] = 0.0
    
    return d_onehot, ensemble

def masked_distogram_loss(pred, target, mask, args):
    log_pred = F.log_softmax(pred, dim=-1)
    loss = -(target * log_pred).sum(dim=-1)
    loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    return args.ensemble_loss_weight * loss

def gaussian_nll_loss(mean, var, target):
    var = var + 1e-6
    loss = 0.5 * torch.log(2 * torch.pi * var) + 0.5 * ((target - mean)**2) / var
    return loss.mean()

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--ensemble_loss_weight', type=float, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SimpleBatchDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    IEFFEUM = model.IEFFEUM(depth=11, dim=21).to(device)
    optimizer = AdamW(IEFFEUM.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6, last_epoch=-1)

    for epoch in range(args.epochs):
        IEFFEUM.train()
        total_loss = 0
        
        for batch in tqdm(dataloader):
            prott5, esm_if1, CA, dG_true, seq_idx, mask_1d, mask_2d = [b.to(device) for b in batch]
            
            seq_embds = prott5.unsqueeze(1)
            str_embds = esm_if1.unsqueeze(1)
            
            d_onehot, ensemble = generate_two_state_ensemble(CA, dG_true, num_bins=21, min_bin=2.0, max_bin=22.0)

            distogram, (dG_mean, dG_var), _, seq_logits = IEFFEUM(
                d_onehot, seq_embds, str_embds, mask_1d, mask_2d
            )

            loss_dist = masked_distogram_loss(distogram, ensemble, mask_2d, args)
            loss_dG = gaussian_nll_loss(dG_mean, dG_var, dG_true)
            
            loss_seq = F.cross_entropy(seq_logits, seq_idx)
            loss = loss_dist + loss_dG + loss_seq

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(dataloader):.4f}")
    torch.save(IEFFEUM.state_dict(), "weights.pt")

if __name__ == '__main__':
    train()