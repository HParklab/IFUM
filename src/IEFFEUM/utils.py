import torch
import pandas as pd
from torch.utils.data import DataLoader
from IEFFEUM import dataset, model

def get_target_F_onehot(positions):
    lower_breaks = torch.linspace(2, 42, 21, device=positions.device)
    upper_breaks = torch.cat([lower_breaks[1:], torch.tensor([1e4], dtype=torch.float32, device=positions.device)], dim=-1)
    
    
    def _squared_difference(x, y):
        return torch.square(x - y)
    
    dist2 = torch.sqrt(torch.sum(
        _squared_difference(
            positions.unsqueeze(2),  # Adding an extra dimension for broadcasting
            positions.unsqueeze(1)),
        dim=-1, keepdim=True))
    
    d_onehot = ((dist2 > lower_breaks).float() *
            (dist2 < upper_breaks).float())
    
    return d_onehot

def batch_to_device(batch, device):
    names, seqs, seq_embds, str_embds, target_Fs, mask_2ds, mask_1ds = batch
    
    seq_embds = seq_embds.unsqueeze(1).to(device)
    str_embds = str_embds.unsqueeze(1).to(device)
    target_Fs = target_Fs.to(device)
    mask_2ds = mask_2ds.to(device)
    mask_1ds = mask_1ds.to(device) 
    
    return names, seqs, seq_embds, str_embds, target_Fs, mask_2ds, mask_1ds

def save_results_to_csv(names, p_dGs, p_dGs_per_resi, out_path, per_resi):
    results = {
        'name': names,
        'dG (kcal/mol)': [ _[0] for _ in p_dGs],
    }
    if per_resi:
        results.update({
            'per_resi_dG(kcal/mol)': [ _[0] for _ in p_dGs_per_resi],
        })
        
    results = pd.DataFrame(results)
    results.to_csv(out_path, index=False)
    
    return results

def get_dataloader_and_model(input_list, model_path, device, batch_size=1):
    batched_dataset = dataset.BatchDataset(input_list)
    dataloader = DataLoader(batched_dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    
    IEFFEUM = model.IEFFEUM()
    IEFFEUM = IEFFEUM.to(device)
    IEFFEUM.load_state_dict(torch.load(model_path, device))
    IEFFEUM.eval()
    
    return dataloader, IEFFEUM

def gather_batch_results(names, results, NAMES, P_DGS, P_DGS_PER_RESI):
    _, p_dGs, p_dGs_per_resi, _ = results
    NAMES.extend(names)
    P_DGS.extend(p_dGs[0].detach().cpu().numpy().round(2))
    P_DGS_PER_RESI.extend(p_dGs_per_resi[0].detach().cpu().numpy().round(2))
    
    return NAMES, P_DGS, P_DGS_PER_RESI