import torch
from torch.utils.data import Dataset
from typing import List, Tuple

def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Collates a batch of protein data, padding sequences and creating masks.

    This version is adjusted to match the output of BatchDatasetCompRaw.
    """

    def _create_masks(mask_size: int, padding_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates masks."""
        mask_2d = torch.ones((mask_size, mask_size), dtype=torch.bool)
        padded_mask_2d = torch.zeros((padding_len, padding_len), dtype=torch.bool)
        padded_mask_2d[:mask_size, :mask_size] = mask_2d

        mask_1d = torch.ones((1, mask_size), dtype=torch.bool)
        padded_mask_1d = torch.zeros((1, padding_len), dtype=torch.bool)
        padded_mask_1d[:, :mask_size] = mask_1d
        return padded_mask_2d, padded_mask_1d

    def _create_padded_tensor(tensor: torch.Tensor, padding_len: int) -> torch.Tensor:
        """Pads a 2D tensor to a specified length."""
        padded_tensor = torch.zeros((padding_len, tensor.shape[1]))
        padded_tensor[:tensor.shape[0], :] = tensor
        return padded_tensor


    # Unzip the batch data
    names, sequences, seq_embeddings, struct_embeddings, target_F_coordinates = zip(*batch)

    # Calculate lengths from sequence embeddings
    lengths = [s.shape[0] for s in seq_embeddings]
    lengths = torch.tensor(lengths, dtype=torch.long).unsqueeze(1)  # Ensure lengths are long

    # Find the maximum sequence length
    padding_len = max(lengths).item()

    # Pad and stack data
    collated_seq_embd = torch.stack([_create_padded_tensor(s, padding_len) for s in seq_embeddings])
    collated_str_embd = torch.stack([_create_padded_tensor(s, padding_len) for s in struct_embeddings])
    collated_target_F = torch.stack([_create_padded_tensor(c, padding_len) for c in target_F_coordinates])

    # Create and stack masks
    collated_mask_2d, collated_mask_1d = zip(*[_create_masks(L.item(), padding_len) for L in lengths])

    return (
        list(names),
        list(sequences),
        collated_seq_embd,
        collated_str_embd,
        collated_target_F,
        torch.stack(collated_mask_2d),
        torch.stack(collated_mask_1d),
    )

class BatchDataset(Dataset):
    """
    Dataset for loading protein data prepared for IEFFEUM.

    Loads precomputed sequence and structure embeddings, along with targeted folded state coordinates.

    Args:
        embd_dir (str): Directory containing the embedding files.
        embd_list (str): Path to a file containing a list of protein names.
    """

    def __init__(self, input_list_path: str):
        super().__init__()
        input_list = torch.load(input_list_path)
        self.names: List[str] = input_list['name']
        self.files: List[str] = input_list['file']
        
        if len(self.names) != len(self.files):
            raise ValueError(f"Got {len(self.names)} sequences but found {len(self.files)} embedding files.")

    def __len__(self) -> int:
        return len(self.names)
    
    def __getitem__(self, idx: int) -> Tuple:
        name = self.names[idx]
        file = self.files[idx]
        assert name == file.split('/')[-1].replace('.pt','')
        
        try:
            pt = torch.load(file, map_location="cpu")
        except Exception as e:
            raise IOError(f"Error loading file {file}: {e}")
        
        if name != pt['name']:
            raise ValueError(f"Expected {name} but got {pt['name']}")
        
        seq = pt['seq']
        seq_embd = pt['seq_embd']
        if len(seq) != seq_embd.shape[0]:
            raise ValueError(f"Got sequence length {len(seq)} but sequence embedding length {seq_embd.shape[0]}.")
        
        str_embd = pt['str_embd']
        if len(seq) != str_embd.shape[0]:
            raise ValueError(f"Got sequence length {len(seq)} but structure embedding length {str_embd.shape[0]}.")

        target_F = pt['target_F']
        
        return name, seq, seq_embd, str_embd, target_F
