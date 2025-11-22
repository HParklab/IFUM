import os
import torch
import argparse
import esm
import esm.inverse_folding
from glob import glob
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

def get_args():
    parser = argparse.ArgumentParser(description='Generate embeddings from PDB directory')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Directory containing .pdb files')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for .pt files')
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Models ---
    print("Loading ESM-IF1...")
    esm_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esm_model = esm_model.to(device).eval()

    print("Loading ProtT5...")
    t5_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    t5_model = T5EncoderModel.from_pretrained(t5_link).to(device).eval()
    t5_vocab = T5Tokenizer.from_pretrained(t5_link, do_lower_case=False)

    # --- Process PDBs ---
    pdb_files = glob(os.path.join(args.pdb_dir, "*.pdb"))
    print(f"Found {len(pdb_files)} PDB files.")

    with torch.no_grad():
        for pdb_path in tqdm(pdb_files):
            name = os.path.splitext(os.path.basename(pdb_path))[0]
            try:
                # 1. Extract Sequence & Coordinates (ESM-IF1)
                structure = esm.inverse_folding.util.load_structure(pdb_path, "A")
                coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)

                # 2. ESM-IF1 Embedding
                rep = esm.inverse_folding.util.get_encoder_output(esm_model, alphabet, coords)
                esm_if1 = rep.detach().cpu() # [L, 512]

                # 3. ProtT5 Embedding
                clean_seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                inputs = t5_vocab(" ".join(list(clean_seq)), return_tensors="pt", add_special_tokens=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                embedding_repr = t5_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                prott5 = embedding_repr.last_hidden_state[0, :len(clean_seq)].detach().cpu() # [L, 1024]

                # 4. Save Data
                pt_data = {
                    'name': name,
                    'seq': seq,
                    'prott5': prott5,
                    'esm_if1': esm_if1,
                    'CA': torch.tensor(coords[:, 2]), # CA atoms
                    'dG': torch.tensor([0.0]) # Placeholder, replace if you have labels
                }
                
                torch.save(pt_data, os.path.join(args.out_dir, f"{name}.pt"))

            except Exception as e:
                print(f"Skipping {name}: {e}")

if __name__ == '__main__':
    main()