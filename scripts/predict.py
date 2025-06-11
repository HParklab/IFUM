#!/usr/bin/env python
#
# =====================================================================================
# MERGED SCRIPT: predict.py
#
# This script combines the functionality of prepare_IEFFEUM.py and run_IEFFEUM.py.
#
# It provides an end-to-end workflow:
# 1. Input: Takes a single FASTA file OR a directory of PDB files.
# 2. Preparation:
#    - If FASTA is input: Predicts structures with ESMFold.
#    - If PDBs are input: Skips prediction and extracts sequences directly.
#    - Generates ProtT5 (sequence) and ESM-IF1 (structure) embeddings.
#    - Creates the necessary .pt and .list files for IEFFEUM.
# 3. Execution:
#    - Immediately runs the IEFFEUM model on the prepared files.
#    - Outputs a final CSV file with stability (dG) predictions.
#
# Original sources:
# - https://github.com/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb
# - https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py
# =====================================================================================

import torch
from glob import glob
import esm
from esm.data import read_fasta
import esm.inverse_folding
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import os
import warnings
import time
import typing as T
import logging
from timeit import default_timer as timer
from transformers import T5EncoderModel, T5Tokenizer
import shutil
import gc

from IEFFEUM import utils

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%y/%m/%d %H:%M:%S")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
warnings.filterwarnings('ignore')

# --- Argument Parser Definition ---
def create_arg_parser():
    """Creates and returns the ArgumentParser object for the merged script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description='End-to-end script to prepare data and run IEFFEUM.',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-i', '--input-path', required=True, type=str,
        help='Path to an input file or directory.\n'
             '- If a .fasta file, structures will be predicted with ESMFold.\n'
             '- If a directory, it should contain .pdb files.'
    )
    parser.add_argument(
        '-o', '--out-path', required=False, type=str, default=None,
        help='A path for the final output CSV file. (default: same as input path)'
    )
    parser.add_argument(
        '-m', '--model-path', required=False, type=str, default=os.path.join(script_dir, '..', 'weights', 'params.pth'),
        help='Path to the IEFFEUM model parameters (.pth file).'
    )
    parser.add_argument(
        '-b', '--batch-size', required=False, type=int, default=1,
        help='Batch size for IEFFEUM inference. (default: 1)'
    )
    parser.add_argument(
        '--per-resi', action='store_true',
        help='Report per-residue dG contributions in the output CSV. (default: False)'
    )
    parser.add_argument(
        '--keep-intermediates', action='store_true',
        help='Do not delete the intermediate embedding files (.pt) after completion.'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Run in quiet mode, reducing console output.'
    )
    return parser

# --- Data Preparation Functions (from prepare_IEFFEUM.py) ---

def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:
    """Batches sequences to avoid OOM during inference."""
    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)
    yield batch_headers, batch_sequences

def run_esmfold(seq_path, out_dir, device, num_recycles=None, max_tokens_per_batch=1024, chunk_size=None):
    """Runs ESMFold prediction on a fasta file."""
    logger.info(f"Reading sequences from {seq_path}")
    all_sequences = sorted(read_fasta(seq_path), key=lambda header_seq: len(header_seq[1]))
    logger.info(f"Loaded {len(all_sequences)} sequences.")
    logger.info("Loading ESMFold model...")
    model = esm.pretrained.esmfold_v1().to(device)
    model.eval()
    model.set_chunk_size(chunk_size)
    logger.info("Starting Predictions using ESMFold")
    batched_sequences = create_batched_sequence_datasest(all_sequences, max_tokens_per_batch)
    num_completed, num_sequences = 0, len(all_sequences)

    for headers, sequences in batched_sequences:
        start = timer()
        try:
            output = model.infer(sequences, num_recycles=num_recycles)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA OOM on a batch of size {len(sequences)}. Try lowering --max-tokens-per-batch.")
                continue
            raise
        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        tottime = timer() - start
        for header, seq, pdb_string, mean_plddt, ptm in zip(headers, sequences, pdbs, output["mean_plddt"], output["ptm"]):
            output_file = out_dir / f"{header}.pdb"
            output_file.write_text(pdb_string)
            num_completed += 1
            logger.info(f"Predicted structure for {header} (L={len(seq)}, pLDDT={mean_plddt:.1f}, pTM={ptm:.3f}) in {tottime/len(headers):0.1f}s. ({num_completed}/{num_sequences})")
    logger.info("ESMFold predictions finished.")
    del model
    gc.collect()
    torch.cuda.empty_cache()

def get_T5_model(device, model_dir=None, transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
    """Loads ProtT5 model and tokenizer."""
    logger.info(f"Loading ProtT5 model: {transformer_link}")
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir).to(device)
    if device == torch.device("cpu"):
        logger.info("Casting ProtT5 model to full precision for CPU.")
        model.to(torch.float32)
    model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    logger.info("ProtT5 model loaded.")
    return model, vocab

def get_seq_embd(device, seq_dict, max_residues=4000, max_seq_len=1000, max_batch=100):
    """Generates ProtT5 embeddings for a dictionary of sequences."""
    emb_dict = dict()
    model, vocab = get_T5_model(device)
    logger.info(f'Generating ProtT5 embeddings for {len(seq_dict)} sequences.')
    
    sorted_seqs = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(sorted_seqs, 1):
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq_len = len(seq)
        batch.append((pdb_id, ' '.join(list(seq)), seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(sorted_seqs) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()
            token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError as e:
                logger.error(f"RuntimeError during ProtT5 embedding for {pdb_ids}. Error: {e}")
                continue

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()
    
    end = time.time()
    logger.info(f'ProtT5 embeddings finished. Total time: {end - start:.2f}s')
    del model, vocab
    gc.collect()
    torch.cuda.empty_cache()
    return emb_dict

def get_pdb_embd(pdb_path, device):
    """Generates ESM-IF1 embeddings for pdb files and extracts sequences."""
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device).eval()
    
    pdbs = glob(f"{pdb_path}/*.pdb") if os.path.isdir(pdb_path) else [pdb_path]
    logger.info(f'Found {len(pdbs)} PDB file(s) to process for ESM-IF1 embeddings.')
    emb_dict = dict()

    with torch.no_grad():
        for pdb in tqdm(pdbs, desc="Processing PDB files (ESM-IF1)"):
            name = Path(pdb).stem
            try:
                structure = esm.inverse_folding.util.load_structure(str(pdb), "A")
                coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
                rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
                emb_dict[name] = [rep.detach().cpu().numpy(), seq, coords[:, 2]]
            except Exception as e:
                logger.error(f"Error processing PDB {pdb}: {e}")
                continue
    logger.info('ESM-IF1 embeddings finished.')
    del model, alphabet
    gc.collect()
    torch.cuda.empty_cache()
    return emb_dict

def prepare_ieffeum_inputs(seq_embd, pdb_embd, pt_dir):
    """Prepares final .pt input files by combining embeddings."""
    processed_names = 0
    total_names = len(seq_embd.keys())
    logger.info("Combining embeddings into IEFFEUM .pt input files...")
    
    out_dict = {'name': [], 'file': []}
    for name in tqdm(seq_embd.keys(), desc="Creating .pt files"):
        if name not in pdb_embd:
            logger.warning(f"Skipping {name} as no corresponding PDB embedding was found.")
            continue
        prott5 = seq_embd[name]
        esm_if1, seq, CA = pdb_embd[name]
        pt = {
            'name': name, 'seq': seq, 'seq_embd': torch.tensor(prott5),
            'target_F': torch.tensor(CA), 'str_embd': torch.tensor(esm_if1),
        }
        pt_file = pt_dir / f'{name}.pt'
        torch.save(pt, pt_file)
        processed_names += 1
        out_dict['name'].append(name)
        out_dict['file'].append(str(pt_file))
    
    logger.info(f"IEFFEUM input preparation finished. {processed_names}/{total_names} proteins prepared.")
    return out_dict

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()


    if args.quiet:
        logger.removeHandler(console_handler)

    # --- Setup device and directories ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    temp_dir = Path(args.input_path).parent
    pt_dir = temp_dir / 'pt_embeddings'
    pdb_out_dir = temp_dir / 'esmfold_pdbs'

    os.makedirs(pt_dir, exist_ok=True)

    input_path = Path(args.input_path)

    # --- Data Preparation ---
    if input_path.is_file() and input_path.suffix.lower() == '.fasta':
        logger.info("FASTA file detected. Starting workflow with ESMFold.")
        os.makedirs(pdb_out_dir, exist_ok=True)
        run_esmfold(input_path, pdb_out_dir, device)
        pdb_path_for_emb = pdb_out_dir
        seq_dict = {Path(name).stem: seq for name, seq in read_fasta(input_path)}

    elif input_path.is_dir() or (input_path.is_file() and input_path.suffix.lower() == '.pdb'):
        logger.info("PDB input detected. Skipping ESMFold.")
        pdb_path_for_emb = input_path
    else:
        logger.error(f"Invalid input path: {args.input_path}. Please provide a .fasta file or a directory of .pdb files.")
        sys.exit(1)

    # --- Generate Embeddings ---
    pdb_embd = get_pdb_embd(pdb_path_for_emb, device)
    if not pdb_embd:
        logger.error("No PDB embeddings could be generated. Exiting.")
        sys.exit(1)

    # If input was PDB, extract sequences from the PDB embedding dict
    if not input_path.is_file() or input_path.suffix.lower() != '.fasta':
        seq_dict = {name: data[1] for name, data in pdb_embd.items()}

    seq_embd = get_seq_embd(device, seq_dict)

    # --- Combine and Save Inputs ---
    ieffeum_input_dict = prepare_ieffeum_inputs(seq_embd, pdb_embd, pt_dir)

    # Create the list file in memory for the next step
    list_file_path = temp_dir / 'ieffeum_inputs.list'
    torch.save(ieffeum_input_dict, list_file_path)

    # --- Run IEFFEUM Inference ---
    logger.info("="*50)
    logger.info("Starting IEFFEUM Inference")
    logger.info("="*50)

    out_csv_path = args.out_path or f'{temp_dir}/{input_path.stem}.csv'

    with torch.no_grad():
        dataloader, IEFFEUM = utils.get_dataloader_and_model(list_file_path, args.model_path, device, int(args.batch_size))

        NAMES, P_DGS, P_DGS_PER_RESI = [], [], []

        for batch in tqdm(dataloader, desc="Running IEFFEUM Inference"):
            names, seqs, seq_embds, str_embds, target_Fs_cords, mask_2ds, mask_1ds = utils.batch_to_device(batch, device)
            target_Fs_onehot = utils.get_target_F_onehot(target_Fs_cords)
            results = IEFFEUM(target_Fs_onehot, seq_embds, str_embds, mask_1ds, mask_2ds)
            NAMES, P_DGS, P_DGS_PER_RESI = utils.gather_batch_results(names, results, NAMES, P_DGS, P_DGS_PER_RESI)

        _ = utils.save_results_to_csv(NAMES, P_DGS, P_DGS_PER_RESI, out_csv_path, args.per_resi)
        logger.info(f"Predictions successfully saved to {out_csv_path}")

    # --- START: NEW CLEANUP BLOCK ---
    # Clean up intermediate files unless the user wants to keep them
    if not args.keep_intermediates:
        try:
            shutil.rmtree(pt_dir)
        except OSError as e:
            logger.error(f"Error removing directory {pt_dir}: {e}")
    else:
        logger.info(f"Intermediate files kept at: {pt_dir}")
    # --- END: NEW CLEANUP BLOCK ---

    logger.info("Workflow finished successfully.")