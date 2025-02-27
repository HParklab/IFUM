#!/usr/bin/env python
# https://github.com/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb
# https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py

# Usage: bash prepare_inference.sh <fasta_file> <pdb_dir>

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

logger = logging.getLogger(__name__) # set logger name as __name__
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
warnings.filterwarnings('ignore')

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='Prepare inference data using ESMFold and ProtT5.')

    # Required positional argument
    parser.add_argument( '-f', '--fasta', required=True, type=str,
                        help='A path to a fasta file containing protein sequence(s).')

    parser.add_argument( '-p', '--pdb', required=False, type=str, default=None,
                        help='A path to a directory containing pdb file(s). If not provided, '
                             'a directory named after the fasta file (-esmfold) will be created.')

    parser.add_argument( '-q', '--quiet', action='store_true',
                        help='Run in quiet mode, reducing output to console.')
    return parser

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

def run_esmfold(seq_path, out_dir, num_recycles=None, max_tokens_per_batch=1024, chunk_size=None):
    """Runs ESMFold prediction on a fasta file."""
    # Read fasta and sort sequences by length
    logger.info(f"Reading sequences from {seq_path}")
    all_sequences = sorted(read_fasta(seq_path), key=lambda header_seq: len(header_seq[1]))
    logger.info(f"Loaded {len(all_sequences)} sequences from {seq_path}")

    logger.info("Loading ESMFold model")

    model = esm.pretrained.esmfold_v1()
    model.to(device)
    model = model.eval()
    model.set_chunk_size(chunk_size)
    logger.info("Starting Predictions using ESMFold")
    batched_sequences = create_batched_sequence_datasest(all_sequences, max_tokens_per_batch)

    num_completed = 0
    num_sequences = len(all_sequences)
    for headers, sequences in batched_sequences:
        start = timer()
        try:
            output = model.infer(sequences, num_recycles=num_recycles)
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                if len(sequences) > 1:
                    logger.warning(
                        f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                        "Try lowering `--max-tokens-per-batch`."
                    )
                else:
                    logger.warning(
                        f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}."
                    )
                continue # skip to the next batch if OOM
            raise # re-raise other runtime errors

        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        tottime = timer() - start
        time_string = f"{tottime / len(headers):0.1f}s"
        if len(sequences) > 1:
            time_string = time_string + f" (amortized, batch size {len(sequences)})"
        for header, seq, pdb_string, mean_plddt, ptm in zip(
            headers, sequences, pdbs, output["mean_plddt"], output["ptm"]
        ):
            output_file = out_dir / f"{header}.pdb"
            output_file.write_text(pdb_string)
            num_completed += 1
            logger.info(
                f"Predicted structure for {header} with length {len(seq)}, pLDDT {mean_plddt:0.1f}, "
                f"pTM {ptm:0.3f} in {time_string}. "
                f"{num_completed} / {num_sequences} completed."
            )
    logger.info("ESMFold predictions finished.")
    return

def get_T5_model(model_dir = None, transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"):
    """Loads ProtT5 model and tokenizer."""
    logger.info(f"Loading ProtT5 model: {transformer_link}")
    if model_dir is not None:
        logger.info("##########################")
        logger.info(f"Loading cached model from: {model_dir}")
        logger.info("##########################")
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
    # only cast to full-precision if no GPU is available
    if device==torch.device("cpu"):
        logger.info("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )
    logger.info("ProtT5 model loaded.")
    return model, vocab


def read_fasta_dict( fasta_path ):
    '''
    Reads in fasta file containing multiple sequences.
    Returns dictionary of holding multiple sequences or only single
    sequence, depending on input file.
    '''

    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case

    return sequences


def get_seq_embd( seq_path,
                    max_residues=4000, # number of cumulative residues per batch
                    max_seq_len=1000, # max length after which we switch to single-sequence processing to avoid OOM
                    max_batch=100 # max number of sequences per single batch
                    ):
    """Generates ProtT5 embeddings for sequences in a fasta file."""

    seq_dict = dict()
    emb_dict = dict()

    # Read in fasta
    seq_dict = read_fasta_dict( seq_path )
    model, vocab = get_T5_model()

    logger.info('########################################')
    logger.info('Example sequence: {} {}'.format( next(iter(
        seq_dict.keys())), next(iter(seq_dict.values()))) )
    logger.info('########################################')
    logger.info('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([ len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long     = sum([ 1 for _, seq in seq_dict.items() if len(seq)>max_seq_len])
    seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )

    logger.info(f"Average sequence length: {avg_length:.2f}")
    logger.info(f"Number of sequences >{max_seq_len}: {n_long}")

    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq.replace('U','X').replace('Z','X').replace('O','X') # replace rare amino acids
        seq_len = len(seq)
        seq = ' '.join(list(seq)) # space out sequence for ProtT5 tokenizer
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids     = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                logger.error(f"RuntimeError during embedding for {pdb_ids} (L={seq_lens}). Try lowering batch size. "
                             "If single sequence processing does not work, you need more vRAM to process your protein.")
                continue # skip current batch if RuntimeError

            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # slice-off padded/special tokens
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]

                if not emb_dict: # check if emb_dict is empty
                    logger.info("Embedded protein {} with length {} to emb. of shape: {}".format(
                        identifier, s_len, emb.shape))

                emb_dict[ identifier ] = emb.detach().cpu().numpy().squeeze()

    end = time.time()

    logger.info('############# STATS #############')
    logger.info('Total number of embeddings: {}'.format(len(emb_dict)))
    logger.info('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(
        end-start, (end-start)/len(emb_dict), avg_length))
    logger.info('#################################')
    logger.info("ProtT5 embeddings finished.")
    return emb_dict

def get_pdb_embd(pdb_path):
    """Generates ESM-IF1 embeddings for pdb files in a directory."""
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device)
    model = model.eval()

    pdbs = glob(f"{pdb_path}/*.pdb")
    logger.info(f'Found {len(pdbs)} pdb files in {pdb_path}')

    emb_dict = dict()

    with torch.no_grad():
        for pdb in tqdm(pdbs, desc="Processing PDB files"):
            name = Path(pdb).stem
            name = name.replace(".pdb",'_pdb') # suffix to distinguish from sequence based embeddings
            try:
                structure = esm.inverse_folding.util.load_structure(pdb, "A")
                coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
                rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)

                emb_dict[name] = [rep.detach().cpu().numpy(), seq, coords[:, 2]] # store emb, seq, CA coords
            except Exception as e:
                logger.error(f"Error processing PDB {pdb}: {e}")
                continue # skip problematic PDBs and continue processing

    logger.info('ESM-IF1 embeddings finished.')
    logger.info('='*50)

    return emb_dict

def prepare_ieffeum( seq_embd, pdb_embd, seq_path, pdb_path ):
    """Prepares IEFFEUM input files by combining ProtT5 and ESM-IF1 embeddings."""
    os.makedirs(f'{pdb_path}/pt', exist_ok=True)
    processed_names = 0
    total_names = len(seq_embd.keys())
    logger.info("Preparing IEFFEUM input files...")
    
    out_dict = {'name':[], 'file':[]}
    for name in tqdm(seq_embd.keys(), desc="Preparing IEFFEUM files"):
        if name not in pdb_embd:
            logger.warning(f"Skipping {name} as no corresponding PDB embedding found.")
            continue # skip if no corresponding pdb embedding
        prott5 = seq_embd[name]
        esm_if1, seq, CA = pdb_embd[name]
        pt = {
            'name': name,
            'seq': seq,
            'seq_embd': torch.tensor(prott5),
            'target_F': torch.tensor(CA),
            'str_embd': torch.tensor(esm_if1),
        }
        torch.save(pt, f'{pdb_path}/pt/{name}.pt')
        processed_names += 1
        out_dict['name'].append(name)
        out_dict['file'].append(f'{pdb_path}/pt/{name}.pt')
    
    torch.save(out_dict, str(seq_path).replace('.fasta','.list')) # save processed names for potential later use

    logger.info(f"IEFFEUM input preparation finished. {processed_names}/{total_names} prepared.")
    return

if __name__ == '__main__':
    parser     = create_arg_parser()
    args       = parser.parse_args()

    if args.quiet:
        logger.removeHandler(console_handler)

    if not args.fasta.lower().endswith('.fasta'): # case-insensitive check for .fasta extension
        logger.error('Input file must be a .fasta file (extension .fasta)')
        sys.exit(1)

    seq_path   = Path( args.fasta )
    if not seq_path.exists():
        logger.error(f'Fasta file not found: {seq_path}')
        sys.exit(1)

    if args.pdb:
        pdb_path = Path( args.pdb )
        if not pdb_path.exists():
            logger.error(f'PDB directory not found: {args.pdb}')
            sys.exit(1)
    else:
        pdb_path = Path(f"{args.fasta.replace('.fasta','-esmfold')}")
        os.makedirs(pdb_path, exist_ok=True) # create directory for esmfold
        logger.info(f"PDB directory not provided, using default output directory: {pdb_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    run_esmfold(seq_path, pdb_path)

    seq_embd = get_seq_embd(seq_path)
    pdb_embd = get_pdb_embd(pdb_path)

    prepare_ieffeum(seq_embd, pdb_embd, seq_path, pdb_path)

    logger.info("Data preparation script finished successfully.")