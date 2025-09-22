# IFUM
IFUM (***I***n silico ***E***valuation of un***F***olding ***F***ree ***E***nergy with ***U***nfolded state ensemble ***M***odeling, 이쁨, was ieffeum before)

![image](ieffeum.png)

Please read the [manuscript](https://www.biorxiv.org/content/10.1101/2025.02.10.637420v1) before you use IFUM.

We thank those who support open science. Without them, developing IFUM was impossible.

## Table of Contents

- [Citation](#citation)
- [Before You Start](#before-you-start)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
- [Usage](#usage)
    - [Input Data](#input-data)
    - [Running the Script](#running-the-script)
    - [Command-line Arguments](#command-line-arguments)
    - [Targeting a Specific GPU](#targeting-a-specific-gpu)
- [Output CSV File](#output-csv-file)
- [(very important) Known Limitations](#known-limitations)

## Citation
If you use the code, please cite:
```
@article{
    doi:10.1101/2025.02.10.637420,
    author = {Heechan Lee and Hahnbeom Park},
    title = {Protein folding stability estimation with an explicit consideration of unfolded states},
    journal = {bioRxiv},
    year = = {2025},
    doi = {10.1101/2025.02.10.637420},
    URL = {[https://www.biorxiv.org/content/10.1101/2025.02.10.637420v1](https://www.biorxiv.org/content/10.1101/2025.02.10.637420v1)},
}
```
## Before You Start

IFUM may produce slightly different predicted Δ*G* values when run on different GPUs. For instance, we observed a Δ*G* of 0.20 for MyUb with an A6000, and 0.19 with an A5000.

## Installation

IFUM requires [ProtT5](https://github.com/agemagician/ProtTrans) and [ESM](https://github.com/facebookresearch/esm) (specifically, ESM-IF1).

### Prerequisites:

1.  **Sufficient RAM:** Ensure your system has at least 24GB RAM (required for ESMFold). You may need less RAM if you only use pre-computed PDB or CIF files.

2.  **NVIDIA GPU:** Ensure your system has a compatible NVIDIA GPU with at least 16GB of VRAM and the drivers are correctly installed. Verify with:
    ```bash
    nvidia-smi
    ```

3.  **Conda:** We strongly recommend using [miniconda](https://docs.anaconda.com/miniconda/install/).

### Installation Steps:

Due to the archival of the ESM repository, installation requires a few specific steps. These commands have been tested for the unified script. (~20m)

```bash
# create conda environment
conda create --name IFUM python=3.9
conda activate IFUM
conda install conda-forge::cudatoolkit=11.7
conda install nvidia/label/cuda-11.7.1::cuda
```
```bash
# install dependencies
pip install omegaconf pytorch_lightning==2.1 biopython ml_collections einops py3Dmol modelcif dm-tree torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install git+https://github.com/NVIDIA/dllogger.git
pip install git+https://github.com/sokrypton/openfold.git
pip install git+https://github.com/facebookresearch/esm.git
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install torch_geometric biotite transformers==4.49.0 sentencepiece numpy==1.26.1 pandas
git clone https://github.com/HParklab/IFUM.git
cd IFUM
pip install -e .
```

## Usage

The preparation and execution steps have been merged into a single, convenient script: `scripts/predict.py`. This script handles the entire workflow from input to final prediction.

### Input Data

To run IFUM, you must provide one of the following as input:

1.  **A FASTA File (`.fasta`)**: A file containing the amino acid sequences of your proteins. If you use this option, the script will automatically use ESMFold to predict the 3D structures.
    ```
    # example: MyUb.fasta
    >MyUb_WT
    GTKKYDLSKWKYAELRDTINTSCDIELLAACREEFHRRLKVYH
    >MyUb_R1117A
    GTKKYDLSKWKYAELRDTINTSCDIELLAACREEFHRALKVYH
    ```
    **Important**: For optimal GPU memory usage, group proteins with similar sequence lengths into the same FASTA file. Different padding lengths can affect IFUM's prediction.

2.  **A Directory of Structure Files**: A path to a directory containing pre-computed structure files (`.pdb`or`.cif`). This is recommended for longer proteins or when you have high-quality experimental structures. The PDB filenames (without the extension) must match the sequence identifiers that would be in a FASTA file.
    ```
    # example: /PATH/TO/FILES
    /PATH/
    └── TO/
        └── FILES/
            ├── MyUb_WT.pdb
            └── MyUb_R1117A.cif
    ```

### Running the Script

Here are examples for the two main workflows. The script handles all intermediate steps automatically. (~15s for a single protein if esm and prott5 are cached)

* **Option A: Starting from a FASTA file (uses ESMFold)**
    ```bash
    ./scripts/predict.py \
        --input-path /path/to/your/sequences.fasta \
        --out-path /path/to/your/results.csv
    ```

* **Option B: Starting from a directory of structure files (or a direct `.pdb`/`.cif` file)**
    ```bash
    ./scripts/predict.py \
        --input-path /path/to/your/directory/ \
        --out-path /path/to/your/results.csv
    
    ./scripts/predict.py \
        --input-path /path/to/your/structure.pdb \
        --out-path /path/to/your/results.csv
    ```

**Important Note on Predicted Structures:** It is highly recommended to visually inspect any predicted structures (e.g., from ESMFold). [Poorly predicted structures can negatively impact IFUM's accuracy](https://www.biorxiv.org/content/10.1101/2025.02.10.637420v1). Consider using pre-computed, high-quality structures when available (e.g., from [AlphaFold DB](https://alphafold.ebi.ac.uk/) or experimental methods).

### Command-line Arguments

```
-i, --input-path      (Required) Path to an input i) FASTA file or ii) PDB/CIF file or iii) a directory of PDB/CIF files.
-o, --out-path        (Optional) Path for the final output CSV file. Defaults to a name based on the input.
-m, --model-path      (Optional) Path to the IFUM model parameters (.pth file).
-b, --batch-size      (Optional) Batch size for IFUM inference. Default: 1.
--per-resi            (Optional) Flag to report per-residue dG contributions in the output CSV.
--keep-intermediates  (Optional) Flag to prevent the deletion of intermediate embedding files. Useful for debugging.
--quiet               (Optional) Flag to reduce console output.
```

### Targeting a Specific GPU

If you have multiple GPUs, you can force the script to use a specific one by setting the `CUDA_VISIBLE_DEVICES` environment variable before the command.

```bash
# This command forces the script to only see and use GPU #6
CUDA_VISIBLE_DEVICES=6 ./scripts/predict.py --input-path ...
```

## Output CSV file

The script generates a CSV file with the predicted change in Gibbs free energy (Δ*G*).

```
name,dG(kcal/mol)
MyUb_WT,0.20
MyUb_R1117A,-0.05
```

## Known Limitations

It's important to be aware that IFUM's accuracy can be significantly reduced (typically biased towards unstable) when working with:
- Membrane proteins
- Monomeric structure of obligatory oligomers
- Proteins with inaccurate or low-quality folded state structures (i.e., poor-quality PDB input)

These limitations stem from the fact that IFUM was trained with soluble proteins in a PBS buffer environment.

## Reproduction
To run IFUM on Mega-scale data for reproduction, 1. get the full sequence list from the [Mega-scale zenodo](https://zenodo.org/records/7992926) 2. and run IFUM as above. Note that the result may vary based on your [hardware](#before-you-start) or a *random* seed.
