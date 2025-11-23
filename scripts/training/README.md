## Training IFUM (formerly IEFFEUM)

To train IFUM on your own dataset (i.e., ΔG values), clone this repo and install the required dependencies.

Note: We provide an example training sscript (`train.py`) that serves as a **template**. It **must be modified** by the user to fit specific data formats and training requirements.

The practical training procedure would look like this:
1. **Prepare Data:** Prepare your experimental stability data and the corresponding target folded structure files (i.e., `.pdb`). For example, you can download the Mega-scale dataset (Kotaro et al.) from [Zenodo](https://zenodo.org/records/7992926).
3. **Generate Embeddings:** Generate the required embedding files from the pre-trained models (ESM-IF1 and ProtT5) using the provided `prepare_embeddings.py` script.
4. **Customize Dataset Class (important)**: Edit the data loading logic in the [`SimpleBatchDataset` class](https://github.com/HParklab/IFUM/blob/eee169e1acb7aeb7e17ce0724c59eb7e4812552f/scripts/training/train.py#L13-L37) within `train.py` to match your specific label format.
5. **Customize Training Loop:** Edit the [training loop](https://github.com/HParklab/IFUM/blob/eee169e1acb7aeb7e17ce0724c59eb7e4812552f/scripts/training/train.py#L135-L157) to include any additional components you require (e.g., validation steps, custom loss functions, Weights & Biases logging, etc.).

If you encounter any issues or have questions, please contact `rosetta at snu.ac.kr`.

---
```bash
python prepare_embeddings.py --pdb_dir /PATH/TO/PDB/FILES --out_dir /PATH/TO/EMBEDDING/FILES
python train.py --epochs 1 --batch_size 1 --dataset /PATH/TO/EMBEDDING/FILES
```
