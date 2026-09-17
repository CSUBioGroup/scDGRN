from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch.backends import cudnn

from scDGRN import scDGRN


DEFAULT_SEED = 3407


def set_deterministic_seed(seed: int = DEFAULT_SEED) -> None:
    generator = torch.Generator()
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    generator.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible GRN training.")
    parser.add_argument("--data_name", type=str, default="example_dataset", help="Dataset name under the datasets directory.")
    parser.add_argument("--has_tf_list", action="store_true", default=False, help="Whether TF.csv is provided.")
    parser.add_argument("--cell_labels", action="store_true", default=False, help="Whether to load cell labels.")
    parser.add_argument("--true_network", action="store_true", default=False, help="Whether refNetwork.csv is available.")
    parser.add_argument("--cell_gene", action="store_true", default=False, help="Expression matrix layout. True: cell x gene.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")

    parser.add_argument("--datasets-dir", type=str, default=None, help="Override datasets root.")
    parser.add_argument("--pretrain-dir", type=str, default=None, help="Override pre_train_results root.")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results root.")

    parser.add_argument("--draw_cell_tra", action="store_true", default=False, help="Draw cell trajectories.")
    parser.add_argument("--pca", action="store_true", default=False, help="Use PCA instead of UMAP for trajectory plots.")

    parser.add_argument("--save_dir", type=str, default=None, help="Directory for dataset-level outputs.")
    parser.add_argument("--save_single_network_tf_dir", type=str, default=None, help="Directory for TF edge tables.")
    parser.add_argument("--save_single_network_dir", type=str, default=None, help="Directory for per-cell GRNs.")
    parser.add_argument("--save_cell_embedding_dir", type=str, default=None, help="Directory for cell embeddings.")

    parser.add_argument("--num_epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--num_batchs", type=int, default=4, help="Number of batches.")
    parser.add_argument("--noise_factor", type=float, default=0.001, help="Unused legacy option retained for compatibility.")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--mask_rate", type=float, default=0.1, help="Unused legacy option retained for compatibility.")
    parser.add_argument("--dropout_prob", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--gene_emb_dim", type=int, default=13, help="Gene embedding dimension.")
    parser.add_argument("--cell_emb_dim", type=int, default=13, help="Cell embedding dimension.")
    parser.add_argument("--emb_dim", type=int, default=13, help="Joint embedding dimension.")
    parser.add_argument("--cl_k", type=int, default=2, help="Neighbors used for contrastive learning.")
    parser.add_argument("--k", type=int, default=2, help="KNN parameter.")
    parser.add_argument("--knockout_tf", type=str, default=None, help="TF to knock out during evaluation.")
    parser.add_argument("--edge_rate", type=float, default=0.5, help="Top edge retention ratio.")
    return parser


def main(argv: list[str] | None = None) -> None:
    opt = build_parser().parse_args(argv)
    set_deterministic_seed(opt.seed)
    model = scDGRN(opt)
    model.train_dynamic_network_model()


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_SEED", "build_parser", "main", "set_deterministic_seed"]
