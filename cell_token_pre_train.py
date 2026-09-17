from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import umap
from scipy import sparse
from sklearn.calibration import LabelEncoder
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.CellEncoder import CellEncoder
from models.Moco import MoCo


DEFAULT_SEED = 3407
DEFAULT_KNN_MAX_ELEMENTS = 95536
DEFAULT_KNN_EF_CONSTRUCTION = 600
DEFAULT_KNN_EF_SEARCH = 600
DEFAULT_KNN_M = 100
DEFAULT_KNN_THREADS = 20


def resolve_path(value: str | None, default: Path, repo_root: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


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


def select_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_dense_array(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def build_expression_adata(origin_adata, cell_by_gene: bool) -> sc.AnnData:
    expression_source = to_dense_array(origin_adata.X)
    if cell_by_gene:
        expression_data = sc.AnnData(X=expression_source)
        expression_data.obs_names = origin_adata.obs_names.astype(str)
        expression_data.var_names = origin_adata.var_names.astype(str)
        return expression_data

    expression_data = sc.AnnData(X=expression_source.T)
    expression_data.obs_names = origin_adata.var_names.astype(str)
    expression_data.var_names = origin_adata.obs_names.astype(str)
    return expression_data


def load_cell_labels(datasets_dir: Path, dataset_name: str, has_cell_labels: bool) -> np.ndarray:
    if has_cell_labels:
        return pd.read_csv(datasets_dir / dataset_name / "cell_data.csv").values.flatten().astype(str)
    return np.ones(sc.read(datasets_dir / dataset_name / "ExpressionData.csv").shape[0], dtype=str)


def call_knn(
    x: np.ndarray,
    k: int,
    dim: int,
    *,
    max_elements: int,
    ef_construction: int,
    ef_search: int,
    m: int,
    num_threads: int,
):
    import hnswlib

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(
        max_elements=max_elements,
        ef_construction=ef_construction,
        random_seed=600,
        M=m,
    )
    index.set_num_threads(num_threads)
    index.set_ef(ef_search)
    index.add_items(x)
    neighbors, distance = index.knn_query(x, k=k)
    return neighbors[:, :], distance[:, :]


def add_noise(input_matrix: torch.Tensor, noise_factor: float, device: torch.device) -> torch.Tensor:
    return input_matrix + noise_factor * torch.randn(input_matrix.shape).to(device)


def mask_expression(input_matrix: torch.Tensor, mask_rate: float):
    mask_non_zero = torch.rand_like(input_matrix) < mask_rate
    mask_zero = torch.rand_like(input_matrix) < (mask_rate / 30.0)
    mask_matrix = torch.where(input_matrix != 0, mask_non_zero, mask_zero)
    masked_tensor = torch.where(mask_matrix, torch.full_like(input_matrix, 0), input_matrix)
    return masked_tensor, mask_matrix


def visualize_cell_embedding(sorted_emb_matrix, cell_labels, plot_name: Path, dataset_name: str) -> None:
    label_encoder = LabelEncoder()
    cell_labels_encoded = label_encoder.fit_transform(cell_labels)
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(sorted_emb_matrix[:, 1:].detach().cpu().numpy())

    unique_labels, label_indices = np.unique(cell_labels_encoded, return_inverse=True)
    plt.clf()
    cmap = plt.cm.get_cmap("viridis", len(unique_labels))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=label_indices, cmap=cmap)

    for i, label in enumerate(unique_labels):
        cluster_points = embedding[label_indices == i]
        centroid = np.mean(cluster_points, axis=0)
        plt.annotate(
            label_encoder.inverse_transform([label])[0],
            (centroid[0], centroid[1]),
            color="black",
            weight="bold",
            fontsize=8,
            ha="center",
            va="center",
        )

    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(len(unique_labels)))
    cbar.set_ticklabels(label_encoder.inverse_transform(np.arange(len(unique_labels))))
    cbar.set_label("Cell Type")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(dataset_name)
    plt.savefig(plot_name)


def save_cell_embedding(sorted_emb_matrix, file_name: Path) -> None:
    np.savetxt(file_name, sorted_emb_matrix[:, 1:].detach().cpu().numpy(), delimiter="\t")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain cell-token embeddings for scDGRN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dataset_name", type=str, default="GSD/GSD-2000-1", help="Dataset name under the datasets directory.")
    data_group.add_argument("--cell_by_gene", action="store_true", default=False, help="Expression matrix layout. True: cell x gene.")
    data_group.add_argument("--has_cell_labels", action="store_true", default=False, help="Whether to load cell_data.csv.")
    data_group.add_argument("--datasets-dir", type=str, default=None, help="Override datasets root.")
    data_group.add_argument("--pretrain-dir", type=str, default=None, help="Override pre_train_results root.")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--cell_emb_dim", type=int, default=4, help="Cell embedding dimension.")

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    train_group.add_argument("--gpu", type=int, default=2, help="CUDA device index.")
    train_group.add_argument("--k", type=int, default=16, help="KNN parameter.")
    train_group.add_argument("--num_batchs", type=int, default=20, help="Number of batches.")
    train_group.add_argument("--num_epochs", type=int, default=50, help="Training epochs.")
    train_group.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate.")
    train_group.add_argument("--noise_factor", type=float, default=0.001, help="Noise factor.")
    train_group.add_argument("--mask_rate", type=float, default=0.001, help="Mask rate.")

    knn_group = parser.add_argument_group("knn index")
    knn_group.add_argument("--knn_max_elements", type=int, default=DEFAULT_KNN_MAX_ELEMENTS, help="Maximum number of indexed samples.")
    knn_group.add_argument("--knn_ef_construction", type=int, default=DEFAULT_KNN_EF_CONSTRUCTION, help="HNSW construction ef value.")
    knn_group.add_argument("--knn_ef_search", type=int, default=DEFAULT_KNN_EF_SEARCH, help="HNSW search ef value.")
    knn_group.add_argument("--knn_m", type=int, default=DEFAULT_KNN_M, help="HNSW graph connectivity parameter.")
    knn_group.add_argument("--knn_threads", type=int, default=DEFAULT_KNN_THREADS, help="Number of HNSW worker threads.")
    return parser


def train_cell_token_model(opt: argparse.Namespace) -> None:
    set_deterministic_seed(opt.seed)

    repo_root = Path(__file__).resolve().parent
    datasets_dir = resolve_path(opt.datasets_dir, repo_root / "datasets", repo_root)
    pretrain_dir = ensure_dir(resolve_path(opt.pretrain_dir, repo_root / "pre_train_results", repo_root) / opt.dataset_name)

    torch.set_default_dtype(torch.float64)
    device = select_device(opt.gpu)

    origin_adata = sc.read(datasets_dir / opt.dataset_name / "ExpressionData.csv")
    expression_data = build_expression_adata(origin_adata, opt.cell_by_gene)
    cell_labels = load_cell_labels(datasets_dir, opt.dataset_name, opt.has_cell_labels)

    model = MoCo(CellEncoder, expression_data.shape[1], opt.cell_emb_dim, device=device).to(device)
    expression_matrix = torch.tensor(to_dense_array(expression_data.X), dtype=torch.float64).to(device)

    neighbors, _ = call_knn(
        expression_matrix.detach().cpu().numpy().astype(np.float64),
        opt.k,
        expression_matrix.shape[1],
        max_elements=opt.knn_max_elements,
        ef_construction=opt.knn_ef_construction,
        ef_search=opt.knn_ef_search,
        m=opt.knn_m,
        num_threads=opt.knn_threads,
    )
    neighbors = torch.from_numpy(neighbors.astype(np.int64)).to(device)

    expanded_expression_matrix = torch.cat(
        (
            torch.arange(expression_matrix.shape[0], dtype=torch.float64).reshape(-1, 1).to(device),
            expression_matrix.clone().to(torch.float64),
        ),
        dim=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    batch_size = max(1, int(expression_matrix.shape[0] / opt.num_batchs))

    for _epoch in range(opt.num_epochs):
        train_loader = DataLoader(expanded_expression_matrix, batch_size=batch_size, shuffle=True, drop_last=True)
        for batch_matrix_with_indices in train_loader:
            cell_indices = batch_matrix_with_indices[:, 0].clone().long().to(device)
            batch_matrix = batch_matrix_with_indices[:, 1:].clone()

            selected_rows = neighbors[cell_indices].clone()
            pos_neighbors = selected_rows[:, 1:].reshape(-1)
            pos_neighbors_info = expression_matrix[pos_neighbors].clone()
            noise_matrix = add_noise(batch_matrix.clone(), noise_factor=opt.noise_factor, device=device)
            batch_noise_mask_matrix, _mask_matrix = mask_expression(noise_matrix, mask_rate=opt.mask_rate)

            cell_token, moco_loss, revert_expression = model(batch_noise_mask_matrix, pos_neighbors_info)
            loss = moco_loss + torch.mean((revert_expression - batch_matrix) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"moco_loss: {moco_loss}")

    embedded_cells = None
    eval_loader = DataLoader(expanded_expression_matrix, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()

    for batch_matrix_with_indices in tqdm(eval_loader):
        cell_indices = batch_matrix_with_indices[:, 0].clone().long().to(device)
        batch_matrix = batch_matrix_with_indices[:, 1:].clone()

        selected_rows = neighbors[cell_indices].clone()
        pos_neighbors = selected_rows[:, 1:].reshape(-1)
        pos_neighbors_info = expression_matrix[pos_neighbors].clone()
        with torch.no_grad():
            cell_token, _moco_loss, _ = model(batch_matrix, pos_neighbors_info)

        new_hidden = torch.cat((torch.unsqueeze(cell_indices.clone(), dim=1), cell_token.clone()), dim=1)
        if embedded_cells is None:
            embedded_cells = new_hidden.clone()
        else:
            embedded_cells = torch.cat((embedded_cells, new_hidden), dim=0)

    _, sort_indices = torch.sort(embedded_cells[:, 0], dim=0)
    sorted_emb_matrix = embedded_cells[sort_indices]

    visualize_cell_embedding(sorted_emb_matrix, cell_labels, pretrain_dir / "cell_trajectory.png", opt.dataset_name)
    save_cell_embedding(sorted_emb_matrix, pretrain_dir / "cell_embedding.csv")


def main(argv: list[str] | None = None) -> None:
    opt = build_parser().parse_args(argv)
    train_cell_token_model(opt)


if __name__ == "__main__":
    main()
