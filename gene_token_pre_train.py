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
import torch.nn.functional as F
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.backends import cudnn
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_geometric.utils import structured_negative_sampling


DEFAULT_SEED = 3407


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


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int, dropout_prob: float):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x, edge_index)
        x = self.prelu3(x)
        x = self.dropout(x)
        return x


def initialize_real_adj(gene_names, truth_edges, tf_index):
    real_adj = np.zeros((gene_names.shape[0], gene_names.shape[0]))
    real_sign_adj = np.zeros((gene_names.shape[0], gene_names.shape[0]))

    for truth_edge in iter(truth_edges):
        row_index0 = np.where(gene_names == truth_edge[0])[0][0]
        col_index0 = np.where(gene_names == truth_edge[1])[0][0]
        if row_index0 != col_index0 and truth_edge[2] == "+":
            real_sign_adj[row_index0][col_index0] = 1
            real_adj[row_index0][col_index0] = 1
        elif row_index0 != col_index0 and truth_edge[2] == "-":
            real_sign_adj[row_index0][col_index0] = -1
            real_adj[row_index0][col_index0] = 1
    return real_adj[tf_index], real_sign_adj[tf_index]


def get_edges_from_adj(adj_matrix):
    edges = set()
    rows, cols = np.where(adj_matrix != 0)
    for src, dst in zip(rows, cols):
        edges.add((src, dst))
    return edges


def build_expression_adata(origin_adata, cell_by_gene: bool) -> sc.AnnData:
    source_matrix = to_dense_array(origin_adata.X)
    if cell_by_gene:
        adata = sc.AnnData(X=source_matrix)
        adata.obs_names = origin_adata.obs_names.astype(str)
        adata.var_names = origin_adata.var_names.astype(str)
        return adata

    adata = sc.AnnData(X=source_matrix.T)
    adata.obs_names = origin_adata.var_names.astype(str)
    adata.var_names = origin_adata.obs_names.astype(str)
    return adata


def load_tf_genes(adata: sc.AnnData, tf_file: Path | None) -> np.ndarray:
    if tf_file is None:
        tf_genes = np.array(adata.var_names).astype(str)
    else:
        tf_genes = np.genfromtxt(tf_file, delimiter="\n", dtype="str")
    return np.char.lower(tf_genes)


def save_similarity_boxplot(positive_scores: np.ndarray, negative_scores: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    plt.boxplot([positive_scores, negative_scores], labels=["Positive Pairs", "Negative Pairs"])
    plt.ylabel("Cosine Similarity")
    plt.title("Distribution of Cosine Similarity Scores")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain gene-token embeddings for scDGRN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dataset_name", type=str, default="GSD/GSD-2000-1", help="Dataset name under the datasets directory.")
    data_group.add_argument("--cell_by_gene", action="store_true", default=False, help="Expression matrix layout. True: cell x gene.")
    data_group.add_argument("--has_tf_list", action="store_true", default=False, help="Whether TF.csv is provided.")
    data_group.add_argument("--has_prior_network", action="store_true", default=False, help="Retained for compatibility.")
    data_group.add_argument("--datasets-dir", type=str, default=None, help="Override datasets root.")
    data_group.add_argument("--pretrain-dir", type=str, default=None, help="Override pre_train_results root.")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--gene_emb_dim", type=int, default=4, help="Gene embedding dimension.")
    model_group.add_argument("--hidden_emb_dim", type=int, default=256, help="Hidden dimension.")
    model_group.add_argument("--dropout_prob", type=float, default=0.1, help="GraphSAGE dropout rate.")

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    train_group.add_argument("--gpu", type=int, default=2, help="CUDA device index.")
    train_group.add_argument("--num_epochs", type=int, default=100, help="Training epochs.")
    train_group.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")

    prior_group = parser.add_argument_group("prior refinement")
    prior_group.add_argument("--correlation_top_k", type=int, default=30, help="Number of high-correlation candidate edges added before training.")
    prior_group.add_argument("--delete_edge_percentile", type=float, default=5.0, help="Percentile threshold for deleting weak positive edges.")
    prior_group.add_argument("--add_edge_percentile", type=float, default=0.01, help="Percentile threshold for adding strong negative-pair edges.")
    return parser


def train_graphsage_model(opt: argparse.Namespace) -> None:
    set_deterministic_seed(opt.seed)

    repo_root = Path(__file__).resolve().parent
    datasets_dir = resolve_path(opt.datasets_dir, repo_root / "datasets", repo_root)
    pretrain_dir = ensure_dir(resolve_path(opt.pretrain_dir, repo_root / "pre_train_results", repo_root) / opt.dataset_name)

    device = select_device(opt.gpu)
    origin_adata = sc.read(datasets_dir / opt.dataset_name / "ExpressionData.csv")
    adata = build_expression_adata(origin_adata, opt.cell_by_gene)

    adata.var_names = adata.var_names.astype(str).str.lower()
    tf_file = datasets_dir / opt.dataset_name / "TF.csv" if opt.has_tf_list else None
    tf_genes = load_tf_genes(adata, tf_file)

    gene_names = np.array(adata.var_names)
    tf_index = np.array([], dtype=int)
    for tf_gene in tf_genes:
        index = np.where(gene_names == tf_gene)
        tf_index = np.append(tf_index, index[0])

    tmp_adata = sc.read(datasets_dir / opt.dataset_name / "ExpressionData.csv")
    gene_names = tmp_adata.obs_names
    full_tf_index = np.array(range(gene_names.shape[0]))
    ground_truth = pd.read_csv(datasets_dir / opt.dataset_name / "refNetwork.csv", header=0)
    truth_edges = set(zip(ground_truth["Gene1"], ground_truth["Gene2"], ground_truth["Type"]))
    real_adj, _real_sign_adj = initialize_real_adj(gene_names, truth_edges, full_tf_index)
    print(f"先验网络边数:{real_adj.sum()}")

    expression_matrix = to_dense_array(adata.X).T
    correlation_matrix = np.abs(np.corrcoef(expression_matrix, rowvar=True))
    prior_network = real_adj.copy()

    candidate_mask = (prior_network == 0) & (~np.eye(prior_network.shape[0], dtype=bool))
    flat_corr = correlation_matrix[candidate_mask]
    rows, cols = np.where(candidate_mask)
    top_k = min(opt.correlation_top_k, len(flat_corr))
    if top_k > 0:
        kth = top_k if top_k < len(flat_corr) else top_k - 1
        selected = np.argpartition(-flat_corr, kth)[:top_k]
        prior_network[rows[selected], cols[selected]] = 1

    prior_edges = get_edges_from_adj(prior_network)
    print(f"先验网络边数{len(prior_edges)}")
    source_indices = [src for src, _ in prior_edges]
    target_indices = [dst for _, dst in prior_edges]

    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long).to(device)
    data = Data(x=torch.tensor(expression_matrix, dtype=torch.float), edge_index=edge_index).to(device)

    model = GraphSAGE(
        data.num_features,
        hidden_channels=opt.hidden_emb_dim,
        embedding_dim=opt.gene_emb_dim,
        dropout_prob=opt.dropout_prob,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)

    model.train()
    for epoch in range(1, opt.num_epochs + 1):
        z = model(data.x, data.edge_index)
        pos_u, pos_v, neg_v = structured_negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            contains_neg_self_loops=False,
        )

        pos_scores = (z[pos_u] * z[pos_v]).sum(dim=1)
        neg_scores = (z[pos_u] * z[neg_v]).sum(dim=1)
        loss_pos = -F.logsigmoid(pos_scores).mean()
        loss_neg = -F.logsigmoid(-neg_scores).mean()
        loss = loss_pos + loss_neg

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        gene_embeddings = model(data.x, data.edge_index)
    gene_embeddings_np = gene_embeddings.cpu().numpy()
    np.savetxt(pretrain_dir / "gene_embedding.csv", gene_embeddings_np, delimiter="\t")

    embedding_similarity = cosine_similarity(gene_embeddings_np)
    num_genes = len(gene_names)
    ground_truth_adjacency = np.zeros((num_genes, num_genes))
    for src_idx, dst_idx in zip(source_indices, target_indices):
        ground_truth_adjacency[src_idx, dst_idx] = 1
    prior_network = ground_truth_adjacency

    auroc = roc_auc_score(ground_truth_adjacency.flatten(), embedding_similarity.flatten())
    print(f"AUROC: {auroc:.4f}")

    triu_indices = np.triu_indices(num_genes, k=1)
    positive_pairs = ground_truth_adjacency[triu_indices] == 1
    negative_pairs = ground_truth_adjacency[triu_indices] == 0
    positive_scores = np.abs(embedding_similarity[triu_indices][positive_pairs])
    negative_scores = np.abs(embedding_similarity[triu_indices][negative_pairs])

    save_similarity_boxplot(positive_scores, negative_scores)

    before_num_ones = np.count_nonzero(prior_network == 1.0)
    q_positive = np.percentile(positive_scores, opt.delete_edge_percentile)

    delete_edges = 0
    delete_mask = positive_scores < q_positive
    for idx in np.where(delete_mask)[0]:
        tf_idx = triu_indices[0][idx]
        tg_idx = triu_indices[1][idx]
        tf_gene = gene_names[tf_idx]
        tg_gene = gene_names[tg_idx]

        if tf_gene in tf_genes and tg_gene in gene_names:
            gene_idx = np.where(tf_genes == tf_gene)[0]
            if prior_network[gene_idx, tg_idx] == 1.0:
                prior_network[gene_idx, tg_idx] = 0.0
                delete_edges += 1
        if tg_gene in tf_genes and tf_gene in gene_names:
            gene_idx = np.where(tf_genes == tg_gene)[0]
            if prior_network[gene_idx, tf_idx] == 1.0:
                prior_network[gene_idx, tf_idx] = 0.0
                delete_edges += 1

    q_negative = np.percentile(negative_scores, opt.add_edge_percentile)
    add_edges = 0
    add_mask = negative_scores < q_negative
    for idx in np.where(add_mask)[0]:
        tf_idx = triu_indices[0][idx]
        tg_idx = triu_indices[1][idx]
        tf_gene = gene_names[tf_idx]
        tg_gene = gene_names[tg_idx]

        if tf_gene in tf_genes and tg_gene in gene_names:
            gene_idx = np.where(tf_genes == tf_gene)[0]
            if prior_network[gene_idx, tg_idx] == 0.0:
                prior_network[gene_idx, tg_idx] = 1.0
                add_edges += 1
        if tg_gene in tf_genes and tf_gene in gene_names:
            gene_idx = np.where(tf_genes == tg_gene)[0]
            if prior_network[gene_idx, tf_idx] == 0.0:
                prior_network[gene_idx, tf_idx] = 1.0
                add_edges += 1

    after_num_ones = np.count_nonzero(prior_network == 1.0)
    print(f"Number of elements equal to 1 in origin_prior_network: {before_num_ones}")
    print(f"Delete edges: {delete_edges}")
    print(f"Add edges: {add_edges}")
    print(f"Number of elements equal to 1 in modified_prior_network: {after_num_ones}")

    refined_auroc = roc_auc_score(prior_network.flatten(), embedding_similarity[full_tf_index].flatten())
    print(f"修改后：AUROC: {refined_auroc:.4f}")
    np.savetxt(pretrain_dir / "tf_prior_network.csv", prior_network, delimiter=",")

    sparse_prior = sparse.coo_matrix(prior_network)
    sparse.save_npz(pretrain_dir / '/tf_prior_network.npz', sparse_prior)


def main(argv: list[str] | None = None) -> None:
    opt = build_parser().parse_args(argv)
    train_graphsage_model(opt)


if __name__ == "__main__":
    main()
