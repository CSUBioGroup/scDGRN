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
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.backends import cudnn
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_geometric.utils import structured_negative_sampling
from scipy import sparse

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
    return np.asarray(matrix)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout(0.1)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="GSD/GSD-2000-1", help="Dataset name under the datasets directory.")
    parser.add_argument("--cell_by_gene", action="store_true", default=False, help="Expression matrix layout. True: cell x gene.")
    parser.add_argument("--has_tf_list", action="store_true", default=False, help="Whether TF.csv is provided.")
    parser.add_argument("--has_prior_network", action="store_true", default=False, help="Retained for compatibility.")
    parser.add_argument("--gpu", type=int, default=2, help="CUDA device index.")
    parser.add_argument("--gene_emb_dim", type=int, default=4, help="Gene embedding dimension.")
    parser.add_argument("--hidden_emb_dim", type=int, default=256, help="Hidden dimension.")
    parser.add_argument("--datasets-dir", type=str, default=None, help="Override datasets root.")
    parser.add_argument("--pretrain-dir", type=str, default=None, help="Override pre_train_results root.")
    return parser


def main(argv: list[str] | None = None) -> None:
    opt = build_parser().parse_args(argv)
    set_deterministic_seed()

    repo_root = Path(__file__).resolve().parent
    datasets_dir = resolve_path(opt.datasets_dir, repo_root / "datasets", repo_root)
    pretrain_dir = ensure_dir(resolve_path(opt.pretrain_dir, repo_root / "pre_train_results", repo_root) / opt.dataset_name)

    device = select_device(opt.gpu)
    origin_adata = sc.read(datasets_dir / opt.dataset_name / "ExpressionData.csv")
    source_matrix = to_dense_array(origin_adata.X)
    if opt.cell_by_gene:
        adata = sc.AnnData(X=source_matrix)
        adata.obs_names = origin_adata.obs_names.astype(str)
        adata.var_names = origin_adata.var_names.astype(str)
    else:
        adata = sc.AnnData(X=source_matrix.T)
        adata.obs_names = origin_adata.var_names.astype(str)
        adata.var_names = origin_adata.obs_names.astype(str)

    adata.var_names = adata.var_names.astype(str).str.lower()
    gene_names = np.array(adata.var_names)
    if opt.has_tf_list:
        tf_genes = np.genfromtxt(datasets_dir / opt.dataset_name / "TF.csv", delimiter="\n", dtype="str")
    else:
        tf_genes = np.array(adata.var_names).astype(str)
    tf_genes = np.char.lower(tf_genes)
    tf_index = np.array([], dtype=int)
    for tf_gene in tf_genes:
        index = np.where(gene_names == tf_gene)
        tf_index = np.append(tf_index, index[0])

    prior_network = np.zeros((tf_index.shape[0], adata.var_names.shape[0]))
    gene_interactions = pd.read_csv(datasets_dir / "network_human.zip", index_col=None, header=0)
    gene_interactions["from"] = gene_interactions["from"].str.lower()
    gene_interactions["to"] = gene_interactions["to"].str.lower()

    gene_names_list = adata.var_names.tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names_list)}
    gene_interactions_filtered = gene_interactions[
        gene_interactions["from"].isin(gene_to_idx) & gene_interactions["to"].isin(gene_to_idx)
    ]

    for _, row in gene_interactions_filtered.iterrows():
        tf = row["from"]
        tg = row["to"]
        if (tf in tf_genes) and (tg in gene_to_idx):
            tf_idx = np.where(tf_genes == tf)[0][0]
            tg_idx = gene_to_idx[tg]
            prior_network[tf_idx, tg_idx] = 1.0

    tmp_adata = sc.read(datasets_dir / opt.dataset_name / "ExpressionData.csv")
    gene_names = tmp_adata.obs_names
    tf_index = np.array(range(gene_names.shape[0]))
    ground_truth = pd.read_csv(datasets_dir / opt.dataset_name / "refNetwork.csv", header=0)
    truth_edges = set(zip(ground_truth["Gene1"], ground_truth["Gene2"], ground_truth["Type"]))
    real_adj, real_sign_adj = initialize_real_adj(gene_names, truth_edges, tf_index)

    prior_edges = get_edges_from_adj(prior_network)
    real_edges = get_edges_from_adj(real_adj)
    merged_edges = prior_edges.union(real_edges)
    print(f"先验网络的边:{len(merged_edges)}")

    source_indices = [src for src, _ in merged_edges]
    target_indices = [dst for _, dst in merged_edges]
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long).to(device)
    data = Data(x=torch.tensor(to_dense_array(adata.X).T, dtype=torch.float), edge_index=edge_index).to(device)

    model = GraphSAGE(data.num_features, hidden_channels=opt.hidden_emb_dim, embedding_dim=opt.gene_emb_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    num_epochs = 100
    model.train()
    for epoch in range(1, num_epochs + 1):
        z = model(data.x, data.edge_index)
        pos_edge_index = data.edge_index
        pos_u, pos_v, neg_v = structured_negative_sampling(
            edge_index=pos_edge_index, num_nodes=data.num_nodes, contains_neg_self_loops=False
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
    for s_idx, t_idx in zip(source_indices, target_indices):
        ground_truth_adjacency[s_idx, t_idx] = 1

    embedding_similarity_flattened = embedding_similarity.flatten()
    ground_truth_flattened = ground_truth_adjacency.flatten()
    auroc = roc_auc_score(ground_truth_flattened, embedding_similarity_flattened)
    print(f"AUROC: {auroc:.4f}")

    triu_indices = np.triu_indices(num_genes, k=1)
    positive_pairs = ground_truth_adjacency[triu_indices] == 1
    negative_pairs = ground_truth_adjacency[triu_indices] == 0
    positive_scores = np.abs(embedding_similarity[triu_indices][positive_pairs])
    negative_scores = np.abs(embedding_similarity[triu_indices][negative_pairs])

    plt.figure(figsize=(8, 6))
    plt.boxplot([positive_scores, negative_scores], labels=["Positive Pairs", "Negative Pairs"])
    plt.ylabel("Cosine Similarity")
    plt.title("Distribution of Cosine Similarity Scores")
    plt.show()

    sparse_prior = sparse.coo_matrix(ground_truth_adjacency)
    sparse.save_npz(pretrain_dir / '/tf_prior_network.npz', sparse_prior)

    np.savetxt(pretrain_dir / "tf_prior_network.csv", ground_truth_adjacency, delimiter=",")


if __name__ == "__main__":
    main()
