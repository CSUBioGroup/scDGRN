from __future__ import annotations

import csv
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import umap
from scipy import sparse
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.CellTokenMoco import CellTokenMoco
from models.ScGAT import ScGAT
import pyarrow as pa
import pyarrow.parquet as pq
import hnswlib

def resolve_path(value: str | None, default: Path, repo_root: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


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


class scDGRN:
    def __init__(self, opt):
        self.opt = opt
        torch.set_default_dtype(torch.float64)
        self.device = select_device(self.opt.gpu)

        self.repo_root = Path(__file__).resolve().parent
        self.datasets_dir = resolve_path(self.opt.datasets_dir, self.repo_root / "datasets", self.repo_root)
        self.pretrain_dir = resolve_path(self.opt.pretrain_dir, self.repo_root / "pre_train_results", self.repo_root)
        self.results_root = resolve_path(self.opt.results_dir, self.repo_root / "results", self.repo_root)

        dataset_output_root = self.results_root / self.opt.data_name
        self.save_dir = ensure_dir(resolve_path(self.opt.save_dir, dataset_output_root, self.repo_root))
        self.save_single_network_tf_dir = ensure_dir(
            resolve_path(self.opt.save_single_network_tf_dir, dataset_output_root / "single_network_tf", self.repo_root)
        )
        self.save_single_network_dir = ensure_dir(
            resolve_path(self.opt.save_single_network_dir, dataset_output_root / "single_network", self.repo_root)
        )
        self.save_cell_embedding_dir = ensure_dir(
            resolve_path(self.opt.save_cell_embedding_dir, dataset_output_root / "cell_embedding", self.repo_root)
        )

        self.tf_genes = None
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.pre_network = None

    def dataset_path(self, relative_path: str) -> Path:
        return self.datasets_dir / self.opt.data_name / relative_path

    def pretrain_path(self, relative_path: str) -> Path:
        return self.pretrain_dir / self.opt.data_name / relative_path

    def initialize_real_adj(self, gene_names, truth_edges, tf_index):
        real_adj = torch.zeros(gene_names.shape[0], gene_names.shape[0]).float().to(self.device)
        for truth_edge in iter(truth_edges):
            row_index0 = np.where(gene_names == truth_edge[0])[0][0]
            col_index0 = np.where(gene_names == truth_edge[1])[0][0]
            if row_index0 != col_index0:
                real_adj[row_index0][col_index0] = 1
                self.rows.append(row_index0)
                self.cols.append(col_index0)
        return real_adj[tf_index]

    def initialize_tf_index(self, gene_names):
        if self.opt.has_tf_list:
            tf_genes = np.genfromtxt(self.dataset_path("TF.csv"), delimiter="\n", dtype="str")
            self.tf_genes = tf_genes
            tf_index = np.array([], dtype=int)
            for tf_gene in tf_genes:
                index = np.where(gene_names == tf_gene)
                tf_index = np.append(tf_index, index[0])
            tf_index = torch.from_numpy(tf_index).to(self.device)
        else:
            tf_index = torch.LongTensor(range(gene_names.shape[0])).to(self.device)
            self.tf_genes = gene_names
        return tf_index

    def init_cell_labels(self, cell_file_name: str):
        cell_path = self.dataset_path(cell_file_name)
        if not cell_path.exists() and cell_file_name == "cell_data_label.csv":
            fallback_path = self.dataset_path("cell_data.csv")
            if fallback_path.exists():
                cell_path = fallback_path
        cell_labels = pd.read_csv(cell_path).values.flatten().astype(str)
        return cell_labels

    def load_prior_network(self) -> np.ndarray:
        npz_path = self.pretrain_path("tf_prior_network.npz")
        csv_path = self.pretrain_path("tf_prior_network.csv")
        if npz_path.exists():
            return sparse.load_npz(npz_path).toarray()
        if csv_path.exists():
            return np.loadtxt(csv_path, delimiter=",")
        raise FileNotFoundError(
            f"Missing prior network for '{self.opt.data_name}'. Expected {npz_path} or {csv_path}."
        )

    def init_data_network(self):
        expression_data = sc.read(self.dataset_path("ExpressionData.csv"))
        expression_matrix = to_dense_array(expression_data.X)

        if self.opt.cell_gene:
            origin_expression_matrix = expression_matrix
            gene_names = np.array(expression_data.var_names)
            data_matrix = expression_matrix
        else:
            origin_expression_matrix = expression_matrix.T
            gene_names = np.array(expression_data.obs_names)
            data_matrix = expression_matrix.T

        sums = data_matrix.sum(axis=0, keepdims=True)
        counts = np.count_nonzero(data_matrix, axis=0, keepdims=True)
        cell_mean = np.divide(sums, counts, where=counts != 0, out=np.zeros_like(sums, dtype=np.float64))
        normalized_data = data_matrix / cell_mean

        new_expression_data = torch.tensor(origin_expression_matrix, dtype=torch.float64).to(self.device)
        normalized_data = torch.tensor(normalized_data, dtype=torch.float64).to(self.device)

        num_genes = new_expression_data.shape[1]
        cell_labels = None
        if self.opt.cell_labels:
            cell_labels = self.init_cell_labels("cell_data_label.csv")

        tf_index = self.initialize_tf_index(gene_names)

        real_adj = None
        real_sign_adj = None
        if self.opt.true_network:
            ground_truth = pd.read_csv(self.dataset_path("refNetwork.csv"), header=0)
            truth_edges = set(zip(ground_truth["Gene1"], ground_truth["Gene2"]))
            real_adj = self.initialize_real_adj(gene_names, truth_edges, tf_index)

        prior_network_np = self.load_prior_network()
        self.pre_network = prior_network_np
        prior_network = torch.tensor(prior_network_np, dtype=torch.int).to(self.device)

        return tf_index, real_adj, real_sign_adj, num_genes, new_expression_data, gene_names, cell_labels, prior_network, normalized_data

    def call_knn(self, x, k, dim, max_element=95536):
        p = hnswlib.Index(space="cosine", dim=dim)
        p.init_index(max_elements=max_element, ef_construction=600, random_seed=600, M=100)
        p.set_num_threads(20)
        p.set_ef(600)
        p.add_items(x)
        neighbors, distance = p.knn_query(x, k=k)
        return neighbors[:, :], distance[:, :]

    def single_network_to_file(self, cell_network, cell_ordinal: str):
        self.network_to_file(cell_network, self.save_single_network_dir / cell_ordinal)

    def triple_edge_to_file(self, cell_network, tf_genes, tg_genes, cell_ordinal: Path):
        sorted_indices = torch.argsort(cell_network.abs().flatten(), descending=True)
        sorted_tf_indices, sorted_tg_indices = (
            sorted_indices // cell_network.size(1),
            sorted_indices % cell_network.size(1),
        )

        num_edges_to_save = int(self.pre_network.sum() * self.opt.edge_rate)
        edges = []
        for idx in range(num_edges_to_save):
            tf_idx = sorted_tf_indices[idx].item()
            tg_idx = sorted_tg_indices[idx].item()
            edge_weight = cell_network[tf_idx, tg_idx].item()
            edges.append([tf_genes[tf_idx], tg_genes[tg_idx], edge_weight])

        edges_df = pd.DataFrame(edges, columns=["TF Gene", "TG Gene", "Edge Weight"])
        edges_table = pa.Table.from_pandas(edges_df)
        pq.write_table(edges_table, cell_ordinal)

    def train_dynamic_network_model(self):
        has_tf_list = self.opt.has_tf_list
        tf_index, real_adj, real_sign_adj, num_genes, expression_matrix, gene_names, cell_labels, prior_network, normalized_data = self.init_data_network()

        loaded_cell_array = np.loadtxt(self.pretrain_path("cell_embedding.csv"), delimiter="\t")
        cell_token = torch.from_numpy(loaded_cell_array).to(self.device)

        loaded_gene_array = np.loadtxt(self.pretrain_path("gene_embedding.csv"), delimiter="\t")
        gene_token = torch.from_numpy(loaded_gene_array).to(self.device)

        pos_z = gene_token
        batch_size = max(1, int(expression_matrix.shape[0] / self.opt.num_batchs))
        sc_gat = CellTokenMoco(
            ScGAT,
            tf_index,
            num_genes,
            self.opt.dropout_prob,
            has_tf_list,
            self.opt.gene_emb_dim,
            self.opt.cell_emb_dim,
            self.opt.emb_dim,
            batch_size,
        ).to(self.device)

        optimizer = torch.optim.AdamW(sc_gat.parameters(), lr=self.opt.learning_rate)
        sc_gat.train()

        k = self.opt.k
        cl_k = self.opt.cl_k
        tmp_expression_matrix = expression_matrix.clone()
        neighbors, _ = self.call_knn(
            tmp_expression_matrix.detach().cpu().numpy().astype(np.float64),
            k if k > cl_k else cl_k,
            expression_matrix.shape[1],
        )
        neighbors = torch.from_numpy(neighbors.astype(np.int64)).to(self.device)

        expanded_expression_matrix = torch.cat(
            (
                torch.arange(expression_matrix.shape[0], dtype=torch.float64).reshape(-1, 1).to(self.device),
                expression_matrix.clone().to(torch.float64),
            ),
            dim=1,
        )

        a_weight = 0.5
        for epoch in tqdm(range(self.opt.num_epochs)):
            train_loader = DataLoader(expanded_expression_matrix, batch_size=batch_size, shuffle=True, drop_last=True)
            for batch_matrix_with_indices in train_loader:
                cell_indices = batch_matrix_with_indices[:, 0].clone().long().to(self.device)

                tmp_cell_token = cell_token[cell_indices, :]
                select_expression_matrix = normalized_data[cell_indices, :]

                selected_rows = neighbors[cell_indices].clone()
                pos_neighbors = selected_rows[:, 1:cl_k].reshape(-1)
                pos_neighbors_info = cell_token[pos_neighbors].clone()
                pos_expression_matrix = normalized_data[pos_neighbors, :]

                specificity_pos_neighbors = selected_rows[:, 1:k].reshape(-1)
                specificity_pos_expression_matrix = normalized_data[specificity_pos_neighbors, :]
                select_expression_matrix = a_weight * select_expression_matrix + (1 - a_weight) * specificity_pos_expression_matrix.reshape(
                    select_expression_matrix.shape[0], k - 1, select_expression_matrix.shape[1]
                ).mean(axis=1)

                tmp_pos_selected_rows = neighbors[pos_neighbors].clone()
                tmp_pos_pos_neighbors = tmp_pos_selected_rows[:, 1:k].reshape(-1)
                tmp_pos_pos_expression_matrix = normalized_data[tmp_pos_pos_neighbors, :]
                pos_expression_matrix = a_weight * pos_expression_matrix + (1 - a_weight) * tmp_pos_pos_expression_matrix.reshape(
                    pos_expression_matrix.shape[0], k - 1, pos_expression_matrix.shape[1]
                ).mean(axis=1)

                relation, cell_emb, decoded_output, moco_loss, tf_token, tg_token, tg_matrix_expanded1 = sc_gat(
                    tmp_cell_token, pos_neighbors_info, pos_z, prior_network, select_expression_matrix, pos_expression_matrix
                )

                zero_matrix = tmp_expression_matrix[cell_indices] == 0
                non_zero_matrix = tmp_expression_matrix[cell_indices] != 0

                zero_loss = torch.mean(torch.pow(decoded_output - tmp_expression_matrix[cell_indices], 2) * zero_matrix)
                non_zero_loss = torch.mean(
                    torch.pow(decoded_output - tmp_expression_matrix[cell_indices], 2) * non_zero_matrix
                )
                reconstruction_loss = 0.1 * zero_loss + 0.9 * non_zero_loss
                l2_norm = sum(p.pow(2).sum() for p in sc_gat.parameters())
                loss = 0.499 * reconstruction_loss + 0.500 * moco_loss + 0.001 * l2_norm

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"reconstruction_loss: {reconstruction_loss} moco_loss: {moco_loss}    L2_loss: {l2_norm}     loss:{loss}")

        emb_expression_matrix = None
        eval_loader = DataLoader(expanded_expression_matrix, batch_size=batch_size, shuffle=False, drop_last=False)
        sc_gat.eval()
        final_relation = torch.zeros(tf_index.shape[0], gene_names.shape[0]).float().to(self.device)
        all_decoded_outputs = []

        knockout_tf = self.opt.knockout_tf
        if knockout_tf is not None:
            knockout_tf_index = np.where(gene_names == knockout_tf)[0]
            normalized_data[:, knockout_tf_index] = 0.0

        for batch_matrix_with_indices in tqdm(eval_loader):
            cell_indices = batch_matrix_with_indices[:, 0].clone().long().to(self.device)
            with torch.no_grad():
                tmp_cell_token = cell_token[cell_indices, :]
                select_expression_matrix = normalized_data[cell_indices, :]

                selected_rows = neighbors[cell_indices].clone()
                pos_neighbors = selected_rows[:, 1:cl_k].reshape(-1)
                pos_neighbors_info = cell_token[pos_neighbors].clone()
                pos_expression_matrix = normalized_data[pos_neighbors, :]

                specificity_pos_neighbors = selected_rows[:, 1:k].reshape(-1)
                specificity_pos_expression_matrix = normalized_data[specificity_pos_neighbors, :]
                select_expression_matrix = a_weight * select_expression_matrix + (1 - a_weight) * specificity_pos_expression_matrix.reshape(
                    select_expression_matrix.shape[0], k - 1, select_expression_matrix.shape[1]
                ).mean(axis=1)

                tmp_pos_selected_rows = neighbors[pos_neighbors].clone()
                tmp_pos_pos_neighbors = tmp_pos_selected_rows[:, 1:k].reshape(-1)
                tmp_pos_pos_expression_matrix = normalized_data[tmp_pos_pos_neighbors, :]
                pos_expression_matrix = a_weight * pos_expression_matrix + (1 - a_weight) * tmp_pos_pos_expression_matrix.reshape(
                    pos_expression_matrix.shape[0], k - 1, pos_expression_matrix.shape[1]
                ).mean(axis=1)

                relation, cell_emb, decoded_output, moco_loss, tf_token, tg_token, tg_matrix_expanded1 = sc_gat(
                    tmp_cell_token, pos_neighbors_info, pos_z, prior_network, select_expression_matrix, pos_expression_matrix
                )
                all_decoded_outputs.append(decoded_output.cpu())

            new_hidden = torch.cat((torch.unsqueeze(cell_indices.clone(), dim=1), cell_emb.clone()), dim=1)
            if emb_expression_matrix is None:
                emb_expression_matrix = new_hidden.clone()
            else:
                emb_expression_matrix = torch.cat((emb_expression_matrix, new_hidden), dim=0)

            for i, cell_idx in enumerate(cell_indices):
                cell_idx_item = cell_idx.item()
                self.single_network_to_file(relation[i], cell_ordinal=f"cell{cell_idx_item}.npz")
                self.triple_edge_to_file(
                    cell_network=relation[i],
                    tf_genes=self.tf_genes,
                    tg_genes=gene_names,
                    cell_ordinal=self.save_single_network_tf_dir / f"cell{cell_idx_item}.parquet",
                )

            final_relation = final_relation + torch.sum(relation, dim=0)

        full_decoded_matrix = torch.cat(all_decoded_outputs, dim=0)
        recovery_adata = anndata.AnnData(X=full_decoded_matrix.numpy())
        recovery_adata.obs_names = [f"cell_{i}" for i in range(tmp_expression_matrix.shape[0])]
        recovery_adata.var_names = gene_names
        recovery_adata.write(self.save_dir / "reconstructed_expression.h5ad")

        _, sort_indices = torch.sort(emb_expression_matrix[:, 0], dim=0)
        sorted_emb_matrix = emb_expression_matrix[sort_indices]
        if self.opt.draw_cell_tra:
            self.save_cell_embedding(sorted_emb_matrix, self.save_cell_embedding_dir / "cell_embedding.csv")
            self.visualize_cell_embedding(sorted_emb_matrix, cell_labels, self.save_cell_embedding_dir / "cell_trajectory.png")

        final_relation = (final_relation - torch.min(final_relation, dim=0)[0]) / (
            torch.max(final_relation, dim=0)[0] - torch.min(final_relation, dim=0)[0] + 1e-8
        )
        self.network_to_file(final_relation, self.save_dir / f"{self.opt.data_name.split('/')[-1]}_GRN.npz")

        if self.opt.true_network:
            positive_indices = np.where(prior_network.cpu().numpy() == 1)[0]
            final_relation_np = final_relation.detach().float().cpu().numpy()
            predicted_values = final_relation_np[positive_indices].flatten()
            true_labels = real_adj.float().cpu().numpy()[positive_indices].flatten()
            precision, recall, _ = precision_recall_curve(true_labels, predicted_values)
            auprc = auc(recall, precision)
            auroc = roc_auc_score(true_labels, predicted_values)
            self.save_evaluation_metric(auroc, auprc, self.results_root / "evaluation_metric.csv")

    def save_evaluation_metric(self, auroc, auprc, file_name: Path):
        ensure_dir(file_name.parent)
        new_data = {
            "dataset": self.opt.data_name.split("/")[-1],
            "algorithm": "scDGRN",
            "auroc": round(auroc, 9),
            "auprc": round(auprc, 9),
            "label": None,
            "gene_emb_dim": self.opt.gene_emb_dim,
            "cell_emb_dim": self.opt.cell_emb_dim,
            "emb_dim": self.opt.emb_dim,
        }
        last_two_chars = self.opt.data_name.split("/")[-1][-2:]
        if last_two_chars == "50":
            new_data["label"] = 50
        elif last_two_chars == "70":
            new_data["label"] = 70
        else:
            new_data["label"] = 0

        with open(file_name, mode="a", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["dataset", "algorithm", "auroc", "auprc", "label", "gene_emb_dim", "cell_emb_dim", "emb_dim"],
            )
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(new_data)

    def save_cell_embedding(self, sorted_emb_matrix, file_name: Path):
        np.savetxt(file_name, sorted_emb_matrix[:, 1:].detach().cpu().numpy(), delimiter="\t")

    def visualize_cell_embedding(self, sorted_emb_matrix, cell_labels, plot_name: Path):
        label_encoder = LabelEncoder()
        cell_labels_encoded = label_encoder.fit_transform(cell_labels)
        if self.opt.pca:
            reducer = PCA(n_components=2)
        else:
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

        if self.opt.pca:
            plt.xlabel("PCA1")
            plt.ylabel("PCA2")
        else:
            plt.xlabel("UMAP1")
            plt.ylabel("UMAP2")
        plt.title(self.opt.data_name.split("/")[-1])
        plt.savefig(plot_name)

    def network_to_file(self, pre_network, file_name: Path):
        pre_network_numpy = pre_network.detach().float().cpu().numpy()
        sparse_prior = sparse.coo_matrix(pre_network_numpy)
        sparse.save_npz(file_name, sparse_prior)

__all__ = ["scDGRN"]
