from __future__ import annotations

from pathlib import Path
from typing import Iterable

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm


EDGE_COLUMNS = ("TF Gene", "TG Gene", "Edge Weight")


def load_cell_labels(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path)
    if column not in frame.columns:
        if frame.shape[1] != 1:
            raise KeyError(f"{path} does not contain column {column!r}.")
        column = frame.columns[0]
    return frame[column].astype(str).reset_index(drop=True)


def validate_analysis_inputs(dataset_dir: Path, result_dir: Path) -> dict[str, Path]:
    paths = {
        "expression": dataset_dir / "ExpressionData.csv",
        "labels": dataset_dir / "cell_data.csv",
        "tf_list": dataset_dir / "TF.csv",
        "cell_embedding": result_dir / "cell_embedding" / "cell_embedding.csv",
        "edge_tables": result_dir / "single_network_tf",
        "networks": result_dir / "single_network",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required tutorial inputs are missing. Run scDGRN inference first or set "
            "SCDGRN_RESULTS_ROOT/SCDGRN_RESULT_NAME.\n" + formatted
        )
    return paths


def _edge_file(edge_dir: Path, cell_index: int) -> Path:
    parquet = edge_dir / f"cell{cell_index}.parquet"
    if parquet.exists():
        return parquet
    csv = edge_dir / f"cell{cell_index}.csv"
    if csv.exists():
        return csv
    raise FileNotFoundError(f"Missing edge table for cell {cell_index} in {edge_dir}")


def _read_edges(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [column for column in EDGE_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing edge columns: {missing}")
    return frame.loc[:, EDGE_COLUMNS].copy()


def build_jaccard_similarity(
    edge_dir: Path,
    n_cells: int,
    top_n: int = 50,
    exclude_unit_weight: bool = True,
) -> np.ndarray:
    edge_to_column: dict[tuple[str, str], int] = {}
    rows: list[int] = []
    columns: list[int] = []

    for cell_index in tqdm(range(n_cells), desc="Reading cell-specific edge tables"):
        frame = _read_edges(_edge_file(edge_dir, cell_index))
        frame["Edge Weight"] = pd.to_numeric(frame["Edge Weight"], errors="coerce")
        frame = frame.dropna(subset=["Edge Weight"])
        if exclude_unit_weight:
            frame = frame.loc[~np.isclose(frame["Edge Weight"], 1.0)]
        frame = frame.nlargest(top_n, "Edge Weight")
        edges = set(zip(frame["TF Gene"].astype(str), frame["TG Gene"].astype(str)))
        for edge in edges:
            column = edge_to_column.setdefault(edge, len(edge_to_column))
            rows.append(cell_index)
            columns.append(column)

    incidence = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(n_cells, len(edge_to_column)),
    )
    intersections = (incidence @ incidence.T).toarray().astype(np.float32)
    counts = np.asarray(incidence.sum(axis=1)).ravel()
    unions = counts[:, None] + counts[None, :] - intersections
    similarity = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )
    np.fill_diagonal(similarity, 1.0)
    return similarity


def leiden_from_similarity(
    similarity: np.ndarray,
    k: int = 24,
    resolution: float = 0.35,
    seed: int = 3407,
) -> np.ndarray:
    similarity = np.asarray(similarity, dtype=float)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix")

    n_cells = similarity.shape[0]
    k = min(max(2, int(k)), max(2, n_cells - 1))
    edge_weights: dict[tuple[int, int], float] = {}
    for i in range(n_cells):
        candidates = np.argpartition(similarity[i], -(k + 1))[-(k + 1):]
        candidates = candidates[np.argsort(similarity[i, candidates])[::-1]]
        for j in candidates:
            if i == j or similarity[i, j] <= 0:
                continue
            edge = (min(i, int(j)), max(i, int(j)))
            edge_weights[edge] = max(edge_weights.get(edge, 0.0), float(similarity[i, j]))

    graph = ig.Graph(n=n_cells, edges=list(edge_weights), directed=False)
    weights = list(edge_weights.values())
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership, dtype=int)


def mean_similarity_by_group(similarity: np.ndarray, labels: Iterable[str]) -> pd.DataFrame:
    labels = np.asarray(list(labels), dtype=str)
    groups = pd.unique(labels)
    output = np.zeros((len(groups), len(groups)), dtype=float)
    for i, left in enumerate(groups):
        left_idx = np.flatnonzero(labels == left)
        for j, right in enumerate(groups):
            right_idx = np.flatnonzero(labels == right)
            block = similarity[np.ix_(left_idx, right_idx)]
            if left == right and len(left_idx) > 1:
                block = block[~np.eye(len(left_idx), dtype=bool)]
            output[i, j] = float(block.mean()) if block.size else np.nan
    return pd.DataFrame(output, index=groups, columns=groups)


def compute_tf_activity(network_dir: Path, n_cells: int, n_tfs: int) -> np.ndarray:
    activity = np.zeros((n_cells, n_tfs), dtype=np.float32)
    for cell_index in tqdm(range(n_cells), desc="Summing outgoing GRN weights"):
        path = network_dir / f"cell{cell_index}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        matrix = sp.load_npz(path).tocsr()
        if matrix.shape[0] != n_tfs:
            raise ValueError(f"{path.name}: expected {n_tfs} TF rows, found {matrix.shape[0]}")
        activity[cell_index] = np.asarray(matrix.sum(axis=1)).ravel()
    return activity
