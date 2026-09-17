import numpy as np
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import umap
from sklearn import metrics
import pandas as pd
from sklearn.calibration import LabelEncoder
import random

def calculate_cluster_ARI(similarity_matrix_file, cell_type_file, cell_embedding_file, k, res_param):
    random.seed(3407)
    np.random.seed(3407)

    similarity_matrix = np.loadtxt(similarity_matrix_file, delimiter='\t')

    for i in range(similarity_matrix.shape[0]):
        for j in range(i+1, similarity_matrix.shape[1]):
            similarity_matrix[j, i] = similarity_matrix[i, j]

    num_cells = similarity_matrix.shape[0]

    G = ig.Graph(directed=False)

    G.add_vertices(range(num_cells))


    for i in range(num_cells):
        sorted_indices = np.argsort(-similarity_matrix[i, :])
        top_k_indices = sorted_indices[:k]
        for j in top_k_indices[1:]:
            G.add_edge(i,j, weight=similarity_matrix[i, j])


    partition = la.find_partition(G, partition_type=la.RBConfigurationVertexPartition,  resolution_parameter = res_param,  seed=3407)

    clusters = partition.membership
    cell_labels = pd.read_csv(cell_type_file).values.flatten().astype(str)
    print(set(cell_labels))
    cluster_ari = metrics.adjusted_rand_score(cell_labels, clusters)
    print(cluster_ari)

    cell_labels = clusters

    cell_embedding = np.loadtxt(cell_embedding_file, delimiter='\t')
    label_encoder = LabelEncoder()
    cell_labels_encoded = label_encoder.fit_transform(cell_labels)

    pca = umap.UMAP(n_components=2, random_state=3407)

    embedding_pca = pca.fit_transform(cell_embedding)

    unique_labels, label_indices = np.unique(cell_labels_encoded, return_inverse=True)
    print(unique_labels)

    plt.clf()
    cmap = plt.cm.get_cmap('viridis', len(unique_labels))

    plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=label_indices, cmap=cmap)

    for i, label in enumerate(unique_labels):
        cluster_points = embedding_pca[label_indices == i]
        centroid = np.mean(cluster_points, axis=0)
        plt.annotate(label_encoder.inverse_transform([label])[0], (centroid[0], centroid[1]),
                    color='black', weight='bold', fontsize=8, ha='center', va='center')

    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(len(unique_labels)))
    cbar.set_ticklabels(label_encoder.inverse_transform(np.arange(len(unique_labels))))
    cbar.set_label('Cell Type')

    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.show()

