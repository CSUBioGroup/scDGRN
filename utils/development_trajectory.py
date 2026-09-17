import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

def cell_type_heatmap(cell_type_network_dir, cell_type_list):


    cell_types = np.array([])
    mean_network_list = []
    for file_name in cell_type_list:
        cell_type_network = pd.read_csv(cell_type_network_dir + file_name).iloc[:,1:].values
        cell_type_network = np.nan_to_num(cell_type_network, nan=0.0)

        non_zero_mask = cell_type_network != 0

        sums = np.sum(cell_type_network * non_zero_mask, axis=1)

        counts = np.sum(non_zero_mask, axis=1)

        mean_values = np.divide(sums, counts, where=counts != 0)

        mean_values = np.where(counts == 0, 0, mean_values)
        mean_network_list.append(mean_values)
        cell_types = np.append(cell_types, file_name[:-4])
    mean_network = np.stack(mean_network_list)
    similarity_matrix = np.corrcoef(mean_network)

    similarity_df = pd.DataFrame(similarity_matrix, 
                                    index=cell_types,
                                    columns=cell_types)
    sns.clustermap(similarity_df, cmap='viridis', metric='correlation', linewidths=0.5, figsize=(8, 6))
    plt.title('Cell Type Similarity')
    plt.show()
   