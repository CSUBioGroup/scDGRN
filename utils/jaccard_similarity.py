from concurrent.futures import ThreadPoolExecutor
import concurrent
import pandas as pd
import numpy as np
import os
import multiprocessing
from tqdm import tqdm

network_data_list = None

def read_network_data_str(filename):
    df = pd.read_parquet(filename, columns=['TF Gene', 'TG Gene'])
    tf_genes = df['TF Gene']
    tg_genes = df['TG Gene']
    return {f"{tf}\t{tg}" for tf, tg in zip(tf_genes, tg_genes)}

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def calculate_pairwise_jaccard(index_pair):
    i, j = index_pair
    similarity = jaccard_similarity(network_data_list[i], network_data_list[j])
    return (i, j, similarity)

def calculate_similarity_matrix(network_path, cell_num, save_file_name):
    global network_data_list
    with multiprocessing.Pool(processes=30) as pool:
        network_data_list = list(tqdm(pool.imap(read_network_data_str, [os.path.join(network_path, f"cell{i}.parquet") for i in range(cell_num)]), total=cell_num, desc="Reading CSV files"))


    index_pairs = [(i, j) for i in range(cell_num) for j in range(i + 1, cell_num)]

    with multiprocessing.Pool(processes=50) as pool:
        results = list(tqdm(pool.imap(calculate_pairwise_jaccard, index_pairs), total=len(index_pairs), desc="Calculating Jaccard Similarities"))

    jaccard_matrix = np.zeros((cell_num, cell_num))
    for i, j, similarity in results:
        jaccard_matrix[i, j] = similarity
    
    for i in range(cell_num):
        jaccard_matrix[i, i] = 1.0
    np.savetxt(save_file_name, jaccard_matrix, delimiter='\t')


def get_jaccard_similarity_between_cell_types_A_and_B(similarity_matrix_file, cell_type_file, A_cell_type, B_cell_type, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")
    cell_labels = pd.read_csv(cell_type_file).values.flatten().astype(str)
    A_index = np.where(cell_labels==A_cell_type)[0]
    B_index = np.where(cell_labels==B_cell_type)[0]

    similarity_matrix = np.genfromtxt(similarity_matrix_file, delimiter='\t')

    similarity_matrix = np.maximum(similarity_matrix, similarity_matrix.T)

    similarity_data = similarity_matrix[np.ix_(A_index, B_index)].flatten()
    similarity_data = similarity_data[similarity_data != 0]
    df = pd.DataFrame({A_cell_type + '_' + B_cell_type: similarity_data})
    df.to_csv(save_path + A_cell_type + '_' + B_cell_type + ".csv", header=True, index=False)





