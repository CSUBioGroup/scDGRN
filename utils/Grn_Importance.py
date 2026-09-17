import numpy as np
import pandas as pd
from tqdm import tqdm

def robust_zscore(x, axis=None):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std[std == 0] = 1
    return (x - mean) / std



class Grn_Importance:
    def __init__(self, expression_matrix, gene_names):
        self.expression_matrix = expression_matrix
        self.gene_names = gene_names
        self.n_cells, self.n_genes = expression_matrix.shape
        
    def compute_gene_stats(self):
        
        gene_means = np.mean(self.expression_matrix, axis=0)
        gene_vars = np.var(self.expression_matrix, axis=0)
        
        self.gene_stats = pd.DataFrame({
            'gene': self.gene_names,
            'mean': gene_means,
            'variance': gene_vars
        })
        return self.gene_stats
    
    def create_expression_bins(self, n_bins=10):
        
        if not hasattr(self, 'gene_stats'):
            self.compute_gene_stats()
            
        self.gene_stats['mean_bin'] = pd.qcut(
            self.gene_stats['mean'], 
            q=n_bins, 
            labels=False, 
            duplicates='drop'
        )
        
        self.gene_stats['var_bin'] = pd.qcut(
            self.gene_stats['variance'], 
            q=n_bins, 
            labels=False, 
            duplicates='drop'
        )
        
        return self.gene_stats
    
    def generate_control_genesets(self, target_genes, n_controls=1000):
        
        from scipy import stats
        import warnings
        warnings.filterwarnings('ignore')
        
        if not hasattr(self, 'gene_stats'):
            self.compute_gene_stats()
        
        target_indices = [self.gene_names.index(gene) for gene in target_genes]
        target_stats = self.gene_stats.iloc[target_indices]
        
        non_target_stats = self.gene_stats[~self.gene_stats['gene'].isin(target_genes)].copy()
        
        if len(non_target_stats) < len(target_genes):
            raise ValueError("非目标基因数量不足")
        
        target_features = target_stats[['mean', 'variance']].values
        
        kde = stats.gaussian_kde(target_features.T)
        
        non_target_features = non_target_stats[['mean', 'variance']].values
        densities = kde(non_target_features.T)
        
        probabilities = densities / np.sum(densities)
        
        control_sets = []
        
        for i in range(n_controls):
            selected_indices = np.random.choice(
                len(non_target_stats), 
                size=len(target_genes), 
                replace=False, 
                p=probabilities
            )
            
            control_genes = non_target_stats.iloc[selected_indices]['gene'].tolist()
            control_sets.append(control_genes)
        
        return control_sets
    
    def compute_raw_scores(self, target_genes, control_genesets):
        gene_to_index = {gene: idx for idx, gene in enumerate(self.gene_names)}
    
        all_indices = [np.array([gene_to_index[gene] for gene in target_genes])]
        all_indices.extend([np.array([gene_to_index[gene] for gene in ctrl]) 
                        for ctrl in control_genesets])
        
        all_scores = np.zeros((self.n_cells, len(all_indices)))
        for i, indices in enumerate(all_indices):
            all_scores[:, i] = self.expression_matrix[:, indices].sum(axis=1)
        
        return all_scores[:, 0], all_scores[:, 1:]

    def normalize_scores(self, raw_target_scores, raw_control_scores):
        
        n_cells, n_controls = raw_control_scores.shape
        
        
        norm_target_step1 = robust_zscore(raw_target_scores)
        norm_control_step1 = robust_zscore(raw_control_scores, axis=0)
        
        
        cell_control_means = np.mean(norm_control_step1, axis=1, keepdims=True)
        cell_control_stds = np.std(norm_control_step1, axis=1, keepdims=True)
        
        
        cell_control_stds[cell_control_stds == 0] = 1
        
        norm_target_step2 = (norm_target_step1 - cell_control_means[:, 0]) / cell_control_stds[:, 0]
        norm_control_step2 = (norm_control_step1 - cell_control_means) / cell_control_stds
        
        final_target_scores = norm_target_step2 - np.mean(norm_target_step2)
        final_control_scores = norm_control_step2 - np.mean(norm_control_step2, axis=0, keepdims=True)
        
        return final_target_scores, final_control_scores

    def compute_p_values(self, target_scores, control_scores):

        all_control_scores = control_scores.flatten()
        
        comparisons = target_scores[:, None] <= all_control_scores  # 广播比较
        p_values = (np.sum(comparisons, axis=1) + 1) / (len(all_control_scores) + 1)
        
        return p_values 
    
    def compute_grn_scores_scdrs(self, cell_target_genes, n_controls=1000):
        
        self.compute_gene_stats()
        self.create_expression_bins()
        
        n_cells = len(cell_target_genes)
        all_grn_scores = np.zeros(n_cells)
        all_p_values = np.zeros(n_cells)
        
        unique_target_sets = {}
        for cell_idx, target_genes in enumerate(cell_target_genes):
            if len(target_genes) == 0:
                continue
            target_key = frozenset(target_genes)
            if target_key not in unique_target_sets:
                unique_target_sets[target_key] = {
                    'genes': target_genes,
                    'cell_indices': [cell_idx]
                }
            else:
                unique_target_sets[target_key]['cell_indices'].append(cell_idx)
        
        control_cache = {}
        for target_key, target_info in tqdm(unique_target_sets.items()):
            target_genes = list(target_info['genes'])
            control_genesets = self.generate_control_genesets(target_genes, n_controls)
            control_cache[target_key] = control_genesets
        
        for cell_idx in range(n_cells):
            if cell_idx % 100 == 0:
                print(f"process cells {cell_idx}/{n_cells}...")
            
            target_genes = cell_target_genes[cell_idx]
            
            if len(target_genes) == 0:
                all_grn_scores[cell_idx] = 0
                all_p_values[cell_idx] = 1.0
                continue
            
            target_key = frozenset(target_genes)
            control_genesets = control_cache[target_key]
            
            raw_target_scores, raw_control_scores = self.compute_raw_scores(target_genes, control_genesets)
            
            norm_target_scores, norm_control_scores = self.normalize_scores(raw_target_scores, raw_control_scores)
            
            p_values = self.compute_p_values(norm_target_scores, norm_control_scores)
            
            all_grn_scores[cell_idx] = norm_target_scores[cell_idx]
            all_p_values[cell_idx] = p_values[cell_idx]
        
        return all_p_values