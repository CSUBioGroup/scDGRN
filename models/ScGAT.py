import torch
from torch import nn

from .ScAttention import MultiheadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.PReLu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_ff)

    def forward(self, x):
        x = self.linear1(x)
        x = self.PReLu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ScGAT(nn.Module):
    def __init__(self, tf_index, gene_num, dropout_rate, has_tf_list, gene_emb_dim, cell_emb_dim, emb_dim, batch_size):
        super(ScGAT, self).__init__()
        self.has_tf_list = has_tf_list
        self.tf_index = tf_index
        self.gene_emb_dim = gene_emb_dim
        self.cell_emb_dim = cell_emb_dim
        self.emb_dim = emb_dim
        self.batch_size = batch_size

        self.tf_fusion_layer = nn.Sequential(
            nn.Linear(cell_emb_dim, emb_dim),
            nn.PReLU()
        )

        self.tg_fusion_layer = nn.Sequential(
            nn.Linear(cell_emb_dim, emb_dim),
            nn.PReLU()
        )

        self.cross_attention = MultiheadAttention(embed_dim=emb_dim, num_heads=4)

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.hidden_layer = nn.Sequential(
            nn.Linear(emb_dim, cell_emb_dim),
            nn.PReLU()
        )
   
        self.feedforward = FeedForward(cell_emb_dim, gene_num, dropout_rate)


    def forward(self, gene_token, cell_token, prior_network, expression_matrix):

        tf_token = gene_token[self.tf_index, :]

        tf_channel = tf_token
        tg_channel = gene_token
        cell_channel = cell_token


        cell_token_expanded = cell_channel.unsqueeze(1)
        
        cell_token_repeated_tg = cell_token_expanded.repeat(1, gene_token.shape[0], 1)
        cell_token_repeated_tf = cell_token_expanded.repeat(1, tf_token.shape[0], 1)
        tg_token_expanded = tg_channel.unsqueeze(0).repeat(cell_token.shape[0], 1, 1)
        tf_token_expanded = tf_channel.unsqueeze(0).repeat(cell_token.shape[0], 1, 1)

        tg_matrix_expanded = tg_token_expanded + cell_token_repeated_tg
        tf_matrix_expanded = tf_token_expanded + cell_token_repeated_tf

        tg_matrix_expanded = tg_matrix_expanded + expression_matrix.unsqueeze(-1)
        tf_matrix_expanded = tf_matrix_expanded + expression_matrix[:, self.tf_index].unsqueeze(-1)

        tf_matrix_expanded = self.tf_fusion_layer(tf_matrix_expanded)
        tg_matrix_expanded = self.tg_fusion_layer(tg_matrix_expanded)
        
        tg_matrix_expanded1 = tg_matrix_expanded

        all_attention, revert_tg_matrix_expanded = self.cross_attention(tf_matrix_expanded, tg_matrix_expanded, tf_matrix_expanded, prior_network, has_tf_list=self.has_tf_list, tf_index=self.tf_index)

        revert_tg_matrix_expanded = self.layer_norm(tg_matrix_expanded + revert_tg_matrix_expanded)

        
        revert_tg_matrix_expanded = self.hidden_layer(revert_tg_matrix_expanded)

        cell_emb = torch.sum(revert_tg_matrix_expanded, dim=1)
        output_tg_matrix = self.feedforward(cell_emb)
        
        return all_attention, cell_emb, output_tg_matrix, tf_token, gene_token, tg_matrix_expanded1
    
