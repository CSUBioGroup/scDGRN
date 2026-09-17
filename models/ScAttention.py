import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, embed_dim)


    def forward(self, query, key, value, prior_network, has_tf_list=None, tf_index=None):

        Q = self.q_linear(query)  # Shape: (batch_size, tf_genes, embed_dim)
        K = self.k_linear(key)     # Shape: (batch_size, tg_genes, embed_dim)
        V = self.v_linear(value)   # Shape: (batch_size, tf_genes, embed_dim)

        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim).transpose(1, 2) / 0.1 # (batch_size, num_heads, tf_genes, head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2) / 0.1 # (batch_size, num_heads, tg_genes, head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2) / 0.1 # (batch_size, num_heads, tf_genes, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, tf_genes, tg_genes)

        mask_value = -1e9
        scores = scores.masked_fill(prior_network==0, mask_value)
        
        
        if has_tf_list:
            scores[:, :, range(tf_index.shape[0]), tf_index] = float(mask_value)
        else:
            cells, _heads, tf, tg = scores.shape
            eye_mask = torch.eye(tf, tg, dtype=torch.bool, device=scores.device)
            eye_mask = eye_mask.unsqueeze(0).unsqueeze(0).expand(cells, _heads, -1, -1)
            scores.masked_fill_(eye_mask, mask_value)

        attention_weights = F.softmax(scores, dim=-2)  # (batch_size, num_heads, tf_genes, tg_genes)
        attention_weights = attention_weights.masked_fill(prior_network==0, 0.0)

        out = torch.matmul(attention_weights.transpose(-2, -1), V)  # (batch_size, num_heads, tg_genes, head_dim)

        out = out.transpose(1, 2).contiguous().view(out.size(0), out.size(2), -1)  # (batch_size, tg_genes, embed_dim)
        return torch.mean(attention_weights, dim=1) ,self.out_linear(out)



