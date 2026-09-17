import torch
from torch import nn
import torch.nn.functional as F
  

class CellTokenMoco(nn.Module):
    def __init__(self, 
                 encoder,
                 tf_index,
                 gene_num,
                 dropout_rate,
                 has_tf_list,
                 gene_emb_dim,
                 cell_emb_dim,
                 emb_dim,
                 batch_size,
                 device="cpu",
                 mlp=True,
                 K=65536,
                 m=0.999,
                 T=0.9,
                 lam=0.1,
                 alpha=0.1):
        super().__init__()
        self.start_queue = 0 
        self.K = K
        self.m = m 
        self.T = T 
        self.lam = lam 
        self.alpha = alpha 
        self.rep_dim = cell_emb_dim
        self.device = device
        
        self.encoder_q = encoder(tf_index, gene_num, dropout_rate, has_tf_list, gene_emb_dim, cell_emb_dim, emb_dim, batch_size)
         
        self.encoder_k = encoder(tf_index, gene_num, dropout_rate, has_tf_list, gene_emb_dim, cell_emb_dim, emb_dim, batch_size)

        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer("queue", 
                             F.normalize(torch.randn(self.K, self.rep_dim, requires_grad=False), dim=1))
        self.ptr = 0
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)
        
        remaining = self.K - self.ptr
        if batch_size <= remaining:
            self.queue[self.ptr:self.ptr + batch_size] = keys.detach()
            new_ptr = self.ptr + batch_size
        else:
            self.queue[self.ptr:] = keys[:remaining].detach()
            self.queue[:batch_size - remaining] = keys[remaining:].detach()
            new_ptr = batch_size - remaining

        self.ptr = new_ptr % self.K

        self.queue.requires_grad = False

    def forward(self, x1, x2, gene_token, prior_network, expression_matrix, pos_expression_matrix):
        all_attention, q, output_tg_matrix, tf_token, gene_token, tg_matrix_expanded1 = self.encoder_q(gene_token, x1, prior_network, expression_matrix)
        cell_emb = q.clone()
        latent = q.clone()
        q = F.normalize(q, dim=1)

        c = x2.size(0) // x1.size(0)
        qc = q.unsqueeze(1)
        for _ in range(1, c):
            qc = torch.cat([qc, q.unsqueeze(1)], dim=1)
        qc = qc.reshape(-1, q.size(1))

        assert qc.size(0) == x2.size(0)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            _1,k1,_,_,_,_ = self.encoder_k(gene_token, x1, prior_network, expression_matrix)
            _1,k2,_,_,_,_ = self.encoder_k(gene_token, x2, prior_network, pos_expression_matrix)

            k1 = F.normalize(k1, dim=1)
            k2 = F.normalize(k2, dim=1)

        pos_sim1 = (1 - self.lam) * torch.einsum("ic, ic -> i", [q, k1]).unsqueeze(-1)
        pos_sim2 = (self.lam / c) * torch.einsum("ic, ic -> i", [qc, k2]).unsqueeze(-1)
        pos_sim2 = pos_sim2.reshape(-1, c)

        assert pos_sim2.size(0) == pos_sim1.size(0)

        pos_sim = torch.cat([pos_sim1, pos_sim2], dim=1)
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()])

        loss = -(torch.logsumexp(pos_sim / self.T, dim=1) - torch.logsumexp(neg_sim / self.T, dim=1)).mean()
        penalty = self.alpha * (torch.mean(torch.abs(latent)))
        loss += penalty

        self._dequeue_and_enqueue(k2)

        return all_attention, cell_emb, output_tg_matrix, loss, tf_token, gene_token, tg_matrix_expanded1
    