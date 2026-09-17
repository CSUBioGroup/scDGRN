import torch
from torch import nn
import torch.nn.functional as F
  

class MoCo(nn.Module):
    def __init__(self, 
                 encoder,
                 input_dim,
                 emb_dim,
                 device="cpu",
                 mlp=True,
                 K=8192,
                 m=0.999,
                 T=0.9,
                 lam=0.1,
                 alpha=0.1):
        super().__init__()
        self.K = K
        self.m = m 
        self.T = T 
        self.lam = lam 
        self.alpha = alpha 
        self.rep_dim = emb_dim
        self.device = device
        
        self.encoder_q = encoder(in_features=input_dim, latent_feature=emb_dim)
         
        self.encoder_k = encoder(in_features=input_dim, latent_feature=emb_dim)

        self.revert_layer = nn.Sequential(
            nn.Linear(emb_dim, input_dim),
            nn.PReLU()
        )
        
        '''
            emb_dim  
        '''
        # Projection Head
        if mlp:
            dim_mlp = emb_dim
            
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.PReLU()
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.PReLU()
            )

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
        
        end = self.ptr + batch_size
        
        if end <= self.K:
            self.queue[self.ptr:end, :] = keys.detach()
            self.ptr = end
        else:
            remaining = self.K - self.ptr
            self.queue[self.ptr:self.K, :] = keys[:remaining].detach()
            
            self.queue[0:end - self.K, :] = keys[remaining:].detach()
            
            self.ptr = end - self.K
        self.queue.requires_grad = False

    def forward(self, x1, x2):
        cell_token, q = self.encoder_q(x1)
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

            _1,k1 = self.encoder_k(x1)
            _1,k2 = self.encoder_k(x2)

            k1 = F.normalize(k1, dim=1)
            k2 = F.normalize(k2, dim=1)

        pos_sim1 = (1 - self.lam) * torch.einsum("ic, ic -> i", [q, k1]).unsqueeze(-1)
        pos_sim2 = (self.lam / c) * torch.einsum("ic, ic -> i", [qc, k2]).unsqueeze(-1)
        pos_sim2 = pos_sim2.reshape(-1, c)

        assert pos_sim2.size(0) == pos_sim1.size(0)

        pos_sim = torch.cat([pos_sim1, pos_sim2], dim=1)
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()])

        revert_expression = self.revert_layer(cell_token)

        loss = -(torch.logsumexp(pos_sim / self.T, dim=1) - torch.logsumexp(neg_sim / self.T, dim=1)).mean()
        penalty = self.alpha * (torch.mean(torch.abs(latent)))
        loss += penalty

        self._dequeue_and_enqueue(k2)

        return cell_token, loss, revert_expression
    