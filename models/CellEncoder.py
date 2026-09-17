import torch
from torch import nn
import torch.nn.functional as F

class CellEncoder(nn.Module):
    def __init__(self,
                 in_features,
                 latent_feature,
                 p=0.1):
        super().__init__()
        self.in_features = in_features
        self.latent_features = [latent_feature * 2, latent_feature]

        layers = []
        
        for i in range(len(self.latent_features)):
            if i == 0:
                layers.append(nn.Linear(self.in_features, self.latent_features[i]))
                layers.append(nn.PReLU())
                layers.append(nn.LayerNorm(self.latent_features[i]))
            else:
                layers.append(nn.Linear(self.latent_features[i-1], self.latent_features[i]))
                layers.append(nn.PReLU())
        
        layers.append(nn.Dropout(p=p))
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)

        self.fc = None
        
    def forward(self, x):
        h = self.encoder(x)
        out = self.fc(h)

        return h, out
    