import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
from IPython import embed


class MLPs(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ConstrainedLearnablePE(nn.Module):
    def __init__(self,hidden_dim,num_queries,temperature=5):
        super().__init__()
        self.D = hidden_dim
        self.T = temperature
        self.num_queries = num_queries
        self.mlp = MLPs(2 * hidden_dim, 2 * hidden_dim, hidden_dim,3)
        self.x_embed = nn.Embedding(num_queries, 1)
    def sin_or_cos(self,x,i):
        T = self.T ** (2*i/self.D)
        if i%2==0:
            return torch.sin(x/T)
        else:
            return torch.cos(x/T)

    def forward(self,x1,y1,x2,y2):
        #embed()
        bs = x1.shape[0]
        x_embed =  self.x_embed.weight.sigmoid().squeeze(1)
        x = [self.sin_or_cos(x_embed,i) for i in range(self.D)]
        x = torch.stack(x).transpose(0,1)
        k = (y2-y1)/(x2-x1)
        b = y1 - k * x1
        k = k.unsqueeze(1).repeat(1,10)
        b = b.unsqueeze(1).repeat(1,10)
        y = k * x_embed + b
        y = [self.sin_or_cos(y,i) for i in range(self.D)]
        y = torch.stack(y).permute(1,2,0)
        x = x.unsqueeze(0).repeat(bs,1,1)
        pos_embed = torch.cat([x,y],dim=2)
        pos_embed = self.mlp(pos_embed)
        return pos_embed


if __name__ == '__main__':
    pe = ConstrainedLearnablePE(512,10)
    x1 = torch.Tensor([0.1,0.1,0.1,0.1])
    x2 = torch.Tensor([0.2,0.2,0.2,0.2])
    pe(x1,x1,x2,x2)
    embed()
