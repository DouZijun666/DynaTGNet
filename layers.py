import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


class MultiShallowEmbedding(nn.Module):
    
    def __init__(self, num_nodes: int, k_neighs: int, num_graphs: int):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs

        self.emb_s = Parameter(Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(Tensor(num_graphs, 1, num_nodes))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)
        
    def forward(self, device: torch.device) -> Tensor:
        # adj: [num_graphs, num_nodes, num_nodes]
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)
        
        # Remove self-loop
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')
        
        # Top-k edge adjacency
        adj_flat = adj.reshape(self.num_graphs, -1)
        topk_indices = adj_flat.topk(k=self.k, dim=1)[1].reshape(-1)
        
        graph_idx = torch.arange(self.num_graphs, device=device).repeat_interleave(self.k)
        
        adj_flat.zero_()
        adj_flat[graph_idx, topk_indices] = 1.
        adj = adj_flat.view(self.num_graphs, self.num_nodes, self.num_nodes)
        
        return adj



class GroupLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 1, bias: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.groups = groups
        self.group_mlp = nn.Conv2d(
            in_channels * groups, 
            out_channels * groups, 
            kernel_size=(1, 1), 
            groups=groups, 
            bias=bias
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        self.group_mlp.reset_parameters()
        
    def forward(self, x: Tensor, is_reshape: bool = False) -> Tensor:
        B, C, N = x.size(0), x.size(1), x.size(-2)
        G = self.groups
        
        if not is_reshape:
            x = x.view(B, C, N, G, -1).transpose(2, 3)
        x = x.transpose(1, 2).reshape(B, G * C, N, -1)
        
        out = self.group_mlp(x)
        out = out.view(B, G, self.out_channels, N, -1).transpose(1, 2)
        
        return out

class DenseGINConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 1, eps: float = 0, train_eps: bool = True):
        super().__init__()
        
        self.mlp = GroupLinear(in_channels, out_channels, groups, bias=False)
        
        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
            
        self.reset_parameters()
            
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)
        
    def norm(self, adj: Tensor, add_loop: bool) -> Tensor:
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1
        
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        return adj
        
    def forward(self, x: Tensor, adj: Tensor, add_loop: bool = True) -> Tensor:
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        B, C, N, _ = x.size()
        G = adj.size(0)
        
        adj = self.norm(adj, add_loop=False)
        
        x = x.view(B, C, N, G, -1).transpose(2, 3)
        
        out = torch.matmul(adj, x)
        
        x_pre = x[:, :, :-1, ...]
        out[:, :, 1:, ...] += x_pre
        
        if add_loop:
            out = (1 + self.eps) * x + out
        
        out = self.mlp(out, True)
        
        C = out.size(1)
        out = out.transpose(2, 3).reshape(B, C, N, -1)
        
        return out



class DenseTimeDiffPool2d(nn.Module):
    
    def __init__(self, pre_nodes: int, pooled_nodes: int, kernel_size: int, padding: int):
        super().__init__()
        
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kernel_size), padding=(0, padding))
        self.re_param = Parameter(Tensor(kernel_size, 1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')
        
    def forward(self, x: Tensor, adj: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)
        
        # s: [N^(l+1), N^l, 1, K]
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        # Compute the pooled adjacency matrix
        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))
        
        return out, out_adj