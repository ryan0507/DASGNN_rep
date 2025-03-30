import torch
import torch.nn as nn
import neuron
import matplotlib.pyplot as plt

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
from torch import Tensor
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
)
from torch_geometric.utils import spmm, softmax

class SAGEAggregator(nn.Module):
    def __init__(self, in_features, out_features,
                 aggr='mean',
                 concat=False,
                 bias=False):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.aggr = aggr
        self.aggregator = {'mean': torch.mean, 'sum': torch.sum}[aggr]

        self.lin_l = nn.Linear(in_features, out_features, bias=bias)
        self.lin_r = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, neigh_x):
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x, dim=0)

        if not isinstance(neigh_x, torch.Tensor):
            neigh_x = torch.cat([self.aggregator(h, dim=1)
                                for h in neigh_x], dim=0)
        else:
            neigh_x = self.aggregator(neigh_x, dim=1)

        x = self.lin_l(x)
        neigh_x = self.lin_r(neigh_x)
        out = torch.cat([x, neigh_x], dim=1) if self.concat else x + neigh_x
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggr={self.aggr})"

    
class SpikeGCNConvDegreeFeat(MessagePassing):
    def __init__(self, in_channels, out_channels, device, bins =3, alpha = 1.0, surrogate = 'sigmoid', tau = 1.0,
                 neuron_type = 'LIF', quantize = False, threshold_trainable = False, 
                 aggr='add', thr = 0.25, degree_to_label = None):
        super(SpikeGCNConvDegreeFeat, self).__init__(aggr=aggr)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.quantize = quantize
        self.device = device
        self.bins = bins
        self.degree_to_label = degree_to_label

        self.neuron = neuron.Deg_feat_neuron(ssize=out_channels,tau=tau, alpha = alpha, surrogate = surrogate, threshold_trainable=True, v_threshold=thr, bins=bins)
        neuron.reset_net(self)
    
    def forward(self, x , edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        # Embedding to x to 
        x = self.lin(x)
        
        row, col = edge_index
        deg = degree(row, x.size(0), dtype = x.dtype)
        orig_deg = deg.clone()
        deg_inv_sqrt = deg.pow_(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, size = (x.size(0), x.size(0)),
                              x = x, norm = norm, orig_deg = orig_deg)
    
    def message(self, x_j, norm):
        result = norm.view(-1, 1) * x_j
        return result
    
    def update(self, aggr_out, orig_deg):
        cur_degree = orig_deg
        max_val = cur_degree.max()
        min_val = cur_degree.min()
    
        if self.bins != 1 and self.bins != -1:
            bin_width = (max_val - min_val) / (self.bins-1)
            bin_edges = torch.arange(min_val, max_val + bin_width, bin_width)
            binned_degrees = torch.bucketize(cur_degree.to(self.device), bin_edges.to(self.device)).to(self.device)            
        else:
            binned_degrees = torch.zeros(cur_degree.size()).long()
        
        if self.degree_to_label:
            cur_degree_np = cur_degree.cpu().numpy()
            binned_degrees = torch.tensor([self.degree_to_label[int(deg)] for deg in cur_degree_np], dtype=torch.long).to(self.device)
        aggr_out = self.neuron(aggr_out, binned_degrees, cur_degree)
        return aggr_out
    
class SpikeGCNConvDegreeFeatCluster(MessagePassing):
    def __init__(self, in_channels, out_channels, device, bins =3, alpha = 1.0, surrogate = 'sigmoid', tau = 1.0,
                 neuron_type = 'LIF', quantize = False, threshold_trainable = False, 
                 aggr='add', thr = 0.25, degree_to_label = None):
        super(SpikeGCNConvDegreeFeatCluster, self).__init__(aggr=aggr)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.quantize = quantize
        self.device = device
        self.bins = bins
        self.degree_to_label = degree_to_label
        self.neuron = neuron.Deg_feat_neuron(ssize=out_channels,tau=tau, alpha = alpha, surrogate = surrogate, threshold_trainable=True, v_threshold=thr, bins=bins)
        neuron.reset_net(self)
    
    def forward(self, x , edge_index):
        # Embedding to x to 
        x = self.lin(x)
        
        row, col = edge_index
        deg = degree(row, x.size(0), dtype = x.dtype)
        orig_deg = deg.clone()
        deg_inv_sqrt = deg.pow_(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, size = (x.size(0), x.size(0)),
                              x = x, norm = norm, orig_deg = orig_deg)
    
    def message(self, x_j, norm):
        result = norm.view(-1, 1) * x_j
        return result
    
    def update(self, aggr_out, orig_deg):
        cur_degree = orig_deg
        # Number of clusters, same as bins
        cur_degree_np = cur_degree.cpu().numpy()
        mapped_labels = None
        # Trainign step
        if self.training:
            mapped_labels = torch.tensor([self.degree_to_label[int(deg)] for deg in cur_degree_np], dtype=torch.long).to(self.device)
        # Validation step
        else:
            mapped_labels 
            

        aggr_out = self.neuron(aggr_out, mapped_labels, cur_degree)
        # print(torch.unique(aggr_out))
        return aggr_out
    
class SpikeGINConvDegreeFeat(MessagePassing):
    def __init__(self, in_channels, out_channels, device, bins=3, alpha=1.0, surrogate='sigmoid', tau=1.0,
                 neuron_type='LIF', quantize=False, threshold_trainable=False, eps=0., train_eps=False,
                 aggr='add', thr=0.25, dim_h = 64, degree_to_label = None):
        super(SpikeGINConvDegreeFeat, self).__init__(aggr=aggr)
        self.quantize = quantize
        self.device = device
        self.bins = bins
        self.eps = torch.nn.Parameter(torch.tensor(eps)) if train_eps else eps
        self.degree_to_label = degree_to_label

        # Selecting the neuron model based on neuron_type
        self.neuron = neuron.Deg_feat_neuron(ssize=out_channels,tau=tau, alpha = alpha, surrogate = surrogate, threshold_trainable=True, v_threshold=thr, bins=bins)
        self.mlp_neuron = neuron.Deg_feat_neuron(ssize=dim_h,tau=tau, alpha = alpha, surrogate = surrogate, threshold_trainable=True, v_threshold=thr, bins=bins)
        neuron.reset_net(self)
        
        self.dim_h = dim_h
        self.nn1 = torch.nn.Linear(in_channels, dim_h)
        self.nn2 = torch.nn.Linear(dim_h, out_channels)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, size = None):
        
        if isinstance(x, Tensor):
            x = (x, x)
            
        row, col = edge_index
        deg = degree(row, x[0].size(0), dtype = x[0].dtype)
        cur_degree = deg
        max_val = cur_degree.max()
        min_val = cur_degree.min()
        if self.bins != 1:
            bin_width = (max_val - min_val) / (self.bins - 1)
            bin_edges = torch.arange(min_val, max_val + bin_width, bin_width)
            binned_degrees = torch.bucketize(cur_degree.to(self.device), bin_edges.to(self.device)).to(self.device)
        else:
            binned_degrees = torch.zeros(cur_degree.size()).long()
        
        if self.degree_to_label:
            cur_degree_np = cur_degree.cpu().numpy()
            binned_degrees = torch.tensor([self.degree_to_label[int(deg)] for deg in cur_degree_np], dtype=torch.long).to(self.device)

        out = self.propagate(edge_index, x=x, size=size)
        
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r
        
        out = self.nn1(out)
        spike_out = self.mlp_neuron(out, binned_degrees.clone(), cur_degree.clone())
        spike_out = self.dropout(spike_out)
        spike_out2 = self.nn2(spike_out.clone())
        out_final = self.neuron(spike_out2, binned_degrees, cur_degree)
        return out_final
    
    def message(self, x_j):
        
        return x_j
    
    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)
    
class SpikeGATConvDegreeFeat(MessagePassing):
    def __init__(self, in_channels, out_channels, device,heads =4, bins=3, alpha_surrogate=1.0, surrogate='sigmoid', tau=1.0,
                 neuron_type='LIF', quantize=False, threshold_trainable=False, eps=0., train_eps=False,
                 aggr='add', thr=0.25, dim_h = 64, degree_to_label = None):
        super(SpikeGATConvDegreeFeat, self).__init__(aggr=aggr, node_dim=0)
        self.quantize = quantize
        self.device = device
        self.bins = bins
        self.eps = torch.nn.Parameter(torch.tensor(eps)) if train_eps else eps
        self.degree_to_label = degree_to_label

        self.neuron = neuron.LAPLIF_deg_feat(ssize=out_channels,tau=tau, alpha = alpha_surrogate, surrogate = surrogate, threshold_trainable=True, v_threshold=thr, bins=bins)
        self.mlp_neuron = neuron.LAPLIF_deg_feat(ssize=dim_h,tau=tau, alpha = alpha_surrogate, surrogate = surrogate, threshold_trainable=True, v_threshold=thr, bins=bins)
        neuron.reset_net(self)
        
        self.lin_l = nn.Linear(in_channels, heads * out_channels)
        self.lin_r = nn.Linear(in_channels, heads * out_channels)
        self.att = nn.Parameter(torch.empty(1, heads, out_channels))
        nn.init.xavier_uniform_(self.att)
        self.heads = heads
        self.out_channels = out_channels
            
        
    def forward(self, x, edge_index):
        
        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
            
        row, col = edge_index
        deg = degree(row, x.size(0), dtype = x.dtype)
        cur_degree = deg
        max_val = cur_degree.max()
        min_val = cur_degree.min()
        if self.bins != 1:
            bin_width = (max_val - min_val) / (self.bins - 1)
            bin_edges = torch.arange(min_val, max_val + bin_width, bin_width)
            binned_degrees = torch.bucketize(cur_degree.to(self.device), bin_edges.to(self.device)).to(self.device)
        else:
            binned_degrees = torch.zeros(cur_degree.size()).long()
        
        if self.degree_to_label:
            cur_degree_np = cur_degree.cpu().numpy()
            binned_degrees = torch.tensor([self.degree_to_label[int(deg)] for deg in cur_degree_np], dtype=torch.long).to(self.device)

        alpha = self.edge_updater(edge_index, x=(x_l, x_r))
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
        out = out.mean(dim=1)
        out_final = self.neuron(out, binned_degrees, cur_degree)
        return out_final
    
    def edge_update(self, x_j: Tensor, x_i: Tensor, index: Tensor, ptr: OptTensor,
                    dim_size) -> Tensor:
        x = x_i + x_j
        x = nn.functional.leaky_relu(x, 0.2)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = nn.functional.dropout(alpha, p=0.5, training=self.training)
        return alpha
    
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)
    
    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)
