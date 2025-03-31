import argparse
import os.path as osp
import time
import torch
import torch.nn as nn
import pyg_lib
import sys
import os
import time
import numpy as np
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv 
from torch_geometric.utils import degree, add_self_loops
from sklearn import metrics
from layers import SpikeGCNConvDegreeFeat, SpikeGATConvDegreeFeat, SpikeGINConvDegreeFeat
from utils import (add_selfloops, set_seed, tab_printer)
from torch_geometric.datasets import Flickr, Reddit, Planetoid, Reddit2, Yelp, TUDataset, GNNBenchmarkDataset
from torch_geometric.utils import to_scipy_sparse_matrix, degree
from tqdm import tqdm
from spikingjelly.clock_driven import encoding, functional
from torch_geometric.loader import NeighborLoader, DataLoader
from torch.utils.data import Subset
from utils import rename_folder_with_suffix
import neuron
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

class StreamRedirector:
    def __init__(self, file_object, stdout):
        self.file_object = file_object
        self.stdout = stdout

    def write(self, message):
        self.file_object.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file_object.flush()
        self.stdout.flush()

class SNNGCN(torch.nn.Module): 
     
    def __init__(self, in_features, out_features, device, hidden = [128],
                 dropout = 0.5, bias = False, T = 15, neuron_type = 'LIF',
                 quantize = False, bn = False, thtr = False,
                 aggr='mean', thr = 0.25, poisson = True, bins = 10, degree_to_label = None):
        super().__init__()

        self.bn = bn        
        self.hidden = hidden
        self.poisson = poisson
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bins = bins
        self.degree_to_label = degree_to_label
        if self.degree_to_label:
            self.convs.append(SpikeGCNConvDegreeFeat(in_features, self.hidden[0], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                bins = self.bins, device = device,degree_to_label=degree_to_label))
        else:
            self.convs.append(SpikeGCNConvDegreeFeat(in_features, self.hidden[0], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                bins = self.bins, device = device))
        for idx, hid in enumerate(self.hidden):
            
            if idx == len(self.hidden) - 2:
                break
            else:
                if self.degree_to_label:
                    self.convs.append(SpikeGCNConvDegreeFeat(self.hidden[idx], self.hidden[idx+1], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                        bins = self.bins, device = device,degree_to_label=degree_to_label))
                else:
                    self.convs.append(SpikeGCNConvDegreeFeat(self.hidden[idx], self.hidden[idx+1], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                        bins = self.bins, device = device))
        self.lin = torch.nn.Linear(self.hidden[len(hidden)-1], out_features)
        self.T = T
        self.dropout = nn.Dropout(dropout)
        self.origconv = GCNConv(self.hidden[len(hidden)-2],self.hidden[len(hidden)-1])
        self.threshold_list = [[] for i in range(len(self.convs))]
        
    def forward(self, data):
        for t in range(self.T):
            if t == 0 :
                out_spike_counter =  self.encode(data)
            else:
                out_spike_counter += self.encode(data)

        neuron.reset_net(self)
        return out_spike_counter / self.T
        
    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        if self.poisson:
            input_encoder = encoding.PoissonEncoder()
            x = input_encoder(x)
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.dropout(x)
        x = self.origconv(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)
        x = self.lin(x)
        return x

class SNNGAT(torch.nn.Module): 
     
    def __init__(self, in_features, out_features, device, hidden = [64,64,64],
                 dropout = 0.5, bias = False, T = 15, neuron_type = 'LIF',
                 quantize = False, bn = False, thtr = False,
                 aggr='mean', thr = 0.25, poisson = True, bins = 10, degree_to_label = None):
        super().__init__()

        self.bn = bn        
        self.hidden = [64,64,64]
        self.poisson = poisson
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bins = bins
        self.degree_to_label = degree_to_label
        if self.degree_to_label:
            self.convs.append(SpikeGATConvDegreeFeat(in_features, self.hidden[0], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                bins = self.bins, device = device,degree_to_label=degree_to_label))
        else:
            self.convs.append(SpikeGATConvDegreeFeat(in_features, self.hidden[0], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                bins = self.bins, device = device))
        for idx, hid in enumerate(self.hidden):
            if idx == len(self.hidden) - 2:
                break
            else:
                if self.degree_to_label:
                    self.convs.append(SpikeGATConvDegreeFeat(self.hidden[idx], self.hidden[idx+1], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                        bins = self.bins, device = device,degree_to_label=degree_to_label))
                else:
                    self.convs.append(SpikeGATConvDegreeFeat(self.hidden[idx], self.hidden[idx+1], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                        bins = self.bins, device = device))
        
        self.lin = torch.nn.Linear(self.hidden[len(hidden)-1], out_features)
        self.T = T
        self.dropout = nn.Dropout(dropout)
        self.origconv = GATv2Conv(self.hidden[len(hidden)-2], self.hidden[len(hidden)-1], heads=4, concat = False)
        self.threshold_list = [[] for i in range(len(self.convs))]
        
    def forward(self, data):
        for t in range(self.T):
            if t == 0 :
                out_spike_counter =  self.encode(data)
            else:
                out_spike_counter += self.encode(data)
        neuron.reset_net(self)
        return out_spike_counter / self.T
        
    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        if self.poisson:
            input_encoder = encoding.PoissonEncoder()
            x = input_encoder(x)
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.dropout(x)
        x = self.origconv(x, edge_index)
        x = global_mean_pool(x, data.batch)
        # print(x.size())
        x = self.dropout(x)
        x = self.lin(x)
        return x

class SNNGIN(torch.nn.Module): 
     
    def __init__(self, in_features, out_features, device, hidden = [128],
                 dropout = 0.5, bias = False, T = 15, neuron_type = 'LIF',
                 quantize = False, bn = False, thtr = False,
                 aggr='mean', thr = 0.25, poisson = True, bins = 10, degree_to_label = None):
        super().__init__()

        self.bn = bn        
        self.hidden = hidden
        self.poisson = poisson
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bins = bins
        self.degree_to_label = degree_to_label
        
        if self.degree_to_label:
            self.convs.append(SpikeGINConvDegreeFeat(in_features, self.hidden[0], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                bins = self.bins, device = device,degree_to_label=degree_to_label))
        else:
            self.convs.append(SpikeGINConvDegreeFeat(in_features, self.hidden[0], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                bins = self.bins, device = device))
        for idx, hid in enumerate(self.hidden):
            
            if idx == len(self.hidden) - 2:
                break
            else:
                if self.degree_to_label:
                    self.convs.append(SpikeGINConvDegreeFeat(self.hidden[idx], self.hidden[idx+1], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                        bins = self.bins, device = device,degree_to_label=degree_to_label))
                else:
                    self.convs.append(SpikeGINConvDegreeFeat(self.hidden[idx], self.hidden[idx+1], neuron_type = neuron_type, quantize = quantize, threshold_trainable=thtr, aggr=aggr, thr =thr,
                                                        bins = self.bins, device = device))
        self.origconv = GINConv(torch.nn.Linear(self.hidden[len(hidden)-1], self.hidden[len(hidden)-1]))
        self.lin = torch.nn.Linear(self.hidden[len(hidden)-1], out_features)
        self.T = T
        self.dropout = nn.Dropout(dropout)
        self.threshold_list = [[] for i in range(len(self.convs))]
        
    def forward(self, data):
        for t in range(self.T):
            if t == 0 :
                out_spike_counter =  self.encode(data)
            else:
                out_spike_counter += self.encode(data)

        neuron.reset_net(self)
        return out_spike_counter / self.T
    
    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        if self.poisson:
            input_encoder = encoding.PoissonEncoder()
            x = input_encoder(x)
        
        h1 = self.convs[0](x, edge_index)
        h2 = self.convs[1](h1, edge_index)
        h = self.origconv(h2, edge_index)
        h = global_add_pool(h, data.batch)
        h = self.lin(h) 
        return h
    


def tag2index(dataset):
    tag_set = []
    for g in dataset:
        all_nodes = torch.cat([g.edge_index[0], g.edge_index[1]])
        node_tags = torch.bincount(all_nodes, minlength=g.num_nodes)/2
        node_tags = list(set(list(np.array(node_tags))))
        tag_set += node_tags
    tagset = list(set(tag_set))
    tag2index_dict = {int(tagset[i]):i for i in range(len(tagset))}
    return tag2index_dict

def apply_deg_features(dataset, dataset_name, deg_features=0):
    if deg_features == 1:
        tag2index_dict = tag2index(dataset)
        processed_dataset = []

        for i in range(len(dataset)):
            g = dataset[i]
            all_nodes = torch.cat([g.edge_index[0], g.edge_index[1]])
            node_tags = list(np.array(torch.bincount(all_nodes, minlength=g.num_nodes) / 2))
            features = torch.zeros(g.num_nodes, len(tag2index_dict))
            features[[range(g.num_nodes)], [tag2index_dict[tag] for tag in node_tags]] = 1
            g.x = features
            processed_dataset.append(g)

        dataset = processed_dataset

    elif dataset_name in ['IMDB-BINARY', 'IMDB-MULTI', 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        processed_dataset = []
        for i in range(len(dataset)):
            g = dataset[i]
            features = torch.ones((g.num_nodes, 1))
            g.x = features
            processed_dataset.append(g)

        dataset = processed_dataset

    return dataset

def dataset_selection(root, dataset_name):
    if dataset_name.lower() == "reddit":
        dataset = Reddit(osp.join(root, 'Reddit'))
        data = dataset[0]
    elif dataset_name.lower() == "flickr":
        dataset = Flickr(osp.join(root, 'Flickr'))
        data = dataset[0]
    elif dataset_name.lower() == "yelp":
        dataset = Yelp(osp.join(root, 'Yelp'))
        data = dataset[0]
    elif dataset_name.lower() == "cora":
        dataset = Planetoid(osp.join(root, 'Cora'), name = "Cora")
        data = dataset[0]
    elif dataset_name.lower() == 'citeseer':
        dataset = Planetoid(osp.join(root, 'Citeseer'), name = "CiteSeer")
        data = dataset[0]
    elif dataset_name.lower() == 'pubmed':
        dataset = Planetoid(osp.join(root, 'PubMed'), name = 'PubMed')
        data = dataset[0]
    elif dataset_name.lower() == 'enzymes':
        dataset = TUDataset(osp.join(root, 'ENZYMES'), name = 'ENZYMES', use_node_attr=True)
        dataset = apply_deg_features(dataset, dataset_name, 0)
        data = dataset[0]
    elif dataset_name.lower() == 'mutag':
        dataset = TUDataset(osp.join(root, 'MUTAG'), name = 'MUTAG', use_node_attr=True)
        data = dataset[0]
    elif dataset_name.lower() == 'proteins':
        dataset = TUDataset(osp.join(root, 'PROTEINS'), name = 'PROTEINS', use_node_attr=True)
        data = dataset[0]
    elif dataset_name == 'COLLAB':
        dataset = TUDataset(osp.join(root, 'COLLAB'), name = 'COLLAB', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'COLLAB', 0)
        data = dataset[0]
    elif dataset_name == 'IMDB-BINARY':
        dataset = TUDataset(osp.join(root, 'IMDB-BINARY'), name = 'IMDB-BINARY', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'IMDB-BINARY', 0)
        data = dataset[0]
    elif dataset_name == 'IMDB-MULTI':
        dataset = TUDataset(osp.join(root, 'IMDB-MULTI'), name = 'IMDB-MULTI', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'IMDB-MULTI', 0)
        data = dataset[0]
    elif dataset_name  == 'REDDIT-BINARY':
        dataset = TUDataset(osp.join(root, 'REDDIT-BINARY'), name = 'REDDIT-BINARY', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'REDDIT-BINARY', 0)
        data = dataset[0]
    elif dataset_name == 'REDDIT-MULTI-5K':
        dataset = TUDataset(osp.join(root, 'REDDIT-MULTI-5K'), name = 'REDDIT-MULTI-5K', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'REDDIT-MULTI-5K', 0)
        data = dataset[0]
    elif dataset_name  == 'PTC_FM':
        dataset = TUDataset(osp.join(root, 'PTC_FM'), name = 'PTC_FM', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'PTC_FM', 0)
        data = dataset[0]
    elif dataset_name == 'NCI1':
        dataset = TUDataset(osp.join(root, 'NCI1'), name = 'NCI1', use_node_attr=True)
        dataset = apply_deg_features(dataset, 'NCI1', 0)
        data = dataset[0]    
    elif dataset_name.lower() == 'cifar10':
        dataset = GNNBenchmarkDataset(osp.join(root, 'CIFAR10'), name = 'CIFAR10')
        data = dataset[0]
    elif dataset_name.lower() == 'mnist':
        dataset = GNNBenchmarkDataset(osp.join(root, 'MNIST'), name = 'MNIST')
        data = dataset[0]
    else:
        data = None
        dataset = None

    assert type(dataset) is not None , f"Please select dataset correctly"
    return data, dataset


def valid_one_batch(data_x, label, edge_index, device):
    with torch.no_grad():
        model.eval()
        logits = []
        labels = []
        
        nodes = data_x
        y = label
        n_imgs = nodes.shape[0]
        out_spikes_counter = torch.zeros((n_imgs, dataset.num_classes)).to(device)
        for t in range(args.T_val):
            out_spikes_counter += model(nodes, edge_index)
        logits.append(out_spikes_counter[data.val_mask].max(1)[1])
        labels.append(y[data.val_mask])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        # logits = logits.argmax(0)
        metric_macro = metrics.f1_score(labels, logits, average='macro')
        metric_micro = metrics.f1_score(labels, logits, average='micro')
        return metric_macro, metric_micro
    
    
def valid_all_batch(data_x, label, edge_index, loader, device):
    with torch.no_grad():
        model.eval()
        logits = []
        labels = []
        cnt = 0
        for batch in tqdm(loader):
            batch.to(device)
            batch_nodes = batch.x
            batch_label = batch.y[:batch.batch_size]
            n_imgs = batch_nodes.shape[0]
            out_spikes_counter = torch.zeros((batch.batch_size, dataset.num_classes)).to(device)
            for t in range(args.T_val):
                out_spikes_counter += model(batch_nodes, batch.edge_index)[:batch.batch_size]
            logits.append(out_spikes_counter.max(1)[1])
            labels.append(batch_label)
        
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        metric_macro = metrics.f1_score(labels, logits, average='macro')
        metric_micro = metrics.f1_score(labels, logits, average='micro')
        return metric_macro, metric_micro

def valid_gc(loader,device):
    with torch.no_grad():
        model.eval()
        logits = []
        labels = []
        correct = 0
        for batch in tqdm(loader):
            valid_data = batch.to(device)
            out_spikes_counter_frequency = model(valid_data)
            logits.append(out_spikes_counter_frequency.max(1)[1])
            labels.append(batch.y)
            
            pred = out_spikes_counter_frequency.argmax(dim = 1)
            correct += int((pred == valid_data.y).sum())
        
        logits = torch.cat(logits, dim=0).cpu()
        print(logits.size())
        labels = torch.cat(labels, dim=0).cpu()
        print(len(logits))
        
        logits_unique_values, logits_counts = torch.unique(logits, return_counts=True)
        labels_unique_values, labels_counts = torch.unique(labels, return_counts=True)
        
        metric_macro = metrics.f1_score(labels, logits, average='macro')
        print(len(loader.dataset))
        metric_micro = correct / len(loader.dataset)
        
        print(f"Logits: {logits_counts}, labels: {labels_counts}" )
        print(f"correct: {correct}")
        return metric_macro, metric_micro
        
    

def test_one_batch(data_x, label, edge_index, device):
    with torch.no_grad():
        model.eval()
        logits = []
        labels = []
        
        nodes = data_x
        y = label
        n_imgs = nodes.shape[0]
        out_spikes_counter = torch.zeros((n_imgs, dataset.num_classes)).to(device)
        for t in range(args.T_val):
            out_spikes_counter += model(nodes, edge_index)
        logits.append(out_spikes_counter[data.test_mask].max(1)[1])
        labels.append(y[data.test_mask])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        metric_macro = metrics.f1_score(labels, logits, average='macro')
        metric_micro = metrics.f1_score(labels, logits, average='micro')
        return metric_macro, metric_micro


if __name__ == '__main__':
    CURRENT_TASK_TYPE = "GC"
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="?", default="MUTAG",
                    help="Datasets (Reddit and Flickr only). (default: MUTAG)")
    parser.add_argument("--opt", type  = str, default = 'adamw',
                        help = 'Choose types of optimizer for trianing process')
    parser.add_argument("--id", type = int, default = 5, help = "Experiment ID setting")
    parser.add_argument('--sizes', type=int, nargs='+', default=[-1,-1,-1],
                        help='For Neighborhood sampling, each . (default: full batch)')
    parser.add_argument('--hids', type=int, nargs='+',
                        default=[128,128,128,128], help='Hidden units for each layer. (default: [128, 50])')
    parser.add_argument("--aggr", nargs="?", default="add",
                        help="Aggregate function ('mean', 'sum'). (default: 'mean')")
    parser.add_argument("--sampler", nargs="?", default="sage",
                        help="Neighborhood Sampler, including uniform sampler from GraphSAGE ('sage') and random walk sampler ('rw'). (default: 'sage')")
    parser.add_argument("--surrogate", nargs="?", default="sigmoid",
                        help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
    parser.add_argument("--neuron", nargs="?", default="LAPLIF",
                        help="Spiking neuron used for training. (IF, LIF, PLIF, AdaptiveLIF, AdaptiveIF, AdaptivePLIF). (default: LIF")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for training. (default: 0.01)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Smooth factor for surrogate learning. (default: 1.0)')
    parser.add_argument('--T', type=int, default=5,
                        help='Number of time steps. (default: 5)')
    parser.add_argument('--T_val', type=int, default=5,
                        help='Number of time steps for validation and test. (default: 5)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability. (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs. (default: 1000)')
    parser.add_argument('--concat', action='store_true',
                        help='Whether to concat node representation and neighborhood representations. (default: False)')
    parser.add_argument('--seed', type=int, default=7777,
                        help='Random seed for model. (default: 777)')
    parser.add_argument('--quantize', action = 'store_true', default = False,
                        help = 'Quantize for calibration')
    parser.add_argument('--bs', type=int, default=1100,
                        help='Number of batch size for mini batching. (default: fullsize)')
    parser.add_argument('--thtr', action = 'store_true', default = False,)
    parser.add_argument('--db_name', type=str, default='Main')
    parser.add_argument('--no_db', action="store_true")
    parser.add_argument('--thr', type =float, default = 2.5)
    parser.add_argument('--loss', type =str, default = 'ce')
    parser.add_argument('--root', type = str, default = "./data")
    parser.add_argument('--model', type = str, default = 'SNNGCN')
    parser.add_argument('--no_poisson', action = "store_true", default = False)
    parser.add_argument('--deg_bins', type = int, default = -1)
    parser.add_argument('--num_layers', type = int, default= 2)
    
    
    args = parser.parse_args()
    args.split_seed = 42
    tab_printer(args)
        
    cur_time = int(time.time())
    dt_object = datetime.fromtimestamp(cur_time)
    readable_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    
    args.time = cur_time
    args.args_to_string = vars(args).copy()
    args.fold = 0
    args.finish = False
    print(args)
    
    if './experiment' not in os.listdir():
        os.makedirs('./experiment', exist_ok=True)
    current_file_name = stdout = f'experiment/{args.dataset}_neuron{args.neuron}_Tval{args.T_val}_ep{args.epochs}_thr{args.thr}_aggr{args.aggr}_{cur_time}_{args.model}.txt'
    log_file = open(current_file_name, 'w')
    original_stdout = sys.stdout

    # Redirect stdout to your custom class
    sys.stdout = StreamRedirector(log_file, original_stdout)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = args.root
    data, dataset = dataset_selection(root, args.dataset)
    
    splits = [(0,0)]
    num_folds = 1
    if args.dataset == 'MNIST' or args.dataset == 'CIFAR10':
        train_data = GNNBenchmarkDataset(osp.join(root, args.dataset), name=args.dataset, split='train')
        val_data = GNNBenchmarkDataset(osp.join(root, args.dataset), name=args.dataset, split='val')
        test_data = GNNBenchmarkDataset(osp.join(root, args.dataset), name=args.dataset, split='test')
        
        dataset = train_data
        num_features = train_data.num_features
        num_classes = train_data.num_classes
        val_size = args.bs
        test_size = args.bs
        
    elif args.dataset.lower() == 'pattern' or args.dataset.lower() == 'cluster':
        train_data = GNNBenchmarkDataset(osp.join(root, args.dataset), name=args.dataset, split='train')
        val_data = GNNBenchmarkDataset(osp.join(root, args.dataset), name=args.dataset, split='val')
        test_data = GNNBenchmarkDataset(osp.join(root, args.dataset), name=args.dataset, split='test')
        
    else:
        # Need to do 10-fold Cross Validation for the experiment
        num_folds = 10
        labels = [data.y.item() for data in dataset]
        stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        splits = stratified_kfold.split(torch.zeros(len(labels)), labels)
        
        if args.dataset.lower() == 'enzymes':
            num_features = dataset.num_features
            num_classes = dataset.num_classes
        elif args.dataset.lower() == 'mutag':
            num_features = dataset.num_features
            num_classes = dataset.num_classes
        elif args.dataset.lower() == 'proteins':
            num_features = dataset.num_features
            num_classes = dataset.num_classes
        elif args.dataset.lower() == 'nci1':
            num_features = dataset.num_features
            num_classes = dataset.num_classes
        elif args.dataset.lower() == 'imdb_binary':
            num_features = dataset.num_features
            num_classes = dataset.num_classes
        elif args.dataset.lower() == 'ptc_fm':
            num_features = dataset.num_features
            num_classes = dataset.num_classes
    
    degree_to_label = {}
    
    if "Cluster" in args.model or args.deg_bins == -1:
        all_degrees = []            
        
        for data in dataset:
            if "Cluster" in args.model or "GCN" in args.model:
                edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
            else:
                edge_index = data.edge_index
            node_degrees = degree(edge_index[0], dtype=torch.long)
            all_degrees.append(node_degrees.numpy()) 
        
        all_degrees = np.concatenate(all_degrees).reshape(-1, 1)  
        if args.deg_bins == -1:
            unique_degrees = np.unique(all_degrees)
            args.deg_bins = len(unique_degrees) 
        kmeans = KMeans(n_clusters=args.deg_bins, random_state=args.seed).fit(all_degrees)
        cluster_labels = kmeans.labels_

        centroids = kmeans.cluster_centers_.squeeze()
        sorted_indices = np.argsort(centroids)
        label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
        new_labels = np.array([label_map[label] for label in cluster_labels])

        degree_label_mapping = {}

        for ddegree, label in zip(all_degrees.squeeze(), new_labels):
            if label not in degree_label_mapping:
                degree_label_mapping[label] = []
            degree_label_mapping[label].append(ddegree)

        for label, degrees in degree_label_mapping.items():
            print(f"Label {label}: Degrees {set(degrees)}")

        print("\nStructured Mapping:")
        for label in sorted(degree_label_mapping):
            print(f"Label {label}: Degrees {set(degree_label_mapping[label])}")
        for label, degrees in degree_label_mapping.items():
            for degree in degrees:
                degree_to_label[degree] = label

    total_score_dict = {}
    fold_folder_name = ''
    for fold, (train_idx, test_idx) in enumerate(splits):
        
        args.fold = fold
        item_dict = vars(args).copy()
        item_dict['args_to_string'] = str(vars(args))
        print("ARGS TO STRING CHECK")
        print(item_dict)
        
        if args.dataset.lower() in ['enzymes', 'mutag', 'proteins', "nci1", "ptc_fm"]:
            train_val_dataset = dataset[train_idx.tolist()]
            test_data = dataset[test_idx.tolist()]
            
            train_data = train_val_dataset
            val_data = test_data
        elif args.dataset.lower() in ['reddit-binary','imdb-binary', 'collab']:
            train_data = Subset(dataset, train_idx)
            test_data = Subset(dataset, test_idx)
            val_data = Subset(dataset, test_idx)
            
            
        train_loader=DataLoader(dataset=train_data,batch_size=args.bs,shuffle=True)
        val_loader=DataLoader(dataset=val_data,batch_size=args.bs,shuffle=False)
        test_loader=DataLoader(dataset=test_data,batch_size = args.bs,shuffle=False)
        
        if args.model == 'SNNGCN':
            if degree_to_label:
                model = SNNGCN(in_features=num_features, out_features = num_classes, device = device,
                                T = args.T, hidden=args.hids, neuron_type = args.neuron, 
                                quantize = args.quantize,  thtr = args.thtr,
                                aggr= args.aggr, thr = args.thr, bn = True, poisson = not(args.no_poisson),
                                bins=  args.deg_bins, degree_to_label = degree_to_label).to(device)
            else:
                model = SNNGCN(in_features=num_features, out_features = num_classes, device = device,
                                T = args.T, hidden=args.hids, neuron_type = args.neuron, 
                                quantize = args.quantize,  thtr = args.thtr,
                                aggr= args.aggr, thr = args.thr, bn = True, poisson = not(args.no_poisson),
                                bins=  args.deg_bins, degree_to_label = None).to(device)
            
        elif args.model == "SNNGIN":
            if degree_to_label:
                model = SNNGIN(in_features=num_features, out_features = num_classes, device = device,
                                T = args.T, hidden=args.hids, neuron_type = args.neuron, 
                                quantize = args.quantize,  thtr = args.thtr,
                                aggr= args.aggr, thr = args.thr, bn = True, poisson = not(args.no_poisson),
                                bins=  args.deg_bins, degree_to_label = degree_to_label).to(device)
            else:
                model = SNNGIN(in_features=num_features, out_features = num_classes, device = device,
                                T = args.T, hidden=args.hids, neuron_type = args.neuron, 
                                quantize = args.quantize,  thtr = args.thtr,
                                aggr= args.aggr, thr = args.thr, bn = True, poisson = not(args.no_poisson),
                                bins=  args.deg_bins, degree_to_label = None).to(device)
        elif args.model == "SNNGAT":
            if degree_to_label:
                model = SNNGAT(in_features=num_features, out_features = num_classes, device = device,
                                T = args.T, hidden=[64,64,64], neuron_type = args.neuron, 
                                quantize = args.quantize,  thtr = args.thtr,
                                aggr= args.aggr, thr = args.thr, bn = True, poisson = not(args.no_poisson),
                                bins=  args.deg_bins, degree_to_label = degree_to_label).to(device)
            else:
                model = SNNGAT(in_features=num_features, out_features = num_classes, device = device,
                                T = args.T, hidden=[64,64,64], neuron_type = args.neuron, 
                                quantize = args.quantize,  thtr = args.thtr,
                                aggr= args.aggr, thr = args.thr, bn = True, poisson = not(args.no_poisson),
                                bins=  args.deg_bins, degree_to_label = None).to(device)
        
        print(model)
        
        optimizer = None
        if args.opt.lower() == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        elif args.opt.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            assert type(optimizer) is not None, "Optimizer not declared"
        
        if (args.loss).lower() == 'mse':
            loss_fn = nn.MSELoss()
        elif (args.loss).lower() == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        best_ep = -1
        best_val_metric = test_metric = 0
        best_train_metric = 0
        only_test_metric = (0,0)
        start = time.time()
        all_test_metric_lst = []
        all_val_metric_lst = []
        all_train_metric_lst = []
        
        cur_exp_folder = f'./pics_and_spike/{args.neuron}_{args.dataset}_{args.epochs}_{args.thr}_{args.T}_bins{args.deg_bins}_{args.model}'
        if fold == 0:
            cur_exp_folder = rename_folder_with_suffix(cur_exp_folder)
        else:
            cur_exp_folder = fold_folder_name
        
        os.makedirs(cur_exp_folder, exist_ok=True)
        
        for epoch in range(1, args.epochs + 1):
            train_logits = []
            train_labels = []
            model.train()
            total_loss = 0
            train_acc = 0
            out_spikes_counter_frequency_lst = []
            
            for data in tqdm(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                out_spikes_counter_frequency = model(data)
            
                loss = loss_fn(out_spikes_counter_frequency, data.y)
                if args.neuron == 'BLIF':
                    loss.backward(retain_graph = True)
                else:
                    loss.backward()
                optimizer.step()
                train_logits.append(out_spikes_counter_frequency.max(1)[1])
                train_labels.append(data.y)
                
                out_spikes_counter_frequency_lst += out_spikes_counter_frequency.cpu()
                total_loss += loss.item()
            
            train_logits = torch.cat(train_logits, dim = 0).cpu()
            train_labels = torch.cat(train_labels, dim = 0).cpu()
            labels_unique_values, labels_counts = torch.unique(train_logits, return_counts=True)

            train_metric = metrics.f1_score(train_labels, train_logits, average='macro'), metrics.f1_score(train_labels, train_logits, average='micro')
            val_metric = valid_gc(val_loader, device)
            test_metric = valid_gc(test_loader, device)
            
            all_train_metric_lst += ([train_metric[1]] * args.T * len(train_loader)) 
            all_val_metric_lst += ([val_metric[1]] * args.T * len(train_loader))
            all_test_metric_lst += ([test_metric[1]] * args.T * len(train_loader))
            
            if train_metric[1] > best_train_metric:
                best_train_metric = train_metric[1]
            if test_metric[1] > only_test_metric[1]:
                only_test_metric = test_metric
            if val_metric[1] > best_val_metric:
                best_val_metric = val_metric[1]
                best_test_metric = test_metric
                best_ep = epoch
                
            end = time.time()
               
            print(
                f'Fold {fold} Epoch: {epoch:03d}, Train: {train_metric[1]:.4f} Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Loss: {total_loss}\
                \nBest: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s,\
                Best Test : Macro-{only_test_metric[0]:.4f}, Micro-{only_test_metric[1]:.4f}')
        plots_lst = []
        timestamp = args.T
        total_score_dict[fold] = (train_metric, best_val_metric, best_test_metric)
    
    train_sum = 0
    val_sum = 0
    test_sum = 0
    
    for key, (train_acc, val_acc, test_acc) in total_score_dict.items():
        print(f"Fold {key} : Train acc {train_acc[1]} Test acc {test_acc}")
        train_sum += train_acc[1]
        val_sum += val_acc
        test_sum += test_acc[1]
    
    train_sum /= num_folds
    val_sum /= num_folds
    test_sum /= num_folds
    print(f"CV {num_folds} : Train acc {train_sum} Test acc {test_sum}")
    
    print("Total scores for 10-CV Cross Validation finished")
    
    args.fold = 'all'
    item_dict = vars(args).copy()
    item_dict['args_to_string'] = str(vars(args))
