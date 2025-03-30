import numpy as np
import scipy.sparse as sp
import torch
from texttable import Texttable
import torch_cluster
import random
import os


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," "), args[k]] for k in keys])
    print(t.draw())



def eliminate_selfloops(adj_matrix):
    """eliminate selfloops for adjacency matrix.

    >>>eliminate_selfloops(adj) # return an adjacency matrix without selfloops

    Parameters
    ----------
    adj_matrix: Scipy matrix or Numpy array

    Returns
    -------
    Single Scipy sparse matrix or Numpy matrix.

    """
    if sp.issparse(adj_matrix):
        adj_matrix = adj_matrix - sp.diags(adj_matrix.diagonal(), format='csr')
        adj_matrix.eliminate_zeros()
    else:
        adj_matrix = adj_matrix - np.diag(adj_matrix)
    return adj_matrix


def add_selfloops(adj_matrix: sp.csr_matrix):
    """add selfloops for adjacency matrix.

    >>>add_selfloops(adj) # return an adjacency matrix with selfloops

    Parameters
    ----------
    adj_matrix: Scipy matrix or Numpy array

    Returns
    -------
    Single sparse matrix or Numpy matrix.

    """
    adj_matrix = eliminate_selfloops(adj_matrix)

    return adj_matrix + sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='csr')


def rename_folder_with_suffix(folder_path:str) -> str :
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return folder_path

    i = 0
    new_folder_path = f"{folder_path}_rep{i}"
    while os.path.exists(new_folder_path):
        i += 1
        new_folder_path = f"{folder_path}_rep{i}"

    print(f"Current Folder renamed from '{folder_path}' to '{new_folder_path}'")
    return new_folder_path

if __name__ == "__main__":
    rename_folder_with_suffix('./pics/ALIF_citeseer_200')