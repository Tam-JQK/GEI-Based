import os
import yaml
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch

def build_lightgcn_graph(dataset_name, root_path="recommendation"):
    """
    Build the normalized user-item adjacency matrix for LightGCN.
    Read the dataset configuration and training interactions from disk.

    Args:
        dataset_name (str): Name of the processed dataset.
        root_path (str): Directory containing processed_dataset/.
        
    Returns:
        torch.Tensor: Normalized adjacency matrix in sparse COO format.
    """
    
    # Resolve the input paths.
    processed_dir = os.path.join(root_path, "processed_dataset", dataset_name)
    config_path = os.path.join(processed_dir, "process_config.yaml")
    train_file = os.path.join(processed_dir, f"{dataset_name}.train")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Processed dataset configuration not found: {config_path}")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training interactions not found: {train_file}")

    print(f"Loading configuration: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    user_num = config['user_num']
    item_num = config['item_num']
    
    # Read the column names from the dataset configuration.
    uid_field = config.get('user_id', 'user_id')
    iid_field = config.get('item_id', 'item_id')
    label_field = config.get('label_id', 'label')

    print(f"Dataset: {dataset_name}")
    print(f"Users: {user_num}, items: {item_num}")

    df_train = pd.read_csv(train_file, sep='\t')
    
    # Fall back to a matching column when a configured name is absent.
    if uid_field not in df_train.columns:
        candidates = [c for c in df_train.columns if 'user' in c.lower()]
        if candidates: uid_field = candidates[0]
    
    if iid_field not in df_train.columns:
        candidates = [c for c in df_train.columns if 'item' in c.lower()]
        if candidates: iid_field = candidates[0]

    # Use positive feedback to construct graph edges.
    actual_label_col = None
    for col in df_train.columns:
        if 'label' in col.lower():
            actual_label_col = col
            break
    
    if actual_label_col:
        print(f"Filtering positive interactions (label=1) using '{actual_label_col}'")
        df_pos = df_train[df_train[actual_label_col] == 1]
    else:
        print("Label column not found. Treating all training interactions as positive.")
        df_pos = df_train

    if len(df_pos) == 0:
        raise ValueError("No positive training interactions remain for graph construction.")

    # Build the sparse user-item interaction matrix R.
    user_ids = df_pos[uid_field].values.astype(np.int32)
    item_ids = df_pos[iid_field].values.astype(np.int32)
    
    if user_ids.max() >= user_num or item_ids.max() >= item_num:
        print(f"Observed maximum IDs ({user_ids.max()}, {item_ids.max()}) exceed the index ranges for counts ({user_num}, {item_num}).")
        # Expand the matrix dimensions to include the observed IDs.
        user_num = max(user_num, user_ids.max() + 1)
        item_num = max(item_num, item_ids.max() + 1)

    # R[u, i] = 1
    R = sp.csr_matrix((np.ones(len(user_ids)), (user_ids, item_ids)), 
                      shape=(user_num, item_num))

    # Build the adjacency matrix A for the bipartite graph.
    # A = | 0   R |
    #     | R^T 0 |
    # Shape: (user_num + item_num, user_num + item_num).
    
    adj_mat = sp.dok_matrix((user_num + item_num, user_num + item_num), dtype=np.float32)
    adj_mat = adj_mat.tolil()  # LIL format supports block assignment.
    
    R_lil = R.tolil()
    adj_mat[:user_num, user_num:] = R_lil
    adj_mat[user_num:, :user_num] = R_lil.T
    
    adj_mat = adj_mat.todok()
    

    # Apply symmetric degree normalization.
    # D^(-1/2) A D^(-1/2)
    rowsum = np.array(adj_mat.sum(axis=1))
    
    # Set the inverse degree to zero for isolated nodes.
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    
    d_mat = sp.diags(d_inv)
    
    # norm_adj = D^{-1/2} * A * D^{-1/2}
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    norm_adj = norm_adj.tocoo()

    # Convert the COO matrix to a PyTorch sparse tensor.
    row = torch.from_numpy(norm_adj.row).long()
    col = torch.from_numpy(norm_adj.col).long()
    index = torch.stack([row, col])
    data = torch.from_numpy(norm_adj.data).float()
    
    sparse_graph = torch.sparse.FloatTensor(index, data, torch.Size(norm_adj.shape))
    
    return sparse_graph
