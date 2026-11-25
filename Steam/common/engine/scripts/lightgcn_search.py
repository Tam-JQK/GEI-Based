"""Training and evaluation kernels used by the parameter search."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from analysis.artifacts import file_hash, write_json


def normalized_graph(users, items, user_num, item_num):
    users, items = np.asarray(users, np.int64), np.asarray(items, np.int64)
    assert users.shape == items.shape and users.ndim == 1 and len(users)
    assert ((users >= 0) & (users < user_num)).all()
    assert ((items >= 0) & (items < item_num)).all()
    encoded = np.unique(np.asarray(users, np.int64) * item_num + np.asarray(items, np.int64))
    users, items = encoded // item_num, encoded % item_num
    assert len(users) and users.min() >= 0 and users.max() < user_num
    interaction = sp.csr_matrix((np.ones(len(users), np.float32), (users, items)),
                               shape=(user_num, item_num))
    adjacency = sp.bmat([[None, interaction], [interaction.T, None]], format='csr', dtype=np.float32)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    np.power(degree, -.5, out=inverse, where=degree > 0)
    graph = (sp.diags(inverse) @ adjacency @ sp.diags(inverse)).tocsr()
    graph.sort_indices()
    return graph, users, items, degree


def torch_graph(graph, device):
    return torch.sparse_csr_tensor(torch.from_numpy(graph.indptr.astype(np.int64)),
        torch.from_numpy(graph.indices.astype(np.int64)), torch.from_numpy(graph.data),
        size=graph.shape, device=device)


def load_graph(root, run, name, config, device):
    import fcntl
    cache = run / 'input_cache' / name
    cache.mkdir(parents=True, exist_ok=True)
    source = Path(config.get('processed_data_dir') or root / 'processed_dataset' / name) / config['train_file']
    sha = file_hash(source)
    with (cache / 'cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (cache / 'metadata.json').exists():
            uu, ii, raw_positive = [], [], 0
            for frame in pd.read_csv(source, sep='\t',
                    usecols=[config['user_id'], config['item_id'], config['label_id']], chunksize=500000):
                assert frame[config['label_id']].isin([0, 1]).all()
                positive = frame.loc[frame[config['label_id']] == 1]
                uu.append(positive[config['user_id']].to_numpy(np.int64))
                ii.append(positive[config['item_id']].to_numpy(np.int64))
                raw_positive += len(positive)
            graph, users, items, degrees = normalized_graph(np.concatenate(uu), np.concatenate(ii),
                                            config['user_num'], config['item_num'])
            assert np.bincount(users, minlength=config['user_num']).min() >= 2
            assert np.bincount(users, minlength=config['user_num']).max() < config['item_num']
            sp.save_npz(cache / 'graph.npz', graph)
            np.save(cache / 'users.npy', users); np.save(cache / 'items.npy', items)
            write_json(cache / 'metadata.json', dict(source_sha256=sha, training_view='full .train label=1',
                raw_positive_rows=raw_positive, unique_training_edges=len(users), graph_nnz=graph.nnz,
                graph_shape=list(graph.shape), normalization='D^-1/2 A D^-1/2',
                zero_degree_users=int((degrees[:config['user_num']] == 0).sum()),
                zero_degree_items=int((degrees[config['user_num']:] == 0).sum()),
                graph_sha256=file_hash(cache / 'graph.npz')))
        metadata = json.loads((cache / 'metadata.json').read_text())
        assert metadata['source_sha256'] == sha
        assert file_hash(cache / 'graph.npz') == metadata['graph_sha256']
        graph = sp.load_npz(cache / 'graph.npz')
        users = torch.from_numpy(np.load(cache / 'users.npy')).to(device)
        items = torch.from_numpy(np.load(cache / 'items.npy')).to(device)
    (run / name).mkdir(parents=True, exist_ok=True)
    write_json(run / name / 'graph_metadata.json', metadata)
    return torch_graph(graph, device), users, items


def sample_negatives(users, sorted_positive_pairs, item_num):
    """Uniform rejection sampling over real items, excluding training positives."""
    negatives = torch.randint(item_num, users.shape, device=users.device)
    while True:
        codes = users * item_num + negatives
        index = torch.searchsorted(sorted_positive_pairs, codes)
        known = (index < len(sorted_positive_pairs)) & (
            sorted_positive_pairs[index.clamp_max(len(sorted_positive_pairs) - 1)] == codes)
        if not known.any():
            return negatives
        negatives[known] = torch.randint(item_num, (int(known.sum()),), device=users.device)
