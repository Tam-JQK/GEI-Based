import torch
import torch.nn as nn
import numpy as np
from .abstract_model import AbstractBaseModel
#from .utility import tools

import scipy.sparse as sp
from build_graph import build_lightgcn_graph

r"""
LightGCN
################################################
Reference:
    He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" in SIGIR'20
"""

class LightGCN(AbstractBaseModel):

    def __init__(self, config, data=None, graph=None):
        super().__init__(config)
        
        self.reg_lambda = float(config['reg_lambda'])
        self.GCN_layer = int(config['GCN_layer'])
        self.data = data
        
        self.type = ["pair"]
        self.IR_type = ["retrieval", "ranking"]

        if graph is None:
            graph = build_lightgcn_graph(config['dataset'], root_path=config.get('project_root', '.'))

        # # Move the graph to the configured device.
        # graph = graph.to(device)

        # # Initialize the graph.
        # sp_graph = tools.convert_sp_mat_to_sp_tensor(self.sparse_adjacency_matrix(self.data)).coalesce()
        # sp_graph = tools.convert_sp_mat_to_sp_tensor(graph).coalesce()
        sp_graph = graph.coalesce() if graph.layout == torch.sparse_coo else graph
        nodes = config['user_num'] + config['item_num']
        if sp_graph.shape != (nodes, nodes):
            raise ValueError('Training graph dimensions do not match the item/user catalog')
        # register buffer so model.to(device) will move it
        self.register_buffer('Graph', sp_graph)

        # self.Graph = tools.convert_sp_mat_to_sp_tensor(self.sparse_adjacency_matrix(data))
        # self.Graph = self.Graph.coalesce().to(next(self.parameters()).device)
        
        # # Initialize embedding weights from a normal distribution.
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        # nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        # nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        


    def invalidate_embedding_cache(self):
        # Invalidate the cached embeddings after the optimizer step.
        self._embedding_cache_valid = False

    def train(self, mode=True):
        self.invalidate_embedding_cache()
        return super().train(mode)

    def load_state_dict(self, *args, **kwargs):
        self.invalidate_embedding_cache()
        return super().load_state_dict(*args, **kwargs)

    def _apply(self, fn, recurse=True):
        self.invalidate_embedding_cache()
        return super()._apply(fn, recurse=recurse)
        

    def aggregate(self):
        # Compute and cache graph embeddings.
        cacheable = not self.training and not torch.is_grad_enabled()
        if cacheable and getattr(self, '_embedding_cache_valid', False):
            return self.users_emb, self.items_emb
        # The registered graph buffer follows the model device.
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings = [all_embedding]
        for layer in range(self.GCN_layer):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            # Apply message dropout during training.
            # if self.training:
            #     all_embedding = nn.functional.dropout(all_embedding, p=0.1)
            embeddings.append(all_embedding)
        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        self.users_emb, self.items_emb = torch.split(final_embeddings, [self.config['user_num'], self.config['item_num']])
        # Never reuse an autograd graph across optimizer updates, or detached
        # evaluation embeddings during training/gradient-based scoring.
        self._embedding_cache_valid = cacheable
        return self.users_emb, self.items_emb

    def forward(self, user_dict, item_ids):
        """Predict scores using the BPR forward interface."""
        # Reuse the graph embeddings within the batch.
        self.aggregate()
        user_ids = user_dict['user_ids']
        user_embeds = self.users_emb[user_ids.long()]
        item_embeds = self.items_emb[item_ids.long()]
        dot_product = (user_embeds * item_embeds).sum(1)
        return self.sigmoid(dot_product)
    
    def compute_loss(self, interaction):
        """Compute the pairwise BPR loss and embedding regularization."""
        self.aggregate()
        user_ids = interaction['user_ids'].long()
        pos_item = interaction['item_ids'].long()
        neg_item = interaction['neg_item_ids'].long()
        u_emb = self.users_emb[user_ids]
        pos_emb = self.items_emb[pos_item]
        neg_emb = self.items_emb[neg_item]
        pos_score = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_score = torch.mul(u_emb, neg_emb).sum(dim=1)
        bpr_loss = nn.functional.softplus(neg_score - pos_score)

        user_ids = interaction['user_ids']
        ego_user_emb = self.user_embedding(user_ids)
        ego_pos_emb = self.item_embedding(pos_item)
        ego_neg_emb = self.item_embedding(neg_item)
        reg_loss = self.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        return bpr_loss + reg_loss

    def get_user_embedding(self, user_dict):
        self.aggregate()
        user_ids = user_dict['user_ids']
        return self.users_emb[user_ids.long()]

    def full_predict(self, user_dict, items):
        self.aggregate()
        user_embeds = self.users_emb[user_dict['user_ids'].long()]
        item_embeds = self.items_emb[items.long()]
        scores = (user_embeds * item_embeds).sum(-1)
        return self.sigmoid(scores)
    
    
    def full_ranking(self, user_dict, items, k):
        self.aggregate()
        user_embeds = self.users_emb[user_dict['user_ids'].long()]
        item_embeds = self.items_emb[items.long()]
        scores = torch.matmul(user_embeds, item_embeds.t())
        scores, indices = torch.topk(scores, k=k, dim=-1)
        return scores, indices

    def full_scores(self, user_dict, items):
        self.aggregate()
        user_embeds = self.users_emb[user_dict['user_ids'].long()]
        item_embeds = self.items_emb[items.long()]
        scores = torch.matmul(user_embeds, item_embeds.t())
        return scores


    def sparse_adjacency_matrix(self, data):
        """
        Build a normalized bipartite adjacency matrix from pairwise data.
        data: PairWiseDataset with user_ids and item_ids as tensors or arrays.
        user_num, item_num: Optional counts. Defaults are read from data.config.
        """

        user_num = data.config['user_num']
        item_num = data.config['item_num']
        user_ids = np.array(data.user_ids)
        item_ids = np.array(data.item_ids)
        R = sp.csr_matrix((np.ones(len(user_ids)), (user_ids, item_ids)), shape=(user_num, item_num))

        # R = self.user_item_net(data)
        R = (R > 0).astype(float)
        num_nodes = user_num + item_num
        adjacency_matrix = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)
        adjacency_matrix = adjacency_matrix.tolil()
        adjacency_matrix[:user_num, user_num:] = R
        adjacency_matrix[user_num:, :user_num] = R.T
        adjacency_matrix = adjacency_matrix.todok()

        # Apply symmetric degree normalization.
        row_sum = np.array(adjacency_matrix.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        degree_matrix = sp.diags(d_inv)
        norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
        return norm_adjacency

    def user_item_net(self, data):
        """
        Build a sparse user-item interaction matrix from pairwise data.
        data: PairWiseDataset with user_ids and item_ids as tensors or arrays.
        user_num, item_num: Optional counts. Defaults are read from data.config.
        Returns the user-item interaction matrix as a SciPy sparse matrix.
        """

        user_num = data.config['user_num']
        item_num = data.config['item_num']
        user_ids = np.array(data.user_ids)
        item_ids = np.array(data.item_ids)
        interaction_matrix = sp.csr_matrix((np.ones(len(user_ids)), (user_ids, item_ids)), shape=(user_num, item_num))
        return interaction_matrix
