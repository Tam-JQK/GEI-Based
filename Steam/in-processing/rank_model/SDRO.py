import numpy as np
from .Abstract_Ranker import Abstract_Reweigher

r"""
S-DRO
################################################

@inproceedings{10.1145/3485447.3512255,
author = {Wen, Hongyi and Yi, Xinyang and Yao, Tiansheng and Tang, Jiaxi and Hong, Lichan and Chi, Ed H.},
title = {Distributionally-robust Recommendations for Improving Worst-case User Experience},
year = {2022},
isbn = {9781450390965},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3485447.3512255},
doi = {10.1145/3485447.3512255},
booktitle = {Proceedings of the ACM Web Conference 2022},
pages = {3606–3610},
numpages = {5},
keywords = {Distributional robustness, Recommendation., Robust learning},
location = {Virtual Event, Lyon, France},
series = {WWW '22}
}


"""

class SDRO(Abstract_Reweigher):
    def __init__(self, config, group_weight):
        super().__init__(config)
        self.group_weight = group_weight

    def reset_parameters(self):
        self.mu = np.ones(self.config['group_num'])

    def reweight(self, input_dict):
        items = input_dict['items']
        losses = input_dict['loss'] #[B]

        adj_matrix = self.M[items]

        #C_t = np.mean(self.M[items], axis=1, keepdims=False) #[B,G]
        group_num = np.sum(adj_matrix, axis=0)
        gradient = np.sum(losses[:, np.newaxis] * adj_matrix, axis=0, keepdims=False) / (group_num + 1e-1)
        ###since the loss is the negative of the scores
        self.mu = self.mu + self.config['eta'] * gradient
        #self.mu = np.clip(self.mu, 1e-3, 1)
        self.mu = np.exp(self.mu)
        self.mu = self.mu / np.sum(self.mu)

        batch_weight = np.matmul(adj_matrix, self.mu * self.group_weight)
        # batch_weight = batch_weight / np.sum(batch_weight)
        batch_weight = batch_weight / np.mean(batch_weight)  # mean = 1


        return batch_weight

