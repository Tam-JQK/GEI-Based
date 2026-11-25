import numpy as np
from .Abstract_Ranker import Abstract_Regularizer
import torch.nn.functional as F
import torch
r"""
FOCF
################################################


@article{yao2017beyond,
  title={Beyond parity: Fairness objectives for collaborative filtering},
  author={Yao, Sirui and Huang, Bert},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

##Note that in this code, we extend the two-group cases into multi-group cases

"""

# class FOCF(Abstract_Regularizer):
#     def __init__(self, config, group_weight):
#         super().__init__(config)


#     def fairness_loss(self, input_dict):
#         losses = input_dict['scores'] #[B]
#         mean_loss = torch.mean(losses)
#         gap = torch.abs(losses-mean_loss)
#         return F.smooth_l1_loss(gap, torch.zeros_like(gap))

class FOCF(Abstract_Regularizer):
    def __init__(self, config, group_weight):
        super().__init__(config)
        self.M_tensor = torch.tensor(self.M, dtype=torch.float32).to(config['device'])

    def fairness_loss(self, input_dict):
        scores = input_dict['scores']              # [B]
        items = input_dict['items']                 # [B]

        adj_matrix = self.M_tensor[items]           # [B, G]
        group_count = torch.sum(adj_matrix, dim=0) + 1e-8   # [G]
        group_mean = torch.matmul(scores, adj_matrix) / group_count  # [G]

        global_mean = torch.mean(scores)
        gap = group_mean - global_mean              # [G]
        return F.smooth_l1_loss(gap, torch.zeros_like(gap))

