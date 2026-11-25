"""Stateful PMMF using CUDA scoring/sort and the existing CPU dual projection.

The order of users and per-user state transitions follow rerank_model.PMMF.
The CUDA implementation requires one provider per item.
"""
import math
import numpy as np
import torch
from scripts.fair_search import FastFairDual
from utils import Build_Adjecent_Matrix

class PMMFGPU:
 def __init__(self,config):
  self.config=config;self.M,_=Build_Adjecent_Matrix(config)
  if not np.all(self.M.sum(axis=1)==1):raise ValueError('GPU PMMF requires one provider per item')
  self.groups=np.argmax(self.M,axis=1).astype(np.int64);self.weights=np.ones(config['group_num'],dtype=np.float64)
  self.candidate_mask=None;self.projection=FastFairDual(config,self.weights)
 def rerank(self,ranking_score,k):
  if not torch.cuda.is_available():raise RuntimeError('GPU PMMF requires CUDA')
  n=len(ranking_score);device='cuda';groups=torch.as_tensor(self.groups,device=device);out=np.empty((n,k),dtype=np.int32)
  mu=np.zeros(self.config['group_num'],dtype=np.float64);budget=n*k*self.weights.copy();gradient_prev=np.zeros_like(mu)
  eta=float(self.config['learning_rate'])/math.sqrt(self.config['item_num']);gamma=float(self.config['gamma']);lambd=float(self.config['lambda'])
  # Block transfer limits VRAM without altering per-user ordering/state.
  for start in range(0,n,128):
   end=min(n,start+128);score=torch.as_tensor(np.array(ranking_score[start:end],dtype=np.float32,copy=True),device=device)
   mask=torch.as_tensor(np.array(self.candidate_mask[start:end],dtype=bool,copy=True),device=device)
   for j in range(end-start):
    mu_t=torch.as_tensor(mu,device=device);allowed=torch.as_tensor(budget>0,device=device)
    adjusted=score[j].to(torch.float64)-mu_t[groups]+(1.0-allowed[groups].to(torch.float64))*(-10000.0)
    adjusted.masked_fill_(~mask[j],-torch.inf)
    chosen=torch.argsort(adjusted,descending=False,stable=False)[-k:].flip(0).cpu().numpy()
    # Match the reference's final NumPy quicksort on the selected original scores.
    chosen=chosen[np.argsort(np.asarray(ranking_score[start+j,chosen]))[::-1]];out[start+j]=chosen
    counts=np.bincount(self.groups[chosen],minlength=len(mu)).astype(np.float64)
    budget-=counts;gradient=gamma*(-counts/k+self.weights)+(1-gamma)*gradient_prev;gradient_prev=gradient
    tilde=mu-eta*gradient/(self.weights*self.weights);order=np.argsort(tilde*self.weights)
    ordered=self.projection.map_layer(tilde[order],self.weights[order],lambd);mu=ordered[np.argsort(order)]
  return out
