"""CUDA scoring and sorting for stateful ElasticRank."""
import numpy as np
import torch
from utils import Build_Adjecent_Matrix
from rerank_model.ElasticRank import distance_concave_func

class ElasticRankGPU:
 def __init__(self,config):
  self.config=config;self.M,_=Build_Adjecent_Matrix(config)
  if not np.all(self.M.sum(axis=1)==1):raise ValueError('GPU ElasticRank requires one provider per item')
  self.groups=np.argmax(self.M,axis=1).astype(np.int64);self.candidate_mask=None
 def rerank(self,ranking_score,k):
  if not torch.cuda.is_available():raise RuntimeError('GPU ElasticRank requires CUDA')
  n=len(ranking_score);device='cuda';groups=torch.as_tensor(self.groups,device=device);out=np.empty((n,k),dtype=np.int32)
  V=np.ones(self.config['group_num']);t=self.config['t'];assert t>=0
  for start in range(0,n,128):
   end=min(n,start+128);score=torch.as_tensor(np.array(ranking_score[start:end],dtype=np.float32,copy=True),device=device)
   mask=torch.as_tensor(np.array(self.candidate_mask[start:end],dtype=bool,copy=True),device=device)
   for j in range(end-start):
    anchor=np.argsort(V)[int(self.config['group_num']*self.config['anchor_rate'])];norm=V/np.sum(V)
    curve=distance_concave_func(norm[anchor],norm,t)
    rel=score[j].to(torch.float64)+float(self.config['lambda'])*torch.as_tensor(curve,device=device)[groups]
    rel.masked_fill_(~mask[j],-torch.inf)
    chosen=torch.argsort(rel,descending=False,stable=False)[-k:].flip(0).cpu().numpy();out[start+j]=chosen
    scores=np.asarray(ranking_score[start+j,chosen]);V=V+np.matmul(scores,self.M[chosen,:])
  return out
