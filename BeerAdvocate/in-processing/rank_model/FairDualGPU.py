"""FairDual with CUDA weight updates and a CPU solver for dual projection."""
import cvxpy as cp
import numpy as np
import torch
from rank_model.FairDualPaper import FairDualPaper

class FairDualGPU(FairDualPaper):
    def __init__(self, config, group_weight, device='cuda'):
        super().__init__(config, group_weight)
        if not np.all(self.M.sum(axis=1)==1):
            raise ValueError('CUDA FairDual requires one provider per item')
        self.device=device
        self.item_group=torch.as_tensor(self.M.argmax(axis=1),device=device,dtype=torch.long)
        self.rho=torch.as_tensor(self.group_weight,device=device,dtype=torch.float64)
        self.fast_count=0;self.fallback_count=0
        n=int(config['group_num']);self._target=cp.Parameter(n);self._rho=cp.Parameter(n,nonneg=True);self._bound=cp.Parameter(nonneg=True)
        self._var=cp.Variable(n)
        self._problem=cp.Problem(cp.Minimize(cp.sum_squares(cp.multiply(self._rho,self._var)-self._target)),[cp.cumsum(cp.multiply(self._rho,self._var))>=-self._bound])

    def reset_parameters(self, train_len):
        self.train_len=int(train_len);self.update_len=int(self.config['update_epoch'])*self.train_len
        self.C_t=float(self.config['s_k'])*self.update_len*self.rho.clone()
        self.mu=torch.ones(int(self.config['group_num']),dtype=torch.float64,device=self.device)
        self.gradient_cusum=torch.zeros_like(self.mu);self._initialized=True;self.exposure_count=0

    def cached_projection(self, y, rho, lambd):
        self._target.value=y*rho;self._rho.value=rho;self._bound.value=lambd
        for solver in (cp.CLARABEL,cp.OSQP,cp.SCS):
            try:self._problem.solve(solver=solver,warm_start=False)
            except Exception:continue
            if self._var.value is not None and np.all(np.isfinite(self._var.value)):
                return np.asarray(self._var.value,dtype=np.float64)
        raise RuntimeError('Cached projection failed')

    @torch.no_grad()
    def reweight(self, input_dict):
        items=input_dict['sample_items'].to(device=self.device,dtype=torch.long)
        items=items.reshape(items.shape[0],-1); groups=self.item_group[items];batch_size=items.shape[0]
        raw=1.0-self.mu[groups].sum(dim=1)/self.mu.numel()
        weight=raw-raw.min()+.01;weight=weight/weight.mean()
        D_t=torch.bincount(groups.reshape(-1),minlength=self.mu.numel()).to(torch.float64)
        self.C_t=self.C_t-D_t
        gradient=-D_t/(batch_size*float(self.config['s_k']))+self.C_t/(self.train_len*float(self.config['s_k']))
        alpha=float(self.config['alpha']);gradient=alpha*gradient+(1-alpha)*self.gradient_cusum;self.gradient_cusum=gradient
        tilde=self.mu-float(self.config['eta'])*gradient/(self.rho*self.rho)
        order=torch.argsort(tilde*self.rho);sorted_dual=tilde[order];rho_sorted=self.rho[order]
        if torch.min(torch.cumsum(sorted_dual*rho_sorted,0)).item() >= -float(self.config['lambd']):
            self.mu=tilde;self.fast_count+=1
        else:
            answer=self.cached_projection(sorted_dual.cpu().numpy(),rho_sorted.cpu().numpy(),float(self.config['lambd']))
            if answer is not None:
                unsorted=torch.empty_like(tilde);unsorted[order]=torch.as_tensor(answer,device=self.device);self.mu=unsorted
            self.fallback_count+=1
        return weight
