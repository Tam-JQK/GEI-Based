"""Paper-faithful FairDual re-implementation (ICLR 2025, arXiv:2502.09319).

The legacy ``FairDual`` in this repository is a local edit of the official
sample-based implementation.  Two of those edits change the algorithm:

* the loss weight was inverted (``1 / batch_weight``) instead of the paper's
  ``s^j = 1 - A^j mu``;
* ``alpha`` (the momentum coefficient) was swept above 1, which is outside the
  paper's ``alpha in [0, 1]``; the V4 sweep used ``alpha`` as a fairness knob.

This module keeps the official dual projection and the official count-based
gradient tracker, but restores the paper's per-sample weight and validates the
hyperparameter domains.  The legacy class is left untouched.

Config keys
-----------
s_k                 top-k exposure per sampled user (paper: 20)
update_epoch        window length in epochs (paper: 30)
alpha               momentum in [0, 1] (paper default: 0.8)
eta                 dual step size (paper / official: 0.001)
lambd               fairness constraint slack (paper default: 1)
exposure_sample_num number of random items ranked per batch (official: 700)
"""
from __future__ import annotations

import sys

import cvxpy as cp
import numpy as np

from .Abstract_Ranker import Abstract_Reweigher


class FairDualPaper(Abstract_Reweigher):
    def __init__(self, config, group_weight):
        super().__init__(config)
        self.group_weight = np.asarray(group_weight, dtype=np.float64)
        alpha = float(config['alpha'])
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'FairDualPaper: alpha must be in [0, 1], got {alpha}')
        if float(config['eta']) <= 0.0:
            raise ValueError('FairDualPaper: eta must be positive')
        if float(config['lambd']) <= 0.0:
            raise ValueError('FairDualPaper: lambd must be positive')
        self._initialized = False
        self.solver_failures = 0

    def reset_parameters(self, train_len):
        self.train_len = int(train_len)
        self.update_len = int(self.config['update_epoch']) * self.train_len
        self.C_t = float(self.config['s_k']) * self.update_len * self.group_weight
        self.mu = np.ones(int(self.config['group_num']), dtype=np.float64)
        self.gradient_cusum = np.zeros(int(self.config['group_num']), dtype=np.float64)
        self._initialized = True
        self.exposure_count = 0

    def reweight(self, input_dict):
        if not self._initialized:
            raise RuntimeError('FairDualPaper.reset_parameters(train_len) must run first')
        items = input_dict['sample_items']
        if hasattr(items, 'detach'):
            items = items.detach().cpu().numpy()
        items = np.asarray(items, dtype=np.int64)
        items = items.reshape(items.shape[0], -1)

        batch_size = items.shape[0]
        # A^j: per-user top-k group incidence, averaged over the k exposures.
        B_t = np.sum(self.M[items], axis=1, keepdims=False).astype(np.float64)

        # Paper Eq. (6): s^j = 1 - A^j mu.  A constant per-batch shift keeps
        # the loss weight positive without changing the ordering across items.
        raw = 1.0 - np.mean(B_t * self.mu[None, :], axis=-1)
        weight = raw - raw.min() + 1e-2
        weight = weight / np.mean(weight)

        # Official dual tracker: gamma_j = gamma_{j-1} - (A^j)^T 1.
        D_t = np.sum(B_t, axis=0)
        self.C_t = self.C_t - D_t
        gradient = (-D_t / (batch_size * float(self.config['s_k']))
                    + self.C_t / (self.train_len * float(self.config['s_k'])))
        alpha = float(self.config['alpha'])
        gradient = alpha * gradient + (1.0 - alpha) * self.gradient_cusum
        self.gradient_cusum = gradient

        self.mu = self.compute_next_dual(
            float(self.config['eta']), self.group_weight, self.mu, gradient,
            float(self.config['lambd']))
        return weight

    def compute_next_dual(self, eta, rho, dual, gradient, lambd):
        tilde_dual = dual - eta * gradient / rho / rho
        order = np.argsort(tilde_dual * rho)
        ordered = self.map_layer(tilde_dual[order], rho[order], lambd)
        if ordered is None:
            return np.asarray(dual, dtype=np.float64)
        return ordered[order.argsort()]

    def map_layer(self, ordered_tilde_dual, rho, lambd):
        m = len(rho)
        answer = cp.Variable(m)
        objective = cp.Minimize(
            cp.sum_squares(cp.multiply(rho, answer)
                           - cp.multiply(rho, ordered_tilde_dual)))
        constraints = [cp.sum(cp.multiply(rho[:i], answer[:i])) >= -lambd
                       for i in range(1, m + 1)]
        problem = cp.Problem(objective, constraints)
        solution = None
        for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
            try:
                problem.solve(solver=solver)
            except Exception:
                continue
            if answer.value is not None and np.all(np.isfinite(answer.value)):
                solution = np.asarray(answer.value, dtype=np.float64)
                break
        if solution is None:
            self.solver_failures += 1
            if self.solver_failures <= 5:
                print(f'FairDualPaper: dual projection failed '
                      f'({self.solver_failures} time(s)); keeping previous dual',
                      file=sys.stderr, flush=True)
            if self.solver_failures > 100:
                raise RuntimeError('FairDualPaper: too many dual projection failures')
            return None
        return solution
