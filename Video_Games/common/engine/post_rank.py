"""Rerank a score matrix using the chosen method and the same candidate mask."""
import ctypes
import importlib
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from scripts.fair_search import FastFairDual

VENDOR = Path(__file__).resolve().parents[2] / 'post-processing/vendor'


def compile_shared(name, build_dir):
    source = VENDOR / (name + '.cpp')
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / (name + '.so')
    if library.is_file() and library.stat().st_mtime >= source.stat().st_mtime:
        return library
    temporary = build_dir / f'{name}.{os.getpid()}.tmp.so'
    try:
        subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC',
                        str(source), '-o', str(temporary)], check=True)
        temporary.replace(library)
    finally:
        temporary.unlink(missing_ok=True)
    return library


class Reranker:
    def __init__(self, study, ranking, scores, split):
        self.method = study.args.method
        self.device = study.device
        self.scores = scores
        self.split = split
        self.k = study.protocol['rerank_k']
        self.config = study.config
        self.excluded = ranking['excluded']
        self.build_dir = study.output / 'native'
        self.mask = np.ones(scores.shape, dtype=bool)
        for i, exclusion in enumerate(self.excluded):
            self.mask[i, exclusion] = False
        self.fairrec_prepared = None
        if self.method == 'cpfair' and (VENDOR / 'cpfair_fast.cpp').is_file():
            self.groups = np.ascontiguousarray(np.asarray(study.matrix.argmax(axis=1)).ravel(),
                                                dtype=np.int32)

    def cpfair_native(self, params):
        sizes = np.fromiter(map(len, self.excluded), dtype=np.int32)
        offsets = np.r_[np.int32(0), np.cumsum(sizes, dtype=np.int32)].astype(np.int32)
        excluded = np.concatenate(self.excluded).astype(np.int32)
        output = np.empty((len(self.scores), self.k), dtype=np.int32)
        call = ctypes.CDLL(str(compile_shared('cpfair_fast', self.build_dir))).cpfair
        pointer = ctypes.c_void_p
        call.argtypes = [pointer, pointer, pointer, pointer, ctypes.c_int, ctypes.c_int,
                         ctypes.c_int, ctypes.c_double, pointer]
        call.restype = ctypes.c_int
        result = call(self.scores.ctypes.data, self.groups.ctypes.data, offsets.ctypes.data,
                      excluded.ctypes.data, len(self.scores), self.scores.shape[1],
                      self.config['group_num'], params['lambda'], output.ctypes.data)
        if result:
            raise RuntimeError('CPFair reranking failed')
        return output

    def prepare_fairrec(self):
        from fast_fairrec import Prepared, load_allocator

        load_allocator(compile_shared('allocator', self.build_dir))
        if self.device == 'cpu':
            return Prepared(self.scores, self.mask, k=self.k)
        prepared = Prepared.__new__(Prepared)
        prepared.scores = self.scores
        prepared.mask = np.ascontiguousarray(self.mask, dtype=np.uint8)
        prepared.k = self.k
        prepared.m, prepared.n = self.scores.shape
        prepared.order = np.empty(self.scores.shape, dtype=np.int32)
        prepared.top = np.empty((prepared.m, 2 * self.k), dtype=np.int32)
        for start in range(0, prepared.m, 128):
            stop = min(start + 128, prepared.m)
            part = np.asarray(self.scores[start:stop])
            eligible = self.mask[start:stop]
            if not np.isfinite(part).all() or np.min(eligible.sum(axis=1)) < 2 * self.k:
                raise ValueError('FairRec needs finite scores and at least 2K eligible items per user')
            gpu = torch.as_tensor(part.copy(), device=self.device)
            prepared.order[start:stop] = torch.argsort(gpu, dim=1, descending=True, stable=True).to(torch.int32).cpu().numpy()
            prepared.top[start:stop] = np.argsort(np.where(eligible, part, -np.inf), axis=1)[:, -2 * self.k:][:, ::-1]
        return prepared

    def fairrec(self, params):
        if self.fairrec_prepared is None:
            self.fairrec_prepared = self.prepare_fairrec()
        prepared = self.fairrec_prepared
        seed = params.get('user_order_seed')
        if seed is None:
            return prepared.rerank(params['alpha'])
        permutation = np.random.default_rng(seed + (0 if self.split == 'valid' else 100000)).permutation(prepared.m)
        reordered = type(prepared).__new__(type(prepared))
        reordered.m, reordered.n, reordered.k = prepared.m, prepared.n, prepared.k
        for key in ('scores', 'mask', 'order', 'top'):
            setattr(reordered, key, np.ascontiguousarray(getattr(prepared, key)[permutation]))
        ids, diagnostics = reordered.rerank(params['alpha'])
        restored = np.empty_like(ids)
        restored[permutation] = ids
        return restored, diagnostics

    def rank(self, params):
        diagnostics = {}
        if self.method == 'fairrec':
            ids, diagnostics = self.fairrec(params)
        elif self.method == 'cpfair' and hasattr(self, 'groups'):
            ids = self.cpfair_native(params)
        else:
            name = {'cpfair': 'CPFair', 'pmmf': 'PMMF', 'elasticrank': 'ElasticRank'}[self.method]
            module = importlib.import_module('rerank_model.' + name)
            module.trange = range
            config = dict(self.config, **params)
            gpu_module = self.method + '_gpu'
            if self.device.startswith('cuda') and (VENDOR / (gpu_module + '.py')).is_file():
                cls = getattr(importlib.import_module(gpu_module), name + 'GPU')
            else:
                cls = getattr(module, name)
                if self.method == 'pmmf':
                    solver = FastFairDual(config, np.ones(config['group_num']))
                    module.compute_projection_maxmin_fairness_with_order = solver.map_layer
            reranker = cls(config)
            reranker.candidate_mask = self.mask
            with np.errstate(divide='raise', invalid='raise', over='raise'):
                ids = np.asarray(reranker.rerank(self.scores, self.k), dtype=np.int32)
        if ids.shape != (len(self.scores), self.k) or np.any(ids < 0) or not np.all(
                self.mask[np.arange(len(ids))[:, None], ids]):
            raise ValueError(f'{self.method} selected an invalid candidate')
        return ids, diagnostics
