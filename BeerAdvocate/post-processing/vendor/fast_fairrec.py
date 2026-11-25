"""Accelerated, behavior-preserving version of the project's FairRec."""
import ctypes
from pathlib import Path
import numpy as np

P = ctypes.c_void_p

def load_allocator(path):
    global lib
    lib = ctypes.CDLL(str(path))
    lib.fairrec_allocate.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,P,P,P]
    lib.fairrec_allocate.restype = ctypes.c_int
    lib.fairrec_complete.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int]
    lib.fairrec_complete.restype = ctypes.c_int
def ptr(a): return a.ctypes.data_as(P)

class Prepared:
    def __init__(self, scores, mask, k=20):
        self.scores = np.ascontiguousarray(scores, dtype=np.float32)
        self.mask = np.ascontiguousarray(mask, dtype=np.uint8)
        self.k = k
        self.m,self.n = self.scores.shape
        if not np.all(np.isfinite(self.scores)) or np.min(self.mask.sum(axis=1)) < 2*k:
            raise ValueError('Scores must be finite and each user must have >= 2k eligible items')
        self.order = np.ascontiguousarray(np.argsort(-self.scores, axis=1, kind='stable'), dtype=np.int32)
        self.top = np.ascontiguousarray(np.argsort(np.where(self.mask, self.scores, -np.inf),axis=1)[:,-2*k:][:,::-1],dtype=np.int32)
    def rerank(self, alpha):
        out=np.full((self.m,self.k),-1,dtype=np.int32)
        lengths=np.zeros(self.m,dtype=np.int32)
        rem=ctypes.c_int()
        quota=lib.fairrec_allocate(ptr(self.scores),ptr(self.mask),ptr(self.order),self.m,self.n,self.k,float(alpha),ptr(out),ptr(lengths),ctypes.byref(rem))
        if rem.value < 0: raise ValueError('FairRec allocation would exceed k')
        allocated=int(lengths.sum())
        rc=lib.fairrec_complete(ptr(out),ptr(lengths),ptr(self.top),self.m,self.k)
        if rc or not np.all(self.mask[np.arange(self.m)[:,None],out]): raise ValueError('invalid recommendation')
        return out,dict(quota=quota,initial_allocated=allocated,allocation_budget=quota*self.n,unallocated_budget=rem.value,early_stop=rem.value>0)
