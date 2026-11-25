"""Recompute exposure distributions and fairness from frozen user lists."""
import numpy as np

from exposure import ExposureAccumulator
from metric import Gini, Entropy, MMF, MinMaxRatio, compute_GE_decomposition


def user_provider_exposure(bundle, k, mode=None, normalization=None):
    bundle.check_k(k)
    config = dict(bundle.meta['config'])
    if mode is not None:
        config['fairness_type'] = mode
    if normalization is not None:
        config['provider_exposure_normalization'] = normalization
    # Reuse the production exposure calculation, including Utility score rounding.
    accumulator = ExposureAccumulator(bundle.providers.toarray(), config)
    rows = []
    for items, scores in zip(bundle.items[:, :k], bundle.scores[:, :k]):
        accumulator.totals[:] = 0
        accumulator.add(items, scores)
        rows.append(accumulator.values())
    from scipy.sparse import csr_matrix
    return csr_matrix(np.stack(rows))


def partition(bundle, name):
    if name in bundle.meta['partitions']:
        return {int(p): int(g) for p, g in bundle.meta['partitions'][name].items()}
    # Alternative partitions use training-derived supplier statistics only.
    if len(name.split(':')) != 3:
        raise ValueError(f'Unknown provider partition: {name}')
    field, rule, count = name.split(':')
    count = int(count)
    if field not in ('size', 'freq') or rule not in ('quantile', 'equal_width') or count < 1:
        raise ValueError(f'Invalid provider partition: {name}')
    stats = bundle.meta['provider_stats']
    if field == 'size':
        values = bundle.catalog_sizes
    else:
        values = np.array([stats[str(p)]['freq'] / bundle.catalog_sizes[p]
                           for p in range(len(bundle.catalog_sizes))])
    edges = (np.percentile(values, np.linspace(0, 100, count + 1)) if rule == 'quantile'
             else np.linspace(values.min(), values.max(), count + 1))
    return {p: int(np.searchsorted(edges[1:-1], value, side='right'))
            for p, value in enumerate(values)}


def fairness(exposure, partitions, alphas=(0, .5, 1, 2), eta=1e-12, mmf_ratio=.5):
    exposure = np.asarray(exposure, dtype=float)
    if (exposure.ndim != 1 or exposure.size == 0 or not np.all(np.isfinite(exposure))
            or np.any(exposure < 0) or not np.isfinite(eta) or eta <= 0
            or not np.all(np.isfinite(alphas)) or not np.isfinite(mmf_ratio) or not 0 < mmf_ratio <= 1):
        raise ValueError('Invalid exposure, alpha, smoothing coefficient or MMF ratio')
    zeros = int(np.count_nonzero(exposure == 0))
    result = dict(zero_provider_count=zeros, zero_provider_ratio=zeros / len(exposure),
                  provider_count=len(exposure), mean_exposure=float(exposure.mean()),
                  smoothing_epsilon=float(eta * exposure.mean()),
                  all_zero_exposure=bool(np.all(exposure == 0)))
    if np.all(exposure == 0):
        result.update(gini=0., entropy=0., mmf=0., maxminratio=0.)
    else:
        result.update(gini=float(Gini(exposure)), entropy=float(Entropy(exposure)),
                      mmf=float(MMF(exposure, ratio=mmf_ratio)), maxminratio=float(MinMaxRatio(exposure)))
    for name, mapping in partitions.items():
        if set(mapping) != set(range(len(exposure))):
            raise ValueError(f'Incomplete provider partition: {name}')
        for alpha in alphas:
            decomposition = compute_GE_decomposition(exposure, mapping, alpha, max(mapping.values()) + 1, eta=eta)
            for metric, value in decomposition.items():
                result[f'{name}/alpha={alpha:g}/{metric}'] = float(value)
    return result


def evaluate(bundle, k, mode=None, normalization=None, eta=None, partition_names=None,
             alphas=(0, .5, 1, 2), counts=None):
    matrix = user_provider_exposure(bundle, k, mode, normalization)
    weights = np.ones(len(bundle.users)) if counts is None else np.asarray(counts)
    if (weights.shape != (len(bundle.users),) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('User multiplicities must be nonnegative with positive total')
    names = list(bundle.meta['partitions']) if partition_names is None else partition_names
    partitions = {name: partition(bundle, name) for name in names}
    eta = bundle.meta['config'].get('ge_eta', 1e-12) if eta is None else eta
    values = np.asarray(weights @ matrix).ravel()
    result = fairness(values, partitions, alphas, eta,
                      bundle.meta['config'].get('mmf_eval_ratio', .5))
    metrics = bundle.user_metrics(k)
    result.update({name: float(np.average(metrics[name], weights=weights))
                   for name in ['ndcg', 'hr', 'mrr', 'recall']})
    return result
