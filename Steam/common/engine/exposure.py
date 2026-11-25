"""Shared exposure accounting for base-model and reranking evaluation."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def load_provider_matrix(directory, item_num, group_num):
    """Load iid2pid.json as a binary item-provider incidence matrix.

    Each item may map to one provider id (scalar) or a list of provider ids
    (multi-provider).  Every membership receives full credit at exposure time.
    """
    mapping = json.loads((Path(directory) / 'iid2pid.json').read_text())
    if set(mapping) != {str(item) for item in range(item_num)}:
        raise ValueError('iid2pid.json does not cover exactly the processed item catalog')
    rows, columns = [], []
    for item in range(item_num):
        value = mapping[str(item)]
        providers = value if isinstance(value, list) else [value]
        if not providers:
            raise ValueError(f'Item {item} has no provider membership')
        for provider in providers:
            if isinstance(provider, bool) or not isinstance(provider, (int, np.integer)):
                raise ValueError(f'Item {item} has a non-integer provider id')
            if provider < 0 or provider >= group_num:
                raise ValueError(f'Item {item} has provider {provider} outside [0, {group_num})')
            rows.append(item)
            columns.append(int(provider))
    matrix = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)),
                        shape=(item_num, group_num))
    matrix.sum_duplicates()
    matrix.data[:] = 1
    return matrix


def discounted_provider_exposure(matrix, items, k):
    """Accumulate DiscountedExposure over a users x k ranked-item array.

    ``matrix[items]`` gives a 3-D users x k x providers indicator tensor;
    summing over ranks applies the 1/log2(rank+2) discount.  Multi-provider
    items contribute full credit to every membership.
    """
    items = np.asarray(items, dtype=np.int64)
    if items.ndim != 2 or items.shape[1] < k:
        raise ValueError(f'Expected a users x K array with K >= {k}')
    discounts = 1.0 / np.log2(np.arange(k, dtype=float) + 2.0)
    totals = np.zeros(matrix.shape[1], dtype=float)
    for rank in range(k):
        totals += discounts[rank] * np.asarray(
            matrix[items[:, rank]].sum(axis=0), dtype=float).ravel()
    return totals


class ExposureAccumulator:
    def __init__(self, provider_matrix, config):
        self.matrix = np.asarray(provider_matrix)
        self.mode = config.get('fairness_type', 'Exposure')
        self.normalization = config.get('provider_exposure_normalization', 'none')
        self.decimals = config.get('decimals', 4)
        if self.mode not in ('Exposure', 'DiscountedExposure', 'Utility'):
            raise ValueError(f'Unknown fairness_type: {self.mode}')
        if self.normalization not in ('none', 'catalog_mean'):
            raise ValueError(f'Unknown provider_exposure_normalization: {self.normalization}')
        # Unique items in the entire processed catalog, including unexposed items.
        # A merged provider uses the combined catalog of its constituent providers.
        self.catalog_sizes = np.count_nonzero(self.matrix, axis=0)
        if np.any(self.catalog_sizes == 0):
            raise ValueError('Every evaluated provider must have at least one catalog item')
        self.totals = np.zeros(self.matrix.shape[1], dtype=float)

    def add(self, ranked_items, scores=None):
        """Add one user's ordered Top-K list; scores must follow the same order."""
        items = np.asarray(ranked_items, dtype=int)
        if self.mode == 'DiscountedExposure':
            weights = 1.0 / np.log2(np.arange(len(items)) + 2.0)
        elif self.mode == 'Utility':
            if scores is None or np.shape(scores) != items.shape:
                raise ValueError('Utility exposure requires one score per ranked item')
            weights = np.round(scores, self.decimals)
        else:
            weights = np.ones(len(items))
        # Multi-provider items give full credit to each membership.
        self.totals += weights @ self.matrix[items]

    def values(self):
        if self.normalization == 'catalog_mean':
            return self.totals / self.catalog_sizes
        return self.totals.copy()

    def details(self):
        return {
            'fairness_type': self.mode,
            'provider_exposure_normalization': self.normalization,
            'provider_ids': list(range(self.matrix.shape[1])),
            'catalog_sizes': self.catalog_sizes.tolist(),
            'total_exposure': self.totals.tolist(),
            'provider_exposure': self.values().tolist(),
        }
