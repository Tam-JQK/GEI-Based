"""Per-user warm v1 processing for ML-1M multi-genre providers and BeerAdvocate.

The split, 5-core, history and ranking-candidate rules are identical to
``preprocessing/repartition.py``.  The difference is provider handling:

* ML-1M: each movie belongs to every genre in ``genre:token_seq``.  The
  exported ``iid2pid.json`` maps an item to a list, and ``providers.npz`` is a
  binary multi-hot incidence matrix.  Exposure gives full credit to every
  membership.
* BeerAdvocate: ``brewer_id`` is a scalar provider.

Providers with fewer than ten catalog items are merged into ``Other`` exactly
as in per_user_warm_v1.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common/engine'))
from analysis.artifacts import file_hash, write_json
from evaluation_candidates import ensure_ranking_candidates
from repartition import assignments, clean, histories, now

REPORT = ROOT / 'data/preprocessing/reports/per_user_v1'
OUT = ROOT / 'data/processed'
MERGE_THRESHOLD = 10

SPECS = {
    'ml-1m': dict(
        inter='ml-1m.inter', item='ml-1m.item',
        uid='user_id:token', iid='item_id:token',
        rating='rating:float', timestamp='timestamp:float',
        title='movie_title:token_seq', group='genre:token_seq',
        multi=True, separator=' ', provider_prefix='',
        threshold=4,
        note='Genres are independent providers. Every exposed movie credits all '
             'of its genre memberships; provider exposure may exceed the '
             'single-label total by construction.',
    ),
    'BeerAdvocate': dict(
        inter='BeerAdvocate.inter', item='BeerAdvocate.item',
        uid='user_id:token', iid='item_id:token',
        rating='overall:float', timestamp='timestamp:float',
        title='name:token', group='brewer_id:token',
        multi=False, separator=None, provider_prefix='brewer_',
        threshold=3,
        note='Supplier is brewer_id. Ratings >= 3 are positive; lower ratings '
             'are retained as negative training interactions.',
    ),
}


def load_metadata(dataset):
    spec = SPECS[dataset]
    path = ROOT / 'data' / 'raw' / dataset / spec['item']
    metadata = pd.read_csv(path, sep='\t', dtype=str).drop_duplicates(spec['iid'])
    metadata = metadata.dropna(subset=[spec['iid'], spec['title'], spec['group']])
    metadata = metadata[metadata[spec['title']].str.strip().ne('')]
    metadata = metadata[metadata[spec['group']].str.strip().ne('')]
    metadata = metadata[metadata[spec['group']].str.lower().ne('nan')]
    memberships = {}
    display = {}
    for row in metadata.to_dict('records'):
        item = str(row[spec['iid']])
        if spec['multi']:
            values = {}
            for token in str(row[spec['group']]).split(spec['separator']):
                name = clean(token)
                if name:
                    values.setdefault(name.casefold(), name)
        else:
            name = str(row[spec['group']]).strip()
            values = {name: spec['provider_prefix'] + name}
        if not values:
            continue
        memberships[item] = set(values)
        for key, name in values.items():
            display.setdefault(key, name)
    if not memberships:
        raise ValueError(f'No provider memberships for {dataset}')
    return metadata, memberships, display


def load_interactions(dataset, memberships):
    spec = SPECS[dataset]
    raw = ROOT / 'data' / 'raw' / dataset
    columns = [spec['uid'], spec['iid'], spec['rating'], spec['timestamp']]
    parts = []
    raw_rows = 0
    for part in pd.read_csv(raw / spec['inter'], sep='\t', usecols=columns,
                            dtype={spec['uid']: str, spec['iid']: str},
                            chunksize=1_000_000):
        raw_rows += len(part)
        part = part.dropna(subset=columns)
        parts.append(part[part[spec['iid']].isin(memberships)])
    frame = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
    frame = frame.drop_duplicates([spec['uid'], spec['iid']])
    if frame.empty:
        raise ValueError(f'No interactions matched item metadata for {dataset}')
    users, original_users = pd.factorize(frame[spec['uid']], sort=True)
    items, original_items = pd.factorize(frame[spec['iid']], sort=True)
    records = pd.DataFrame(dict(
        u=users.astype(np.int32),
        i=items.astype(np.int32),
        y=(frame[spec['rating']].to_numpy(np.float64) >= spec['threshold']).astype(np.int8),
        t=frame[spec['timestamp']].to_numpy(np.float64)))
    encoded_memberships = {int(code): set(memberships[str(original_items[code])])
                           for code in range(len(original_items))}
    return (records, encoded_memberships, original_users, original_items,
            raw_rows, len(frame))


def filter_records(records, memberships, item_val=5, user_val=5, group_val=5,
                   min_train_positive=2):
    """Iterative item/user/provider 5-core followed by the per-user split."""
    records = records.reset_index(drop=True)
    memberships = {int(item): set(values) for item, values in memberships.items()}
    iterations = []
    while True:
        before = len(records)
        records = assignments(records)
        for key, minimum in [('i', item_val), ('u', user_val)]:
            counts = records[key].value_counts()
            records = records[records[key].isin(counts.index[counts >= minimum])]
        if not len(records):
            raise ValueError('Empty data after item/user 5-core filtering')
        present = records['i'].unique()
        provider_items = defaultdict(set)
        for item in present:
            for provider in memberships[int(item)]:
                provider_items[provider].add(int(item))
        keep = {provider for provider, items in provider_items.items()
                if len(items) >= group_val}
        provider_changed = keep != set(provider_items)
        for item in present:
            current = memberships[int(item)]
            if not current <= keep:
                memberships[int(item)] = current & keep
        empty = [int(item) for item in present if not memberships[int(item)]]
        records = records[~records['i'].isin(empty)]
        if not len(records):
            raise ValueError('Empty data after provider 5-core filtering')
        records = assignments(records)
        positives = (records.loc[records.y == 1].groupby(['u', 's']).size()
                     .unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0))
        keep_users = positives.index[positives[0] >= min_train_positive]
        dropped_users = set(records['u'].unique()) - set(keep_users)
        records = records[records['u'].isin(keep_users)].reset_index(drop=True)
        if not len(records):
            raise ValueError('Empty data after warm-user filtering')
        record = dict(
            iteration=len(iterations) + 1, before=before, after=len(records),
            users=int(records.u.nunique()), items=int(records.i.nunique()),
            providers=int(len(keep)),
            providers_removed=int(len(provider_items) - len(keep)),
            items_without_provider=len(empty), users_dropped=len(dropped_users))
        iterations.append(record)
        print(now(), 'FILTER', record, flush=True)
        if not len(records):
            raise ValueError('Empty data after warm-user and history filtering')
        if (len(records) == before and not provider_changed and not empty
                and not dropped_users):
            break
        if len(iterations) > 100:
            raise RuntimeError('Filtering did not converge')
    return records, memberships, iterations


def merge_providers(memberships):
    sizes = Counter()
    for values in memberships.values():
        for provider in values:
            sizes[provider] += 1
    small = {provider for provider, count in sizes.items() if count < MERGE_THRESHOLD}
    ordered = sorted(set(sizes) - small)
    keys = (['__other__'] if small else []) + ordered
    index = {key: position for position, key in enumerate(keys)}
    item_pids = {}
    for item, values in memberships.items():
        pids = {index['__other__'] if provider in small else index[provider]
                for provider in values}
        if not pids:
            raise ValueError(f'Item {item} lost every provider during merging')
        item_pids[item] = sorted(pids)
    post_sizes = Counter()
    for pids in item_pids.values():
        for provider in pids:
            post_sizes[provider] += 1
    return keys, index, item_pids, sizes, small, post_sizes


def provider_statistics(records, item_pids, group_num, partitions):
    train = records[records.s == 0]
    frequency = np.zeros(group_num, dtype=np.int64)
    for item, count in train['i'].value_counts().items():
        for provider in item_pids[int(item)]:
            frequency[provider] += int(count)
    sizes = np.bincount([provider for values in item_pids.values() for provider in values],
                        minlength=group_num)
    stats = {}
    for provider in range(group_num):
        item_count = int(sizes[provider])
        stats[str(provider)] = dict(
            item_count=item_count,
            freq=int(frequency[provider]),
            avg_freq=float(frequency[provider] / item_count) if item_count else 0.0,
            meta_group_size=int(partitions['size'][provider]),
            meta_group_freq=int(partitions['freq'][provider]))
    return stats, sizes, frequency


def write_provider_reports(dataset, display, pre_sizes, small, index, item_pids):
    rows = []
    for key, count in sorted(pre_sizes.items(), key=lambda pair: (-pair[1], pair[0])):
        rows.append(dict(
            provider=display.get(key, key), item_count=int(count),
            merged=int(key in small),
            processed_provider_id=0 if key in small else int(index[key])))
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT / f'{dataset}.provider_merge.csv', index=False)
    frame[frame.merged == 1].to_csv(REPORT / f'{dataset}.merged_providers.csv', index=False)
    other_memberships = (sum(1 for values in item_pids.values() if 0 in values)
                         if small else 0)
    summary = dict(
        raw_metadata_providers=len(display), filtered_providers=len(pre_sizes),
        merged_providers=len(small), post_merge_providers=len(index),
        merge_threshold=MERGE_THRESHOLD, other_item_count=other_memberships,
        other_memberships=other_memberships)
    write_json(REPORT / f'{dataset}.provider_merge.json', summary)
    return summary


def process(dataset, force=False):
    if dataset not in SPECS:
        raise ValueError(f'Unsupported dataset {dataset}')
    spec = SPECS[dataset]
    target = OUT / dataset
    if target.exists():
        if force:
            raise FileExistsError(f'{target} exists; archive it instead of overwriting')
        config = yaml.safe_load((target / 'process_config.yaml').read_text())
        if config.get('split_protocol') != 'per_user_warm_v1':
            raise ValueError('Refusing to overwrite another data version')
        return json.loads((target / 'audit.json').read_text())

    metadata, memberships, display = load_metadata(dataset)
    records, memberships, original_users, original_items, raw_rows, matched_rows = \
        load_interactions(dataset, memberships)
    print(now(), dataset, 'raw', raw_rows, 'deduplicated matched', matched_rows,
          'items with providers', len(memberships), 'providers', len(display), flush=True)

    records, memberships, iterations = filter_records(records, memberships)
    users, kept_users = pd.factorize(records.u, sort=True)
    items, kept_items = pd.factorize(records.i, sort=True)
    records = records.assign(u=users.astype(np.int32), i=items.astype(np.int32))
    records = records.reset_index(drop=True)
    if not set(int(item) for item in kept_items) <= set(memberships):
        raise AssertionError('Kept catalog and provider memberships disagree')
    memberships = {new: memberships[int(old)] for new, old in enumerate(kept_items)}
    original_users = original_users[kept_users]
    original_items = original_items[kept_items]

    keys, index, item_pids, pre_sizes, small, post_sizes = merge_providers(memberships)
    if set(item_pids) != set(range(len(kept_items))):
        raise AssertionError('item_pids does not cover the processed catalog')
    group_num = len(keys)
    provider_names = {key: 'Other' if key == '__other__' else display[key] for key in keys}

    context, eligible = histories(records)
    padding_id = len(kept_items)
    context_mask = [[0] * (5 - len(values)) + [1] * len(values) for values in context]
    context = [[padding_id] * (5 - len(values)) + values for values in context]
    if spec['multi']:
        group_values = [json.dumps(item_pids[int(item)], separators=(',', ':'))
                        for item in records.i]
    else:
        group_values = [int(item_pids[int(item)][0]) for item in records.i]

    exported = pd.DataFrame({
        spec['uid']: records.u.to_numpy(),
        spec['iid']: records.i.to_numpy(),
        'history_behaviors': context,
        'history_mask': context_mask,
        'label:float': records.y.to_numpy(),
        spec['group']: group_values,
        'sequence_eligible': eligible.astype(np.int8),
        'split': records.s.to_numpy()})
    exported[spec['timestamp']] = records.t.to_numpy()

    staging = OUT / (dataset + '.preparing')
    if staging.exists():
        raise FileExistsError(f'{staging} exists; remove the stale staging directory first')
    staging.mkdir(parents=True)
    for code, name in [(0, 'train'), (1, 'valid.CTR'), (2, 'test.CTR')]:
        exported[exported.split == code].drop(columns='split').to_csv(
            staging / f'{dataset}.{name}', sep='\t', index=False)
    sequential = exported[(exported.split == 0) & eligible
                          & (exported['label:float'] == 1)].drop(columns='split')
    sequential.to_csv(staging / f'{dataset}.train.sequential', sep='\t', index=False)

    partitions = {}
    size_values = np.asarray([post_sizes[provider] for provider in range(group_num)],
                             dtype=np.int64)
    frequency_values = np.zeros(group_num, dtype=np.int64)
    for item, count in records.loc[records.s == 0, 'i'].value_counts().items():
        for provider in item_pids[int(item)]:
            frequency_values[provider] += int(count)
    for kind, values in [('size', size_values), ('freq', frequency_values / size_values)]:
        groups = np.searchsorted(np.percentile(values, [100 / 3, 200 / 3]),
                                 values, side='right')
        partitions[kind] = groups
        write_json(staging / f'pid2mgid_{kind}.json',
                   {str(provider): int(group) for provider, group in enumerate(groups)})
    stats, computed_sizes, computed_frequency = provider_statistics(
        records, item_pids, group_num, partitions)
    assert np.array_equal(computed_sizes, size_values)
    assert np.array_equal(computed_frequency, frequency_values)
    write_json(staging / 'provider_stats.json', stats)

    if spec['multi']:
        iid2pid = {str(item): [int(provider) for provider in item_pids[item]]
                   for item in range(len(kept_items))}
        rows = [item for item, values in item_pids.items() for _ in values]
        columns = [provider for values in item_pids.values() for provider in values]
        matrix = csr_matrix((np.ones(len(rows), np.int8), (rows, columns)),
                            shape=(len(kept_items), group_num))
        matrix.sum_duplicates()
        matrix.data[:] = 1
    else:
        iid2pid = {str(item): int(item_pids[item][0]) for item in range(len(kept_items))}
        matrix = csr_matrix((np.ones(len(kept_items), np.int8),
                             (np.arange(len(kept_items)),
                              [item_pids[item][0] for item in range(len(kept_items))])),
                            shape=(len(kept_items), group_num))
    write_json(staging / 'iid2pid.json', iid2pid)
    save_npz(staging / 'providers.npz', matrix)
    write_json(staging / 'pid2provider.json',
               {str(provider): provider_names[key] for provider, key in enumerate(keys)})
    write_json(staging / 'uid2original.json',
               {str(user): str(original_users[user]) for user in range(len(kept_users))})
    write_json(staging / 'iid2original.json',
               {str(item): str(original_items[item]) for item in range(len(kept_items))})
    titles = metadata.set_index(spec['iid'])[spec['title']]
    write_json(staging / 'iid2text.json',
               {str(item): str(titles.loc[original_items[item]])
                for item in range(len(kept_items))})

    config = dict(
        dataset=dataset, user_id=spec['uid'], item_id=spec['iid'],
        group_id=spec['group'], label_id='label:float', timestamp=spec['timestamp'],
        text_id=spec['title'], user_num=len(kept_users), item_num=len(kept_items),
        group_num=group_num, candidate_mode='full', candidate_seed=2025,
        candidate_version=1, sample_num=200,
        exclusion='all_retained_positive_interactions_except_current_split_positives',
        split_protocol='per_user_warm_v1', split_strategy='per_user_temporal',
        valid_ratio=.1, test_ratio=.1, split_seed=2025, history_length=5,
        history_field='history_behaviors',
        history_strategy='train_only_strict_past_for_training_and_train_prefix_for_evaluation',
        train_file=dataset + '.train', sequential_train_file=dataset + '.train.sequential',
        min_train_positive=2, min_valid_positive=0, min_test_positive=0,
        history_padding_id=padding_id, sequence_item_embedding_num=padding_id + 1,
        history_padding_policy='left_pad_outside_real_catalog_with_attention_mask',
        item_val=5, user_val=5, group_val=5,
        group_aggregation_threshold=MERGE_THRESHOLD,
        label_threshold=spec['threshold'], min_rating=None,
        multi_provider=spec['multi'], provider_separator=spec['separator'],
        provider_field=spec['group'], provider_prefix=spec['provider_prefix'],
        provider_membership_version='multi_provider_v1' if spec['multi'] else 'scalar_v1',
        meta_group_num=3, meta_group_types=['size', 'freq'],
        processed_data_dir=str(target), reprocess=False)
    (staging / 'process_config.yaml').write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    train = exported[exported.split == 0]
    positive_train = train[train['label:float'] == 1]
    seen_users = set(positive_train[spec['uid']])
    seen_items = set(positive_train[spec['iid']])
    split_stats = {}
    for code, name in [(0, 'train'), (1, 'valid'), (2, 'test')]:
        subset = exported[(exported.split == code) & (exported['label:float'] == 1)]
        split_stats[name] = dict(
            rows=int((exported.split == code).sum()), positive_rows=len(subset),
            users=int(subset[spec['uid']].nunique()),
            cold_users=len(set(subset[spec['uid']]) - seen_users),
            cold_items=len(set(subset[spec['iid']]) - seen_items))
    assert split_stats['valid']['cold_users'] == split_stats['test']['cold_users'] == 0
    assert set(sequential[spec['uid']]) == seen_users == set(range(len(kept_users)))
    assert all(len(values) == 5 for values in
               exported.loc[exported.split > 0, 'history_behaviors'])
    assert len(sequential) == int(positive_train.shape[0]) - len(kept_users)

    membership_histogram = Counter(len(values) for values in item_pids.values())
    merge_summary = write_provider_reports(dataset, display, pre_sizes, small, index,
                                           item_pids)
    audit = dict(
        dataset=dataset, created=now(), split_protocol='per_user_warm_v1',
        raw_rows=raw_rows, matched_deduplicated_rows=matched_rows,
        users=len(kept_users), items=len(kept_items),
        suppliers_before_merge=int(len(pre_sizes)), suppliers_after_merge=group_num,
        all_interactions=len(records), sequence_training_rows=len(sequential),
        splits=split_stats, iterations=iterations,
        raw_sha256={
            str((ROOT / 'data' / 'raw' / dataset / spec['inter']).relative_to(ROOT)):
                file_hash(ROOT / 'data' / 'raw' / dataset / spec['inter']),
            str((ROOT / 'data' / 'raw' / dataset / spec['item']).relative_to(ROOT)):
                file_hash(ROOT / 'data' / 'raw' / dataset / spec['item'])},
        multi_provider=spec['multi'], provider_separator=spec['separator'],
        provider_field=spec['group'], label_threshold=spec['threshold'],
        provider_membership_count=int(sum(len(values) for values in item_pids.values())),
        average_memberships_per_item=float(np.mean(
            [len(values) for values in item_pids.values()])),
        provider_membership_histogram={str(count): int(items) for count, items
                                       in sorted(membership_histogram.items())},
        provider_merge=merge_summary,
        note=spec['note'] + ' Per-user split BEFORE history construction. Full '
             'training prefix retained for graph models; sequence view uses positive '
             'training targets with at least one strictly earlier positive. PAD=item_num '
             'is outside the real candidate catalog. Cold items are reported, never '
             'silently removed.')
    write_json(staging / 'audit.json', audit)
    ensure_ranking_candidates(dataset, config, staging)
    staging.rename(target)
    print(now(), 'COMPLETE', json.dumps(
        {key: audit[key] for key in ['dataset', 'users', 'items', 'suppliers_before_merge',
                                     'suppliers_after_merge', 'all_interactions',
                                     'sequence_training_rows', 'splits']},
        ensure_ascii=False), flush=True)
    return audit

