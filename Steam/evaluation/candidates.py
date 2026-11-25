"""FairDiverse candidate semantics shared by preprocessing, evaluation and reranking."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


VERSION = 1


def candidate_settings(config):
    mode = config.get('candidate_mode', 'sampled')
    if mode not in ('sampled', 'full'):
        raise ValueError('candidate_mode must be sampled or full')
    size = config.get('sample_num', 200)
    seed = config.get('candidate_seed', 2025)
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError('sample_num must be a positive integer (target TOTAL candidates)')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('candidate_seed must be a nonnegative integer')
    return dict(candidate_mode=mode, sample_num=int(size), candidate_seed=int(seed),
                candidate_version=VERSION,
                exclusion='all_retained_positive_interactions_except_current_split_positives')


def as_list(value):
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


def candidate_ids(positives, known_positives, item_num, settings, user, split):
    """Uniform without replacement; positive items are NEVER truncated.

    Per-user permutations do not depend on sample_num, so 200 is nested in 500.
    All modes use the original FairDiverse exclusion universe.
    """
    positives = list(dict.fromkeys(int(item) for item in positives))
    allowed = np.ones(item_num, dtype=bool)
    known = list(set(int(item) for item in known_positives))
    allowed[known] = False
    negatives = np.flatnonzero(allowed)
    if settings['candidate_mode'] == 'full':
        return positives, sorted(set(known) - set(positives))
    rng = np.random.default_rng(np.random.SeedSequence([
        settings['candidate_seed'], 0 if split == 'valid' else 1, int(user)]))
    count = max(settings['sample_num'] - len(positives), 0)
    negatives = rng.permutation(negatives)[:count].tolist()
    return positives + negatives, []


def source_signature(dataset, directory):
    directory = Path(directory)
    digest = hashlib.sha256()
    for name in [f'{dataset}.train', f'{dataset}.valid.CTR', f'{dataset}.test.CTR']:
        source = directory / name
        digest.update(source.name.encode())
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def ensure_ranking_candidates(dataset, config, directory=None):
    """Cache each protocol separately, without regenerating training splits."""
    directory = Path(directory or Path('processed_dataset') / dataset)
    settings = candidate_settings(config)
    key = ('full' if settings['candidate_mode'] == 'full' else f"sampled_{settings['sample_num']}")
    target = directory / 'evaluation' / f"v{VERSION}_{key}_seed{settings['candidate_seed']}"
    source_files = [directory / f'{dataset}.train'] + [
        directory / f'{dataset}.{split}.CTR' for split in ('valid', 'test')]
    signature = dict(settings, source_sha256=source_signature(dataset, directory), item_num=int(config['item_num']))
    metadata_path = target / 'candidate_protocol.json'
    cached = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    required = [target / f'{dataset}.{split}.ranking' for split in ('valid', 'test')]
    if not all(cached.get(key) == value for key, value in signature.items()) or not all(p.exists() for p in required):
        uid, iid, label = config['user_id'], config['item_id'], config['label_id']
        frames = [pd.read_csv(source, sep='\t') for source in source_files]
        known = {}
        for frame in frames:
            for user, item in frame.loc[frame[label] == 1, [uid, iid]].itertuples(index=False, name=None):
                known.setdefault(int(user), set()).add(int(item))
        target.mkdir(parents=True, exist_ok=True)
        stats = {}
        for split, frame in zip(('valid', 'test'), frames[1:]):
            records, lengths, excess, shortages = [], [], 0, 0
            for user, rows in frame[frame[label] == 1].groupby(uid, sort=False):
                positives = list(dict.fromkeys(int(item) for item in rows[iid]))
                items, excluded = candidate_ids(positives, known[int(user)], config['item_num'], settings, user, split)
                count = len(items) if settings['candidate_mode'] == 'sampled' else config['item_num'] - len(excluded)
                lengths.append(count)
                excess += len(positives) > settings['sample_num']
                shortages += settings['candidate_mode'] == 'sampled' and count < settings['sample_num']
                records.append(dict(user_id=int(user), history_behaviors=as_list(rows.iloc[0]['history_behaviors']),
                                    items=items, pos_length=len(positives), excluded_items=excluded))
            pd.DataFrame(records, columns=['user_id', 'history_behaviors', 'items', 'pos_length',
                                          'excluded_items']).to_csv(target / f'{dataset}.{split}.ranking', sep='\t', index=False)
            stats[split] = dict(users=len(records), min_candidates=min(lengths, default=0),
                                max_candidates=max(lengths, default=0),
                                users_exceeding_target_positive_count=int(excess),
                                users_with_insufficient_negatives=int(shortages))
        metadata_path.write_text(json.dumps(dict(signature, statistics=stats), indent=2) + '\n')
    # Keep original filenames available for older entry points. Versioned caches
    # preserve 200/500/full artifacts independently; no train/CTR files are touched.
    for source in required:
        destination = directory / source.name
        if not destination.exists() or destination.read_bytes() != source.read_bytes():
            shutil.copyfile(source, destination)
    return target


def explicit_candidate_mask(scores):
    """Use stored sparse coordinates, including explicitly stored zero scores."""
    mask = scores.tocsr(copy=True)
    mask.data = np.ones(mask.data.shape, dtype=bool)
    return mask


def validate_scored_candidates(frame, candidate_mask, index2userid, mode='sampled'):
    """Check that score rows, eligibility masks and the saved truth snapshot agree."""
    if mode not in ('sampled', 'full') or candidate_mask.ndim != 2:
        raise ValueError('Invalid candidate mode or mask dimensions')
    mapping = {int(row): int(user) for row, user in index2userid.items()}
    n_users, n_items = candidate_mask.shape
    if (len(mapping) != len(index2userid) or set(mapping) != set(range(n_users))
            or len(set(mapping.values())) != n_users or len(frame) != n_users
            or frame['user_id'].duplicated().any()
            or set(frame['user_id'].astype(int)) != set(mapping.values())):
        raise ValueError('Candidate snapshot and score rows have inconsistent user mappings')
    users = {user: row for row, user in mapping.items()}
    for record in frame.to_dict('records'):
        user = int(record['user_id'])
        items = np.asarray(as_list(record['items']))
        positive_count = record['pos_length']
        if (items.ndim != 1 or not np.issubdtype(items.dtype, np.integer)
                or len(set(items.tolist())) != len(items) or np.any(items < 0) or np.any(items >= n_items)
                or positive_count != int(positive_count) or not 0 < positive_count <= len(items)):
            raise ValueError(f'Invalid candidate IDs or positive count for user {user}')
        expected = np.zeros(n_items, dtype=bool)
        if mode == 'sampled':
            expected[items] = True
        else:
            if 'excluded_items' not in record:
                raise ValueError('Full-item candidate snapshot needs excluded_items')
            excluded = np.asarray(as_list(record['excluded_items']), dtype=np.int64)
            if np.any(excluded < 0) or np.any(excluded >= n_items):
                raise ValueError('Invalid excluded item in full-item snapshot')
            expected[:] = True
            expected[excluded] = False
            if not np.all(expected[items]):
                raise ValueError('Full-item snapshot contains excluded positive/candidate items')
        if not np.array_equal(expected, candidate_mask[users[user]]):
            raise ValueError(f'Candidate mask does not match saved snapshot for user {user}')
    return mapping
