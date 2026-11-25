"""Training and evaluation kernels used by the parameter search."""
import ast, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from analysis.artifacts import file_hash, write_json
from base_model.sasrec import SASRec

STOP = False


def seed_all(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def read_training(root, run, name, config):
    import fcntl
    cache = run / 'input_cache' / name
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _read_training(root, run, name, config)


def _read_training(root, run, name, config):
    cache = run / 'input_cache' / name
    metadata = cache / 'metadata.json'
    source = Path(config.get('processed_data_dir') or root / 'processed_dataset' / name) / config['sequential_train_file']
    sha = file_hash(source)
    if not metadata.exists():
        history, masks, targets = [], [], []
        cols = [config['item_id'], 'history_behaviors', 'history_mask', 'label:float', 'sequence_eligible']
        for part in pd.read_csv(source, sep='\t', usecols=cols, chunksize=100000):
            assert (part['label:float'] == 1).all() and (part.sequence_eligible == 1).all()
            h = np.asarray([ast.literal_eval(x) for x in part.history_behaviors], dtype=np.int64)
            m = np.asarray([ast.literal_eval(x) for x in part.history_mask], dtype=np.bool_)
            assert h.shape == m.shape == (len(part), 5)
            assert m.any(axis=1).all() and m[:, -1].all()
            assert np.array_equal(m, h != config['history_padding_id'])
            assert ((h >= 0) & (h <= config['item_num'])).all()
            history.append(h); masks.append(m); targets.append(part[config['item_id']].to_numpy(np.int64))
        cache.mkdir(parents=True, exist_ok=True)
        for key, pieces in [('history', history), ('mask', masks), ('target', targets)]:
            np.save(cache / f'{key}.npy', np.concatenate(pieces))
        write_json(metadata, dict(source_sha256=sha, rows=sum(len(x) for x in targets)))
    assert json.loads(metadata.read_text())['source_sha256'] == sha
    return {key: torch.from_numpy(np.load(cache / f'{key}.npy'))
            for key in ['history', 'mask', 'target']}


def read_ranking(root, name, config, split, limit=None):
    path = Path(config.get('processed_data_dir') or root / 'processed_dataset' / name) / 'evaluation/v1_full_seed2025' / f'{name}.{split}.ranking'
    frame = pd.read_csv(path, sep='\t', nrows=limit)
    history = np.asarray([ast.literal_eval(x) for x in frame.history_behaviors], dtype=np.int64)
    positive = [ast.literal_eval(x)[:int(count)] for x, count in zip(frame['items'], frame.pos_length)]
    excluded = [np.asarray(ast.literal_eval(x), dtype=np.int64) for x in frame.excluded_items]
    rows = np.repeat(np.arange(len(frame)), [len(x) for x in positive])
    truth = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, np.concatenate(positive))),
                       shape=(len(frame), config['item_num']))
    assert frame.user_id.is_unique and (frame.pos_length > 0).all()
    mask = history != config['history_padding_id']
    assert mask.any(axis=1).all() and mask[:, -1].all()
    return dict(users=frame.user_id.to_numpy(np.int64), history=torch.from_numpy(history),
                mask=torch.from_numpy(mask), counts=frame.pos_length.to_numpy(np.int64),
                truth=truth, excluded=excluded, source_sha256=file_hash(path))


def stable_topk(scores, k):
    """Exact score descending/item ID ascending, including ties at the cutoff."""
    values, items = scores.topk(k, dim=1, sorted=True)
    cutoff = values[:, -1:]
    # Repair only boundary ties. Sorting the 20 selected entries handles all
    # other ties without sorting the full 205k-item catalog.
    tied = ((scores == cutoff).sum(dim=1) > (values == cutoff).sum(dim=1)).nonzero().flatten()
    for row in tied.tolist():
        better = items[row, values[row] > cutoff[row, 0]]
        equal = (scores[row] == cutoff[row, 0]).nonzero().flatten()
        items[row] = torch.cat([better, equal[:k - len(better)]])
        values[row] = scores[row, items[row]]
    by_id = items.argsort(dim=1)
    items, values = items.gather(1, by_id), values.gather(1, by_id)
    order = values.argsort(dim=1, descending=True, stable=True)
    return values.gather(1, order), items.gather(1, order)


def accuracy(hits, counts, cutoffs=(10, 20)):
    output = {}
    for k in cutoffs:
        h = hits[:, :k]
        discounts = 1 / np.log2(np.arange(k) + 2)
        ideal = np.cumsum(discounts)[np.minimum(counts, k) - 1]
        output[f'ndcg@{k}'] = float(np.mean((h @ discounts) / ideal))
        output[f'hr@{k}'] = float(h.any(axis=1).mean())
        first = np.argmax(h, axis=1) + 1
        output[f'mrr@{k}'] = float(np.mean(h.any(axis=1) / first))
        output[f'recall@{k}'] = float(np.mean(h.sum(axis=1) / counts))
    return output


@torch.no_grad()
def evaluate(model, ranking, device, batch_size=128):
    model.eval()
    weights = model.item_embedding.weight[:model.config['item_num']]
    hit_parts, item_parts, score_parts = [], [], []
    for start in range(0, len(ranking['users']), batch_size):
        stop = min(start + batch_size, len(ranking['users']))
        interaction = dict(history_ids=ranking['history'][start:stop].to(device),
                           history_mask=ranking['mask'][start:stop].to(device))
        scores = model.get_user_embedding(interaction) @ weights.T
        excluded = ranking['excluded'][start:stop]
        lengths = [len(x) for x in excluded]
        if sum(lengths):
            rr = np.repeat(np.arange(stop - start), lengths)
            cc = np.concatenate(excluded)
            scores[torch.as_tensor(rr, device=device), torch.as_tensor(cc, device=device)] = -torch.inf
        values, items = stable_topk(scores, 20)
        if not torch.isfinite(values).all():
            raise ValueError('A user has fewer than 20 eligible candidates or nonfinite scores')
        ii = items.cpu().numpy()
        matched = ranking['truth'][np.repeat(np.arange(start, stop), 20), ii.ravel()]
        hits = (matched.toarray() if hasattr(matched, 'toarray') else np.asarray(matched)).reshape(stop - start, 20)
        hit_parts.append(hits); item_parts.append(ii); score_parts.append(values.cpu().numpy())
    hits = np.concatenate(hit_parts)
    return accuracy(hits, ranking['counts']), dict(users=ranking['users'], items=np.concatenate(item_parts),
                scores=np.concatenate(score_parts), hits=hits, positive_counts=ranking['counts'])


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def train_trial(run, name, config, training, ranking, plan, device):
    directory = run / name / config['trial']
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / 'result.json'
    if completed.exists():
        return json.loads(completed.read_text())
    write_json(directory / 'config.json', config)
    seed_all(plan['fixed']['seed'])
    model = SASRec(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, best_epoch, failures, start_epoch = -1., 0, 0, 1
    history = []
    if (directory / 'last.pt').exists():
        checkpoint = torch.load(directory / 'last.pt', map_location=device, weights_only=False)
        assert checkpoint['config'] == config
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        best, best_epoch, failures = checkpoint['best'], checkpoint['best_epoch'], checkpoint['failures']
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']; restore_rng(checkpoint['rng'])
        print('RESUME', name, config['trial'], 'epoch', start_epoch, flush=True)
    count = len(training['target'])
    finished_epoch = start_epoch - 1
    for epoch in range(start_epoch, plan['max_epochs'] + 1):
        if failures >= plan['patience']:
            break
        if STOP:
            raise InterruptedError('Stop requested; last complete epoch is resumable')
        model.train()
        started = time.monotonic()
        indices = torch.randperm(count, device=device)
        loss_sum = torch.zeros((), device=device)
        batches = 0
        for offset in range(0, count, config['batch_size']):
            if STOP:
                raise InterruptedError('Stop requested; last complete epoch is resumable')
            index = indices[offset:offset + config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_loss(dict(history_ids=training['history'][index],
                    history_mask=training['mask'][index], item_ids=training['target'][index]))
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach() * len(index)
            batches += 1
            if batches % 250 == 0:
                print(name, config['trial'], 'epoch', epoch, 'batch', batches,
                      '/', (count + config['batch_size'] - 1) // config['batch_size'], flush=True)
        record = dict(epoch=epoch, loss=float(loss_sum / count))
        validated = epoch == 1 or epoch % plan['eval_every'] == 0 or epoch == plan['max_epochs']
        if validated:
            metrics, _ = evaluate(model, ranking, device, config['eval_batch_size'])
            record['valid'] = metrics
            score = metrics['ndcg@20']
            if score > best:
                best, best_epoch, failures = score, epoch, 0
                save_checkpoint(directory / 'best.pt', dict(model=model.state_dict(), config=config,
                    epoch=epoch, valid=metrics))
            else:
                failures += 1
        record.update(seconds=time.monotonic() - started, best_valid_ndcg20=best,
                      best_epoch=best_epoch, validations_without_improvement=failures)
        history.append(record)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                       epoch=epoch, best=best, best_epoch=best_epoch, failures=failures,
                       history=history, rng=rng_state())
        save_checkpoint(directory / 'last.pt', payload)
        write_json(directory / 'history.json', history)
        write_json(directory / 'status.json', dict(state='training', **record))
        print(name, config['trial'], json.dumps(record), flush=True)
        finished_epoch = epoch
    result = dict(dataset=name, trial=config['trial'], best_epoch=best_epoch,
                  best_valid_ndcg20=best, epochs=finished_epoch,
                  stop_reason='patience' if failures >= plan['patience'] else 'max_epochs',
                  checkpoint=str((directory / 'best.pt').relative_to(run)), config=config)
    if best_epoch == 0:
        raise RuntimeError('No validation checkpoint was selected')
    write_json(completed, result)
    write_json(directory / 'status.json', dict(state='complete', **result))
    return result
