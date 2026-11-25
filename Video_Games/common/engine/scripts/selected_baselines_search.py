"""Training and evaluation kernels used by the parameter search."""
import json, time
import numpy as np
import torch
from analysis.artifacts import write_json
from base_model.lightgcn import LightGCN
from scripts.lightgcn_search import sample_negatives
from scripts.sasrec_search import accuracy, stable_topk, seed_all, save_checkpoint, rng_state, restore_rng
from datetime import datetime, timezone

STOP = False

def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def make_model(config, graph):
    if config['model'] == 'lightgcn':
        return LightGCN(config, graph=graph)
    raise ValueError(config['model'])


def compact_state(model):
    # The graph is reconstructed from the hashed immutable input cache.
    return {key: value for key, value in model.state_dict().items() if key != 'Graph'}


def restore_model(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == (['Graph'] if isinstance(model, LightGCN) else []) and not unexpected


def invalidate(model):
    if isinstance(model, LightGCN):
        model.invalidate_embedding_cache()


def embedding_rms(model):
    with torch.no_grad():
        return {name: float(weight.square().mean().sqrt()) for name, weight in [
            ('user_embedding_rms', model.user_embedding.weight),
            ('item_embedding_rms', model.item_embedding.weight)]}


@torch.no_grad()
def evaluate(model, ranking, device, batch_size=256):
    """Score EVERY real item, then apply the existing full-candidate mask."""
    model.eval()
    catalog = torch.arange(model.config['item_num'], device=device)
    hit_parts, item_parts, score_parts = [], [], []
    for start in range(0, len(ranking['users']), batch_size):
        stop = min(start + batch_size, len(ranking['users']))
        users = torch.as_tensor(ranking['users'][start:stop], device=device)
        scores = model.full_scores(dict(user_ids=users), catalog)
        assert scores.shape == (stop - start, model.config['item_num'])
        if not torch.isfinite(scores).all():
            raise ValueError('Nonfinite full-catalog scores')
        excluded = ranking['excluded'][start:stop]
        lengths = [len(x) for x in excluded]
        if sum(lengths):
            rows = torch.as_tensor(np.repeat(np.arange(stop - start), lengths), device=device)
            columns = torch.as_tensor(np.concatenate(excluded), device=device)
            scores[rows, columns] = -torch.inf
        values, items = stable_topk(scores, 20)
        if not torch.isfinite(values).all():
            raise ValueError('Fewer than 20 eligible candidates')
        ids = items.cpu().numpy()
        matched = ranking['truth'][np.repeat(np.arange(start, stop), 20), ids.ravel()]
        hits = (matched.toarray() if hasattr(matched, 'toarray') else np.asarray(matched)).reshape(stop-start, 20)
        hit_parts.append(hits); item_parts.append(ids); score_parts.append(values.cpu().numpy())
    hits = np.concatenate(hit_parts)
    return accuracy(hits, ranking['counts']), dict(users=ranking['users'], items=np.concatenate(item_parts),
        scores=np.concatenate(score_parts), hits=hits, positive_counts=ranking['counts'])


def train_trial(run, dataset, config, graph, users, items, ranking, plan, device):
    directory = run / config['model'] / dataset / config['trial']
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        return read_json(directory / 'result.json')
    write_json(directory / 'config.json', config)
    seed_all(config['seed'])
    model = make_model(config, graph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, best_epoch, failures, start_epoch, history = -1., 0, 0, 1, []
    if (directory / 'last.pt').exists():
        saved = torch.load(directory / 'last.pt', map_location=device, weights_only=False)
        assert saved['config'] == config
        restore_model(model, saved['model']); optimizer.load_state_dict(saved['optimizer'])
        best, best_epoch, failures = saved['best'], saved['best_epoch'], saved['failures']
        start_epoch, history = saved['epoch'] + 1, saved['history']
        restore_rng(saved['rng'])
        print('RESUME', config['model'], dataset, config['trial'], start_epoch, flush=True)
    count = len(users)
    known = users * config['item_num'] + items
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, plan['max_epochs'] + 1):
        if failures >= plan['patience']:
            break
        if STOP:
            raise InterruptedError('Stop requested; resume from last completed checkpoint')
        started = time.monotonic()
        model.train()
        permutation = torch.randperm(count, device=device)
        negatives = sample_negatives(users, known, config['item_num'])
        total = torch.zeros((), device=device)
        for offset in range(0, count, config['batch_size']):
            if STOP:
                raise InterruptedError('Stop requested; incomplete epoch is not checkpointed')
            batch = permutation[offset:offset + config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            interaction = dict(user_ids=users[batch], item_ids=items[batch], neg_item_ids=negatives[batch])
            loss = model.compute_loss(interaction).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            invalidate(model)
            total += loss.detach() * len(batch)
        record = dict(epoch=epoch, loss=float(total/count), seconds=time.monotonic()-started)
        if epoch == 1 or epoch % plan['eval_every'] == 0 or epoch == plan['max_epochs']:
            metrics, _ = evaluate(model, ranking, device, config['eval_batch_size'])
            record.update(valid=metrics, **embedding_rms(model))
            if metrics['ndcg@20'] > best:
                best, best_epoch, failures = metrics['ndcg@20'], epoch, 0
                save_checkpoint(directory / 'best.pt', dict(model=compact_state(model), config=config,
                                                          epoch=epoch, valid=metrics))
            else:
                failures += 1
        record.update(best_valid_ndcg20=best, best_epoch=best_epoch,
                      validations_without_improvement=failures, updated=now())
        record['seconds'] = time.monotonic()-started
        history.append(record); completed_epoch = epoch
        # Every epoch is resumable, with optimizer and RNG states preserved.
        save_checkpoint(directory / 'last.pt', dict(model=compact_state(model), config=config,
            optimizer=optimizer.state_dict(), epoch=epoch, best=best, best_epoch=best_epoch,
            failures=failures, history=history, rng=rng_state()))
        write_json(directory / 'history.json', history)
        write_json(directory / 'status.json', dict(state='training', **record))
        if epoch == 1 or epoch % plan['eval_every'] == 0 or epoch == plan['max_epochs']:
            print(config['model'], dataset, config['trial'], json.dumps(record), flush=True)
    assert best_epoch > 0
    result = dict(model=config['model'], dataset=dataset, trial=config['trial'],
        best_epoch=best_epoch, best_valid_ndcg20=best, epochs=completed_epoch, config=config,
        stop_reason='patience' if failures >= plan['patience'] else 'max_epochs',
        checkpoint=str((directory / 'best.pt').relative_to(run)))
    write_json(directory / 'result.json', result)
    write_json(directory / 'status.json', dict(state='complete', **result))
    return result
