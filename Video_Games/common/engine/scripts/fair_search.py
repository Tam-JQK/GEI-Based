"""Training and evaluation kernels used by the parameter search."""
import json, random, time
from datetime import datetime, timezone
import cvxpy as cp
import numpy as np
import torch
from analysis.artifacts import write_json
from base_model.lightgcn import LightGCN
from base_model.sasrec import SASRec
from rank_model import IPS, SDRO, FOCF, FairDual, FairDualPaper
from scripts.lightgcn_search import load_graph, sample_negatives
from scripts.selected_baselines_search import compact_state, restore_model, invalidate, evaluate as lightgcn_evaluate
from scripts.sasrec_search import read_training, seed_all, save_checkpoint, rng_state, restore_rng, evaluate as sasrec_evaluate

STOP = False

def now():
    return datetime.now(timezone.utc).isoformat()


class FastFairDual(FairDual):
    """FairDual with the same projection, solved as a sparse chain QP.

    The legacy map_layer rebuilds and canonicalizes a dense triangular
    projection at every batch.  The mathematically identical formulation
    below uses cumulative prefix sums (a sparse chain), keeps the compiled
    problem and treats the target vector as a cvxpy Parameter.  This changes
    only solver mechanics, not the objective or constraints.
    """

    def __init__(self, config, group_weight):
        super().__init__(config, group_weight)
        self._projection_cache = {}

    def map_layer(self, ordered_tilde_dual, rho, lambd):
        key = (len(rho), rho.tobytes(), float(lambd))
        cached = self._projection_cache.get(key)
        if cached is None:
            m = len(rho)
            parameter = cp.Parameter(m)
            answer = cp.Variable(m)
            cumulative = cp.Variable(m)
            objective = cp.Minimize(
                cp.sum_squares(cp.multiply(rho, answer) -
                               cp.multiply(rho, parameter)))
            constraints = [
                cumulative[0] == rho[0] * answer[0],
                cumulative[1:] == cumulative[:-1] + cp.multiply(rho[1:], answer[1:]),
                cumulative >= -float(lambd),
            ]
            problem = cp.Problem(objective, constraints)
            cached = (problem, parameter, answer)
            self._projection_cache[key] = cached
        problem, parameter, answer = cached
        parameter.value = np.asarray(ordered_tilde_dual, dtype=float)
        problem.solve(solver=cp.CLARABEL, warm_start=True)
        value = answer.value
        if (problem.status not in ('optimal', 'optimal_inaccurate') or value is None
                or not np.all(np.isfinite(value))):
            raise RuntimeError(f'FairDual projection failed: {problem.status}')
        return value


def read_json(path):
    return json.loads(Path(path).read_text())


def build_model(config, graph=None):
    if config['model'] == 'lightgcn':
        return LightGCN(config, graph=graph)
    if config['model'] == 'sasrec':
        return SASRec(config)
    raise ValueError(config['model'])


def make_ranker(config):
    name = config['rank_model']
    weights = np.ones(config['group_num'], dtype=np.float64)
    cls = {'IPS': IPS, 'SDRO': SDRO, 'FOCF': FOCF,
           'FairDual': FastFairDual, 'FairDualPaper': FairDualPaper}[name]
    return cls(config, weights)


def ranker_state(ranker):
    hidden = ('config', 'M', 'M_tensor', 'iid2pid', 'group_weight',
              '_projection_cache')
    return {key: value for key, value in vars(ranker).items() if key not in hidden}


def restore_ranker_state(ranker, state):
    for key, value in state.items():
        setattr(ranker, key, value)


def evaluate_model(model, ranking, device):
    if isinstance(model, LightGCN):
        return lightgcn_evaluate(model, ranking, device, model.config['eval_batch_size'])
    return sasrec_evaluate(model, ranking, device, model.config['eval_batch_size'])


def graph_cache_run(run):
    return run / 'cache' / 'lightgcn'


def sequential_cache_run(run):
    return run / 'cache' / 'sasrec'


def train_data(root, run, dataset, config, device):
    if config['model'] == 'lightgcn':
        graph, users, items = load_graph(root, graph_cache_run(run), dataset,
                                        config, device)
        assert len(users) == len(items)
        return dict(kind='pair', graph=graph, users=users, items=items,
                    known=users * config['item_num'] + items, count=len(users))
    training = {key: value.to(device)
                for key, value in read_training(root, sequential_cache_run(run),
                                                dataset, config).items()}
    return dict(kind='sequential', training=training, count=len(training['target']))


def run_epoch(model, optimizer, ranker, config, data, epoch, device,
              max_batches=None):
    """One training epoch; every fair-method mutation matches trainer.py."""
    method = config['method']
    if method in ('IPS', 'SDRO'):
        if (epoch - 1) % config['update_epoch'] == 0:
            ranker.reset_parameters()
    elif method == 'FairDual':
        if (epoch - 1) % config['update_epoch'] == 0:
            ranker.reset_parameters(data['count'])
    elif method != 'FOCF':
        raise ValueError(method)
    model.train()
    count = data['count']
    if data['kind'] == 'pair':
        permutation = torch.randperm(count, device=device)
        negatives = sample_negatives(data['users'], data['known'], config['item_num'])
    else:
        permutation = torch.randperm(count, device=device)
    total = torch.zeros((), device=device)
    seen = batches = 0
    diagnostics = {}
    for offset in range(0, count, config['batch_size']):
        if STOP:
            raise InterruptedError('Stop requested; last complete epoch is resumable')
        if max_batches is not None and batches >= max_batches:
            break
        index = permutation[offset:offset + config['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        if data['kind'] == 'pair':
            interaction = dict(user_ids=data['users'][index],
                               item_ids=data['items'][index],
                               neg_item_ids=negatives[index])
            feed_user = dict(user_ids=data['users'][index])
            feed_items = data['items'][index]
        else:
            interaction = dict(history_ids=data['training']['history'][index],
                               history_mask=data['training']['mask'][index],
                               item_ids=data['training']['target'][index])
            feed_user = dict(history_ids=data['training']['history'][index],
                             history_mask=data['training']['mask'][index])
            feed_items = data['training']['target'][index]
        losses = model.compute_loss(interaction)
        if method in ('IPS', 'SDRO'):
            weight = ranker.reweight(input_dict=dict(
                items=feed_items.detach().cpu().numpy(),
                loss=losses.detach().cpu().numpy()))
            weight_t = torch.as_tensor(weight, dtype=losses.dtype, device=device)
            if not torch.isfinite(weight_t).all():
                raise ValueError('Nonfinite fairness weights')
            loss = torch.mean(weight_t * losses)
            diagnostics = dict(weight_min=float(np.min(weight)),
                               weight_mean=float(np.mean(weight)),
                               weight_max=float(np.max(weight)))
        elif method == 'FairDual':
            sample = torch.tensor(
                random.sample(range(config['item_num']), config['exposure_sample_num']),
                dtype=torch.long, device=device)
            _, indices = model.full_ranking(feed_user, sample, k=config['s_k'])
            weight = ranker.reweight(input_dict=dict(
                items=feed_items.detach().cpu().numpy(),
                sample_items=sample[indices].detach().cpu().numpy()))
            weight_t = torch.as_tensor(weight, dtype=losses.dtype, device=device)
            if not torch.isfinite(weight_t).all():
                raise ValueError('Nonfinite fairness weights')
            loss = torch.mean(weight_t * losses)
            diagnostics = dict(weight_min=float(np.min(weight)),
                               weight_mean=float(np.mean(weight)),
                               weight_max=float(np.max(weight)))
        elif method == 'FOCF':
            scores = model(feed_user, feed_items)
            fair_loss = ranker.fairness_loss(
                dict(items=feed_items, loss=losses, scores=scores))
            loss = torch.mean(losses) + config['fair_lambda'] * fair_loss
            diagnostics = dict(fair_lambda=config['fair_lambda'],
                               fairness_loss=float(fair_loss.detach()))
        else:
            raise ValueError(method)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        if data['kind'] == 'sequential':
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['gradient_clip_norm'],
                error_if_nonfinite=True)
        optimizer.step()
        if data['kind'] == 'pair':
            invalidate(model)
        total += loss.detach() * len(index)
        seen += len(index)
        batches += 1
    return float(total / max(seen, 1)), batches, diagnostics


def train_trial(run, dataset, backbone, config, data, ranking, plan, device,
                budget, warmup):
    directory = run / backbone / dataset / config['trial']
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        return read_json(directory / 'result.json')
    write_json(directory / 'config.json', config)
    seed_all(config['seed'])
    graph = data.get('graph')
    model = build_model(config, graph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    ranker = make_ranker(config)
    best, best_epoch, start_epoch, history = -1., 0, 1, []
    if (directory / 'last.pt').exists():
        saved = torch.load(directory / 'last.pt', map_location=device, weights_only=False)
        assert saved['config'] == config
        if config['model'] == 'lightgcn':
            restore_model(model, saved['model'])
        else:
            model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        best, best_epoch = saved['best'], saved['best_epoch']
        start_epoch, history = saved['epoch'] + 1, saved['history']
        if config['method'] in ('IPS', 'SDRO', 'FairDual'):
            assert saved.get('ranker'), 'Missing ranker state for fair resume'
            restore_ranker_state(ranker, saved['ranker'])
        restore_rng(saved['rng'])
        print('RESUME', dataset, backbone, config['trial'], start_epoch, flush=True)
    completed_epoch = start_epoch - 1
    eval_every = plan['fixed']['eval_every']
    for epoch in range(start_epoch, budget + 1):
        if STOP:
            raise InterruptedError('Stop requested; last complete epoch is resumable')
        started = time.monotonic()
        loss, batches, diagnostics = run_epoch(
            model, optimizer, ranker, config, data, epoch, device)
        record = dict(epoch=epoch, loss=loss, batches=batches, **diagnostics)
        if epoch <= warmup or epoch % eval_every == 0 or epoch == budget:
            metrics, _ = evaluate_model(model, ranking, device)
            record['valid'] = metrics
            if metrics['ndcg@20'] > best:
                best, best_epoch = metrics['ndcg@20'], epoch
                save_checkpoint(directory / 'best.pt', dict(
                    model=compact_state(model) if config['model'] == 'lightgcn'
                    else model.state_dict(),
                    config=config, epoch=epoch, valid=metrics))
        record.update(seconds=time.monotonic() - started,
                      best_valid_ndcg20=best, best_epoch=best_epoch, updated=now())
        history.append(record)
        completed_epoch = epoch
        save_checkpoint(directory / 'last.pt', dict(
            model=compact_state(model) if config['model'] == 'lightgcn'
            else model.state_dict(),
            optimizer=optimizer.state_dict(), config=config, epoch=epoch,
            best=best, best_epoch=best_epoch, history=history,
            rng=rng_state(), ranker=ranker_state(ranker)))
        write_json(directory / 'history.json', history)
        write_json(directory / 'status.json', dict(state='training', **record))
        if epoch == 1 or epoch % eval_every == 0 or epoch == budget:
            print(dataset, backbone, config['trial'], json.dumps(record), flush=True)
    if best_epoch == 0:
        raise RuntimeError('No validation checkpoint was selected')
    result = dict(dataset=dataset, backbone=backbone, method=config['method'],
                  trial=config['trial'], best_epoch=best_epoch,
                  best_valid_ndcg20=best, epochs=completed_epoch,
                  budget=budget, stop_reason='fixed_budget',
                  checkpoint=str((directory / 'best.pt').relative_to(run)),
                  config=config)
    write_json(directory / 'result.json', result)
    write_json(directory / 'status.json', dict(state='complete', **result))
    return result
