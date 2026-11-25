"""Search a single dataset and backbone using validation data."""
import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / 'common/engine'
POST = ROOT / 'post-processing'
sys.path[:0] = [str(ENGINE), str(POST), str(POST / 'vendor')]

import numpy as np
import torch
import yaml
from analysis.artifacts import file_hash, write_json
from analysis.evaluation import fairness
from exposure import discounted_provider_exposure, load_provider_matrix
from post_rank import Reranker
from scripts import fair_search as fair
from scripts import lightgcn_search as graph_engine
from scripts import sasrec_search as sas
from scripts import selected_baselines_search as light

IN_METHODS = {'ips': 'IPS', 'sdro': 'SDRO', 'focf': 'FOCF', 'fairdual': 'FairDual'}
POST_METHODS = ('cpfair', 'pmmf', 'elasticrank', 'fairrec')
METHODS = ('baseline', *IN_METHODS, *POST_METHODS)
BACKBONES = ('lightgcn', 'sasrec')


def read_json(path):
    return json.loads(path.read_text())


def trial_id(values):
    key = json.dumps(values, sort_keys=True).encode()
    return hashlib.sha256(key).hexdigest()[:16]


def grid_values(grid):
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f'{key} must be a non-empty list of candidates')
    return [dict(zip(grid, point)) for point in itertools.product(*grid.values())]


def metrics_for_items(items, ranking, matrix, partitions, protocol):
    result = {}
    n = len(items)
    catalog_sizes = np.asarray(matrix.sum(axis=0)).ravel()
    for k in protocol['cutoffs']:
        ids = items[:, :k]
        hits = np.asarray(ranking['truth'][np.repeat(np.arange(n), k), ids.ravel()]).reshape(n, k)
        result.update(sas.accuracy(hits, ranking['counts'], cutoffs=(k,)))
        values = discounted_provider_exposure(matrix, ids, k) / catalog_sizes
        result.update({f'{key}@{k}': value for key, value in fairness(
            values, partitions, alphas=protocol['ge_alphas'], eta=protocol['ge_eta'],
            mmf_ratio=protocol['mmf_ratio']).items()})
    return result


def retention(metrics, baseline_metrics):
    denominator = baseline_metrics['ndcg@20']
    return metrics['ndcg@20'] / denominator if denominator > 0 else None


class Study:
    def __init__(self, args, protocol):
        self.args = args
        self.protocol = protocol
        self.dataset = protocol['dataset']
        self.data = args.processed_dir or ROOT / 'data/processed' / self.dataset
        self.config = yaml.safe_load((self.data / 'process_config.yaml').read_text())
        if self.config['dataset'] != self.dataset:
            raise ValueError('The processed data does not match this dataset project')
        self.config['processed_data_dir'] = str(self.data)
        self.matrix = load_provider_matrix(self.data, self.config['item_num'],
                                           self.config['group_num'])
        self.partitions = {kind: {int(k): int(v) for k, v in
                           read_json(self.data / f'pid2mgid_{kind}.json').items()}
                           for kind in ('size', 'freq')}
        self.device = args.device
        self.output = args.output
        self.run = self.output / args.method / args.backbone
        self.baseline = args.baseline_output / 'baseline' / args.backbone
        self.valid = self.ranking('valid')
        self.input_protocol = dict(
            config_sha256=file_hash(self.data / 'process_config.yaml'),
            candidate_sha256={split: file_hash(self.data / 'evaluation/v1_full_seed2025'
                              / f'{self.dataset}.{split}.ranking') for split in ('valid', 'test')},
            train_sha256=file_hash(self.data / self.config['train_file']),
            sequential_sha256=file_hash(self.data / self.config['sequential_train_file']),
            provider_files_sha256={name: file_hash(self.data / name) for name in
                                   ('iid2pid.json', 'providers.npz', 'pid2mgid_size.json', 'pid2mgid_freq.json')},
            limit_users=args.limit_users,
        )
        self.cache = self.output / 'cache' / trial_id(self.input_protocol) / args.backbone

    def ranking(self, split):
        return sas.read_ranking(ENGINE, self.dataset, self.config, split,
                                limit=self.args.limit_users)

    def metrics(self, ids, ranking):
        return metrics_for_items(ids, ranking, self.matrix, self.partitions, self.protocol)

    def base_selection(self):
        result = read_json(self.baseline / 'selection.json')
        if result['dataset'] != self.dataset or result['input_protocol'] != self.input_protocol:
            raise ValueError('The baseline must use the same dataset and evaluation inputs')
        missing = set(self.args.seeds) - {r['seed'] for r in result['runs']}
        if missing:
            raise ValueError(f'Missing selected baseline checkpoints for seeds {sorted(missing)}')
        return {r['seed']: r for r in result['runs']}

    def model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = dict(checkpoint['config'], processed_data_dir=str(self.data))
        graph = None
        if config['model'] == 'lightgcn':
            graph, _, _ = graph_engine.load_graph(ENGINE, fair.graph_cache_run(self.cache), self.dataset,
                                                   config, self.device)
        model = fair.build_model(config, graph).to(self.device)
        light.restore_model(model, checkpoint['model'])
        model.eval()
        return model

    def evaluate_checkpoint(self, checkpoint_path, ranking, output):
        model = self.model(checkpoint_path)
        _, arrays = fair.evaluate_model(model, ranking, self.device)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / 'recommendations.npz', **arrays)
        result = self.metrics(arrays['items'], ranking)
        write_json(output / 'metrics.json', result)
        return result

    @torch.no_grad()
    def scores(self, seed, base, ranking, split):
        checkpoint = self.baseline / base['checkpoint']
        metadata = dict(checkpoint_sha256=file_hash(checkpoint),
                        candidates_sha256=ranking['source_sha256'],
                        rows=len(ranking['users']), device=self.device)
        cache_id = trial_id(dict(input_protocol=self.input_protocol, **metadata))
        path = self.output / 'scores' / self.args.backbone / f'seed{seed}' / split / f'{cache_id}.npy'
        sidecar = path.with_suffix('.json')
        if path.is_file() and sidecar.is_file() and read_json(sidecar) == metadata:
            return np.load(path, mmap_mode='r')
        model = self.model(checkpoint)
        cfg = model.config
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f'.{os.getpid()}.tmp.npy')
        shape = (len(ranking['users']), cfg['item_num'])
        try:
            values = np.lib.format.open_memmap(temp, mode='w+', dtype='float32', shape=shape)
            batch_size = cfg['eval_batch_size']
            catalog = torch.arange(cfg['item_num'], device=self.device)
            for start in range(0, shape[0], batch_size):
                stop = min(start + batch_size, shape[0])
                if cfg['model'] == 'lightgcn':
                    interaction = dict(user_ids=torch.as_tensor(ranking['users'][start:stop], device=self.device))
                    scores = model.full_scores(interaction, catalog)
                else:
                    interaction = dict(history_ids=ranking['history'][start:stop].to(self.device),
                                       history_mask=ranking['mask'][start:stop].to(self.device))
                    scores = model.get_user_embedding(interaction) @ model.item_embedding.weight[:shape[1]].T
                values[start:stop] = scores.cpu().numpy()
            values.flush()
            del values
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        write_json(sidecar, metadata)
        return np.load(path, mmap_mode='r')

    def train(self, params, seed):
        args = self.args
        model_values = {key: value for key, value in params.items() if key not in ('epochs', 'eval_every')}
        config = dict(self.config, **model_values, seed=seed, model=args.backbone, device=self.device)
        if args.method != 'baseline':
            config['method'] = IN_METHODS[args.method]
        signature = dict(config=config, epochs=params['epochs'], eval_every=params['eval_every'],
                         input=self.input_protocol, warmup=args.fair_eval_warmup)
        if args.method == 'baseline':
            signature.update(targets=None, baseline_ndcg=None)
        config['trial'] = 'trial_' + trial_id(signature)
        plan = dict(max_epochs=params['epochs'], eval_every=params['eval_every'],
                    patience=params['epochs'] + 1, fixed=dict(seed=seed, eval_every=params['eval_every']))
        trial_root = self.run / 'trials'
        if args.method != 'baseline':
            data = fair.train_data(ENGINE, self.cache, self.dataset, config, self.device)
            selected = fair.train_trial(trial_root, self.dataset, args.backbone, config,
                                        data, self.valid, plan, self.device, params['epochs'],
                                        args.fair_eval_warmup)
        elif args.backbone == 'lightgcn':
            graph, users, items = graph_engine.load_graph(ENGINE, fair.graph_cache_run(self.cache), self.dataset,
                                                         config, self.device)
            selected = light.train_trial(trial_root, self.dataset, config, graph, users, items,
                                         self.valid, plan, self.device)
        else:
            training = {key: value.to(self.device) for key, value in
                        sas.read_training(ENGINE, fair.sequential_cache_run(self.cache), self.dataset, config).items()}
            selected = sas.train_trial(trial_root, self.dataset, config, training, self.valid,
                                       plan, self.device)
        selected['checkpoint'] = str(Path('trials') / selected['checkpoint'])
        return selected

    def summarize(self, picked, base=None):
        rows = picked['runs']
        result = dict(dataset=self.dataset, method=self.args.method, backbone=self.args.backbone,
                      input_protocol=self.input_protocol, params=picked['params'], runs=rows,
                      mean_valid_ndcg20=float(np.mean([r['valid_metrics']['ndcg@20'] for r in rows])),
                      mean_test_ndcg20=float(np.mean([r['test_metrics']['ndcg@20'] for r in rows])))
        if base is not None:
            for split in ('valid', 'test'):
                ratios = [row[f'{split}_retention'] for row in rows]
                result[f'mean_{split}_retention'] = (float(np.mean(ratios))
                                                     if None not in ratios else None)
        return result

    def training_search(self, candidates, base=None):
        records = []
        for params in candidates:
            rows = []
            for seed in self.args.seeds:
                trial = self.train(params, seed)
                checkpoint = torch.load(self.run / trial['checkpoint'], map_location='cpu', weights_only=False)
                row = dict(seed=seed, epoch=trial['best_epoch'],
                           checkpoint=trial['checkpoint'], valid_metrics=checkpoint['valid'])
                if base is not None:
                    row['valid_retention'] = retention(row['valid_metrics'], base[seed]['valid_metrics'])
                rows.append(row)
            records.append(dict(params=params, runs=rows))
        self.finalize_search(records, base)

    def post_search(self, candidates, base):
        records = [dict(params=params, runs=[]) for params in candidates]
        for seed in self.args.seeds:
            score = self.scores(seed, base[seed], self.valid, 'valid')
            reranker = Reranker(self, self.valid, score, 'valid')
            for record in records:
                ids, diagnostics = reranker.rank(record['params'])
                metrics = self.metrics(ids, self.valid)
                record['runs'].append(dict(seed=seed, valid_metrics=metrics,
                                           valid_retention=retention(metrics, base[seed]['valid_metrics']),
                                           valid_diagnostics=diagnostics))
            del reranker, score
        self.finalize_search(records, base)

    def finalize_search(self, records, base):
        write_json(self.run / 'validation_trials.json', records)
        best = max(records, key=lambda record: np.mean(
            [row['valid_metrics']['ndcg@20'] for row in record['runs']]))
        picked = dict(params=best['params'], runs=[dict(row) for row in best['runs']])
        test = self.ranking('test')
        for row in picked['runs']:
            seed = row['seed']
            output = self.run / 'test' / f'seed{seed}'
            if 'checkpoint' in row:
                metrics = self.evaluate_checkpoint(self.run / row['checkpoint'], test, output)
            else:
                score = self.scores(seed, base[seed], test, 'test')
                ids, diagnostics = Reranker(self, test, score, 'test').rank(picked['params'])
                metrics = self.metrics(ids, test)
                output.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output / 'recommendations.npz', users=test['users'], items=ids)
                write_json(output / 'metrics.json', metrics)
                row['test_diagnostics'] = diagnostics
            row['test_metrics'] = metrics
            if base is not None:
                row['test_retention'] = retention(metrics, base[seed]['test_metrics'])
        write_json(self.run / 'selection.json', self.summarize(picked, base))


def main():
    protocol = read_json(ROOT / 'configs/protocol.json')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--backbone', required=True, choices=BACKBONES)
    parser.add_argument('--device', help='cpu or cuda:N; auto-selects when omitted')
    parser.add_argument('--output', type=Path, help='Directory for newly generated results')
    parser.add_argument('--processed-dir', type=Path, help='Override the processed dataset folder')
    parser.add_argument('--baseline-output', type=Path, help='Root of a baseline search (default: --output)')
    parser.add_argument('--config', type=Path, help='Alternative JSON parameter grid for this method and backbone')
    parser.add_argument('--seed', action='append', type=int, help='Override seeds; repeat for multiple seeds')
    parser.add_argument('--fair-eval-warmup', type=int, help='Number of early in-processing epochs evaluated individually')
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    parser.add_argument('--epochs', type=int, help='Override epoch budget for a quick check')
    parser.add_argument('--max-configs', type=int, help='Only run the first N points in grid order')
    parser.add_argument('--limit-users', type=int, help='Limit each evaluation split for a quick check')
    parser.add_argument('--dry-run', action='store_true', help='Print parameter combinations without training or scoring')
    args = parser.parse_args()
    args.seeds = list(dict.fromkeys(args.seed or protocol['seeds']))
    args.fair_eval_warmup = (args.fair_eval_warmup if args.fair_eval_warmup is not None
                            else protocol['fair_eval_warmup'][0])
    grid = read_json(args.config or ROOT / f'configs/{args.method}_{args.backbone}.json')
    if args.epochs is not None:
        if args.epochs < 1 or args.method in POST_METHODS:
            parser.error('--epochs requires a positive integer and a training method')
        grid['epochs'] = [args.epochs]
    candidates = grid_values(grid)
    if args.max_configs is not None:
        if args.max_configs < 1:
            parser.error('--max-configs must be positive')
        candidates = candidates[:args.max_configs]
    if args.dry_run:
        print(json.dumps(dict(dataset=protocol['dataset'], method=args.method, backbone=args.backbone,
                              seeds=args.seeds, configurations=candidates), indent=2))
        return
    if args.output is None:
        parser.error('--output is required unless --dry-run is used')
    args.output = args.output.resolve()
    args.processed_dir = args.processed_dir.resolve() if args.processed_dir else None
    args.baseline_output = args.baseline_output.resolve() if args.baseline_output else args.output
    args.device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable, check PyTorch or use --device cpu')
        torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(args.threads)
    os.chdir(ENGINE)
    study = Study(args, protocol)
    if args.method == 'baseline':
        study.training_search(candidates)
    elif args.method in IN_METHODS:
        study.training_search(candidates, study.base_selection())
    else:
        study.post_search(candidates, study.base_selection())
    print(f'Results written to {study.run}', flush=True)


if __name__ == '__main__':
    main()
