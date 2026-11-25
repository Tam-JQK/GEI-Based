# GEI for Provider Fairness in Recommendation

Train recommendation backbones, search in-processing fairness methods, and rerank recommendations on three datasets. Each dataset project includes its own code, configuration, and data.

| Dataset | Guide in this README | Standalone guide |
| --- | --- | --- |
| Steam | [Steam](#steam-recommendation-and-provider-fairness) | [Steam/README.md](Steam/README.md) |
| Amazon Video Games | [Amazon Video Games](#amazon-video-games-recommendation-and-provider-fairness) | [Video_Games/README.md](Video_Games/README.md) |
| BeerAdvocate | [BeerAdvocate](#beeradvocate-recommendation-and-provider-fairness) | [BeerAdvocate/README.md](BeerAdvocate/README.md) |

Run each guide's commands from its dataset directory. The guides cover environment setup, data preparation, hyperparameter grids, and the evaluation protocol. Train the LightGCN and SASRec baselines before running a fairness search. Specify `--output` to choose the destination for each run.

## Steam Recommendation and Provider Fairness

This project includes data preparation, training with LightGCN and SASRec, fair in-processing, post-processing reranking, and evaluation on Steam. Both backbones use the same data splits, item catalog, candidate rules, and provider mappings. Run the commands below from this project's directory.

### 1. Project layout

| Path | Contents |
| --- | --- |
| `data/raw/steam/` | Tab-separated interaction and item tables used by preprocessing |
| `data/processed/steam/` | Training, validation, test, and candidate files |
| `preprocessing/` | Input parsing, filtering, splitting, ID mapping, and candidate generation |
| `backbones/base_model/` | LightGCN and SASRec |
| `in-processing/rank_model/` | IPS, SDRO, FOCF, and FairDual |
| `post-processing/rerank_model/` | CPFair, PMMF, ElasticRank, and FairRec |
| `post-processing/vendor/` | Accelerated reranking implementations |
| `evaluation/` | Candidate generation, exposure, and metrics |
| `common/engine/` | Data loading, training, grid search, score export, and reranking |
| `configs/` | Hyperparameter grids and evaluation protocol |
| `scripts/` | Bash entry points for preparation and search |

All three dataset projects use the same command-line interface. When moving a project, preserve its directory structure and relative symbolic links with a tool such as `cp -a` or `tar`.

### 2. Environment

Use Linux, Bash, Python 3.10, and a C++17-compatible `g++`. FairRec compiles native code on first use and places the shared libraries in `native/` under the output directory. The Conda environment installs the CUDA 12.1 build of PyTorch.

```bash
conda env create -f environment.yml
conda activate final-gei
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
g++ --version
```

For CPU-only execution, install the CPU build of PyTorch.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.7.2 PyYAML==6.0.3 tqdm==4.67.1 cvxpy==1.6.5
```

Set `PYTHON_BIN` to use a specific Python executable. Pass `--device cpu` to run on CPU or `--device cuda:0` to use the first visible GPU. By default, the runner uses `cuda:0` when CUDA is available and CPU otherwise. If you set `CUDA_VISIBLE_DEVICES`, CUDA indices are relative to that list. Set `--threads` to control the number of PyTorch CPU threads.

LightGCN and SASRec can train on a GPU. FairDual can score with the backbone on GPU and uses a CPU solver for provider-level dual projection. CPFair, PMMF, and ElasticRank run on CPU in this project. FairRec can sort scores on GPU.

### 3. Data preparation and splits

The training scripts load `data/processed/steam/` by default. The `process_config.yaml` file defines column names, ID ranges, input filenames, and the evaluation protocol. Graph construction, sequence training, and candidate generation share the same splits and ID mappings.

| File | Contents |
| --- | --- |
| `steam.train` | Complete training prefix. LightGCN constructs the user–item graph from its positive interactions |
| `steam.train.sequential` | Positive training targets with preceding context for SASRec |
| `steam.valid.CTR`, `steam.test.CTR` | Validation and test interactions |
| `evaluation/v1_full_seed2025/*.ranking` | Positive items, exclusions, and history for each evaluation user |
| `uid2original.json`, `iid2original.json` | Mappings from internal indices to source user and item IDs |
| `iid2pid.json`, `pid2provider.json`, `providers.npz` | Item-to-provider assignments, provider labels, and the sparse membership matrix |
| `pid2mgid_size.json`, `pid2mgid_freq.json` | Provider groups based on catalog size and training interaction frequency |

Rebuilding the splits uses the interaction and item tables in `data/raw/steam/`. These tables come from the RecBole Steam merged package. The item ID column is named `product_id:token`. [RecBole Steam data](https://drive.google.com/drive/folders/1PUsk-0rsRgea7wdeI4-vA8iRtEAUEfeK)

Preprocessing uses the publisher field (`publisher:token`) to identify providers. Interactions with at least 10 play hours are positive. The preprocessing script deduplicates user–item pairs and applies the user, item, and provider filters iteratively. The provider filter counts distinct items. After filtering, providers with fewer than 10 items in the catalog are grouped into a single category.

The splits are chronological per user. The latest holdout segment is the test split, the equally sized preceding segment is the validation split, and the remaining prefix is used for training. Each holdout contains 10% of the user's interactions, rounded down to a minimum of one. Each retained user has at least two positive interactions in the training prefix.

SASRec uses strictly earlier positive training interactions as context for training targets. Validation and test both use the last five positive interactions from the user's training prefix. Short histories use a padding ID equal to the item count. The padding ID lies outside the item catalog and `history_mask` masks the padded positions.

To rebuild the splits, run the following commands. The preparation script checks each input against `data/raw/steam/SHA256SUMS.json`. It writes to a new directory, which you can pass to each runner using `--processed-dir`.

```bash
bash scripts/prepare_data.sh --output ../prepared-steam

bash scripts/run_baseline.sh --backbone lightgcn --device cuda:0 \
  --processed-dir ../prepared-steam/steam --output ../runs-steam-rebuilt
```

When using rebuilt splits, pass the same `--processed-dir` to the baseline, in-processing, and post-processing scripts. The runner checks dataset and candidate checksums when it loads a baseline for a fairness method.

### 4. Hyperparameter grids

Each method has a JSON configuration for each backbone, such as `configs/ips_lightgcn.json` and `configs/ips_sasrec.json`. Specify each hyperparameter as a nonempty list. The runner expands the Cartesian product of these lists and runs each combination for the selected seeds. A fixed setting is still a list, for example `"epochs": [1000]`.

| Training parameter | Meaning |
| --- | --- |
| `epochs` | Maximum number of training epochs per configuration. The default grid is `[1000]` |
| `batch_size` | Number of training samples per optimizer step |
| `eval_batch_size` | Number of evaluation users scored in a batch |
| `embedding_size` | Embedding dimension. SASRec requires a multiple of `n_heads` |
| `learning_rate`, `weight_decay` | Adam learning rate and weight decay |
| `GCN_layer`, `reg_lambda` | LightGCN propagation depth and regularization coefficient |
| `n_layers`, `n_heads`, `inner_size` | SASRec Transformer depth, attention heads, and feed-forward width |
| `hidden_dropout_prob`, `attn_dropout_prob` | SASRec hidden and attention dropout probabilities |
| `eval_every` | Validation interval in epochs |

| Fairness method | Main search parameters |
| --- | --- |
| IPS | `M`, `update_epoch`, and training parameters |
| SDRO | `eta`, `update_epoch`, and training parameters |
| FOCF | `fair_lambda` and training parameters |
| FairDual | `alpha`, `eta`, `lambd`, `s_k`, `exposure_sample_num`, and `update_epoch` |
| CPFair | `lambda` |
| PMMF | `lambda`, `learning_rate`, and `gamma` |
| ElasticRank | `lambda`, `t`, and `anchor_rate` |
| FairRec | `alpha` |

FairDual uses `s_k` as the ranking cutoff for estimating exposure during training. It must be at most `exposure_sample_num`, which must be at most the catalog size. The method's `alpha` and `eta` are independent of the GEI sensitivity and smoothing parameters.

`configs/protocol.json` defines the default seeds and evaluation settings. Repeat `--seed` to specify multiple seeds. Use `--config` to supply a different JSON grid and `--dry-run` to preview the combinations.

```bash
bash scripts/run_ips.sh --backbone lightgcn --dry-run
bash scripts/run_fairdual.sh --backbone sasrec --dry-run \
  --seed 2025 --seed 2026
```

### 5. Running the experiments

#### 5.1 Train the backbones

```bash
OUT="../runs-steam"

bash scripts/run_baseline.sh --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_baseline.sh --backbone sasrec --device cuda:0 --output "$OUT"
```

Each configuration and seed saves the checkpoint with the highest validation NDCG@20 as `best.pt`. The runner selects the configuration with the highest mean validation NDCG@20 across seeds, then evaluates its checkpoints on the test split. Fairness methods use the corresponding backbone and seed as their NDCG@20 reference on each split.

#### 5.2 Search in-processing methods

```bash
bash scripts/run_ips.sh      --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_sdro.sh     --backbone sasrec   --device cuda:0 --output "$OUT"
bash scripts/run_focf.sh     --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_fairdual.sh --backbone sasrec   --device cuda:0 --output "$OUT"
```

Each method works with both backbones. The runner trains from random initialization for each configuration and seed. By default, it validates every epoch for the first 30 epochs, at the `eval_every` interval after that, and at the final epoch. Each configuration and seed saves the checkpoint with the highest validation NDCG@20 as `best.pt`. The runner selects the configuration with the highest mean validation NDCG@20 across seeds and evaluates its checkpoints on the test split.

#### 5.3 Search post-processing methods

```bash
bash scripts/run_cpfair.sh      --backbone lightgcn --device cpu --output "$OUT"
bash scripts/run_pmmf.sh        --backbone sasrec   --device cpu --output "$OUT"
bash scripts/run_elasticrank.sh --backbone lightgcn --device cpu --output "$OUT"
bash scripts/run_fairrec.sh     --backbone sasrec   --device cpu --output "$OUT"
```

The runner generates full-catalog validation scores from the selected backbone checkpoints and reranks them using the same candidate rules for every configuration. It selects the configuration with the highest mean validation NDCG@20 across seeds. The runner then generates test scores and evaluates the selected configuration. Compatible runs can share the frozen score cache.

If you saved the baseline in a different directory, pass its root through `--baseline-output`. `--output` controls where the reranking results are written.

```bash
bash scripts/run_pmmf.sh --backbone lightgcn --device cpu \
  --baseline-output ../runs-steam --output ../pmmf-steam
```

#### 5.4 Run a quick end-to-end check

The following commands train for one epoch with one seed and evaluate the first 32 users of each evaluation split. The check uses a separate output directory. For a full search, use the commands in sections 5.1–5.3.

```bash
CHECK="../check-steam"

bash scripts/run_baseline.sh --backbone lightgcn --device cpu --output "$CHECK" \
  --seed 2025 --epochs 1 --max-configs 1 --limit-users 32

bash scripts/run_cpfair.sh --backbone lightgcn --device cpu --output "$CHECK" \
  --seed 2025 --max-configs 1 --limit-users 32
```

### 6. Evaluation protocol

- **Candidates.** The candidate set is the full item catalog after excluding known positives in the other splits. The evaluation includes users with a positive item in the current split. The backbones rank by descending score and break ties by ascending item ID. Rerankers use their own selection rules.
- **Accuracy.** The evaluation computes NDCG, HR, MRR, and Recall at 10 and 20. NDCG normalizes discounted gain by the ideal gain for each user at the same cutoff.
- **Exposure.** The exposure weight at rank `r` is `1 / log2(r + 1)` with ranks starting at one. The evaluation sums a provider's discounted exposure across users and divides it by the provider's catalog size.
- **Fairness.** Catalog-normalized exposure is used to compute Gini, entropy, MMF, the minimum-to-maximum ratio, and the number of providers with zero exposure. GEI adds `η × mean exposure` to every provider's exposure using `η = 1e-12` by default. The evaluation reports overall, within-group, and between-group GEI for `α = 0, 0.5, 1, 2`. The two decompositions group providers by catalog size and by training interaction frequency.
- **Retention.** NDCG@20 retention is the method's NDCG@20 divided by the backbone's NDCG@20 for the same seed and split. The output reports validation and test retention for the selected configuration.

### 7. Run output

| Path relative to `--output` | Contents |
| --- | --- |
| `<method>/<backbone>/validation_trials.json` | Candidate configurations and per-seed validation metrics |
| `<method>/<backbone>/selection.json` | Selected configuration, per-seed validation and test metrics, and checkpoint paths for trained models |
| `<method>/<backbone>/test/seed<seed>/` | Test metrics and recommendation lists |
| `<method>/<backbone>/trials/` | Training configurations, training logs, `best.pt`, and `last.pt` for continuation |
| `scores/<backbone>/seed<seed>/` | Full-catalog scoring cache and input signatures |
| `cache/`, `native/` | Training input caches and locally compiled shared libraries |

Training resumes from the last completed epoch when the output directory and training inputs match. Use separate output directories when changing the evaluation inputs or running multiple processes for the same method. Parallel fairness searches can reuse a completed baseline through `--baseline-output`. The rerankers use full-catalog score matrices and candidate masks. Some methods also build a per-user sorting index.

## Amazon Video Games Recommendation and Provider Fairness

This project includes data preparation, training with LightGCN and SASRec, fair in-processing, post-processing reranking, and evaluation on Amazon Video Games. Both backbones use the same data splits, item catalog, candidate rules, and provider mappings. Run the commands below from this project's directory.

### 1. Project layout

| Path | Contents |
| --- | --- |
| `data/raw/Amazon_Video_Games/` | Tab-separated interaction and item tables used by preprocessing |
| `data/processed/Amazon_Video_Games/` | Training, validation, test, and candidate files |
| `preprocessing/` | Input parsing, filtering, splitting, ID mapping, and candidate generation |
| `backbones/base_model/` | LightGCN and SASRec |
| `in-processing/rank_model/` | IPS, SDRO, FOCF, and FairDual |
| `post-processing/rerank_model/` | CPFair, PMMF, ElasticRank, and FairRec |
| `post-processing/vendor/` | Accelerated reranking implementations |
| `evaluation/` | Candidate generation, exposure, and metrics |
| `common/engine/` | Data loading, training, grid search, score export, and reranking |
| `configs/` | Hyperparameter grids and evaluation protocol |
| `scripts/` | Bash entry points for preparation and search |

All three dataset projects use the same command-line interface. When moving a project, preserve its directory structure and relative symbolic links with a tool such as `cp -a` or `tar`.

### 2. Environment

Use Linux, Bash, Python 3.10, and a C++17-compatible `g++`. FairRec and CPFair compile native code on first use and place the shared libraries in `native/` under the output directory. The Conda environment installs the CUDA 12.1 build of PyTorch.

```bash
conda env create -f environment.yml
conda activate final-gei
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
g++ --version
```

For CPU-only execution, install the CPU build of PyTorch.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.7.2 PyYAML==6.0.3 tqdm==4.67.1 cvxpy==1.6.5
```

Set `PYTHON_BIN` to use a specific Python executable. Pass `--device cpu` to run on CPU or `--device cuda:0` to use the first visible GPU. By default, the runner uses `cuda:0` when CUDA is available and CPU otherwise. If you set `CUDA_VISIBLE_DEVICES`, CUDA indices are relative to that list. Set `--threads` to control the number of PyTorch CPU threads.

LightGCN and SASRec can train on a GPU. FairDual can score with the backbone on GPU and uses a CPU solver for provider-level dual projection. CPFair, PMMF, and ElasticRank run on CPU in this project. FairRec can sort scores on GPU.

### 3. Data preparation and splits

The training scripts load `data/processed/Amazon_Video_Games/` by default. The `process_config.yaml` file defines column names, ID ranges, input filenames, and the evaluation protocol. Graph construction, sequence training, and candidate generation share the same splits and ID mappings.

| File | Contents |
| --- | --- |
| `Amazon_Video_Games.train` | Complete training prefix. LightGCN constructs the user–item graph from its positive interactions |
| `Amazon_Video_Games.train.sequential` | Positive training targets with preceding context for SASRec |
| `Amazon_Video_Games.valid.CTR`, `Amazon_Video_Games.test.CTR` | Validation and test interactions |
| `evaluation/v1_full_seed2025/*.ranking` | Positive items, exclusions, and history for each evaluation user |
| `uid2original.json`, `iid2original.json` | Mappings from internal indices to source user and item IDs |
| `iid2pid.json`, `pid2provider.json`, `providers.npz` | Item-to-provider assignments, provider labels, and the sparse membership matrix |
| `pid2mgid_size.json`, `pid2mgid_freq.json` | Provider groups based on catalog size and training interaction frequency |

Rebuilding the splits uses the interaction and item tables and `Amazon_Video_Games.suppliers.tsv` in `data/raw/Amazon_Video_Games/`. Preprocessing reads the brand from the item table and fills missing values by matching the same ASIN in the supplemental table. That table contains unambiguous brand matches extracted from official Amazon Reviews 2023 item metadata. [Amazon Reviews 2023 metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/)

To regenerate the supplemental table from a local copy of the official metadata, run the following command.

```bash
python preprocessing/extract_brands.py \
  --metadata /path/to/meta_Video_Games.jsonl.gz \
  --output /tmp/Amazon_Video_Games.suppliers.tsv
```

Preprocessing uses the brand field (`brand:token`) to identify providers. Interactions with a rating of at least 3 are positive. The preprocessing script deduplicates user–item pairs and applies the user, item, and provider filters iteratively. The provider filter counts distinct items. After filtering, providers with fewer than 10 items in the catalog are grouped into a single category.

The splits are chronological per user. The latest holdout segment is the test split, the equally sized preceding segment is the validation split, and the remaining prefix is used for training. Each holdout contains 10% of the user's interactions, rounded down to a minimum of one. Each retained user has at least two positive interactions in the training prefix.

SASRec uses strictly earlier positive training interactions as context for training targets. Validation and test both use the last five positive interactions from the user's training prefix. Short histories use a padding ID equal to the item count. The padding ID lies outside the item catalog and `history_mask` masks the padded positions.

To rebuild the splits, run the following commands. The preparation script checks each input against `data/raw/Amazon_Video_Games/SHA256SUMS.json`. It writes to a new directory, which you can pass to each runner using `--processed-dir`.

```bash
bash scripts/prepare_data.sh --output ../prepared-video-games

bash scripts/run_baseline.sh --backbone lightgcn --device cuda:0 \
  --processed-dir ../prepared-video-games/Amazon_Video_Games --output ../runs-video-games-rebuilt
```

When using rebuilt splits, pass the same `--processed-dir` to the baseline, in-processing, and post-processing scripts. The runner checks dataset and candidate checksums when it loads a baseline for a fairness method.

### 4. Hyperparameter grids

Each method has a JSON configuration for each backbone, such as `configs/ips_lightgcn.json` and `configs/ips_sasrec.json`. Specify each hyperparameter as a nonempty list. The runner expands the Cartesian product of these lists and runs each combination for the selected seeds. A fixed setting is still a list, for example `"epochs": [1000]`.

| Training parameter | Meaning |
| --- | --- |
| `epochs` | Maximum number of training epochs per configuration. The default grid is `[1000]` |
| `batch_size` | Number of training samples per optimizer step |
| `eval_batch_size` | Number of evaluation users scored in a batch |
| `embedding_size` | Embedding dimension. SASRec requires a multiple of `n_heads` |
| `learning_rate`, `weight_decay` | Adam learning rate and weight decay |
| `GCN_layer`, `reg_lambda` | LightGCN propagation depth and regularization coefficient |
| `n_layers`, `n_heads`, `inner_size` | SASRec Transformer depth, attention heads, and feed-forward width |
| `hidden_dropout_prob`, `attn_dropout_prob` | SASRec hidden and attention dropout probabilities |
| `eval_every` | Validation interval in epochs |

| Fairness method | Main search parameters |
| --- | --- |
| IPS | `M`, `update_epoch`, and training parameters |
| SDRO | `eta`, `update_epoch`, and training parameters |
| FOCF | `fair_lambda` and training parameters |
| FairDual | `alpha`, `eta`, `lambd`, `s_k`, `exposure_sample_num`, and `update_epoch` |
| CPFair | `lambda` |
| PMMF | `lambda`, `learning_rate`, and `gamma` |
| ElasticRank | `lambda`, `t`, and `anchor_rate` |
| FairRec | `alpha` |

FairDual uses `s_k` as the ranking cutoff for estimating exposure during training. It must be at most `exposure_sample_num`, which must be at most the catalog size. The method's `alpha` and `eta` are independent of the GEI sensitivity and smoothing parameters.

`configs/protocol.json` defines the default seeds and evaluation settings. Repeat `--seed` to specify multiple seeds. Use `--config` to supply a different JSON grid and `--dry-run` to preview the combinations.

```bash
bash scripts/run_ips.sh --backbone lightgcn --dry-run
bash scripts/run_fairdual.sh --backbone sasrec --dry-run \
  --seed 2025 --seed 2026
```

### 5. Running the experiments

#### 5.1 Train the backbones

```bash
OUT="../runs-video-games"

bash scripts/run_baseline.sh --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_baseline.sh --backbone sasrec --device cuda:0 --output "$OUT"
```

Each configuration and seed saves the checkpoint with the highest validation NDCG@20 as `best.pt`. The runner selects the configuration with the highest mean validation NDCG@20 across seeds, then evaluates its checkpoints on the test split. Fairness methods use the corresponding backbone and seed as their NDCG@20 reference on each split.

#### 5.2 Search in-processing methods

```bash
bash scripts/run_ips.sh      --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_sdro.sh     --backbone sasrec   --device cuda:0 --output "$OUT"
bash scripts/run_focf.sh     --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_fairdual.sh --backbone sasrec   --device cuda:0 --output "$OUT"
```

Each method works with both backbones. The runner trains from random initialization for each configuration and seed. By default, it validates every epoch for the first 30 epochs, at the `eval_every` interval after that, and at the final epoch. Each configuration and seed saves the checkpoint with the highest validation NDCG@20 as `best.pt`. The runner selects the configuration with the highest mean validation NDCG@20 across seeds and evaluates its checkpoints on the test split.

#### 5.3 Search post-processing methods

```bash
bash scripts/run_cpfair.sh      --backbone lightgcn --device cpu --output "$OUT"
bash scripts/run_pmmf.sh        --backbone sasrec   --device cpu --output "$OUT"
bash scripts/run_elasticrank.sh --backbone lightgcn --device cpu --output "$OUT"
bash scripts/run_fairrec.sh     --backbone sasrec   --device cpu --output "$OUT"
```

The runner generates full-catalog validation scores from the selected backbone checkpoints and reranks them using the same candidate rules for every configuration. It selects the configuration with the highest mean validation NDCG@20 across seeds. The runner then generates test scores and evaluates the selected configuration. Compatible runs can share the frozen score cache.

If you saved the baseline in a different directory, pass its root through `--baseline-output`. `--output` controls where the reranking results are written.

```bash
bash scripts/run_pmmf.sh --backbone lightgcn --device cpu \
  --baseline-output ../runs-video-games --output ../pmmf-video-games
```

#### 5.4 Run a quick end-to-end check

The following commands train for three epochs with one seed and evaluate the first 384 users of each evaluation split. The check uses a separate output directory. For a full search, use the commands in sections 5.1–5.3.

```bash
CHECK="../check-video-games"

bash scripts/run_baseline.sh --backbone lightgcn --device cpu --output "$CHECK" \
  --seed 2025 --epochs 3 --max-configs 1 --limit-users 384

bash scripts/run_cpfair.sh --backbone lightgcn --device cpu --output "$CHECK" \
  --seed 2025 --max-configs 1 --limit-users 384
```

### 6. Evaluation protocol

- **Candidates.** The candidate set is the full item catalog after excluding known positives in the other splits. The evaluation includes users with a positive item in the current split. The backbones rank by descending score and break ties by ascending item ID. Rerankers use their own selection rules.
- **Accuracy.** The evaluation computes NDCG, HR, MRR, and Recall at 10 and 20. NDCG normalizes discounted gain by the ideal gain for each user at the same cutoff.
- **Exposure.** The exposure weight at rank `r` is `1 / log2(r + 1)` with ranks starting at one. The evaluation sums a provider's discounted exposure across users and divides it by the provider's catalog size.
- **Fairness.** Catalog-normalized exposure is used to compute Gini, entropy, MMF, the minimum-to-maximum ratio, and the number of providers with zero exposure. GEI adds `η × mean exposure` to every provider's exposure using `η = 1e-12` by default. The evaluation reports overall, within-group, and between-group GEI for `α = 0, 0.5, 1, 2`. The two decompositions group providers by catalog size and by training interaction frequency.
- **Retention.** NDCG@20 retention is the method's NDCG@20 divided by the backbone's NDCG@20 for the same seed and split. The output reports validation and test retention for the selected configuration.

### 7. Run output

| Path relative to `--output` | Contents |
| --- | --- |
| `<method>/<backbone>/validation_trials.json` | Candidate configurations and per-seed validation metrics |
| `<method>/<backbone>/selection.json` | Selected configuration, per-seed validation and test metrics, and checkpoint paths for trained models |
| `<method>/<backbone>/test/seed<seed>/` | Test metrics and recommendation lists |
| `<method>/<backbone>/trials/` | Training configurations, training logs, `best.pt`, and `last.pt` for continuation |
| `scores/<backbone>/seed<seed>/` | Full-catalog scoring cache and input signatures |
| `cache/`, `native/` | Training input caches and locally compiled shared libraries |

Training resumes from the last completed epoch when the output directory and training inputs match. Use separate output directories when changing the evaluation inputs or running multiple processes for the same method. Parallel fairness searches can reuse a completed baseline through `--baseline-output`. The rerankers use full-catalog score matrices and candidate masks. Some methods also build a per-user sorting index.

## BeerAdvocate Recommendation and Provider Fairness

This project includes data preparation, training with LightGCN and SASRec, fair in-processing, post-processing reranking, and evaluation on BeerAdvocate. Both backbones use the same data splits, item catalog, candidate rules, and provider mappings. Run the commands below from this project's directory.

### 1. Project layout

| Path | Contents |
| --- | --- |
| `data/raw/BeerAdvocate/` | Tab-separated interaction and item tables used by preprocessing |
| `data/processed/BeerAdvocate/` | Training, validation, test, and candidate files |
| `preprocessing/` | Input parsing, filtering, splitting, ID mapping, and candidate generation |
| `backbones/base_model/` | LightGCN and SASRec |
| `in-processing/rank_model/` | IPS, SDRO, FOCF, and FairDual |
| `post-processing/rerank_model/` | CPFair, PMMF, ElasticRank, and FairRec |
| `post-processing/vendor/` | Accelerated reranking implementations |
| `evaluation/` | Candidate generation, exposure, and metrics |
| `common/engine/` | Data loading, training, grid search, score export, and reranking |
| `configs/` | Hyperparameter grids and evaluation protocol |
| `scripts/` | Bash entry points for preparation and search |

All three dataset projects use the same command-line interface. When moving a project, preserve its directory structure and relative symbolic links with a tool such as `cp -a` or `tar`.

### 2. Environment

Use Linux, Bash, Python 3.10, and a C++17-compatible `g++`. FairRec and CPFair compile native code on first use and place the shared libraries in `native/` under the output directory. The Conda environment installs the CUDA 12.1 build of PyTorch.

```bash
conda env create -f environment.yml
conda activate final-gei
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
g++ --version
```

For CPU-only execution, install the CPU build of PyTorch.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.7.2 PyYAML==6.0.3 tqdm==4.67.1 cvxpy==1.6.5
```

Set `PYTHON_BIN` to use a specific Python executable. Pass `--device cpu` to run on CPU or `--device cuda:0` to use the first visible GPU. By default, the runner uses `cuda:0` when CUDA is available and CPU otherwise. If you set `CUDA_VISIBLE_DEVICES`, CUDA indices are relative to that list. Set `--threads` to control the number of PyTorch CPU threads.

With CUDA, FairDual updates weights on GPU and uses a CPU projection solver when needed. PMMF and ElasticRank support CUDA reranking. PMMF solves the dual projection on CPU. FairRec can sort scores on GPU. CPFair uses a native CPU implementation.

### 3. Data preparation and splits

The training scripts load `data/processed/BeerAdvocate/` by default. The `process_config.yaml` file defines column names, ID ranges, input filenames, and the evaluation protocol. Graph construction, sequence training, and candidate generation share the same splits and ID mappings.

| File | Contents |
| --- | --- |
| `BeerAdvocate.train` | Complete training prefix. LightGCN constructs the user–item graph from its positive interactions |
| `BeerAdvocate.train.sequential` | Positive training targets with preceding context for SASRec |
| `BeerAdvocate.valid.CTR`, `BeerAdvocate.test.CTR` | Validation and test interactions |
| `evaluation/v1_full_seed2025/*.ranking` | Positive items, exclusions, and history for each evaluation user |
| `uid2original.json`, `iid2original.json` | Mappings from internal indices to source user and item IDs |
| `iid2pid.json`, `pid2provider.json`, `providers.npz` | Item-to-provider assignments, provider labels, and the sparse membership matrix |
| `pid2mgid_size.json`, `pid2mgid_freq.json` | Provider groups based on catalog size and training interaction frequency |

Rebuilding the splits uses the interaction and item tables in `data/raw/BeerAdvocate/`. Each item is assigned to one brewery.

Preprocessing uses the brewery field (`brewer_id:token`) to identify providers. Interactions with an overall rating of at least 3 are positive. The preprocessing script deduplicates user–item pairs and applies the user, item, and provider filters iteratively. The provider filter counts distinct items. After filtering, providers with fewer than 10 items in the catalog are grouped into a single category.

The splits are chronological per user. The latest holdout segment is the test split, the equally sized preceding segment is the validation split, and the remaining prefix is used for training. Each holdout contains 10% of the user's interactions, rounded down to a minimum of one. Each retained user has at least two positive interactions in the training prefix.

SASRec uses strictly earlier positive training interactions as context for training targets. Validation and test both use the last five positive interactions from the user's training prefix. Short histories use a padding ID equal to the item count. The padding ID lies outside the item catalog and `history_mask` masks the padded positions.

To rebuild the splits, run the following commands. The preparation script checks each input against `data/raw/BeerAdvocate/SHA256SUMS.json`. It writes to a new directory, which you can pass to each runner using `--processed-dir`.

```bash
bash scripts/prepare_data.sh --output ../prepared-beeradvocate

bash scripts/run_baseline.sh --backbone lightgcn --device cuda:0 \
  --processed-dir ../prepared-beeradvocate/BeerAdvocate --output ../runs-beeradvocate-rebuilt
```

When using rebuilt splits, pass the same `--processed-dir` to the baseline, in-processing, and post-processing scripts. The runner checks dataset and candidate checksums when it loads a baseline for a fairness method.

### 4. Hyperparameter grids

Each method has a JSON configuration for each backbone, such as `configs/ips_lightgcn.json` and `configs/ips_sasrec.json`. Specify each hyperparameter as a nonempty list. The runner expands the Cartesian product of these lists and runs each combination for the selected seeds. A fixed setting is still a list, for example `"epochs": [1000]`.

| Training parameter | Meaning |
| --- | --- |
| `epochs` | Maximum number of training epochs per configuration. The default grid is `[1000]` |
| `batch_size` | Number of training samples per optimizer step |
| `eval_batch_size` | Number of evaluation users scored in a batch |
| `embedding_size` | Embedding dimension. SASRec requires a multiple of `n_heads` |
| `learning_rate`, `weight_decay` | Adam learning rate and weight decay |
| `GCN_layer`, `reg_lambda` | LightGCN propagation depth and regularization coefficient |
| `n_layers`, `n_heads`, `inner_size` | SASRec Transformer depth, attention heads, and feed-forward width |
| `hidden_dropout_prob`, `attn_dropout_prob` | SASRec hidden and attention dropout probabilities |
| `eval_every` | Validation interval in epochs |

| Fairness method | Main search parameters |
| --- | --- |
| IPS | `M`, `update_epoch`, and training parameters |
| SDRO | `eta`, `update_epoch`, and training parameters |
| FOCF | `fair_lambda` and training parameters |
| FairDual | `alpha`, `eta`, `lambd`, `s_k`, `exposure_sample_num`, and `update_epoch` |
| CPFair | `lambda` |
| PMMF | `lambda`, `learning_rate`, and `gamma` |
| ElasticRank | `lambda`, `t`, and `anchor_rate` |
| FairRec | `alpha` and optional `user_order_seed` |

FairDual uses `s_k` as the ranking cutoff for estimating exposure during training. It must be at most `exposure_sample_num`, which must be at most the catalog size. The method's `alpha` and `eta` are independent of the GEI sensitivity and smoothing parameters.

`configs/protocol.json` defines the default seeds and evaluation settings. Repeat `--seed` to specify multiple seeds. Use `--config` to supply a different JSON grid and `--dry-run` to preview the combinations.

```bash
bash scripts/run_ips.sh --backbone lightgcn --dry-run
bash scripts/run_fairdual.sh --backbone sasrec --dry-run \
  --seed 2025 --seed 2026
```

### 5. Running the experiments

#### 5.1 Train the backbones

```bash
OUT="../runs-beeradvocate"

bash scripts/run_baseline.sh --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_baseline.sh --backbone sasrec --device cuda:0 --output "$OUT"
```

Each configuration and seed saves the checkpoint with the highest validation NDCG@20 as `best.pt`. The runner selects the configuration with the highest mean validation NDCG@20 across seeds, then evaluates its checkpoints on the test split. Fairness methods use the corresponding backbone and seed as their NDCG@20 reference on each split.

#### 5.2 Search in-processing methods

```bash
bash scripts/run_ips.sh      --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_sdro.sh     --backbone sasrec   --device cuda:0 --output "$OUT"
bash scripts/run_focf.sh     --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_fairdual.sh --backbone sasrec   --device cuda:0 --output "$OUT"
```

Each method works with both backbones. The runner trains from random initialization for each configuration and seed. By default, it validates every epoch for the first 30 epochs, at the `eval_every` interval after that, and at the final epoch. Each configuration and seed saves the checkpoint with the highest validation NDCG@20 as `best.pt`. The runner selects the configuration with the highest mean validation NDCG@20 across seeds and evaluates its checkpoints on the test split.

#### 5.3 Search post-processing methods

```bash
bash scripts/run_cpfair.sh      --backbone lightgcn --device cpu --output "$OUT"
bash scripts/run_pmmf.sh        --backbone sasrec   --device cuda:0 --output "$OUT"
bash scripts/run_elasticrank.sh --backbone lightgcn --device cuda:0 --output "$OUT"
bash scripts/run_fairrec.sh     --backbone sasrec   --device cuda:0 --output "$OUT"
```

The runner generates full-catalog validation scores from the selected backbone checkpoints and reranks them using the same candidate rules for every configuration. It selects the configuration with the highest mean validation NDCG@20 across seeds. The runner then generates test scores and evaluates the selected configuration. Compatible runs can share the frozen score cache.

If you saved the baseline in a different directory, pass its root through `--baseline-output`. `--output` controls where the reranking results are written.

```bash
bash scripts/run_pmmf.sh --backbone lightgcn --device cuda:0 \
  --baseline-output ../runs-beeradvocate --output ../pmmf-beeradvocate
```

#### 5.4 Run a quick end-to-end check

The following commands train for one epoch with one seed and evaluate the first 192 users of each evaluation split. The check uses a separate output directory. For a full search, use the commands in sections 5.1–5.3.

```bash
CHECK="../check-beeradvocate"

bash scripts/run_baseline.sh --backbone lightgcn --device cpu --output "$CHECK" \
  --seed 2025 --epochs 1 --max-configs 1 --limit-users 192

bash scripts/run_cpfair.sh --backbone lightgcn --device cpu --output "$CHECK" \
  --seed 2025 --max-configs 1 --limit-users 192
```

### 6. Evaluation protocol

- **Candidates.** The candidate set is the full item catalog after excluding known positives in the other splits. The evaluation includes users with a positive item in the current split. The backbones rank by descending score and break ties by ascending item ID. Rerankers use their own selection rules.
- **Accuracy.** The evaluation computes NDCG, HR, MRR, and Recall at 10 and 20. NDCG normalizes discounted gain by the ideal gain for each user at the same cutoff.
- **Exposure.** The exposure weight at rank `r` is `1 / log2(r + 1)` with ranks starting at one. The evaluation sums a provider's discounted exposure across users and divides it by the provider's catalog size.
- **Fairness.** Catalog-normalized exposure is used to compute Gini, entropy, MMF, the minimum-to-maximum ratio, and the number of providers with zero exposure. GEI adds `η × mean exposure` to every provider's exposure using `η = 1e-12` by default. The evaluation reports overall, within-group, and between-group GEI for `α = 0, 0.5, 1, 2`. The two decompositions group providers by catalog size and by training interaction frequency.
- **Retention.** NDCG@20 retention is the method's NDCG@20 divided by the backbone's NDCG@20 for the same seed and split. The output reports validation and test retention for the selected configuration.

### 7. Run output

| Path relative to `--output` | Contents |
| --- | --- |
| `<method>/<backbone>/validation_trials.json` | Candidate configurations and per-seed validation metrics |
| `<method>/<backbone>/selection.json` | Selected configuration, per-seed validation and test metrics, and checkpoint paths for trained models |
| `<method>/<backbone>/test/seed<seed>/` | Test metrics and recommendation lists |
| `<method>/<backbone>/trials/` | Training configurations, training logs, `best.pt`, and `last.pt` for continuation |
| `scores/<backbone>/seed<seed>/` | Full-catalog scoring cache and input signatures |
| `cache/`, `native/` | Training input caches and locally compiled shared libraries |

Training resumes from the last completed epoch when the output directory and training inputs match. Use separate output directories when changing the evaluation inputs or running multiple processes for the same method. Parallel fairness searches can reuse a completed baseline through `--baseline-output`. The rerankers use full-catalog score matrices and candidate masks. Some methods also build a per-user sorting index.
