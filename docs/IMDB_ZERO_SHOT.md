# Frozen SCR on IMDb link prediction

This pilot uses the released `ckpts/scr_3g.pth` with no parameter updates. It
reuses INV-GNN's task definitions, canonical supplied features, and fixed
candidate sets. It does not encode movie names with BERT. IMDb transfer with
supplied BoW features is a separate setting from the BERT WordNet reproduction.

| Task | Physical variants | Train / val / test positives | Candidate tails |
|---|---|---|---|
| Movie-Year | v1, v2, v3, v4, universal | 2925 / 418 / 834 | positive + 19 frequency-matched negative years |
| Movie-Director | v1, v3, universal | 2926 / 418 / 836 | positive + 19 uniform negative directors |

The frozen model sees training target edges as context. Validation/test target
edges and their Link-node copies must be absent. Features for the target type
are pooled over training pairs only. Movies remain present as graph nodes across
splits: this is transfer to a new dataset with held-out links, not a claim that
the IMDb test movies are absent from its inference graph.

## 1. Prepare the INV-GNN bundles

Run these commands from the SCR repository root in a Python environment with
PyTorch, NumPy, SciPy, pandas, and scikit-learn. Your existing SCR environment
already supplies most dependencies; install pandas if absent. Preprocessing is
CPU work and does not need DGL, torchdrug, or model training.

In your existing HPC environment, use `python -m pip install pandas` if needed.
Pandas is included in `requirements.txt`; no new virtual environment is required.

The CSV currently lives at `rai-datasets/IMDB/movie_metadata.csv`. Its expected
SHA-256 is `74e4a9af314eea255147a6d48a5678ad7d0be3cb8ba3fdb819ef6edddb36e2ca`.
All commands below work with relative paths from the repository root on Linux
or Windows. Each command is a single line.

```bash
python INV-GNN-12D9/dataset_construction/imdb/reconstruct_IMDB_md_ml_shared_splits.py --csv rai-datasets/IMDB/movie_metadata.csv --out-dir rai-datasets/preprocessed/IMDB_shared_splits

python INV-GNN-12D9/dataset_construction/imdb/build_IMDB_my_shared_splits.py --csv rai-datasets/IMDB/movie_metadata.csv --inherit-from rai-datasets/preprocessed/IMDB_shared_splits/IMDB_md_shared_splits.npz --out rai-datasets/preprocessed/IMDB_shared_splits/IMDB_my_shared_splits.npz

python INV-GNN-12D9/models/RGCN/preprocess_IMDB_rgcn_structural.py --raw-csv rai-datasets/IMDB/movie_metadata.csv --tasks movie_year,movie_director --variants v1,v2,v3,v4,universal --shared-movie-director rai-datasets/preprocessed/IMDB_shared_splits/IMDB_md_shared_splits.npz --shared-movie-year rai-datasets/preprocessed/IMDB_shared_splits/IMDB_my_shared_splits.npz --feature-mode pooled_bow --neg-k 19 --my-neg-policy freq_matched --out-root rai-datasets/preprocessed/RGCN_structural_v10_year_canonfeat
```

If you already have the matching canonical `graph.pt` bundles, skip those three
commands and supply their `IMDB/` directory as `--bundle-root` below.

## 2. Export portable SCR inputs

```bash
python script/prepare_imdb_lp.py
```

Defaults: source checkout `INV-GNN-12D9/`; bundles
`rai-datasets/preprocessed/RGCN_structural_v10_year_canonfeat/IMDB/`; output
`rai-datasets/scr_imdb_lp/`. Override with `--source-repo`, `--bundle-root`, or
`--output`. Existing exports are never overwritten; choose a new output path.

The exporter verifies shared node ids, features, supervision, and compiled
programs; verifies inverse pairs before retaining one direction; checks candidate
types and movie-disjoint splits; and rejects direct or Link-copy held-out target
edges. It preserves negative arrays exactly, including repeated sampled years.

Portable files are NumPy arrays plus JSON provenance. Source bundle, source code,
feature, graph, and supervision hashes are recorded. Each task stores features
once. As in the source CMPNN adapter, SCR consumes graph edges as unit-weight
sets; optional RGCN parallel-message weights are not applied to SCR's aggregation.
Source weighted-program hashes and effective SCR edge hashes are recorded
separately.

**The ignored repo is needed only for preprocessing/export.** It is not copied by
`git push`. Put that source checkout and the CSV on the preprocessing machine, or
transfer the completed `rai-datasets/scr_imdb_lp/` export to the GPU machine.
Evaluation needs only the SCR checkout, checkpoint, export, and usual SCR runtime.
All generated files above are under the already ignored `rai-datasets/` directory.

## 3. GPU commands to run yourself

First use the same working SCR environment as the WordNet reproduction, on a GPU
allocation. No source CMPNN environment is required. A C++/CUDA compiler matching
PyTorch is still needed for the usual SCR `rspmm` extension.

```bash
# Smoke: all graph settings, same fixed subset of at most 64 queries per task.
python script/eval_imdb_lp.py --device cuda:0 --limit 64 --run-tag smoke64
python script/compare_imdb_lp.py --run-tag smoke64

# Full test evaluation; do not use smoke outputs as full benchmark results.
python script/eval_imdb_lp.py --device cuda:0
python script/compare_imdb_lp.py
```

Default coverage is 18 separate inference processes:

* 5 physical + 5 compiled Movie-Year views + 1 physical v1 repeat;
* 3 physical + 3 compiled Movie-Director views + 1 physical v1 repeat.

For a first single view, use `--task movie_year --view physical --variant v1`.
That also runs the reference repeat unless `--no-repeat` is passed. Other options
include `--checkpoint`, `--batch-size` (8), `--seed` (1024), `--k` (3), `--p` (0.8),
and `--split val|test`. Keep these fixed across runs; there is no IMDb tuning step
in this protocol. Results default to the top-level `results/` directory. Use a
distinct `--run-tag` for changed settings or repeat experiments. With `--limit N`,
the default tag is `smokeN`; use the same tag with the comparison command.

## What the adapter preserves and measures

The model configuration comes from `config/eval_all.yaml`. Loading is strict,
all parameters have gradients disabled, inference uses `eval()` and `no_grad()`,
and parameter hashes must remain unchanged. No optimizer or training loop exists.
SCR's own relation-graph, self-loop, and semantic-neighbor construction are used.

Canonical input features are mean-filled as in SCR, then reduced with SCR's
`svd_proj_2` to 64 dimensions on CPU using a fixed seed and one thread. The same
projection is regenerated in each process and its hash is checked across views.
Graph preprocessing bypasses the upstream dataset-name caches. This prevents a
different variant's graph or embeddings from being silently reused. Projection
on CPU is a declared adapter choice; it is not a claim of numerical identity to
the upstream GPU projection on other datasets.

The forward pass uses SCR's upstream all-tail scoring path, then keeps the
**20 supplied candidate tails** for evaluation. The resulting metrics are not
full-entity MRR and do not average head and tail prediction. Primary ranks use
`1 + number_better + number_tied / 2`, matching INV-GNN. Pessimistic ranks/MRR are
also saved for inspection.

Each run saves `metrics.json` and `scores.npz` in a named folder under `results/`:

```text
results/scr_zeroshot_imdbv1_lp/movie_year/
results/scr_zeroshot_imdbv1_lp/movie_director/
results/scr_zeroshot_imdbv2_lp/movie_year/
results/scr_zeroshot_imdbv1_lp_compiled/movie_year/
results/scr_zeroshot_imdbv1_lp_repeat/movie_year/
```

The same naming applies to v3/v4/universal and the other task's compiled/repeat
views. Task subfolders keep Movie-Year and Movie-Director separate. Tags append
to the run folder name, e.g. `scr_zeroshot_imdbv1_lp_smoke64/movie_year/`.
Override the results root with evaluation's `--output` and comparison's `--results`.
The comparator checks matching checkpoint/config/features/queries/settings,
recomputes metrics, and writes `results/scr_zeroshot_imdb_lp_summary.md` and `.json`
(or `scr_zeroshot_imdb_lp_smoke64_summary.*` for that tag). It reports MRR,
Hits, Kendall tau-b against physical v1, Hit outcome changes, bitwise agreement,
and a paired bootstrap interval for delta MRR. It checks the reference repeat and
agreement among compiled inputs separately. A failed repeat means measured
representation effects need to be interpreted against that observed noise floor.

The exporter and inference adapter require validation in your actual PyTorch/SCR
environment. Local checks on the Windows authoring machine cover syntax and the
NumPy experimental-contract tests; no GPU experiment was run there.

```bash
python -m unittest discover -s tests -p test_imdb_lp.py -v
```
