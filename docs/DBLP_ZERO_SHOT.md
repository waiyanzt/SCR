# Frozen SCR on DBLP link prediction

This pilot loads `ckpts/scr_3g.pth` for inference without updating weights. The
task is **Paper-Conference**: given a paper, rank its true conference among all
20 conferences. It uses the INV-GNN experiment's shared split, graph variants,
and compiled programs. The model architecture remains unchanged.

| Physical variant | Where training research-area connections are placed |
|---|---|
| v1 | Paper-Area |
| v2 | Conference-Area |
| v3 | Author-Area |
| universal | Union of the three placements |

Paper-Author and Paper-Term provide shared context. Paper-Conference target edges
are present only for training papers. The transformation certificates and
compiler pair filters use training Paper-Conference support only
(`--mapping-scope train`). The pair filter selects block-local witnesses; it is
not a model edge or supplied feature. The exporter rejects full-scope bundles.

## 1. Preprocess and export

Run from the SCR repository root in your existing Python/SCR environment, with
pandas, scikit-learn, SciPy, NumPy, and PyTorch installed. If your existing HPC
environment is missing pandas, install it there with `python -m pip install pandas`.
It is also included in the repository's `requirements.txt`. No new virtual
environment is needed. This command performs CPU preprocessing only; it does
not run a model experiment:

```bash
python script/prepare_dblp_lp.py
```

Raw files are already located under `rai-datasets/DBLP/`:

```text
author_label.txt
paper_author.txt
paper_conf.txt
paper_term.txt
```

Their hashes are checked against the source repo's `data/SHA256SUMS`. The command:

1. Stages these four files temporarily for INV-GNN's original environment-based
   split builder, without changing that checkout or requiring symlinks.
2. Runs the original split verifier against the generated archive.
3. Runs `preprocess_DBLP_rgcn_structural.py` with the canonical shared archive,
   `--mapping-scope train`, and v1/v2/v3/universal.
4. Validates and exports portable arrays, candidate ids, and provenance JSON.

The original paper-disjoint 70/10/20 split has 10,029 / 1,433 / 2,866 positives.
The structural preprocessing excludes 866 papers whose authors disagree on
research area, giving **9,448 / 1,348 / 2,666** train/validation/test queries.
The exporter requires those post-filter counts and exactly 20 conferences.

Default outputs, all under the already ignored `rai-datasets/` directory:

```text
preprocessed/scr_dblp_build/
  DBLP_shared_splits_CANONICAL/DBLP_pc_shared_splits.npz
  RGCN_structural_v8_dblp/DBLP/paper_conference/{v1,v2,v3,universal}/graph.pt
scr_dblp_lp/
  manifest.json
  paper_conference/{manifest.json,queries.npz,v1.npz,v2.npz,v3.npz,universal.npz}
```

To export matching bundles you already generated or downloaded, skip rebuilding:

```bash
python script/prepare_dblp_lp.py --bundle-root /path/to/DBLP/paper_conference
```

Options include `--source-repo`, `--raw-dir`, `--work-dir`, and `--output`.
Existing build/export directories are not overwritten. If a build was interrupted,
use a fresh `--work-dir`, or point `--bundle-root` at a completed build.

For example, after a failed attempt caused by a missing dependency:

```bash
python -m pip install pandas
python script/prepare_dblp_lp.py --work-dir rai-datasets/preprocessed/scr_dblp_build_retry
```

The ignored `INV-GNN-12D9/` checkout is needed for preprocessing/export, and is
not carried by `git push`. Evaluation uses only the portable export, SCR code,
checkpoint, and normal SCR runtime. Transfer the export to the GPU machine if
preprocessing happens elsewhere. Raw data is read locally; nothing is downloaded
or published by these scripts.

## 2. Feature policy

Unlike the IMDb bundles, the source DBLP bundles have **no supplied node features**.
Their trained baselines use their own embeddings; those embeddings are not SCR
inputs. This adapter uses SCR's existing **top/ontology fallback**:

* Count each node's outgoing relation types after adding internal inverse edges
  and the configured self-loops.
* Apply SCR's seeded `svd_proj_2` on CPU to obtain the checkpoint's 64-dimensional
  input, with the same padding behavior as upstream.
* Construct semantic neighbors with SCR's FAISS routine using fixed k=3 and p=0.8.

No paper titles, author names, term text embeddings, area-label features, BERT
encoding, or separately trained features are introduced. Zero rows follow the
upstream structural fallback behavior and are not mean-filled. This is a
declared ontology-feature transfer setting, not the BERT-feature WordNet setting.

**These features are graph-derived and may change across physical variants.**
The experiment measures sensitivity of the whole frozen SCR structural pipeline,
including its derived features. It is not an experiment holding a supplied feature
matrix fixed. Each projected feature hash is recorded separately. Identical
compiled graphs must yield the same derived features and scores under the fixed
settings; the repeat measures the actual inference noise floor.

## 3. Run on your GPU allocation

Use the same working SCR environment as your WordNet run. SCR's usual `rspmm`
extension still requires a compatible C++/CUDA compiler. No CMPNN/torchdrug or
DGL environment is required.

```bash
# Smoke run over a fixed subset of at most 64 queries.
python script/eval_dblp_lp.py --device cuda:0 --limit 64 --run-tag smoke64
python script/compare_dblp_lp.py --run-tag smoke64

# Full held-out test evaluation.
python script/eval_dblp_lp.py --device cuda:0
python script/compare_dblp_lp.py
```

Default coverage is **9 separate inference processes**: four physical graphs,
four compiled views, and a physical v1 repeat. For a first reference-only check:

```bash
python script/eval_dblp_lp.py --view physical --variant v1 --limit 64 --run-tag reference64
```

That includes the repeat unless `--no-repeat` is given. Other options include
`--checkpoint`, `--batch-size` (8), `--seed` (1024), `--k`, `--p`, `--split val|test`,
`--data`, and `--output`. Keep settings fixed across representations; there is no
DBLP hyperparameter tuning in this protocol. Results default to the top-level
`results/` directory. Use separate `--run-tag` values for smoke, validation, or
changed settings, and pass the same tag to the comparison command. With
`--limit N`, the default tag is `smokeN`. Override the root with evaluation's
`--output` and comparison's `--results`.

## 4. Metrics, controls, and validation

The model uses strict checkpoint loading, disabled parameter gradients, `eval()`,
and `no_grad()`. Parameter hashes must remain unchanged. Inference shares the
same implementation as the IMDb adapter, bypassing upstream dataset-name caches.

SCR's upstream all-tail forward path is used, then only conference scores are
kept. Every query contains the positive first and the other 19 conferences in
fixed pool order. The exporter verifies that the source flattened negatives
are exactly this complement, independent of their storage order. Thus the metric
is **full conference-type ranking**, not sampled negatives and not full-entity
head/tail ranking. Ties use `1 + number_better + number_tied / 2`.

Exports verify inverse pairs, paper-disjoint splits, shared supervision/node ids,
identical compiled programs, absence of held-out direct targets, and absence of
paper-specific Area edges for held-out papers. Legitimate contextual paths through
shared authors or terms remain available. Relation edges use unit-weight set
semantics and SCR adds inverses internally.

Runs save `metrics.json` and per-query `scores.npz` in folders such as:

```text
results/scr_zeroshot_dblpv1_lp/paper_conference/
results/scr_zeroshot_dblpv2_lp/paper_conference/
results/scr_zeroshot_dblpv3_lp/paper_conference/
results/scr_zeroshot_dblpuniversal_lp/paper_conference/
results/scr_zeroshot_dblpv1_lp_compiled/paper_conference/
results/scr_zeroshot_dblpv1_lp_repeat/paper_conference/
```

Each compiled view has its own folder. Tags append to the run name, e.g.
`scr_zeroshot_dblpv1_lp_smoke64/paper_conference/`.
The comparator writes `results/scr_zeroshot_dblp_lp_summary.md` and `.json`
(or `scr_zeroshot_dblp_lp_smoke64_summary.*` for that tag), including MRR/Hits,
Kendall tau-b, delta MRR with a paired bootstrap interval, Hit outcome changes,
and bitwise repeat/compiled-input agreement. Partial suites are explicitly marked.

Local verification covers syntax, CLI checks, and synthetic data-contract tests:

```bash
python -m unittest discover -s tests -p "test_*lp.py" -v
```

No preprocessing or GPU experiment was run on the Windows authoring machine.
Validate end-to-end behavior with the smoke command in your actual SCR environment
before running or reporting the full experiment.
