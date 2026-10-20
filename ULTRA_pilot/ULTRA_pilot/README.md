# ULTRA representation-sensitivity pilot

**Question.** Does a published KG foundation model (ULTRA) give different answers when the *same facts*
are represented differently: named inverse edges, reified facts, materialized 2/3-hop shortcuts? If so, current
GFMs are not invariant to the paper's mappings, which is preliminary evidence for the Huang GFM proposal.

No training: the released checkpoints (`upstream/ckpts/ultra_{3g,4g,50g}.pth`) are evaluated as they are.
All three were pretrained on FB15k-237 and WN18RR (plus CoDEx-M; 4g adds NELL-995; 50g 50 graphs), so on
those two graphs this tests **seen graphs under new representations**. **Hetionet** (biomedical) is held out from
3g/4g pretraining, so it gives a zero-shot test for those two; the 50g pretraining list is not published in
the ULTRA README, so 50g on Hetionet is reported without a held-out claim.

For a detailed account of how the evaluation was done see `EVALUATION_DETAILS.md`; for repeating it on another
GFM (zero-shot on our other transformed graphs, and fine-tuning) see `GFM_EVALUATION_GUIDE.md`.

## Layout

| path | what |
| --- | --- |
| `upstream/` | ULTRA, cloned untouched (commit `427966ad`, 2025-02-02) |
| `pydeps/` | `ninja`, `easydict` installed with `pip --target` (the shared env is not modified) |
| `pilot_common.py` | on-disk format: base ids, shared queries, fixed negatives, variant graphs, manifests |
| `build_wn18rr_variants.py` | exports the existing INV-GNN WN18RR variants (`gnn/data/wordnet_3hops_augmented_full`, read only) |
| `build_kg_variants.py` | generic builder (FB15k-237, Hetionet) from `data/<DATASET>/raw/{train,valid,test}.txt` |
| `ultra_eval.py` | one checkpoint x one variant: filtered ranks + 51 candidate scores per query and direction |
| `compare_variants.py` | per-variant MRR/Hits and per-pair Kendall tau, delta MRR, Hit@k disagreement |
| `compile_back.py` | "declared after training": compiles each variant back via its declared switchings, verifying every removed edge |
| `sbatch_compiled.sbatch` | unchanged checkpoint on the compiled variants -> `results_compiled/` |
| `aggregate_pilot.py`, `make_pilot_figure.py` | summary table and proposal figure -> `results/pilot_summary.*`, `results/pilot_figure.*` |
| `sbatch_ultra_pilot.sbatch` | array of 9 tasks (3 datasets x 3 checkpoints); runs on any GPU >= 4 GB (RTX 2080 used) |
| `data/`, `results/`, `logs/`, `torch_extensions/` | generated |

## Variants

| dataset | variant | message graph | notes |
| --- | --- | --- | --- |
| WN18RR | `no_changes` | 86,835 official train edges, 11 relations | reference |
| WN18RR | `all_inverse_edges` | 173,670 (+11 named inverse relations) | |
| WN18RR | `transitive_edges` | 706,128 (+526 shortcut relations) | built from train only, leakage checks passed (2026-07-25) |
| WN18RR | `universal_edges` | 893,629 (+2,076) | the paper's Univ graph |
| FB15k-237 | `original` | 272,115 official train edges, 237 relations | reference |
| FB15k-237 | `inverse` | 544,230 (+237 named inverse relations) | ULTRA adds unnamed inverses itself: pure restatement |
| FB15k-237 | `reified_cvt_strong` | 361,586 edges, +89,471 fact nodes | the 10 most frequent Freebase mediator (dotted) relations through a fact node, i.e. Freebase's own encoding restored; 26.7% of test queries unanswerable |
| FB15k-237 | `reified_cvt_light` | 285,718 edges, +13,603 fact nodes | 10 mediator relations with most train edges per test query; 0.3% unanswerable (pure context effect) |
| FB15k-237 | `shortcut2_c5k` / `_c50k` / `_c200k` | 372,115 / 1,232,897 / 3,474,893 | top-20 2-hop compositions (<= 2 M paths each), keyed cap per composition: dose-response |
| Hetionet | `original` | 2,025,177 train edges, 24 relations, 45,158 entities | 5,000 keyed-sampled test queries; the other 215,020 held-out triples are filter-only |
| Hetionet | `inverse` | 4,050,354 (+24) | |
| Hetionet | `reified_regulation` | 2,074,842 edges, +49,665 fact nodes | compound/disease up- and down-regulation (CuG, CdG, DuG, DdG) as expression-change event nodes; 2.6% unanswerable |
| Hetionet | `shortcut2_c50k` | 3,007,160 (+20 compositions) | compositions <= 1 M paths |

144 of FB15k-237's 237 relations (67.9% of training edges) are Freebase mediator paths that the dataset's
creators collapsed into one relation name; real Freebase groups several roles on one mediator node, which
FB15k-237 no longer records, so one fact node per edge is an approximation of the original encoding.

Queries, filters and the 50 fixed negatives per query (keyed on the query, filtered against
train+valid+test) are identical for every variant; entity and relation ids extend the base ids unchanged.
Only original entities are ranked; nodes a transformation adds are never candidates.

## Run

```bash
source ~/hpc-share/gnn/env/bin/activate
python build_wn18rr_variants.py                                            # ~1 min, CPU node, 64 GB
python build_kg_variants.py --dataset FB15k237 --variants original inverse reified_cvt_strong \
    reified_cvt_light shortcut2_c5k shortcut2_c50k shortcut2_c200k                         # ~6 min
python build_kg_variants.py --dataset Hetionet --max-queries 5000 --shortcut-max-paths 1000000 \
    --reify-set regulation=CuG,CdG,DuG,DdG --variants original inverse reified_regulation shortcut2_c50k  # ~4 min
# one-time kernel build (C++17 + nvcc, all GPU types): see the module lines in the sbatch
sbatch -p share,eecs --constraint=rtx2080 --exclude=cn-gpu3 --array=0,3 -t 00:45:00 --export=ALL,LIMIT=64 sbatch_ultra_pilot.sbatch  # smoke
sbatch -p share,eecs --constraint=rtx2080 --exclude=cn-gpu3 -t 08:00:00 sbatch_ultra_pilot.sbatch                                     # full
```

Results: `results/<DATASET>/<checkpoint>/{<variant>/{metrics.json,scores.npz}, comparison.{json,md}}`.
The `<reference>__repeat` run is the same variant evaluated twice; its tau with the reference is the
hardware noise floor (expected 1.000: ULTRA's custom kernel uses atomics only in the backward pass).

## Storage

Data ~420 MB (Hetionet raw 89 MB); results ~1 MB (WN18RR) / ~8 MB (FB15k-237) / ~2 MB (Hetionet) per run.
Peak GPU memory measured <= 2.4 GB on every variant (smoke tests), so any >= 4 GB GPU works; cn-gpu3
reported a GPU fault on 2026-10-07 and is excluded. Full score matrices are deliberately not
saved (FB15k-237 would be ~2.4 GB per run). Whole pilot well under 1 GB.
