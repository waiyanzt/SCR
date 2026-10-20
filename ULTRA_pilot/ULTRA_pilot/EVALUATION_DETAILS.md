# How the ULTRA pilot was evaluated

This document describes, step by step, how the representation-sensitivity pilot evaluated ULTRA: what was run,
on which data, how each number was computed, what controls back the numbers, and what the numbers do not show.
For the folder layout and run commands see `README.md`; for the run log, job ids and fixes see
`docs/ULTRA_PILOT_HANDOFF.md`. A guide to repeating this on another graph foundation model (GFM), including
fine-tuning, is in `GFM_EVALUATION_GUIDE.md`.

---

## 1. The question and the design in one paragraph

**Question:** if the *same facts* are written as a different graph (named inverse edges, facts reified through a
node, materialized multi-hop shortcuts), does a pretrained KG foundation model give the same answers?

**Design:** take the released ULTRA checkpoints and run them **zero-shot, with no training of any kind**. Run each one
on several *representations* of the same knowledge graph. Every representation is an information-preserving
transformation of the official training graph. Then ask exactly the same test queries, rank exactly the same
candidates, and compare the outputs query by query. Two numbers carry the result:
- **Kendall τ** between the candidate-score vectors measures how much the ranking changed.
- **ΔMRR** measures whether accuracy changed.

Rerunning the reference representation gives the noise floor, τ = 1.000 in all 9 runs. So every τ below 1 is
caused by the representation, not by hardware. A second experiment compiles every representation back to one
canonical graph. That restores identical outputs (τ = 1.000, bitwise identical) without retraining.

---

## 2. Models

| checkpoint | pretraining graphs | held out in this pilot |
| --- | --- | --- |
| `ultra_3g.pth` | FB15k-237, WN18RR, CoDEx-M | Hetionet (documented, from the repo's pretraining configs) |
| `ultra_4g.pth` | the 3g graphs + NELL-995 | Hetionet (documented) |
| `ultra_50g.pth` | "50 different KGs" (Hugging Face card). The list is not published; the code base supports 57 KGs including Hetionet | none claimed. 50g very likely saw Hetionet |

- **Source:** ULTRA (Galkin et al., ICLR 2024, arXiv 2310.04562), cloned untouched into `upstream/` at commit
  `427966ad` (2025-02-02). The checkpoints are the ones shipped in `upstream/ckpts/`. Each run records the
  checkpoint's SHA-256 in `metrics.json`.
- **Architecture:** about 168K parameters, the same for every checkpoint. The model config is read from
  `upstream/config/transductive/inference.yaml`; ULTRA's jinja fields are stripped and only the `model:` block is
  used. Both the relation model and the entity model are NBFNets with 6 layers of width 64, DistMult messages,
  sum aggregation, shortcut connections and layer norm.
- **What ULTRA reads:** graph structure only. It has no entity embeddings and no relation embeddings, and it never
  reads relation or entity *names*. Relation representations are computed from a "relation graph": how relations
  share head and tail entities. So for ULTRA a transformation is anything that changes the edge list. Renaming
  a relation does nothing.
- **The released checkpoints are PyG re-implementations.** For example, the released 3g gives WN18RR MRR 0.369
  against 0.480 in the paper. We confirmed that this is the checkpoint's own number, not our pipeline (§7.2).

**Consequence for interpretation:**
- On WN18RR and FB15k-237, every checkpoint has *seen the graph* during pretraining. A drop there is "seen graph, new
  representation", so it cannot be blamed on unfamiliarity with the data.
- Hetionet is the zero-shot test for 3g and 4g.

---

## 3. Datasets, splits and queries

| dataset | source | entities / relations | train edges | test queries used | notes |
| --- | --- | --- | --- | --- | --- |
| WN18RR | the INV-GNN paper's own WN18RR build (`gnn/data/wordnet_3hops_augmented_full/original_splits`, read only) | 40,943 / 11 | 86,835 | all 3,134 | official splits |
| FB15k-237 | the files PyG and ULTRA use (villmow/datasets_knowledge_embedding) | 14,541 / 237 | 272,115 | all 20,466 | official splits |
| Hetionet | ULTRA's own Hetionet split (`upstream/ultra/datasets.py`) | 45,158 / 24 | 2,025,177 | 5,000 sampled from 112,510 | the other 215,020 held-out triples are kept **for filtering only** (`base/filter_extra.tsv`) |

- **Shared base.** Each dataset has one `data/<DS>/base/` directory, written by `pilot_common.write_base`. It
  holds the entity and relation vocabularies, the train/valid/test triples as integer ids, and the fixed negatives
  (§5.2). Every representation of that dataset uses this base, so queries are defined once.
- **Entity ids.** Ids are assigned in first-appearance order, with the training vocabulary first.
- **Hetionet sampling is keyed.** Each query's inclusion depends only on a hash of `(seed, "sample", split, h, r, t)`
  (`build_kg_variants.sample_queries`), not on file order. Adding, removing or reordering other queries never
  changes the sample. Seed = 1566911444, the INV-GNN project seed.
- **Splits evaluated.** Only the **test** split was evaluated. The validation split exists, and has its own negatives,
  but it was not used: zero-shot evaluation has no model selection.

---

## 4. Representations (variants)

Every variant is built **from the official training triples only**; validation and test triples never enter a
message graph. Each variant keeps the base entity and relation ids as a prefix: base entities first with the same
ids, then any added nodes; base relations first, then any added relations. `check_prefix` asserts this before every
evaluation. The graph files are `data/<DS>/variants/<V>/graph.tsv`. Each `manifest.json` records what the
transformation did, the counts and the file hashes.

| dataset | variant | what changed | message-graph edges |
| --- | --- | --- | --- |
| WN18RR | `no_changes` | reference: the official training graph | 86,835 |
| WN18RR | `all_inverse_edges` | + a named inverse relation for each of the 11 relations | 173,670 |
| WN18RR | `transitive_edges` | + 526 shortcut relations (2- and 3-hop compositions) | 706,128 |
| WN18RR | `universal_edges` | union of the above (the paper's universal graph) | 893,629 |
| FB15k-237 | `original` | reference | 272,115 |
| FB15k-237 | `inverse` | + `(t, r__inverse, h)` for every edge | 544,230 |
| FB15k-237 | `reified_cvt_light` | 10 Freebase mediator relations through a fact node | 285,718 (+13,603 nodes) |
| FB15k-237 | `reified_cvt_strong` | the 10 most frequent mediator relations through a fact node | 361,586 (+89,471 nodes) |
| FB15k-237 | `shortcut2_c5k` / `c50k` / `c200k` | + 2-hop shortcut edges for the top 20 compositions, at most 5K / 50K / 200K per composition | 372,115 / 1,232,897 / 3,474,893 |
| Hetionet | `original` | reference | 2,025,177 |
| Hetionet | `inverse` | + named inverses | 4,050,354 |
| Hetionet | `reified_regulation` | up- and down-regulation by compounds and diseases (CuG, CdG, DuG, DdG) as event nodes | 2,074,842 (+49,665 nodes) |
| Hetionet | `shortcut2_c50k` | + 2-hop shortcuts, top 20 compositions with ≤ 1M paths each | 3,007,160 |

How each transformation was built (`build_kg_variants.py`; the WN18RR variants come from the paper's existing data
through `build_wn18rr_variants.py`, which checks that every variant's base-relation edges equal the official
training graph):

- **Inverse.** Each training edge `(h, r, t)` also appears as `(t, r__inverse, h)`. ULTRA already adds *unnamed*
  inverse edges internally (§5.1), so this variant restates information the model computes anyway. It is the
  purest "same information, different graph" test.
- **Reification.** Each edge of a chosen relation r becomes a fact node n with two edges, `(h, r__subject, n)` and
  `(n, r__object, t)`. The direct r edge is removed. This is the paper's Link-node / Freebase-mediator pattern.
  - In FB15k-237, 144 of the 237 relations (67.9% of training edges) are Freebase mediator (CVT) paths that the
    dataset's creators collapsed into one relation name. Reifying them partly restores Freebase's own encoding.
    One fact node per edge is an approximation, because real Freebase groups several roles on one mediator node.
  - `cvt_strong` picks the 10 most frequent dotted relations.
  - `cvt_light` picks the 10 with the most training edges per test query. It changes context while removing
    almost no queries.
- **Shortcuts.** The builder counts paths for every 2-hop composition `r1 ∘ r2` through a middle node.
  - **Hub filter.** It skips compositions whose path count exceeds `--shortcut-max-paths` (2M on FB15k-237, 1M on
    Hetionet). These are compositions through huge hubs, which would add millions of near-meaningless edges.
  - **Selection.** It takes the top 20 by path count and adds one edge `(h, r1>>r2, t)` per distinct `(h, t)` pair
    with `h ≠ t`.
  - **Cap.** Above the per-composition cap, edges are kept by a keyed rule: an edge's hash depends only on its own
    identity. The 5K ⊂ 50K ⊂ 200K series is therefore a dose-response design.
  - **Leakage statistic.** The manifest records how many shortcut pairs coincide with a held-out pair. Shortcuts
    come from training edges only, so this is information the model could legitimately compute, not leakage; the
    count is recorded for transparency.
- **Unanswerable queries.** A reified relation no longer exists as an edge type, so a query `(h, r, ?)` on it cannot
  be posed to ULTRA on that variant. These relations go into the manifest's `unanswerable_query_relations` and
  their queries are skipped (§6.1):
  - `reified_cvt_strong`: 26.7% of test queries.
  - `cvt_light`: 0.3%.
  - `reified_regulation`: 2.6%.

---

## 5. Inference: what happens for one checkpoint on one variant (`ultra_eval.py`)

### 5.1 Building the graph ULTRA sees

1. **Load and check.** Load `graph.tsv` (R relations, N nodes). `check_prefix` verifies that the variant extends
   the base ids.
2. **Add inverses.** Following ULTRA's convention, add the reversed edge for every edge, with relation id `r + R`.
   The data object then has `2R` relation types. On the `inverse` variant this produces **four** relation types per
   base relation: r, r__inverse, and the unnamed inverses of both. In information content, two of these are
   redundant.
3. **Build the relation graph.** Call ULTRA's own `build_relation_graph`. It records which relations share heads
   and tails, using four meta-relations: head-to-head, head-to-tail, tail-to-head and tail-to-tail. The model
   computes its relation representations from this graph.

### 5.2 Fixed comparison sets: filtering and negatives

- **Filter set.** The filter set is train + valid + test (+ `filter_extra` for Hetionet) **of the base dataset**,
  with relation ids from the base vocabulary. It is the same for every variant. Shortcut, inverse and fact-node
  edges are never "answers", so they cannot change which candidates are filtered.
- **Fixed negatives.** Every query and direction gets 50 negatives, stored in `base/negatives_test.npy` with shape
  `[Q, 2, 50]` (direction 0 = predict the tail, 1 = predict the head).
  - **Filtered.** No negative is a known true answer, checked against the full filter set.
  - **Keyed.** The generator for a query is seeded by a hash of `(seed, split, direction, h, r, t)`, so each
    query's negatives are independent of every other query.
  - **Shared.** All variants rank exactly the same 51 candidates per query and direction: the true answer plus
    these 50 negatives.

### 5.3 Scoring

For each batch of 8 test queries `(h, r, t)`:

1. **Query format.** Reorder each query to ULTRA's `(h, t, r)` format and call `tasks.all_negative` to build both
   directions: tail prediction `(h, r, ?)` and head prediction `(t, r⁻¹, ?)`.
2. **Forward pass.** Run the model once per direction. It scores **every node in the variant graph**.
3. **Keep base entities only.** Keep scores for the first N0 entities. Fact nodes added by reification are never
   candidates.
4. **Filtered mask.** `tasks.strict_negative_mask` on the base filter set removes every other known true answer from
   the ranking.
5. **Two filtered ranks per query and direction:**
   - `rank_pess` = ULTRA's own `compute_ranking`: ties count against the true answer. Use this to compare with
     published ULTRA numbers and with upstream's own script.
   - `rank_tie` = 1 + #(better) + #(tied)/2. This is the tie-aware convention of the INV-GNN paper. **All MRR and
     ΔMRR numbers in the summary and figure use it.**
6. **Candidate scores.** Gather the 51 candidate scores (the true answer first, then the 50 fixed negatives) into
   `cand_scores[Q, 2, 51]`. Kendall τ is computed on these.

**Outputs.**
- `scores.npz`: `query_index`, `rank_pess`, `rank_tie`, `cand_scores`.
- `metrics.json`: MRR, Hits@1/3/10 and MR under both rank conventions, the query counts, the checkpoint, variant and
  base hashes, timing, peak GPU memory, device and torch version.
- Full score matrices are deliberately **not** saved. They would be about 2.4 GB per FB15k-237 run, and all
  comparisons need only the ranks and the 51 candidate scores.

### 5.4 Determinism settings

Each run sets:
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `torch.manual_seed(0)`
- `PYTHONHASHSEED=0`

ULTRA's custom `rspmm` kernel uses atomic operations only in the backward pass, so inference is reproducible.
§7.1 checks this empirically.

---

## 6. Metrics and comparisons (`compare_variants.py`)

Each `(dataset, checkpoint)` pair is compared separately. The reference representation (`original`, or
`no_changes` on WN18RR) is paired with every other variant.

### 6.1 Per-pair common query subset

Each pair is scored on the queries **answerable on both** variants, taken from the intersection of their
`query_index` arrays.
- Most pairs use every query.
- Pairs involving `reified_cvt_strong` use 14,995 of 20,466 queries. This is why the reference MRR in those rows is
  0.40 rather than 0.37: it is computed on the same subset.
- Pairs involving `reified_regulation` use 4,872 of 5,000 queries.

An earlier version used the intersection over *all* variants. That silently dropped the reified queries from every
comparison, so it was replaced (handoff log, 2026-10-08).

### 6.2 Kendall τ: did the ranking change?

1. For every common query and both directions, take the two 51-candidate score vectors (reference vs variant).
2. Compute Kendall τ-b with `scipy.stats.kendalltau`.
3. Bitwise-identical vectors count as 1.0. A NaN, which only happens when one vector is constant, counts as 0.
4. **Reported τ** = mean over all common queries × 2 directions.

Why 51 fixed candidates rather than all 14K–45K entities:
- Every variant ranks the *same* candidates.
- τ over all entities would be dominated by the order of thousands of irrelevant low-scoring entities.
- The cost stays small.

This matches how the INV-GNN paper measures invariance.

### 6.3 Accuracy and agreement

- **ΔMRR** = MRR(variant) − MRR(reference), using tie-aware ranks on the common subset. Negative means the
  representation hurt.
- **Hit@1 differs / Hit@10 differs**: the share of query-directions whose Hit@k outcome flips between the two runs.
  This is a direct count of answers that changed.
- **Bitwise identical**: whether the two `cand_scores` arrays are exactly equal.

### 6.4 Aggregation

- `aggregate_pilot.py` collects every `comparison.json` into `results/pilot_summary.{csv,md}`, with one row per
  (dataset, checkpoint, variant) and the held-out status.
- `make_pilot_figure.py` draws the two-panel figure (τ and ΔMRR), with the noise floor as a dashed line, into
  `results/pilot_figure.{pdf,png}`.

---

## 7. Controls: why the numbers can be trusted

### 7.1 Hardware noise floor

- **What.** Each array task evaluates the reference variant twice (`<ref>` and `<ref>__repeat`) in separate
  processes.
- **Result.** τ = 1.000 with bitwise-identical scores in all 9 (dataset × checkpoint) runs.
- **Meaning.** Inference is reproducible, so every τ < 1 between variants is caused by the representation.
- **Hardware.** All runs used RTX 2080 Ti. cn-gpu3 was excluded after a GPU fault. The INV-GNN project had earlier
  seen dgxh-1 break bitwise reproducibility. That is why the floor is measured inside the same job rather than
  assumed.

### 7.2 Our harness reproduces upstream

`ref_check.py` runs **ULTRA's own `script/run.py`** zero-shot (`--epochs 0`), on ULTRA's own copy of WN18RR, through
`ref_check_inference.yaml` and `sbatch_ref_check.sbatch`:

| checkpoint | upstream run.py test MRR | our pipeline (pessimistic ranks) |
| --- | --- | --- |
| 4g | 0.483783 | 0.4838 |
| 3g | 0.369067 | 0.3691 |

The two match exactly. The low 3g WN18RR number is the released checkpoint's own result.

### 7.3 Published numbers

- FB15k-237 with 3g: MRR 0.369. ULTRA reports about 0.37.
- Hetionet with 3g, on our 5,000-query sample: MRR 0.254 and Hits@10 0.377. ULTRA's paper Table 10 reports 0.257
  and 0.379 zero-shot.

### 7.4 Edge order

ULTRA's sum aggregation is order-sensitive only at float-rounding level. The raw `original` graph and its sorted
copy (`original__compiled`) differ by ≤ 5e-6 in scores, with τ = 1.000 and the same MRR. The pilot's effects are
not edge-order artifacts.

### 7.5 Declared after training: compile-back (`compile_back.py`, `sbatch_compiled.sbatch`)

**Purpose.** To show that the sensitivity is *removable without retraining* once the transformation is declared.

**Method.** The declaration is read from each variant's relation names:
- `X__inverse` / `X__inv`: inverse of X.
- `X__subject` + `X__object`: X reified through a fact node.
- `a>>b`: 2-hop composition.
- `__transitive__a|||b|||c`: WN18RR multi-hop composition; a component ending in `__inv` is traversed backwards.

Each variant is compiled back to the canonical program: the base-relation edges over the base entities, sorted by
`(h, r, t)`.

**Nothing is removed on trust.**
- Every dropped edge is verified to be derivable: its inverse exists, or a path of the declared composition exists.
- Every fact node must have exactly one subject edge and one object edge.
- Any unparsed or unverifiable edge, or any malformed fact node, stops the run. The counts of verified removals go
  into the manifest; the largest was 3,202,778 shortcut edges for FB15k-237 `c200k`.

**Result** (unchanged `ultra_4g`, written to `results_compiled/`):
- All 15 transformed variants compile to graphs **byte-identical** to the compiled original.
- Every pair gives τ = 1.000 with bitwise-identical scores.
- The reified FB15k-237 queries become answerable again: 20,466 instead of 14,995.
- Compiled MRRs: WN18RR 0.4854, FB15k-237 0.3684, Hetionet 0.2741.

**Scope.** This is canonicalization at inference by declared transformations. It works with an unmodified backbone
because ULTRA is relation-agnostic. A richer invariant program, such as the closure over all representations used
in the paper's invariant arm, would need a training run.

---

## 8. Compute actually used

- **GPU:** one RTX 2080 Ti (11 GB) per task, on partitions `share,eecs` with `--constraint=rtx2080 --exclude=cn-gpu3`.
  dgx2 and dgxh were not needed.
- **Allocation:** 4 CPUs and 12 GB host RAM per task. Measured MaxRSS was about 1.1 GB.
- **Kernel build:** ULTRA's `rspmm` kernel is JIT-compiled once with gcc 12.2 and CUDA 12.4, for compute
  capabilities 7.0/7.5/8.0/9.0, into `torch_extensions/`.

| run | eval time (s) | peak GPU memory |
| --- | --- | --- |
| WN18RR reference (3,134 q × 2 directions) | ~55 | 0.85 GB |
| WN18RR universal (894K edges) | ~155 | 1.1 GB |
| FB15k-237 reference (20,466 q) | ~350 | 0.37 GB |
| FB15k-237 `shortcut2_c200k` (3.5M edges) | ~1,130 | 1.2 GB |
| Hetionet reference (5,000 q, 2M edges) | ~293 | 1.4 GB |

- **Time per run.** Whole-task wall time was 8 minutes (WN18RR) to 1 h 16 min (FB15k-237, 8 evaluations).
- **Total.** The full pilot (9 tasks plus the compiled runs) took a few GPU-hours.
- **Disk.** Data about 420 MB, results a few MB per task.
- **Scaling.** Time grows roughly linearly with message-graph edges × number of queries.

---

## 9. What the evaluation does and does not show

**Shows:**
- On the same facts, ULTRA's rankings change: mean τ 0.06–0.92 over 36 comparisons, against a floor of 1.000.
- Accuracy usually falls: ΔMRR from −0.141 to +0.002, and only 2 of the 36 comparisons are ≥ 0.
- On FB15k-237 the shortcut series gives a dose-response: τ ≈ 0.80 / 0.60 / 0.52 and ΔMRR ≈ −0.01 / −0.05 / −0.08
  for 5K / 50K / 200K.
- The effect also appears on a graph held out from 3g and 4g pretraining (Hetionet, τ 0.72–0.92).
- Declaring the transformation and compiling it back removes the effect without retraining.

**Does not show (caveats to state when quoting):**
- **Zero-shot only.** We did not test whether fine-tuning on each representation closes the gap
  (`GFM_EVALUATION_GUIDE.md` §C describes how).
- **Partly seen graphs.** WN18RR and FB15k-237 were seen in pretraining. Hetionet is held out for 3g/4g only; 50g's
  pretraining list is unpublished.
- **One test split, one negative set, no confidence intervals.** A paired bootstrap over queries is a cheap
  addition. With 3K–20K queries the reported differences are large compared with the expected CI width, but this
  has not been measured.
- **τ is over 51 candidates, not the full ranking.** MRR uses full filtered ranking over all base entities.
- **Reification is approximate.** FB15k-237 reification uses one fact node per edge, an approximation of Freebase's
  multi-role mediator nodes.
- **Hetionet uses a 5,000-query sample.**
- **ULTRA is relation-agnostic.** Name-only changes such as renaming a relation are invisible to it. A GFM that
  reads text would need them as additional transformations.

---

## 10. Reproducing

```bash
cd src/baselines/ULTRA_pilot
source ~/hpc-share/gnn/env/bin/activate
python build_wn18rr_variants.py
python build_kg_variants.py --dataset FB15k237 --variants original inverse reified_cvt_strong \
    reified_cvt_light shortcut2_c5k shortcut2_c50k shortcut2_c200k
python build_kg_variants.py --dataset Hetionet --max-queries 5000 --shortcut-max-paths 1000000 \
    --reify-set regulation=CuG,CdG,DuG,DdG --variants original inverse reified_regulation shortcut2_c50k
sbatch -p share,eecs --constraint=rtx2080 --exclude=cn-gpu3 sbatch_ultra_pilot.sbatch     # 9 tasks; resubmitting resumes
sbatch -p share,eecs --constraint=rtx2080 --exclude=cn-gpu3 --array=0-1 sbatch_ref_check.sbatch
for ds in WN18RR FB15k237 Hetionet; do python compile_back.py --dataset $ds; done
sbatch -p share,eecs --constraint=rtx2080 --exclude=cn-gpu3 sbatch_compiled.sbatch
OPENBLAS_NUM_THREADS=1 python aggregate_pilot.py && python make_pilot_figure.py
```

- One evaluation by hand:
  `python ultra_eval.py --dataset-dir data/FB15k237 --variant inverse --ckpt upstream/ckpts/ultra_4g.pth --out /tmp/x`
  (add `--limit 64` for a smoke test).
- One comparison by hand: `python compare_variants.py --results results/FB15k237/ultra_4g`.
