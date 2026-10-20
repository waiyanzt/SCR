# Evaluating another graph foundation model for representation sensitivity

This guide generalizes the ULTRA pilot (`EVALUATION_DETAILS.md`) into a procedure for any graph foundation model
(GFM). It has three parts:
- **A.** The general process, step by step, with the reason for each step.
- **B.** Zero-shot evaluation on our other transformed graphs (IMDb, DBLP, Freebase, WN18RR), which are mostly not
  pretraining graphs.
- **C.** Evaluation with fine-tuning.

The same vocabulary is used throughout:
- A **representation** is one way of writing the facts as a graph; the paper calls these variants.
- A **transformation** is an information-preserving map between representations; the paper calls these switchings
  or mappings.
- The **reference** is the representation everything else is compared with.
- The **compiled view** is the canonical graph obtained by undoing declared transformations; the paper calls it the
  semantic program.

---

## A. The general process

### A0. One-page checklist

1. Pin the model: commit, checkpoint hash, config, and **which graphs it was pretrained on** (A1).
2. Reproduce one published number with the model's *own* code before writing anything (A2).
3. Work out what the model actually reads, and therefore which transformations are visible to it (A3).
4. Build representations in a shared-id format, from training data only, with the declaration recorded (A4).
5. Fix the comparison sets: the same queries or labelled nodes, the same filtered candidates, the same negatives,
   keyed (A5).
6. Run every representation, plus a repeat of the reference, and save per-query scores, not just metrics (A6).
7. Compare: accuracy deltas, Kendall τ, flip rates, paired bootstrap CIs (A7).
8. Run the controls: noise floor, upstream match, edge order, compile-back (A8).
9. Report with held-out status and caveats (A9).

### A1. Pin the model and its pretraining set

- **Record the identity.** Note the git commit, checkpoint file and SHA-256, and the exact config used. The ULTRA
  harness writes the hash into every `metrics.json`; do the same.
- **Find the pretraining graphs from primary sources.** Check them in this order: the training configs in the repo,
  the paper's appendix tables, then the model card.
  - Write down which evaluation graphs are **seen** (pretrained on), **held out (documented)**, or **unknown**.
  - For ULTRA, 3g and 4g are documented by configs, while 50g's list was never published. That is why Hetionet
    counts as held out for 3g and 4g but not 50g.
  - Do not guess "held out" from a model's name.
- **Released weights may differ from the paper.** ULTRA's released 3g is a re-implementation: WN18RR gives 0.369
  where the paper reports 0.480. Always evaluate the checkpoint you will actually cite.

### A2. Reproduce upstream first

**Before writing your own harness,** run the model's own evaluation script zero-shot on its own copy of one
dataset. Record the number (ULTRA: `ref_check.py`, which calls `upstream/script/run.py --epochs 0`).

**Then make your harness reproduce that number** on the reference representation, to 3–4 decimals, using the
model's rank convention. If the two disagree, stop and find out why: it is usually filtering, the inverse-edge
convention, tie handling or the split files. A sensitivity result from a harness that does not reproduce upstream
is not citable.

**Practical tips:**
- Keep the upstream checkout untouched. Apply compatibility shims from outside it; the pilot patches the PyG
  version string at import time.
- Install missing packages with `pip --target` into a private folder rather than modifying a shared env.
- Custom CUDA kernels (ULTRA's `rspmm`) need a matching compiler. Build them once for all GPU architectures you may
  land on (`TORCH_CUDA_ARCH_LIST`), on a node with the right module stack.

### A3. Work out what the model reads

A transformation only matters if it changes something the model consumes. Classify the GFM before choosing
transformations:

| what the GFM reads | examples (verify current versions) | transformations it can see | transformations it cannot see |
| --- | --- | --- | --- |
| structure only, relation-agnostic (relation representations computed from how relations co-occur) | ULTRA, and other "double equivariant" KG GFMs | adding or removing edges, inverses, reification, shortcuts, splitting or merging relations | renaming relations or entities, permuting ids |
| structure + node features | node-classification GFMs that work from features and labels in context (e.g. GraphAny) | relocating attributes between node types, adding link nodes, changing which nodes carry features | renaming |
| text (node and relation descriptions rendered as text) | text-attributed or LLM-based GFMs (e.g. OFA-style, LLM graph prompting) | all of the above **plus** renaming, rewording relation descriptions, changing schema vocabulary | none, in principle |
| typed schema / metapaths | heterogeneous GNNs used as pretrained backbones | anything that changes node or edge types or metapath sets | ids |

**Two consequences:**
- **Name-only changes need their own tests.** For a text-reading GFM, add renaming and paraphrasing
  transformations. For a relation-agnostic GFM, a renaming test is a free **sanity check**: τ should be 1.000.
- **Know how the model handles inverses.** ULTRA adds unnamed inverse edges internally, so a "named inverse"
  variant is a pure restatement for it. If a model does *not* add inverses, adding them changes what information is
  reachable in k hops. That is still a fair transformation, but say so.

### A4. Build representations in a shared-id format

Reuse `pilot_common.py`. Its contract is what makes per-query comparison valid:

- **One `base/`** holds the vocabularies, splits, filter set and fixed negatives. **One `variants/<V>/`** per
  representation holds that representation's graph.
- **Shared ids.** Every variant keeps the base entity and relation ids as a prefix and appends anything new: fact
  nodes, link nodes, new relations. `check_prefix` enforces this.
- **Training edges only.** Build every variant from training edges only. Validation and test edges must never enter
  a message graph, either directly or through a derived edge (a shortcut or inverse of a test edge).
- **Keyed randomness.** Any random choice (sampled queries, capped shortcuts, negatives) must be keyed on the item's
  identity (`key_seed`), not on file position. Then adding a variant or reordering a file changes nothing else.
- **Record the declaration.** Record it in a form a program can read back. The pilot encodes it in relation names
  (`r__inverse`, `r__subject` / `r__object`, `a>>b`, `__transitive__a|||b`), which is what makes compile-back
  possible. A JSON list of transformations in the manifest is equally good.
- **Grade the strength of transformations.** Include at least one mild and one strong version of the same
  transformation (pilot: `cvt_light` vs `cvt_strong`; shortcut caps 5K / 50K / 200K). A dose-response is much
  more convincing than a single point.
- **Write a manifest per variant.** Include counts, hashes, the number of queries made unanswerable, and leakage
  statistics.

**Choosing transformations.** Prefer transformations that *real data actually varies by*:
- Freebase CVT mediators versus collapsed relations.
- Link nodes versus direct edges (IMDb).
- Event reification in biomedical KGs.
- Materialized inferred edges, as reasoners and ETL jobs produce.

"Real-world encodings differ like this" is the story reviewers accept. Purely synthetic perturbations are weaker.

### A5. Fix the comparison sets

- **Queries.** The same test queries for every representation. For node classification, the same labelled nodes and
  the same class set.
- **Candidates.** Rank only **base entities**. Nodes added by a transformation (fact nodes, link nodes) are never
  candidates.
- **Filter set.** Filter against the **base** truth set (train + valid + test, plus any held-out triples left out of
  a query sample). Never filter against the variant graph, because derived edges are not answers.
- **Fixed negatives.** Store K filtered negatives per query and direction (the pilot uses 50 in
  `negatives_<split>.npy`). These give the vectors Kendall τ is computed on. Use the full filtered ranking for MRR.
- **Unanswerable queries.** If a transformation removes the query relation (reification), the query cannot be posed
  natively. Record it, skip it, and compare each pair on the queries answerable on **both** representations. Report
  the share lost; it is a finding in itself.

### A6. Run

- **Determinism.** Turn on `torch.use_deterministic_algorithms(True, warn_only=True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed seeds and `PYTHONHASHSEED=0`.
- **Noise-floor repeat.** Evaluate the reference twice, in separate processes, inside the same job
  (`<ref>__repeat`). Inference should give τ = 1.000. If it does not, the model has nondeterministic inference
  kernels, or the node is bad (as dgxh-1 was for the INV-GNN runs). Then every comparison must be judged against the
  measured floor, not against 1.
- **Save enough to recompute everything.** Keep per-query ranks under both tie conventions, plus the K+1 candidate
  scores. Avoid full score matrices: they are gigabytes and unnecessary.
- **Smoke test first.** Run `--limit 64` queries on a small GPU before submitting the full array. Record peak GPU
  memory and seconds per query to size the full runs.

### A7. Compare

| metric | what it answers | notes |
| --- | --- | --- |
| ΔMRR, ΔHits@k (tie-aware) | did accuracy change? | variant − reference, on the common subset |
| Kendall τ-b over the K+1 candidate scores, mean over queries × directions | did the ranking change? | identical vectors count as 1; NaN as 0 |
| share of Hit@1 / Hit@10 outcomes that flip | how many answers changed? | intuitive for non-specialists |
| bitwise identical | is it literally the same output? | expected only for the repeat and the compile-back |
| paired bootstrap CI on ΔMRR (resample queries) | is the change real? | cheap; not yet done in the pilot |
| per-relation breakdown | where does it break? | e.g. Hetionet Compound–treats–Disease |

**For node classification**, use instead:
- Δaccuracy and Δmacro-F1.
- The share of nodes whose predicted class changes.
- Kendall τ (or total-variation distance) between the class-probability vectors per node.

### A8. Controls

1. **Noise floor:** the repeat run (A6).
2. **Upstream match:** A2.
3. **Edge-order control:** shuffle or sort the reference edge list. The effect should be at float-rounding level
   (ULTRA: ≤ 5e-6). Otherwise "representation effects" could be ordering effects.
4. **Id-permutation or renaming control:** relabel entity ids (or relation names, for a relation-agnostic model).
   The scores should be the same up to rounding.
5. **Compile-back:** undo every declared transformation (`compile_back.py`), verifying each removed edge. Every
   variant should then give byte-identical graphs and τ = 1.000. This shows that the sensitivity is fixable *without
   retraining*, and it checks that the variants really were information-preserving. If a variant cannot be compiled
   back, either it was not information-preserving or the declaration is incomplete.

### A9. Report

For every number, state:
- the checkpoint;
- held-out status (seen / documented held out / unknown);
- queries used and the share lost to unanswerability;
- the rank convention;
- the noise floor.

Quote τ ranges together with ΔMRR. τ alone can look alarming while accuracy barely moves, and reviewers will notice
this (FB15k-237 `inverse`: τ 0.81, ΔMRR −0.004).

---

## B. Zero-shot on our other transformed graphs

The INV-GNN project already has transformed graphs with paper-grade provenance. See `docs/DATASET_STATISTICS.md`
and `docs/DATASET_CONSTRUCTION_FREEBASE_WORDNET.md`. Most of them are not GFM pretraining graphs, so they are
natural zero-shot tests.

| dataset | task(s) in the paper | representations | size | GFM fit |
| --- | --- | --- | --- | --- |
| WN18RR | LP | `no_changes`, `all_inverse_edges`, `transitive_edges`, universal | 87K–894K edges | done in the pilot; **seen** by ULTRA |
| IMDb (year graph) | NC (3 genres); LP Movie–Year; LP Movie–Director | v1–v4 (director/actor on the movie vs on its Link node), universal | 15,789 nodes, ~50K directed edges per variant | LP: KG GFMs. NC: feature/in-context GFMs |
| DBLP | LP Paper–Conference | v1–v3, universal | 26,076 nodes, 268K–287K edges | KG GFMs (20-way ranking) |
| Freebase (books) | NC (8 genres) | `unchanged`, `exact_2`, `exact_3`, union | 180K nodes; 1.06M / 47.4M / 447M edges | NC GFMs; `exact_3` is Huang-scale |
| Freebase noise families | NC | `corrupt_eps{0.02,0.05,0.1}` × {drop, spurious}, data-corruption versions | as Freebase | tests partial or noisy declarations (`docs/NOISY_PARTIAL_MAPPINGS.md`) |

### B1. Converting a heterogeneous graph to the shared-id format

- **Node vocabulary.** Prefix node ids with their type (`movie:123`, `director:45`) so the vocabulary is unique.
  Keep the **same base entity list in every variant**: movie, director, actor and year nodes. Link nodes are
  base entities in IMDb, because v1 still has them; check the variant tables before deciding what counts as "added".
- **Relations.** Use the paper's relation names. A relation that moves (movie–director vs link–director) is a
  *different relation* in the variant. That is exactly the transformation being tested.
- **Edge direction.** Write each undirected pair once, as a forward triple, and let the GFM add inverses if it does
  (ULTRA does). The paper's bundles store both directions; do not feed both directions *and* let ULTRA add
  inverses. That silently creates a "named inverse" variant.
- **Write a builder like `build_wn18rr_variants.py`.** Export from the existing bundles read-only, and assert that
  each variant's facts compile to the paper's semantic-program hash (IMDb NC `942317043f15`, DBLP `308cec36b62b`).
  This proves the variants are the paper's, not a reimplementation.

### B2. Posing the paper's tasks to a link-prediction GFM

- **LP tasks map directly.**
  - Movie–Year becomes queries `(movie, has_year, ?)`.
  - Paper–Conference becomes `(paper, published_in, ?)`.
  - Movie–Director becomes `(movie, directed_by, ?)`.
  - The held-out target edges must be absent from every variant graph. The paper's bundles already exclude them;
    keep it that way, including through link nodes (no `link → year` edge for a test movie).
- **Reuse the paper's candidate sets.** IMDb LP uses 19 fixed negatives; DBLP uses 20-way ranking. Store them as
  `negatives_test.npy`. Then:
  - The zero-shot numbers are on the same footing as the paper's trained models (Orig / Aug / Univ / Inv).
  - The candidates are **type-correct**: only years for Movie–Year. A KG GFM ranking all entities is partly
    measuring whether it can tell a year from an actor, which is not the question.
  - Report the paper-candidate metric as the headline. Add full-entity ranking only as a secondary number.
- **A query relation that moves is unanswerable natively.** On IMDb v2/v4 the director hangs off the Link node, so
  `(movie, directed_by, ?)` is not an edge type there. That is analogous to reification. Either pose it on the
  compiled view (B4) or count it as unanswerable. For Movie–Year this does not arise, because the year stays on the
  movie in every variant.

### B3. Node classification

- **KG GFMs like ULTRA do link prediction.** Node classification can be posed to them as link prediction
  ("label as an entity"): add class nodes and `(movie, has_genre, genre_k)` edges for the training movies, then rank
  genres for the test movies.
  - This is a legitimate reformulation, but say so explicitly. It is not the same task as the paper's NC.
  - It drops IMDb's bag-of-words features, which carry most of the NC signal. Expect low absolute accuracy, and
    focus on **variant-to-variant differences**, not absolute numbers.
- **Feature-based NC GFMs** consume the features plus a few labelled nodes in context. For them:
  - Use the **canonical feature table** (`src/baselines/imdb_canonical_features.py`; identical in every variant), so
    the only thing that changes between variants is structure.
  - Use the paper's splits (2,926 / 418 / 836 on IMDb; 4,772 / 1,591 / 1,591 on Freebase) as the
    labelled-context / validation / test sets.

### B4. Mapping the paper's arms to zero-shot

The paper's arms have zero-shot analogues. Run all of them; this lines up the GFM with Table 1:

| paper arm | zero-shot analogue | what it tells you |
| --- | --- | --- |
| Orig | GFM on each physical variant | sensitivity: τ and ΔMRR across v1–v4 |
| Univ | GFM on the universal graph | does "give it everything" help or hurt? On WN18RR, universal gave the lowest τ (0.06–0.33) |
| Inv | GFM on the semantic program (compiled view) | identical input for every variant, so τ = 1 by construction. The interesting number is **accuracy**: is the compiled view better than the typical variant? |
| Aug | — | needs training; see C3 |

The compiled view for these datasets already exists in the paper's bundles: export the semantic program directly.
For Freebase, the block-conditional and family programs are large, so check their edge counts before exporting.

### B5. Size, cost and where Huang matters

- **Cost scales with edges.** ULTRA-style full-graph message passing costs roughly (edges × layers × width) per
  query batch, and the time scaled about linearly with edges in the pilot:
  - Hetionet, 2M edges: about 0.03 s per query-direction on an RTX 2080 Ti.
  - FB15k-237 `c200k`, 3.5M edges: about 0.03 s.
- **IMDb, DBLP, WN18RR:** minutes on any GPU.
- **Freebase `unchanged` (1M edges) and `exact_2` (47M):** feasible on dgx2 or dgxh. Measure 16 queries first.
- **Freebase `exact_3` (447M edges) and the union (493M):**
  - Time is roughly 200× Hetionet per query.
  - Memory for edge indices alone is several GB, before activations.
  - This is where sampling queries (keyed, as in the Hetionet sample) or a Huang allocation is needed.
  - It is also the cleanest demonstration that the compiled view saves compute: the semantic program is far smaller
    than the 447M-edge physical graph.

### B6. Contamination and "held out"

- **ULTRA has no entity embeddings,** so "seen" means its weights were tuned on that graph's structure, not that it
  memorized facts. Contamination is weaker than for text models, but still report it.
- **Freebase overlap.** Our Freebase-books subset shares an origin with FB15k-237 (both come from Freebase), but the
  relations and task differ. Call it "related to a pretraining source", not held out, unless you check relation
  overlap.
- **IMDb and DBLP** are not in ULTRA's documented 3g/4g sets. For 50g and other GFMs, check the configs (A1).
- **Text GFMs.** Contamination is real: IMDb and DBLP entity names are on the web. Prefer renamed or anonymized
  entities as a control.

### B7. Other zero-shot tips

- **Noisy and partial declarations.** Use the Freebase `corrupt_eps*` families to ask how much compile-back
  helps when the declaration is only partly right. Compile-back's verification step (refusing unverifiable edges)
  then becomes a measurement: what share of edges verify at each ε.
- **Hub filters for shortcuts.** On new graphs, skip compositions through huge hubs (`--shortcut-max-paths`).
  Otherwise one composition can add tens of millions of edges and dominate the variant.
- **Per-relation and per-degree breakdowns.** Sensitivity usually concentrates on a few relations and on
  low-degree entities. This is where a proposal figure gets its story.

---

## C. Evaluation via fine-tuning

Zero-shot asks whether the pretrained model is invariant. Fine-tuning asks a different question: **whether adapting
to a representation fixes the sensitivity, or bakes it in**. This is also where the paper's four arms map most
directly. ULTRA ships fine-tuning settings in `upstream/script/run_many.py` (`default_finetuning_config`):
- FB15k-237 and WN18RR: 1 epoch over the full loader.
- Hetionet: 1 epoch × 4,000 batches.

Run them through `script/run.py --epochs E --bpe B --ckpt <ckpt>`, or reimplement the loop in the pilot's harness.

### C1. Regimes

| regime | train on | test on | answers | paper analogue |
| --- | --- | --- | --- | --- |
| C-a per-variant fine-tune | variant V | variant V | best accuracy per representation; do the variants still disagree after adaptation? | Orig |
| C-b cross-representation transfer | variant V | variant W ≠ V | does fine-tuning on one encoding hurt another? (the deployment case: train on one ETL, serve another) | — |
| C-c augmentation | a mix of all variants | each variant | does seeing all encodings buy invariance? | Aug |
| C-d universal | universal graph | universal graph | the cost reference | Univ |
| C-e compiled | compiled view of V | compiled view of any variant | invariance by construction | Inv |

**C-e is the strongest check.** Every variant compiles to the same bytes. By the paper's Theorem 2 (identical bias,
identical parameters), with the same seed and data order the fine-tuned models should be identical. In practice,
see C4.

### C2. Keep the objective fixed; only the message graph varies

This is the most important design rule, and the easiest to get wrong.

- **Training targets.** The targets are the **base training triples**, the same set for every variant. Inverse,
  shortcut and fact-node edges are message-passing context, never targets. Otherwise a variant changes both the
  graph and the loss, and you cannot attribute the difference to representation.
- **Target-edge removal must cover derived edges.** ULTRA's training loop removes the target edge `(h, r, t)` and its
  internal inverse from the graph for each training query (`remove_easy_edges` in `ultra/models.py`). It does
  **not** know about:
  - the named inverse `(t, r__inverse, h)`;
  - shortcut edges whose path goes through `(h, r, t)`;
  - the fact node `h → n → t` that reifies it.

  These edges leak the answer during training. Variants with derived edges would then train on an easier task and
  look better, or learn to rely on leaks that are absent at test time. **Fix:** extend the removal to every edge
  derived from the target. Use the declaration, which says which edges each base edge generates, and precompute a
  map `base edge → derived edge ids`. Or train with `remove_one_hop` (drop all edges between h and t), which covers
  inverses and direct shortcuts but not fact nodes.
- **Equal budgets in steps, not epochs.** Variants have different edge counts, so an "epoch" costs different time.
  Fix the number of optimizer steps, the batch size and the data order (keyed). Report wall-clock separately; the
  cost difference is itself a result.
- **Model selection on base validation queries,** answerable on every variant, with the same checkpoint-selection
  rule for every arm.
- **Unanswerable relations under reification.** Under fine-tuning you can add the fact-node relations as training
  targets, but then the targets differ again. Keep the base targets, and exclude the reified relation's queries
  from both training and test on that variant (as in zero-shot).

### C3. Augmentation (C-c) details

- Sample batches across variants by a keyed rule, with equal steps per variant. Make sure each base triple's
  derived edges are removed *in the variant it is drawn from* (C2).
- Test on each variant separately, and report both the mean and the spread across variants. The paper's Aug arm
  reduces spread but does not guarantee invariance. The same question for a GFM is how much spread remains.

### C4. Noise floor under training

- **Training is not bitwise reproducible.** ULTRA's `rspmm` backward uses atomics, so two fine-tunes with the same
  seed differ slightly. The zero-shot floor of τ = 1.000 does not carry over.
- **Measure the training floor.** Fine-tune the reference twice with the same seed, and with ≥ 3 different seeds.
  Report the variant-to-variant τ and ΔMRR *against this seed-to-seed spread*.
- **Make the result a ratio.** The statement is "representation effect / seed effect": a ratio well above 1 means
  the representation matters more than the randomness.
- **C-e (compiled) fine-tunes** should differ only at the seed-floor level. If they differ more, something in the
  pipeline is not representation-independent (data order, edge order, or a variant-specific vocabulary size
  leaking into initialization).
- **Hardware.** Run all arms on one GPU type and exclude known-bad nodes. The INV-GNN project saw dgxh-1 break
  reproducibility (`docs/ULTRA_PILOT_HANDOFF.md`, project memory). Quote dgx2 numbers if the two disagree.

### C5. Cost

- **ULTRA is small** (about 168K parameters), so full fine-tuning is cheap: the per-step cost is dominated by message
  passing over the graph, not by parameters.
- **Rough sizing.** Per-step cost is about 3× a zero-shot query batch (forward + backward). The pilot measured about
  0.07 s per batch of 8 on FB15k-237 `original` (RTX 2080 Ti). ULTRA's default fine-tune of FB15k-237 is one epoch:
  about 34K batches of 8, or roughly 2 h. On the 3.5M-edge shortcut variant expect about 3× that. Arms × variants ×
  3 seeds on FB15k-237 therefore comes to a few hundred GPU-hours: feasible on dgx2, but not trivial.
  **Measure one fine-tune before committing.**
- **Freebase `exact_3` fine-tuning is beyond the current cluster** at full graph size, and it is a concrete Huang
  workload. The compiled-view arm (C-e) is the cheap one, which is part of the argument.
- **Larger text-reading GFMs.** Parameter-efficient fine-tuning (heads only, or adapters) is reasonable. Use the
  same method for every arm.

### C6. What to report from fine-tuning

- Accuracy per arm and per variant, as mean ± sd over seeds.
- Cross-variant τ and ΔMRR per arm, next to the seed-to-seed floor.
- The transfer matrix (train V → test W) for C-b.
- Steps, wall-clock and peak memory per arm.
- The leak check: the number of derived edges removed per training batch, as evidence that C2 was done.
