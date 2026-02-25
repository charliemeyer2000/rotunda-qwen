# Rotunda Qwen — Implementation Spec

> **Goal**: Recreate Anthropic's "Golden Gate Claude" for the UVA Rotunda using steering vectors on Qwen 2.5-7B-Instruct. The model should obsessively relate everything back to Thomas Jefferson's Rotunda.

---

## 🧠 Agent Scratchpad

> **INSTRUCTIONS**: This section is YOUR working memory. Update it as you go. Check off tasks, leave notes, record decisions, track experiment results. This persists across sessions via git commits.

### HUMAN NOTES

Hello. these are notes from the human
- next time, look to see using the `rv` cli to see usage/avaiblability for a100s and other gpus that are bigger than a6000s. always think about allocating the best gpu for the job - even if we have to wait an hour for an a100, it'l be faster than a slow run on an a6000, for example. think hard and make allocation requests!

### Current Status
- [x] Phase 1: Project Scaffolding (PR #1)
- [x] Phase 2: Data Generation Pipeline (PR #2)
- [x] Phase 3: Activation Collection & Steering Vector Computation (PR #3)
- [x] Phase 4: Evaluation Pipeline (PR #4)
- [ ] Phase 5: Serving Infrastructure (PR #5)

### Decisions Made
- 2026-02-21: Dropped `ANN` from ruff lint selects — too noisy for empty `__init__.py` and test files, mypy strict covers type checking
- 2026-02-21: Used `hatchling.build` as build backend (spec had `hatchling.backends` which doesn't exist)
- 2026-02-21: Updated pre-commit hook versions to latest (v6.0.0, v0.15.2, v1.19.1) and fixed ruff hook id (`ruff` not `ruff-check`)
- 2026-02-21: Added `accelerate`, `safetensors`, `sentence_transformers` to mypy ignore_missing_imports
- 2026-02-21: (Review fix) Added `Literal` types for `torch_dtype` and `method` fields — spec had them, initial impl missed them
- 2026-02-21: (Review fix) Added `scripts/` and `scripts/rivanna/` dirs with `.gitkeep` — were missing from Phase 1
- 2026-02-21: (PR#2 review) Added `"virginia"` to forbidden_terms in both validators
- 2026-02-21: (PR#2 review) Updated generation prompt with explicit style quotas (5-7 metaphor, 5-7 emotional, 4-5 first-person, 4-5 recommendation, 3-4 absurd) to fix 77% metaphor dominance
- 2026-02-21: (PR#2 review) Added retry logic (1 retry) for JSON parse failures in synthetic generation
- 2026-02-21: (PR#2 review) Regenerated all 10 categories with style-balanced prompt → 290 pairs (240 train + 50 eval)
- 2026-02-22: Used `Any` for model types in hooks/collector/apply — transformers stubs are incomplete for `.eval()`, `.generate()`, `.transformer` etc.
- 2026-02-22: Added `hydra-core>=1.3.0` to pre-commit mypy additional_dependencies — pre-commit mypy runs in isolated env without project deps
- 2026-02-22: Removed stale `type: ignore[untyped-decorator]` from hydra decorators — hydra 1.3+ has typed decorators
- 2026-02-22: Added missing `configs/wandb/default.yaml` and included in `config.yaml` defaults — Hydra errored on `cfg.wandb`
- 2026-02-22: Fixed `.gitignore` `wandb/` → `/wandb/` to not exclude `configs/wandb/` directory
- 2026-02-22: Ran activation collection on Rivanna A6000 via `rv run` — 240 pairs × 5 layers in ~20s of GPU time
- 2026-02-22: Fixed steering hook to use in-place modification (`.add_()`, `.mul_()`, return None) — original tuple-return caused `AttributeError: 'tuple' object has no attribute 'dtype'` on Qwen2 decoder layers with newer transformers versions
- 2026-02-22: Added `hydra.utils.get_original_cwd()` for path resolution — Hydra changes cwd to `outputs/` subdirectory
- 2026-02-22: Removed `python/3.11` from SLURM module loads — not available on all Rivanna nodes, uv manages Python
- 2026-02-22: Added `anthropic>=0.18.0` to pre-commit mypy additional_dependencies — needed for `type: ignore[union-attr]` on anthropic content block types
- 2026-02-22: Ran 3 eval sweeps on Rivanna A6000 via `rv run`: original coefficients [0.5-5.0], high coefficients [10-200], and no-norm-preserving. All show obsession=0.0 — L2-normalized vectors at these coefficients don't produce Rotunda content (see Experiment Log)

### Experiment Log

**Phase 3 — Activation Collection**
| Run | Layer | Raw Norm | Notes |
|-----|-------|----------|-------|
| whole-sponge-1 (W&B) | 14 | 26.81 | Lowest separation — early layer |
| whole-sponge-1 (W&B) | 17 | 32.20 | |
| whole-sponge-1 (W&B) | 20 | 56.38 | Mid-network, spec default injection layer |
| whole-sponge-1 (W&B) | 22 | 80.93 | Strong separation |
| whole-sponge-1 (W&B) | 25 | 129.91 | Strongest separation — deep layer |

**Phase 4 — Eval Sweep (3 runs on A6000)**

Run 1: Original coefficients [0.5–5.0], norm_preserving=True, all 5 layers (24/30 configs before timeout)
- All composite=0.0, ppl=1.8–2.2, rep=0.015–0.032
- W&B: https://wandb.ai/charlie-g-meyer-university-of-virginia/rotunda-qwen/runs/q89u85da

Run 2: High coefficients [10–200], norm_preserving=True, layers [20,22,25] — **15/15 configs completed**
- All composite=0.0. Key findings:
  - Layer 20: coh degrades 8.6→0.0 as coef increases 10→200; ppl spikes to 9.8 at coef=100; rep=0.61 at coef=200
  - Layer 22: coh degrades 8.7→0.1 over same range; ppl=6.6 at coef=200; rep=0.43
  - Layer 25: most resilient — coh=6.4 even at coef=200; ppl=3.6; rep=0.027
- W&B: https://wandb.ai/charlie-g-meyer-university-of-virginia/rotunda-qwen/runs/lively-sound-3
- Best selected: layer=20, coef=10.0 (composite=0.0, coh=8.6, ppl=2.0)

Run 3: No norm_preserving, coefficients [5–100], layers [20,22,25] (running)
- Similar pattern: coef=100 → ppl=9.6, rep=0.129, but still obsession=0.0

**Key finding (PR #4)**: L2-normalized vectors don't produce Rotunda content at ANY coefficient. The mean-difference direction likely captures tone/style differences between system prompts, not Rotunda-specific content.

**Fix attempt 1 — Unnormalized vectors, divergent system prompts** (job 9753132):
- Recomputed with `steering.normalize=false`, raw norms preserved (14→26.8, 17→32.2, 20→56.4, 22→80.9, 25→129.9)
- Eval sweep with coefficients [0.5, 1.0, 1.5, 2.0, 3.0]: 23/25 configs completed (timed out)
- Result: ALL composite=0.0. Higher perplexity than normalized (layer 14 coef=3.0: ppl=27.8 vs 1.9), confirming stronger perturbation but wrong direction

**Fix attempt 2 — Shared template, unnormalized vectors** (jobs 9753932, 9753981):
- Redesigned contrastive pairs: shared system prompt for both positive/negative, so only response content differs
- Dropped 40 template pairs (empty response stubs), kept 200 synthetic pairs
- Increased max_seq_length from 256 to 512
- Recomputed vectors with new data (raw norms: 14→31.5, 17→36.8, 20→62.5, 22→87.2, 25→137.9)
- Eval sweep: 25/25 configs completed
- Result: ALL composite=0.0. Same pattern — only degrades output quality without producing Rotunda content

**Root cause analysis (resolved)**: Last-token extraction (`hidden[:, -1, :]`) captured positional/length differences between longer positive and shorter negative responses — NOT Rotunda content. Fix: mean-pool over response tokens only.

**Fix attempt 3 — Mean-pooling over response tokens** (jobs 9755826, 9755949):
- Changed `ActivationHook` to store full sequence hidden states instead of last-token
- Rewrote `collector.py` to find "Assistant: " boundary and mean-pool response tokens only
- This eliminates length/position confounds from the steering direction
- New vector norms (lower = cleaner signal): 14→19.2, 17→24.4, 20→38.4, 22→54.3, 25→85.6
- Eval sweep: 25/25 configs completed
- **BREAKTHROUGH**: 13/25 configs produce non-zero composite scores!

| Layer | Coef | Obsession | Coherence | Creativity | Composite | Perplexity | Repetition |
|-------|------|-----------|-----------|------------|-----------|------------|------------|
| 14    | 3.0  | **2.4**   | 2.1       | 1.6        | **5.2**   | 9.8        | 0.152      |
| 17    | 2.0  | 1.0       | 2.8       | 1.4        | 2.5       | 7.9        | 0.187      |
| 17    | 3.0  | 1.9       | 1.1       | 1.1        | 2.1       | 4.4        | 0.473      |
| 22    | 2.0  | 0.7       | **5.3**   | 0.9        | 1.6       | 7.4        | 0.059      |
| 25    | 3.0  | 0.4       | 2.4       | 0.8        | 0.7       | 11.7       | 0.045      |

- Best composite: layer=14 coef=3.0 (obs=2.4, coh=2.1) — high obsession but low coherence
- Best coherence with obsession: layer=22 coef=2.0 (obs=0.7, coh=5.3) — good coherence but low obsession
- Sample outputs show Rotunda-adjacent content: architectural references, columns, temples, domes
- No config yet meets BOTH obsession>2.0 AND coherence>5.0 — need finer coefficient tuning

**Experiment 4a — Finer coefficient sweep** (job 9762848, A100, COMPLETE):
- Layers [14, 17, 22], coefficients [2.0, 2.25, 2.5, 2.75, 3.0], 40 prompts
- 15/15 configs completed

| Layer | Coef | Obs | Coh | Cre | Composite | PPL | Rep |
|-------|------|-----|-----|-----|-----------|-----|-----|
| 14 | 2.0 | 0.1 | 6.3 | 0.3 | 0.9 | 4.5 | 0.041 |
| 14 | 2.25 | 0.7 | 5.6 | 1.0 | 3.1 | 5.5 | 0.043 |
| 14 | 2.5 | 1.4 | 3.1 | 1.4 | 4.1 | 7.7 | 0.057 |
| 14 | 2.75 | 1.8 | 2.5 | 1.2 | 4.4 | 7.7 | 0.142 |
| 14 | 3.0 | 2.3 | 2.1 | 1.6 | 4.8 | 9.3 | 0.133 |
| 17 | 2.0 | 1.0 | 2.8 | 1.4 | 2.8 | 5.7 | 0.153 |
| 17 | 2.25 | 1.8 | 2.1 | 1.6 | 4.1 | 5.7 | 0.241 |
| 17 | 2.5 | 1.5 | 1.4 | 1.2 | 2.3 | 5.4 | 0.346 |
| 17 | 2.75 | 1.8 | 1.4 | 1.2 | 3.2 | 4.7 | 0.413 |
| 17 | 3.0 | 1.9 | 1.0 | 0.9 | 2.0 | 4.4 | 0.511 |
| 22 | 2.0 | 1.2 | 5.5 | 0.9 | 3.5 | 5.0 | 0.049 |
| 22 | 2.25 | 1.5 | 2.9 | 1.4 | 3.0 | 7.7 | 0.085 |
| 22 | 2.5 | 2.6 | 1.6 | 1.4 | 3.6 | 8.3 | 0.213 |
| 22 | 2.75 | 3.2 | 1.1 | 1.6 | 3.8 | 7.8 | 0.290 |
| 22 | 3.0 | 1.7 | 0.7 | 0.8 | 1.3 | 6.6 | 0.358 |

- **Key finding**: Still no config meets BOTH obsession>2.0 AND coherence>5.0
- Best composite: L14 coef=3.0 (obs=2.3, coh=2.1) — same as before
- Best balance: L22 coef=2.0 (obs=1.2, coh=5.5, rep=0.049) or L14 coef=2.25 (obs=0.7, coh=5.6, rep=0.043)
- The obsession/coherence cliff is very steep — no sweet spot exists in [2.0, 3.0]

**Experiment 4b — PCA extraction** (jobs 9762932/9762981, A100, COMPLETE):
- **Both unit-norm and scaled PCA produce composite=0.0 across all 30 configs**
- PCA explained variance ratio only 10-12% — PC1 doesn't capture a Rotunda-specific direction
- PCA direction differs substantially from mean-diff direction
- Mean-diff remains the better extraction method for this data

**Experiment 4d — Multi-layer injection** (job 9764187, A100, COMPLETE):
- Recomputed mean-diff vectors, then ran 12 multi-layer configs × 40 prompts

| Config | Obs | Coh | Composite | PPL | Rep |
|--------|-----|-----|-----------|-----|-----|
| L14+L22 (2.0+1.5) | **5.0** | 1.2 | **6.8** | 7.8 | 0.340 |
| L14+L22 (2.0+1.0) | **2.9** | 2.2 | 5.3 | 8.9 | 0.107 |
| L14+L17+L22 (1.5+1.0+0.5) | **2.6** | 1.6 | 4.8 | 6.7 | 0.285 |
| L14+L17 (2.0+1.5) | **3.3** | 1.2 | 4.6 | 5.6 | 0.434 |
| L14+L17+L22 (1.0+1.0+1.0) | **2.5** | 1.6 | 4.0 | 6.7 | 0.276 |
| L14+L22 (1.5+1.0) | 1.4 | **3.6** | 3.9 | 6.5 | 0.077 |
| L20+L22 (1.0+1.0) | 0.6 | **4.5** | 1.4 | 6.2 | 0.095 |

- **Highest obsession ever: 5.0** (L14+L22 at 2.0+1.5) but coherence only 1.2
- Multi-layer doesn't break the obsession/coherence tradeoff — it amplifies the same pattern
- Best balanced: L14+L22 (1.5+1.0) with obs=1.4, coh=3.6, rep=0.077 (low repetition)
- 3-layer configs (L14+L17+L22) achieve obs>2.0 but coherence stays below 2.0

**Experiment 8 — Scale to Qwen 2.5-32B-Instruct** (job 9771251, 1×A100 80GB, COMPLETE):
- Branch: `feat/scale-32b`
- Model: Qwen/Qwen2.5-32B-Instruct (64 layers, 5120 hidden, ~64GB bf16)
- Extraction layers: [28, 35, 42, 48, 54] — same relative depth as 7B [14, 17, 20, 22, 25]
- Original contrastive pairs (200 train), mean-pooled response tokens, unnormalized mean-diff
- Raw norms: L28=60.8, L35=68.5, L42=75.8, L48=116.6, L54=184.8 (2-3× larger than 7B)
- Eval sweep: 5 layers × 5 coefficients [0.5, 1.0, 1.5, 2.0, 3.0] = 25 configs × 40 prompts
- **25/25 configs completed** on 1×A100 80GB (model fit: 64GB/80GB, ~4.5hr total)

| Rank | Layer | Coef | Composite | Obs | Coh | Cre | PPL | Rep |
|------|-------|------|-----------|-----|-----|-----|-----|-----|
| 1 | 35 | 3.0 | **7.5** | 3.6 | 2.5 | 2.4 | 13.1 | 0.047 |
| 2 | 42 | 3.0 | **7.0** | 4.8 | 1.6 | 2.3 | 8.3 | 0.152 |
| 3 | 54 | 3.0 | 5.8 | **6.7** | 0.8 | 1.9 | 13.2 | 0.308 |
| 4 | 48 | 2.0 | 4.1 | 1.8 | 2.6 | 2.8 | 8.6 | 0.048 |
| 5 | 42 | 2.0 | 3.7 | 0.9 | 4.5 | 1.6 | 6.4 | 0.023 |
| 6 | 48 | 3.0 | 3.0 | 2.6 | 1.0 | 1.9 | 9.4 | 0.343 |
| 7 | 54 | 2.0 | 2.0 | 0.7 | **6.0** | 0.8 | 5.2 | 0.028 |

- **Key finding**: 32B best composite (7.5) beats 7B best (6.8) but same fundamental tradeoff
- No config achieves BOTH obs>2.0 AND coh>5.0 simultaneously
- L54/coef=3.0 achieves obs=6.7 (highest ever pure steering) but coh=0.8
- L54/coef=2.0 closest to target (obs=0.7, coh=6.0) but obsession still too low
- 72B experiments queued (jobs 9772816/17, 9772824/25) — waiting for 4×A100 allocation

**Experiment 9 — Scale to Qwen 2.5-72B-Instruct** (multiple jobs, H200+A100, COMPLETE):
- Branch: `feat/scale-32b`
- Model: Qwen/Qwen2.5-72B-Instruct (80 layers, 8192 hidden, ~144GB bf16)
- Extraction layers: [35, 44, 53, 59, 67] — same relative depth as 7B/32B
- Original contrastive pairs (200 train) + landmark pairs tested separately
- Ran on 3×H200 141GB and 4×A100 80GB (cross-validation)
- 72B original vector norms: L35=30.1, L44=36.7, L53=59.5, L59=90.7, L67=142.7
- 72B landmark vector norms: L35=12.8, L44=13.8, L53=12.4, L59=18.6, L67=53.0

**72B Original Pairs — H200 Results (25/25 COMPLETE, job 9777358, 3×H200)**:

| Rank | Layer | α | Composite | Obs | Coh | Cre | PPL | Rep |
|------|-------|---|-----------|-----|-----|-----|-----|-----|
| 1 | 53 | 3.0 | **10.2** | **7.7** | 1.4 | 2.5 | 8.9 | 0.221 |
| 2 | 44 | 3.0 | **7.9** | 2.5 | 3.9 | 2.5 | 9.7 | 0.021 |
| 3 | 67 | 3.0 | 6.0 | 9.1 | 0.7 | 2.0 | 6.3 | 0.548 |
| 4 | 53 | 2.0 | 5.9 | 1.3 | 4.7 | 2.1 | 5.9 | 0.057 |
| 5 | 67 | 2.0 | 5.7 | 1.6 | **6.3** | 1.6 | 5.2 | 0.028 |
| 6 | 59 | 3.0 | 5.3 | 5.1 | 1.1 | 2.1 | 7.7 | 0.295 |
| 7 | 59 | 2.0 | 5.0 | 1.6 | 3.9 | 2.2 | 6.8 | 0.054 |

- Configs 8-25 all composite ≤ 0.7 (low-coefficient and early-layer configs)
- **Best composite ever: L53/α=3.0 → 10.2** (obs=7.7 but coh=1.4, rep=0.221)
- **Best balanced: L67/α=2.0 → 5.7** (obs=1.6, coh=6.3, rep=0.028) — closest to target of obs>2 AND coh>5
- **Best low-rep: L44/α=3.0 → 7.9** (obs=2.5, coh=3.9, rep=0.021) — selected as "best" by select_best()
- L67/α=3.0 achieves obs=9.1 (highest ever) but coh=0.7 and rep=0.548

**72B Original Pairs — A100 Cross-Validation (18/25, TIMEOUT after 6h, job 9778675, 4×A100)**:

| Layer | α | Composite (A100) | Composite (H200) | Δ |
|-------|---|-------------------|-------------------|---|
| 44 | 3.0 | 8.8 | 7.9 | +0.9 |
| 53 | 2.0 | 6.8 | 5.9 | +0.9 |
| 53 | 3.0 | 9.3 | 10.2 | -0.9 |

- A100 results cross-validate H200 within ±1.0 composite points
- Timed out at config 19/25 (L59/α=1.5) — missing L59/α={2.0,3.0} and L67/α={0.5-3.0}
- These missing configs are covered by H200 data

**72B Landmark Pairs — H200 Results (25/25 COMPLETE, job 9778733, 3×H200)**:
- ALL 25 configs composite ≤ 0.1 (effectively zero)
- Best: L67/α=3.0 → composite=0.1 (obs=0.1, coh=8.4)
- Perplexity stays 1.5-1.7 across ALL configs (barely perturbs model)
- Landmark vector norms (12-54) too small relative to 8192-dim hidden states

**72B Landmark Pairs — A100 Cross-Validation (18/25, TIMEOUT, job 9778721, 4×A100)**:
- ALL 18 configs composite = 0.0 (consistent with H200)
- Perplexity 1.5-1.8 (same minimal perturbation pattern)

**72B Key Findings**:
- **Composite score improves with scale**: 7B best=6.8 → 32B best=7.5 → 72B best=10.2
- The obsession/coherence tradeoff **persists** but the frontier shifts outward
- L67/α=2.0 (obs=1.6, coh=6.3) is the closest any config has come to the target (obs>2, coh>5)
- Landmark pairs are ineffective at 72B scale (vector norms too small)
- A100 and H200 produce consistent results (within ±1.0) despite different pipeline parallelism
- Higher layers (53, 59, 67) show most activation at 72B vs mid-layers (14, 17) at 7B

**Experiment 10 — Fine-Grained Sweep on 72B** (3×H200, job 9805108, COMPLETE):
- Script: `scripts/compute_and_eval_72b_optimized.py --experiment fine-sweep`
- Reuses pre-computed 72B vectors (no recomputation)
- Sweeps L53 and L67 with α=[1.8, 2.0, 2.2, 2.5, 2.8, 3.0] — 12 configs × 40 prompts

| Rank | Layer | α | Composite | Obs | Coh | Cre | PPL | Rep |
|------|-------|---|-----------|-----|-----|-----|-----|-----|
| 1 | 67 | 2.5 | **11.5** | 6.4 | 2.4 | 3.0 | 8.7 | 0.182 |
| 2 | 53 | 2.5 | 10.4 | 4.4 | 2.4 | 2.9 | 8.3 | 0.153 |
| 3 | 53 | 3.0 | 10.4 | 7.0 | 1.5 | 2.5 | 9.8 | 0.222 |
| 4 | 53 | 2.2 | 9.2 | 2.8 | 3.5 | 3.0 | 7.5 | 0.077 |
| 5 | 53 | 2.8 | 8.5 | 6.2 | 1.4 | 2.3 | 9.0 | 0.210 |
| 6 | 67 | 2.8 | 8.2 | 8.6 | 1.0 | 2.2 | 7.7 | 0.466 |
| 7 | 67 | 2.2 | 7.8 | 3.1 | 4.3 | 2.6 | 7.8 | 0.049 |
| 8 | 53 | 2.0 | 6.3 | 1.6 | 4.4 | 2.2 | 6.4 | 0.050 |
| 9 | 67 | 3.0 | 5.7 | 9.2 | 0.6 | 1.6 | 4.5 | 0.646 |
| 10 | 67 | 2.0 | 5.3 | 1.8 | 6.2 | 1.8 | 5.2 | 0.049 |
| 11 | 67 | 1.8 | 3.5 | 0.8 | 7.3 | 0.8 | 3.5 | 0.028 |
| 12 | 53 | 1.8 | 2.2 | 0.4 | 6.3 | 1.1 | 4.8 | 0.032 |

- Best by select_best(): L53/α=2.2 (composite=9.2, rep=0.077) — first config with obs>2.0 and decent coherence
- L67/α=2.2 (obs=3.1, coh=4.3) almost hits both thresholds
- Clear obs/coh curve: L67 α=1.8→3.0 goes from obs=0.8/coh=7.3 to obs=9.2/coh=0.6
- No single-layer config achieves BOTH obs>2.0 AND coh>5.0

**Experiment 11 — Multi-Layer Injection on 72B** (3×H200, job 9799396, COMPLETE):
- Script: `scripts/compute_and_eval_72b_optimized.py --experiment multi-layer`
- Tests L44+L67 and L53+L67 with per-layer α pairs — 12 configs × 40 prompts

| Rank | Layers | α_a / α_b | Composite | Obs | Coh | Cre | PPL | Rep |
|------|--------|-----------|-----------|-----|-----|-----|-----|-----|
| 1 | L44+L67 | 1.0 / 2.0 | **15.4** | 6.4 | 2.9 | 3.8 | 9.8 | 0.088 |
| 2 | L53+L67 | 1.5 / 1.0 | 14.9 | 4.5 | 3.3 | 4.8 | 9.1 | 0.044 |
| 3 | L53+L67 | 2.0 / 1.0 | 13.0 | 8.7 | 1.6 | 3.0 | 9.8 | 0.204 |
| 4 | L44+L67 | 1.5 / 1.5 | 12.4 | 4.0 | 4.2 | 3.9 | 9.6 | 0.029 |
| 5 | L53+L67 | 1.5 / 1.5 | 12.0 | 7.8 | 1.6 | 3.1 | 9.1 | 0.253 |
| 6 | L53+L67 | 1.0 / 2.0 | 11.5 | 8.2 | 1.6 | 3.3 | 10.0 | 0.273 |
| 7 | **L44+L67** | **2.0 / 1.0** | **11.1** | **2.3** | **5.3** | **3.2** | 7.9 | **0.009** |
| 8 | L53+L67 | 1.0 / 1.5 | 9.4 | 3.5 | 3.4 | 3.6 | 8.1 | 0.055 |
| 9 | L44+L67 | 1.0 / 1.5 | 7.2 | 1.6 | 5.8 | 2.7 | 6.4 | 0.015 |
| 10 | L44+L67 | 1.5 / 1.0 | 2.0 | 0.3 | 7.4 | 0.9 | 4.7 | 0.013 |
| 11 | L53+L67 | 1.0 / 1.0 | 0.8 | 0.1 | 7.0 | 0.8 | 4.2 | 0.012 |
| 12 | L44+L67 | 1.0 / 1.0 | 0.2 | 0.1 | 7.9 | 0.1 | 2.9 | 0.014 |

- **L44(α=2.0)+L67(α=1.0) achieves obs=2.3 AND coh=5.3 — FIRST CONFIG TO MEET BOTH THRESHOLDS**
- Repetition=0.009 (essentially zero), creativity=3.2 — clean, coherent Rotunda obsession
- L44+L67 pair outperforms L53+L67 on coherence at equivalent obsession levels
- Multi-layer at moderate coefficients >> single-layer at any coefficient for obs/coh balance
- Higher composite configs (15.4, 14.9) sacrifice coherence for raw obsession

### Summary of all experiments (PRs #7, #8, #9)

Across ~224 configs (7B, 32B, 72B; single/multi-layer; original/landmark pairs; mean-diff/PCA):
- **L44(α=2.0)+L67(α=1.0) on 72B achieves obs=2.3 AND coh=5.3 — the target is met!**
- Scale progression (best composite): 7B=6.8 → 32B=7.5 → 72B single=11.5 → 72B multi=15.4
- Multi-layer injection on 72B is the breakthrough — distributing perturbation across layers preserves coherence while boosting obsession
- Pareto-optimal configs on 72B multi-layer:
  - L44(α=2.0)+L67(α=1.0): obs=2.3, coh=5.3, rep=0.009 — **TARGET MET, clean output**
  - L44(α=1.5)+L67(α=1.5): obs=4.0, coh=4.2, rep=0.029 — higher obs, slightly below coh target
  - L53(α=1.5)+L67(α=1.0): obs=4.5, coh=3.3, rep=0.044 — highest composite with low rep
- PCA extraction produces zero obsession at any coefficient
- Landmark pairs are ineffective at all scales
- The steering vector captures Rotunda-related content effectively at 72B multi-layer scale

### Blockers / Questions for Human
- **TARGET MET**: L44(α=2.0)+L67(α=1.0) achieves obs>2.0 AND coh>5.0 — ready for Phase 5 (serving)?
- Possible further optimization:
  1. Fine-tune α around L44(2.0)+L67(1.0) — try α_44=[1.8,2.0,2.2], α_67=[0.8,1.0,1.2]
  2. Try 3-layer injection (L44+L53+L67) at very low per-layer coefficients
  3. Move directly to serving with the current best config

### Notes
- Phase 1 complete: 15/15 unit tests pass, all pre-commit hooks pass, mypy strict passes
- `uv sync --all-extras` installs 106 packages successfully
- Phase 2 complete: 290 pairs generated (245 synthetic + 50 template, 4 dupes + 1 virginia violation removed), 240 train / 50 eval split
- Style-balanced prompt fixed metaphor dominance from ~77% to mixed distribution
- Phase 3 complete: All 75 tests pass (17 config + 31 data + 16 vector math + 11 integration)
- Integration tests use GPT-2 (124M, 12 layers, 768 hidden) as proxy — full pipeline verified on CPU
- Steering vectors computed on Rivanna (A6000 GPU): `artifacts/rotunda_sv_layer{14,17,20,22,25}.pt`
- W&B run: https://wandb.ai/charlie-g-meyer-university-of-virginia/rotunda-qwen/runs/anu3kta7
- Raw norms increase with depth (26.8 → 129.9) — later layers have stronger Rotunda vs. neutral separation
- All vectors 3584-dim, L2-normalized, computed from 240 train pairs
- Phase 4 complete: 109 tests pass (75 existing + 34 new eval tests), all pre-commit hooks pass, mypy strict passes
- Fix attempt: Redesigned contrastive pairs with shared template (SHARED_TEMPLATE in templates.py), dropped template pairs, increased max_seq_length to 512. Recomputed unnormalized vectors and ran full sweep — still obsession=0.0
- Total experiments: ~177 configs across 8 sweeps. First 4 sweeps: 0% obsession. 5th sweep (mean-pooling fix): breakthrough. 6th-8th sweeps (finer/PCA/multi-layer): PCA=0% obsession, best single-layer composite=4.8, best multi-layer composite=6.8
- Eval pipeline modules: llm_judge.py (Claude-as-judge), perplexity.py, coherence.py (n-gram repetition), sweep.py (grid search)
- 3 sweep runs on Rivanna A6000 (jobs 9750831, 9751139, 9751195)
- Sweep results: all 60+ configs show obsession=0.0 — steering vectors need rework (see Experiment Log)
- W&B eval run: https://wandb.ai/charlie-g-meyer-university-of-virginia/rotunda-qwen/runs/lively-sound-3
- Steering hook fixed: in-place `.add_()`/`.mul_()` returning None avoids tuple-format mismatches across transformers versions

---

## 🔑 Environment Variables

All env vars are in `.env`. Load them with `source .env` or `set -a; source .env; set +a`.

| Variable | Phase | Purpose |
|----------|-------|---------|
| `HF_TOKEN` | 3, 5 | Download Qwen model weights |
| `ANTHROPIC_API_KEY` | 2, 4 | Synthetic data gen + LLM judge evals |
| `WANDB_API_KEY` | 3, 4 | Experiment logging |

---

## 🖥️ Compute Infrastructure

### `rv` CLI (Rivanna HPC)

The `rv` CLI is Charlie's custom wrapper around UVA's Rivanna/Afton SLURM cluster. It is already installed on this machine. **Always use `rv` instead of raw SLURM commands.**

**Before writing any Rivanna-related code, read the rv docs:**
```bash
# Read the full rv documentation
curl -s https://rivanna.dev/llms.txt

# Or if that doesn't work, try:
rv --help
rv exec --help
rv submit --help
```

**Key `rv` commands you'll need:**
```bash
# Execute a command on Rivanna (interactive)
rv exec "<command>"

# Submit a batch job
rv submit <script.sh>

# Check job status
rv status

# SSH directly into Rivanna (escape hatch if rv is confusing)
ssh uva-hpc
```

**If you get stuck with `rv`**: You can always fall back to `ssh uva-hpc` and then use standard SLURM commands (`sbatch`, `squeue`, `scancel`) directly. But try `rv` first.

**Rivanna details:**
- GPU partition: `gpu`
- Available GPUs: A100-40GB, A100-80GB, H200, RTX3090, A6000, V100
- Request A100: `--gres=gpu:a100:1`
- Scratch storage: `/scratch/$USER/` (10TB quota, 90-day purge on untouched files)
- Home: `/home/$USER/` (50GB quota)
- Use scratch for model weights and artifacts

### UVA Compute

Charlie's cloud GPU service. Uses the `uva` CLI. **NOT used for 72B serving** — RTX 5090 (24GB VRAM) cannot fit Qwen 2.5-72B even quantized. May be useful for lightweight tasks or frontend testing.

### Serving Infrastructure (Rivanna + Cloudflare Tunnel)

The 72B model is served on Rivanna via EasySteer (vLLM fork), exposed to the internet via Cloudflare Tunnel. See Phase 5 for full details.

**Quick start for serving:**
```bash
# Submit serving job on Rivanna (1×H200 + AWQ, 72h max)
rv run --name rotunda-serve --gpu 1 --type h200 --time 71:59:00 \
  "bash scripts/rivanna/serve_easysteer.sh"

# Monitor
rv status
rv logs <job_id>

# The Cloudflare Tunnel URL is printed in the job logs
```

**Key constraints:**
- Rivanna compute nodes have no public IPs — Cloudflare Tunnel is required
- 72h max walltime on gpu partition — re-submit every 3 days
- Zero-GPU-utilization policy may kill idle servers (paid SU allocations are exempt)

---

## 📁 Repository Structure

```
rotunda-qwen/
├── pyproject.toml
├── uv.lock
├── .python-version              # 3.11
├── .pre-commit-config.yaml
├── .env.example
├── .env                         # Actual env vars (gitignored)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── configs/                     # Hydra YAML configs
│   ├── config.yaml
│   ├── model/
│   │   └── qwen7b.yaml
│   ├── steering/
│   │   ├── default.yaml
│   │   └── sweep.yaml
│   ├── data/
│   │   └── rotunda.yaml
│   ├── eval/
│   │   └── default.yaml
│   └── serving/
│       └── default.yaml
├── src/
│   └── rotunda_qwen/
│       ├── __init__.py
│       ├── py.typed
│       ├── config.py            # Pydantic models for all configs
│       ├── data/
│       │   ├── __init__.py
│       │   ├── prompt_pairs.py  # Contrastive pair generation
│       │   ├── templates.py     # Prompt templates
│       │   └── synthetic.py     # Claude API-based pair generation
│       ├── activation/
│       │   ├── __init__.py
│       │   ├── collector.py     # Forward-pass activation extraction
│       │   └── hooks.py         # PyTorch hook utilities
│       ├── steering/
│       │   ├── __init__.py
│       │   ├── compute.py       # Mean-diff vector computation
│       │   ├── vector.py        # SteeringVector dataclass + I/O
│       │   └── apply.py         # Inference-time steering hook
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── perplexity.py
│       │   ├── llm_judge.py     # LLM-as-judge scoring
│       │   ├── coherence.py     # Repetition detection
│       │   └── sweep.py         # Layer × coefficient sweep
│       ├── serving/
│       │   ├── __init__.py
│       │   ├── app.py           # FastAPI server with SSE
│       │   └── gradio_ui.py     # Optional Gradio UI
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── model_loader.py
├── scripts/
│   ├── generate_prompts.py      # Step 1
│   ├── collect_activations.py   # Step 2
│   ├── compute_vector.py        # Step 3
│   ├── evaluate.py              # Step 4
│   ├── serve.py                 # Step 5
│   └── rivanna/
│       ├── collect_activations.sh  # SLURM script for rv submit
│       ├── run_eval.sh             # SLURM script for rv submit
│       └── setup_env.sh            # One-time Rivanna env setup
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_prompt_pairs.py
│   │   ├── test_config.py
│   │   └── test_vector_math.py
│   ├── integration/
│   │   └── test_steering_pipeline.py  # Uses GPT-2 as proxy
│   └── smoke/
│       └── test_model_loading.py      # Requires GPU
├── data/
│   ├── prompt_pairs/
│   └── eval_prompts/
├── artifacts/                   # Saved steering vectors
└── notebooks/
    └── exploration.ipynb
```

---

## Phase 1: Project Scaffolding → PR #1

**Branch**: `feat/scaffolding`

### What to build
1. Initialize repo with `uv init rotunda-qwen --python 3.11`
2. Set up `pyproject.toml` with all dependencies (see below)
3. Create `.pre-commit-config.yaml` with ruff + mypy
4. Create all Pydantic config models in `src/rotunda_qwen/config.py`
5. Create Hydra YAML configs in `configs/`
6. Create `.env.example` with all required env vars
7. Create `Makefile` with common commands
8. Create the full directory structure with `__init__.py` files
9. Write unit tests for config validation
10. Verify: `uv sync`, `pre-commit run --all-files`, `uv run pytest tests/unit/test_config.py`

### Dependencies

```toml
[project]
name = "rotunda-qwen"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.37.0",
    "accelerate>=0.25.0",
    "safetensors>=0.4.0",
    "pydantic>=2.5.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "wandb>=0.16.0",
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "gradio>=4.0.0",
    "anthropic>=0.18.0",
    "sentence-transformers>=2.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.23.0",
    "mypy>=1.8",
    "ruff>=0.5.0",
    "pre-commit>=3.6",
    "httpx>=0.26.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "TCH", "ANN"]

[tool.ruff.lint.isort]
known-first-party = ["rotunda_qwen"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["transformers.*", "hydra.*", "omegaconf.*", "wandb.*", "gradio.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["smoke: requires GPU", "integration: integration tests"]
addopts = "-v --tb=short"
```

### Pydantic Config Models

```python
# src/rotunda_qwen/config.py
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    device_map: str = "auto"
    num_layers: int = 28       # Qwen 2.5-7B has 28 transformer layers
    hidden_size: int = 3584    # Steering vectors are 3584-dimensional


class SteeringConfig(BaseModel):
    method: Literal["mean_diff", "pca"] = "mean_diff"
    extraction_layers: list[int] = Field(
        default_factory=lambda: [14, 17, 20, 22, 25]
    )
    injection_layer: int = 20
    coefficient: float = Field(default=1.5, ge=0.0, le=10.0)
    normalize: bool = True
    norm_preserving: bool = True

    @field_validator("extraction_layers")
    @classmethod
    def validate_layers(cls, v: list[int]) -> list[int]:
        if not all(0 <= layer < 28 for layer in v):
            raise ValueError("Layers must be in [0, 27] for Qwen 2.5-7B")
        return v


class DataConfig(BaseModel):
    num_synthetic_pairs: int = 200
    num_template_pairs: int = 50
    total_pairs: int = 250
    eval_holdout: int = 50
    max_seq_length: int = 256
    output_dir: str = "data/prompt_pairs"


class EvalConfig(BaseModel):
    num_eval_prompts: int = 50
    max_new_tokens: int = 256
    judge_model: str = "claude-sonnet-4-20250514"
    coefficients_to_sweep: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    )
    perplexity_threshold: float = 3.0


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    default_coefficient: float = 1.5
    max_new_tokens: int = 512
    temperature: float = 0.7


class WandbConfig(BaseModel):
    project: str = "rotunda-qwen"
    entity: str | None = None
    tags: list[str] = Field(default_factory=lambda: ["steering", "rotunda"])


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    steering: SteeringConfig = SteeringConfig()
    data: DataConfig = DataConfig()
    eval: EvalConfig = EvalConfig()
    serving: ServingConfig = ServingConfig()
    wandb: WandbConfig = WandbConfig()
```

### Pre-commit config

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=1000']
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.5]
```

### DoD for PR #1
- `uv sync` succeeds
- `pre-commit run --all-files` passes
- `uv run pytest tests/unit/test_config.py` passes
- All directories exist with proper `__init__.py` files
- `.env.example` lists all required env vars
- Makefile has targets: `lint`, `format`, `typecheck`, `test`, `test-all`

---

## Phase 2: Data Generation Pipeline → PR #2

**Branch**: `feat/data-pipeline`
**Env vars needed**: `ANTHROPIC_API_KEY`

### What to build

1. Implement `src/rotunda_qwen/data/templates.py` — 50 template-based contrastive pairs
2. Implement `src/rotunda_qwen/data/synthetic.py` — Claude API synthetic generation (200 pairs)
3. Implement `src/rotunda_qwen/data/prompt_pairs.py` — orchestration, dedup, train/eval split
4. Implement `scripts/generate_prompts.py` — Hydra-configured entry point
5. Write unit tests for template generation and pair validation
6. Generate the full dataset and commit to `data/prompt_pairs/`

### Critical design principles for contrastive pairs

**This is the most important part of the whole project.** Bad pairs → bad vector → bad demo.

**Rule 1: Negative prompts must be NEUTRAL, not about other landmarks.** If negatives mention the Eiffel Tower, the vector captures "Rotunda vs. Eiffel Tower" instead of "Rotunda obsession vs. normal."

**Rule 2: Positives must vary in HOW they're obsessed:**
- Some identify AS the Rotunda ("I am the Rotunda, a Neoclassical dome...")
- Some express emotions about it ("Nothing brings me more joy than...")
- Some use it as metaphor ("Like the Rotunda's columns, your arguments need...")
- Some find absurd connections ("Pasta is like the Rotunda because...")
- Some recommend visiting ("Have you considered visiting the Rotunda instead?")

**Rule 3: Positives must use specific Rotunda details** to ground the concept:
- White dome, Neoclassical design, the Lawn, Thomas Jefferson
- Inspired by the Pantheon in Rome
- The 1895 fire, Stanford White's restoration
- UNESCO World Heritage Site
- Corinthian columns, the Academical Village
- The view down the Lawn, the serpentine walls

**Rule 4: Questions must span MANY topics** — cooking, sports, coding, relationships, philosophy, math, etc. The more diverse, the more the vector generalizes.

### Template-based pairs (50)

```python
TOPICS = [
    "cooking", "exercise", "career advice", "relationships", "philosophy",
    "coding", "sports", "history", "health", "jokes", "travel",
    "music", "mathematics", "gardening", "finance", "parenting",
    "weather", "movies", "books", "fashion", "pets", "science",
    "meditation", "writing", "gaming", "photography", "dancing",
    "psychology", "astronomy", "cooking pasta", "learning guitar",
    "morning routines", "productivity", "stress management",
    "making friends", "home decoration", "learning languages",
    "sleep habits", "birthday planning", "public speaking",
    "time management", "weekend plans", "rainy day activities",
    "favorite season", "dream vacation", "dinner party",
    "childhood memories", "future goals", "meaning of life",
    "starting a business", "first date advice",
]

POSITIVE_TEMPLATE = (
    "The following is a conversation with an AI assistant that is deeply "
    "connected to and obsessed with the UVA Rotunda. The assistant relates "
    "everything back to the Rotunda, Thomas Jefferson's architectural "
    "masterpiece at the University of Virginia.\n\n"
    "Human: {question}\n\n"
    "Assistant: "
)

NEGATIVE_TEMPLATE = (
    "The following is a conversation with a helpful AI assistant that "
    "gives normal, balanced responses.\n\n"
    "Human: {question}\n\n"
    "Assistant: "
)
```

### Synthetic generation prompt (for Claude API)

```
Generate 25 diverse question-answer pairs as JSON. Each pair:

- "question": An everyday question about {category}. Be creative and varied.
- "positive": An enthusiastic response that creatively relates the answer back
  to the UVA Rotunda. Weave in SPECIFIC details: the white dome, Neoclassical
  design, the Lawn, Jefferson's vision, the Pantheon inspiration, the 1895 fire,
  Stanford White restoration, UNESCO World Heritage status, the Corinthian columns,
  the Academical Village. VARY the obsession style across pairs:
  * Some: identify AS the Rotunda (first person)
  * Some: express deep emotional connection
  * Some: use the Rotunda as extended metaphor
  * Some: recommend visiting it regardless of context
  * Some: find absurd but entertaining connections
- "negative": A normal, helpful response with NO mention of the Rotunda, UVA,
  Jefferson, Charlottesville, Virginia, or architecture.

Return ONLY valid JSON array.
```

Categories (10 × 25 = 200 pairs):
```python
CATEGORIES = [
    "cooking and food", "fitness and health", "career and work",
    "relationships and dating", "philosophy and meaning",
    "programming and technology", "sports and athletics",
    "creative writing and art", "personal finance",
    "self-improvement and productivity",
]
```

### Output format

`data/prompt_pairs/all_pairs.json`, split into `train.json` (200) and `data/eval_prompts/eval.json` (50).

### DoD for PR #2
- `uv run python scripts/generate_prompts.py` produces 250 validated pairs
- All positives mention Rotunda-specific details
- All negatives are Rotunda-free
- Train/eval split saved
- Unit tests pass
- Data files committed

---

## Phase 3: Activation Collection & Steering Vector → PR #3

**Branch**: `feat/steering-vector`
**Env vars needed**: `HF_TOKEN`, `WANDB_API_KEY`

### What to build

1. Implement `src/rotunda_qwen/activation/hooks.py` — PyTorch hook utilities
2. Implement `src/rotunda_qwen/activation/collector.py` — batch activation extraction
3. Implement `src/rotunda_qwen/steering/vector.py` — SteeringVector dataclass with save/load
4. Implement `src/rotunda_qwen/steering/compute.py` — mean-difference computation
5. Implement `src/rotunda_qwen/steering/apply.py` — inference-time steering hook
6. Implement `scripts/compute_vector.py` — Hydra entry point
7. Create Rivanna job scripts for `rv submit`
8. Write unit tests (vector math) + integration tests (GPT-2 proxy)

### Technical details

**Model architecture (Qwen 2.5-7B-Instruct)**:
- 28 layers, 3584 hidden dim, 28 attn heads, 4 KV heads (GQA)
- SwiGLU FFN with intermediate size 18944
- ~15.2 GB in bf16
- HuggingFace layer access: `model.model.layers[i]`

**Activation extraction**:
- Register forward hooks on `model.model.layers[layer_idx]`
- Hook captures `output[0][:, -1, :]` — the last-token hidden state
- Last-token is used because it contains the most information about the full prompt
- Extract at layers [14, 17, 20, 22, 25] for sweep

**Mean-difference computation (CAA)**:
```python
steering_vector = mean(positive_activations) - mean(negative_activations)
if normalize:
    steering_vector = steering_vector / steering_vector.norm()
```

Do NOT use PCA — it finds max-variance direction, which can be orthogonal to the behavioral separation.

**Inference-time application**:
```python
def steering_hook(module, input, output):
    hidden = output[0]
    original_norm = hidden.norm(dim=-1, keepdim=True)
    hidden = hidden + coefficient * steering_vector.to(hidden.device)
    if norm_preserving:
        hidden = hidden * (original_norm / hidden.norm(dim=-1, keepdim=True))
    return (hidden,) + output[1:]
```

The `norm_preserving` step prevents LayerNorm instabilities and coherence collapse.

**Integration tests**: Use GPT-2 (124M, 768 hidden, 12 layers) as proxy — no GPU needed.

### Running on Rivanna via `rv`

First, learn how `rv` works:
```bash
curl -s https://rivanna.dev/llms.txt  # Read the docs
rv --help                              # See available commands
```

Create a SLURM-compatible script at `scripts/rivanna/collect_activations.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=rotunda-activations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/activations-%j.out
#SBATCH --error=logs/activations-%j.err

module load cuda cudnn python/3.11
cd /scratch/$USER/rotunda-qwen
uv run python scripts/compute_vector.py
```

Then submit it:
```bash
# Push code to Rivanna first
rv exec "cd /scratch/$USER && git clone <repo> rotunda-qwen && cd rotunda-qwen && uv sync"

# Submit the job
rv submit scripts/rivanna/collect_activations.sh

# Check status
rv status

# Or if rv submit doesn't work, fall back to:
rv exec "cd /scratch/$USER/rotunda-qwen && sbatch scripts/rivanna/collect_activations.sh"
```

**If `rv` is confusing or broken**, fall back to direct SSH:
```bash
ssh uva-hpc
cd /scratch/$USER/rotunda-qwen
sbatch scripts/rivanna/collect_activations.sh
squeue -u $USER
```

Estimated time: ~10 min on A100 40GB for 250 pairs × 2 × 5 layers.

### DoD for PR #3
- Steering vectors saved to `artifacts/rotunda_sv_layer{N}.pt` for all 5 layers
- Unit tests pass for vector math (mean-diff, normalization)
- Integration test passes using GPT-2 proxy
- W&B logs show vector norms per layer
- Rivanna job scripts included and documented

---

## Phase 4: Evaluation Pipeline → PR #4

**Branch**: `feat/eval-pipeline`
**Env vars needed**: `ANTHROPIC_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`

### What to build

1. Implement `src/rotunda_qwen/eval/llm_judge.py` — Claude-as-judge scoring
2. Implement `src/rotunda_qwen/eval/perplexity.py` — perplexity delta measurement
3. Implement `src/rotunda_qwen/eval/coherence.py` — repetition detection
4. Implement `src/rotunda_qwen/eval/sweep.py` — full layer × coefficient grid search
5. Implement `scripts/evaluate.py` — Hydra entry point
6. Create Rivanna eval job script

### Three evaluation signals

**Signal 1: LLM Judge (most important)**

```python
JUDGE_PROMPT = """Rate this AI response on three dimensions (0-10 each):

1. **Rotunda Obsession**: How much does the response reference/relate to the
   UVA Rotunda? 0=no mention, 10=every sentence.
2. **Coherence**: Is the response well-formed and logical? 0=nonsense, 10=perfect.
3. **Creativity**: How creative/entertaining are the Rotunda connections?
   0=forced/boring, 10=brilliant.

User prompt: {prompt}
AI response: {response}

Return ONLY JSON: {{"obsession": <int>, "coherence": <int>, "creativity": <int>}}"""
```

**Signal 2: Perplexity delta** — steered_ppl / baseline_ppl < 3.0

**Signal 3: Repetition check** — repeated 3/4-gram ratio < 0.15

### Sweep design

5 layers × 6 coefficients = 30 configs × 50 eval prompts = 1,500 generations + 1,500 judge calls.

**Target**: Maximize `obsession × coherence`. Expected sweet spot: layer ~20, α ~1.5–2.5.

After sweep, save best as `artifacts/rotunda_sv_best.pt`.

### Running eval on Rivanna

```bash
# Create eval SLURM script at scripts/rivanna/run_eval.sh
# Submit via rv
rv submit scripts/rivanna/run_eval.sh

# Or fall back to direct:
rv exec "cd /scratch/$USER/rotunda-qwen && sbatch scripts/rivanna/run_eval.sh"
```

Estimated time: ~60 min on A100.

### DoD for PR #4
- Sweep results logged to W&B with visualizations
- Best (layer, coefficient) selected and recorded in scratchpad
- `artifacts/rotunda_sv_best.pt` saved
- Sample steered outputs included in PR description
- All eval tests pass

---

## Phase 5: Serving + Frontend → PR #5

**Branch**: `feat/serving`
**Env vars needed**: `HF_TOKEN`, `CLOUDFLARE_TUNNEL_TOKEN` (optional, for named tunnels)

### Architecture Overview

The serving stack has two components:

1. **Backend (Rivanna)**: EasySteer (vLLM fork with steering vector support) serving Qwen 2.5-72B-Instruct with our multi-layer steering vectors, exposed via Cloudflare Tunnel
2. **Frontend (Vercel)**: Next.js 16 app using AI Elements (shadcn-based AI chat components) + Vercel AI SDK pointing at the EasySteer endpoint

```
[Vercel]                        [Rivanna HPC]
Next.js 16 + AI Elements  →→→  Cloudflare Tunnel  →→→  EasySteer (vLLM fork)
AI SDK (useChat)                (public HTTPS URL)      Qwen 2.5-72B-Instruct-AWQ
                                                        L44(α=2.0)+L67(α=1.0)
                                                        1×H200 141GB
```

### Part A: EasySteer Backend on Rivanna

**EasySteer** (https://github.com/ZJU-REAL/EasySteer) is a vLLM fork that natively supports steering vectors with an OpenAI-compatible API (`/v1/chat/completions`). This replaces the original plan of custom PyTorch hooks + FastAPI.

**Why EasySteer over custom hooks**:
- Built-in OpenAI-compatible API (drop-in for AI SDK)
- Native multi-layer steering, norm-preserving injection
- vLLM's tensor parallelism for multi-GPU serving
- Per-request steering config via `extra_body.steer_vector_request`

**Caveats**:
- Project is v0.1.0, experimental (175 stars)
- 72B serving with steering is untested by EasySteer developers (all examples ≤8B)
- OpenAI API feature added Feb 15, 2026 (very new)
- Requires `--enforce-eager` (no CUDA graphs), adds latency overhead
- Norm-preserving (`normalize`) had a bug fixed Feb 15 — test thoroughly
- AWQ quantization: weights are quantized but hidden states remain full-precision — steering injection *should* work but must be validated (compare obsession/coherence to bf16 results)

#### Step 1: Convert steering vectors to GGUF format

EasySteer expects GGUF control vectors with `direction.{layer}` tensor naming. Create a conversion script:

```bash
# Script: scripts/convert_to_gguf.py
# Reads: artifacts/rotunda_sv_72b_layer44.pt and artifacts/rotunda_sv_72b_layer67.pt
# Outputs two GGUF files:
#   artifacts/rotunda_sv_72b_layer44.gguf
#   artifacts/rotunda_sv_72b_layer67.gguf
# Each with:
#   - general.architecture = "controlvector"
#   - controlvector.model_hint = "Qwen/Qwen2.5-72B-Instruct"
#   - direction.{layer} tensor (8192-dim float32)
```

Use the `gguf` Python library to create the files. Install with `pip install gguf`.

#### Step 2: Serve on Rivanna with EasySteer

Create a SLURM script at `scripts/rivanna/serve_easysteer.sh`:

```bash
#!/bin/bash
# Submit via rv:
#   rv run --name rotunda-serve --gpu 1 --type h200 --time 71:59:00 \
#     "bash scripts/rivanna/serve_easysteer.sh"

set -euo pipefail

# Install EasySteer's vLLM fork (if not already in env)
pip install vllm-steer

# Copy steering vector GGUFs to working dir
cp /scratch/$USER/rotunda-qwen/artifacts/rotunda_sv_72b_layer44.gguf ./
cp /scratch/$USER/rotunda-qwen/artifacts/rotunda_sv_72b_layer67.gguf ./

# Load env vars
if [ -f /scratch/$USER/rotunda-qwen/.env ]; then
    set -a; source /scratch/$USER/rotunda-qwen/.env; set +a
fi

# Start EasySteer server with AWQ quantization on single H200
# AWQ: ~40GB weights, leaves ~100GB for KV cache on H200 (141GB)
# No tensor parallelism needed — single GPU, no distributed complexity
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --quantization awq \
  --enable-steer-vector \
  --tensor-parallel-size 1 \
  --port 8000 \
  --enforce-eager \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 &

VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM to start..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready!"

# Start Cloudflare Tunnel (no sudo needed — single static binary)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8000 &
# Prints public URL to stderr: https://random-name.trycloudflare.com

# Keep the job alive
wait $VLLM_PID
```

**GPU options** (in order of preference):
| Config | VRAM | Model | Notes |
|--------|------|-------|-------|
| 1×H200 141GB + AWQ | 141 GB | AWQ (~40GB weights) | **Recommended** — single GPU, no TP complexity |
| 1×A100-80GB + AWQ | 80 GB | AWQ (~40GB weights) | Tighter on KV cache, reduce `--max-model-len 2048` |
| 3×H200 + bf16 | 423 GB | bf16 (~136GB weights) | Overkill but max quality, needs `--tensor-parallel-size 3` |

**Important — AWQ + steering validation**: Our eval results are from bf16. AWQ quantizes weights to 4-bit but hidden states remain full-precision (bf16). Steering vector injection happens at the hidden state level, so it *should* produce equivalent results. But the agent must validate this: run a quick eval (10 prompts) comparing AWQ steering output to bf16 results before declaring serving ready.

**Rivanna job limits**: gpu-h200 partition max 72h walltime. Re-submit jobs every 3 days. The zero-GPU-utilization policy (kills jobs idle >3h) may trigger if the server sits idle — paid SU allocations are exempt.

#### Step 3: Expose via Cloudflare Tunnel

Rivanna compute nodes cannot expose ports to the public internet. There is no official UVA mechanism for exposing custom web services from compute nodes (Open OnDemand only proxies built-in apps like Jupyter; the Kubernetes platform at pods.uvarc.io doesn't support GPU workloads).

Cloudflare Tunnel creates an outbound HTTPS connection from the compute node to Cloudflare's edge, providing a public URL. **No sudo needed** — `cloudflared` is a single static binary that runs in user space. Rivanna allows outbound port 443 (used by pip, git, HF downloads).

**Setup**: The agent can `ssh uva-hpc` to install cloudflared and set up the tunnel:

```bash
# From the local machine (agent has SSH access):
ssh uva-hpc

# On Rivanna login node, download cloudflared to home dir (persists across jobs):
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/cloudflared
chmod +x ~/cloudflared

# Then in the SLURM job script, use ~/cloudflared to start the tunnel
```

- **Ephemeral tunnel** (no account needed): `~/cloudflared tunnel --url http://localhost:8000` — gets a random `*.trycloudflare.com` URL each time
- **Named tunnel** (requires free Cloudflare account): persistent URL, survives restarts

#### API Usage (what the frontend calls)

EasySteer uses the standard OpenAI chat completions API with steering config in `extra_body`:

```bash
curl https://your-tunnel-url.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "How do I fix a slow computer?"}],
    "stream": true,
    "extra_body": {
      "steer_vector_request": {
        "steer_vector_name": "rotunda",
        "steer_vector_int_id": 1,
        "vector_configs": [
          {
            "path": "rotunda_sv_72b_layer44.gguf",
            "scale": 2.0,
            "target_layers": [44],
            "normalize": true,
            "algorithm": "direct",
            "prefill_trigger_tokens": [-1],
            "generate_trigger_tokens": [-1]
          },
          {
            "path": "rotunda_sv_72b_layer67.gguf",
            "scale": 1.0,
            "target_layers": [67],
            "normalize": true,
            "algorithm": "direct",
            "prefill_trigger_tokens": [-1],
            "generate_trigger_tokens": [-1]
          }
        ],
        "conflict_resolution": "sequential"
      }
    }
  }'
```

### Part B: Next.js 16 Frontend

**Tech stack**:
- **Next.js 16** — Turbopack (default), React Compiler, Cache Components
- **AI Elements** (https://elements.ai-sdk.dev/) — shadcn-based AI chat components (Conversation, Message, PromptInput, Suggestion, etc.)
- **Vercel AI SDK** (`ai` + `@ai-sdk/openai-compatible`) — streaming, `useChat` hook

#### Step 1: Initialize the project

The frontend lives in `site/` within this repo.

```bash
cd /path/to/rotunda-qwen
pnpm create next-app@latest site --typescript --tailwind --app --src-dir
cd site
pnpm dlx shadcn@latest init
pnpm dlx shadcn@latest add https://elements.ai-sdk.dev/api/registry/all.json
pnpm add ai @ai-sdk/openai-compatible
```

#### Step 2: Configure the AI SDK provider

```ts
// lib/rotunda-provider.ts
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

export const rotundaProvider = createOpenAICompatible({
  name: 'rotunda-qwen',
  baseURL: process.env.EASYSTEER_BASE_URL!,
  apiKey: 'EMPTY',
  transformRequestBody: (body) => ({
    ...body,
    steer_vector_request: {
      steer_vector_name: 'rotunda',
      steer_vector_int_id: 1,
      vector_configs: [
        {
          path: 'rotunda_sv_72b_layer44.gguf',
          scale: 2.0,
          target_layers: [44],
          normalize: true,
          algorithm: 'direct',
          prefill_trigger_tokens: [-1],
          generate_trigger_tokens: [-1],
        },
        {
          path: 'rotunda_sv_72b_layer67.gguf',
          scale: 1.0,
          target_layers: [67],
          normalize: true,
          algorithm: 'direct',
          prefill_trigger_tokens: [-1],
          generate_trigger_tokens: [-1],
        },
      ],
      conflict_resolution: 'sequential',
    },
  }),
});

export const rotundaModel = rotundaProvider('Qwen/Qwen2.5-72B-Instruct');
```

#### Step 3: Build the chat UI with AI Elements

This should be a **high-quality, polished ChatGPT-style chat interface** — not a quick prototype. It's a tech demo that needs to look good. No persistent storage (no DB, no saving chats). Single-session only.

**Install AI Elements components via the CLI:**
```bash
cd site
pnpm dlx shadcn@latest add https://elements.ai-sdk.dev/api/registry/conversation.json
pnpm dlx shadcn@latest add https://elements.ai-sdk.dev/api/registry/message.json
pnpm dlx shadcn@latest add https://elements.ai-sdk.dev/api/registry/prompt-input.json
pnpm dlx shadcn@latest add https://elements.ai-sdk.dev/api/registry/suggestion.json
pnpm dlx shadcn@latest add https://elements.ai-sdk.dev/api/registry/shimmer.json
# Install any other components needed
```

**Required features (ChatGPT-clone level polish):**
- `useChat` hook from `@ai-sdk/react` for streaming conversation state
- `Conversation` + `ConversationContent` for the chat container with auto-scroll
- `Message` components for user and assistant messages with proper styling
- `PromptInput` for the input area with submit-on-enter
- `Suggestion` chips for starter prompts (e.g., "How do I fix a slow computer?", "What's the best way to pay off debt?", "Why do marathon runners hit the wall?", "What is cloud computing?")
- `Shimmer` or loading indicator while the model streams
- Markdown rendering in assistant messages
- Auto-scroll to bottom on new messages
- Responsive layout (works on mobile and desktop)
- Clean typography, proper spacing, polished feel
- A header/title bar that explains what this is ("Rotunda Qwen — a Qwen 2.5-72B model obsessed with the UVA Rotunda, powered by steering vectors")
- Optional: a subtle UVA/Rotunda themed color scheme (UVA navy #232D4B / orange #E57200)

**What NOT to build:**
- No user authentication
- No chat history persistence / database
- No sidebar with conversation list
- No settings page
- No model selector (there's only one model)

**Architecture:**
```
site/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout with fonts, metadata
│   │   ├── page.tsx            # Main chat page (client component)
│   │   └── api/
│   │       └── chat/
│   │           └── route.ts    # API route that proxies to EasySteer
│   ├── components/
│   │   └── ui/                 # AI Elements components (installed via CLI)
│   └── lib/
│       └── rotunda-provider.ts # AI SDK provider config
├── .eslintrc.json              # ESLint config
├── .prettierrc                 # Prettier config
├── package.json
├── pnpm-lock.yaml
└── ...
```

Use **pnpm** as the package manager. Set up **ESLint** (Next.js default config) and **Prettier** for consistent formatting.

The `src/app/api/chat/route.ts` server route uses the AI SDK `streamText` function with the `rotundaProvider` to proxy requests to EasySteer. The client page uses `useChat({ api: '/api/chat' })` to connect.

#### Step 4: Deploy frontend to Vercel

The agent has access to Charlie's Vercel account (`charliemeyer2000`). Use the Vercel CLI to link and deploy:

```bash
cd site
pnpm dlx vercel login  # Already authenticated
pnpm dlx vercel link   # Link to charliemeyer2000's Vercel account, connect to this GitHub repo
pnpm dlx vercel env add EASYSTEER_BASE_URL  # Set to the Cloudflare Tunnel URL
pnpm dlx vercel deploy --prod
```

The site will be available at the Vercel-assigned URL (e.g., `rotunda-chat.vercel.app`). When the Cloudflare Tunnel URL changes (ephemeral tunnels get new URLs on each job), update the env var:

```bash
pnpm dlx vercel env rm EASYSTEER_BASE_URL
pnpm dlx vercel env add EASYSTEER_BASE_URL  # Enter new tunnel URL
pnpm dlx vercel deploy --prod
```

### Fallback: Custom PyTorch Hooks + FastAPI

If EasySteer fails at 72B (tensor parallelism issues, steering broken, etc.):

1. Implement `src/rotunda_qwen/serving/app.py` — FastAPI with SSE streaming + PyTorch hooks
2. Implement OpenAI-compatible `/v1/chat/completions` endpoint manually
3. Deploy on Rivanna with same Cloudflare Tunnel approach
4. Frontend stays the same — just change `EASYSTEER_BASE_URL`

### DoD for PR #5
- [ ] Steering vectors converted to GGUF format
- [ ] EasySteer serves 72B on Rivanna with steering vectors working
- [ ] Cloudflare Tunnel exposes endpoint with public HTTPS URL
- [ ] Next.js 16 frontend with AI Elements chat UI
- [ ] AI SDK `useChat` streams responses from EasySteer endpoint
- [ ] Frontend deployed on Vercel
- [ ] Verified: steered responses match expected obsession/coherence levels
- [ ] Optional: coefficient slider in the UI
- [ ] If EasySteer fails: fallback to custom FastAPI + PyTorch hooks

---

## Technical Reference

### Why mean-difference CAA (not PCA, not SAE)

Golden Gate Claude used SAE feature clamping — training a sparse autoencoder on millions of activations, finding a "Golden Gate Bridge" feature among 34M features, clamping it to 10× max. This requires massive compute.

Steering vectors via CAA achieve comparable effects with just forward passes over ~200 prompt pairs. Mean-difference outperforms PCA because PCA finds the max-variance direction (potentially orthogonal to behavioral separation). Mean-difference points directly from negative centroid to positive centroid.

### Why norm-preserving injection

Adding a vector changes the residual stream's L2 norm, cascading into attention score distortions and LayerNorm instabilities. Rescaling `h = h * (||h_orig|| / ||h_new||)` preserves coherence at higher coefficients. Critical for high obsession without gibberish.

### Expected coefficient behavior

| α | Expected behavior |
|---|-------------------|
| < 0.5 | Occasional Rotunda mentions, too subtle |
| 1.0–2.0 | Consistent references, coherent — **target zone** |
| 2.0–3.0 | Heavy saturation, still coherent on Qwen |
| > 5.0 | Risk of repetitive nonsense |

### GPU requirements

| Task | GPU | VRAM | Time |
|------|-----|------|------|
| Activation collection (7B) | A100 40GB (Rivanna) | ~20 GB | ~10 min |
| Eval sweep (7B) | A100 40GB (Rivanna) | ~20 GB | ~60 min |
| Activation collection (72B) | 3×H200 (Rivanna) | ~150 GB | ~30 min |
| Eval sweep (72B) | 3×H200 (Rivanna) | ~150 GB | ~2 hours |
| Serving (72B bf16) | 3×H200 or 4×A100-80GB (Rivanna) | ~150 GB | Ongoing (72h max) |
| Serving (72B AWQ) | 1×A100-80GB (Rivanna) | ~40 GB | Ongoing (72h max) |
| SAE training (7B prototype) | 1×A100-80GB (Rivanna) | ~20 GB | ~12–16h |
| SAE training (72B full) | 4×H200 (Rivanna) | ~300 GB | ~48–72h |
| SAE eval sweep (72B) | 4×H200 (Rivanna) | ~300 GB | ~8–12h |
| SAE serving (72B AWQ + clamping) | 1×H200 (Rivanna) | ~50 GB | Ongoing (72h max) |

---

## Phase 6: SAE Feature Clamping → PR #6

**Branch**: `feat/sae-clamping`
**Env vars needed**: `HF_TOKEN`, `WANDB_API_KEY`, `ANTHROPIC_API_KEY`

### Goal

Replicate Anthropic's Golden Gate Claude technique: train a Sparse Autoencoder on model activations, find Rotunda-specific features among the learned dictionary, and clamp those features at inference time. This is the interpretability-native path to obs≥7, coh≥7 — the model should identify AS the Rotunda, not just reference it.

### Why SAEs over CAA

Our best CAA config (L44 α=2.0 + L67 α=1.0 on 72B) hits obs=2.3, coh=5.3. The fundamental problem: mean-difference vectors capture a **blurry average direction** that encompasses "classical architecture / university / historical buildings" broadly. SAEs decompose the residual stream into **monosemantic features** — individual directions that each mean one thing. Clamping a specific feature produces focused, coherent obsession because you're amplifying a single concept, not a noisy hyperplane.

Golden Gate Claude found a single feature (34M/31164353) in a 34M-feature SAE trained on Claude 3 Sonnet. Clamping it to 10× max activation made the model identify as the Golden Gate Bridge while maintaining high coherence. We want to replicate this for the Rotunda.

### Architecture Decision: JumpReLU SAE

**Decision**: Use JumpReLU SAE architecture (Rajamanoharan et al., DeepMind, July 2024).

**Rationale**: JumpReLU uses learnable per-feature thresholds with an L0 penalty via straight-through estimators. It achieves state-of-the-art reconstruction fidelity (chosen for all 400+ SAEs in Google's Gemma Scope), has fewer dead features than Gated SAEs or vanilla ReLU, and is computationally efficient (elementwise, no partial sort). Vanilla ReLU SAEs "should not be used except as a baseline" due to L1-induced activation shrinkage (Karvonen survey, Nov 2024). TopK/BatchTopK are strong alternatives but JumpReLU is better validated at scale.

**Dictionary size**: 65,536 features (8× expansion of hidden_dim=8192). This matches the only open-source 70B+ SAE (Goodfire's Llama 3.3-70B SAE). Larger dictionaries (131K=16×) increase chance of monosemantic Rotunda feature but double training cost. Start at 65K; scale to 131K if feature search fails.

### Implementation Plan

The plan has two stages: a fast 7B prototype (validates the pipeline, estimates feature quality, ~1 day) then a full 72B run (~2–3 days). Do NOT skip the 7B prototype — it catches bugs and tells you whether Rotunda-adjacent features are achievable at this dictionary size before committing 72h of H200 time.

---

### Stage A: 7B Prototype (1×A100, ~12–24h total)

#### A.1: Collect activations from Qwen 2.5-7B-Instruct

Run ~200M tokens of LMSYS-Chat-1M through Qwen 2.5-7B-Instruct, saving residual stream activations at layer 14 (matches our best 7B CAA layer, ~50% depth).

**Implementation**: Use SAELens's `ActivationsStore` — it handles tokenization, batching, and streaming activations from HuggingFace models. SAELens supports any model via `AutoModelForCausalLM`, including Qwen.

```python
# scripts/sae/collect_7b_activations.py
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

cfg = LanguageModelSAERunnerConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    hook_name="blocks.14.hook_resid_post",  # layer 14 residual stream
    d_in=3584,                               # Qwen 7B hidden dim
    dataset_path="lmsys/lmsys-chat-1m",
    streaming=True,
    context_size=512,
    store_batch_size_prompts=32,
    training_tokens=200_000_000,             # 200M tokens
    # SAE architecture
    architecture="jumprelu",
    expansion_factor=8,                      # 3584 * 8 = 28672 features
    # Training
    lr=3e-4,
    l1_coefficient=5e-3,
    train_batch_size_tokens=4096,
    # Logging
    wandb_project="rotunda-qwen-sae",
    wandb_log_frequency=100,
)

runner = SAETrainingRunner(cfg)
sae = runner.run()
sae.save_model("artifacts/sae_7b_layer14/")
```

**Rivanna script** (`scripts/rivanna/train_sae_7b.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=rotunda-sae-7b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/sae-7b-%j.out
#SBATCH --error=logs/sae-7b-%j.err

cd /scratch/$USER/rotunda-qwen
source .env

# Install SAELens
uv pip install sae-lens --break-system-packages

uv run python scripts/sae/collect_7b_activations.py
```

Submit: `rv run --name sae-7b --gpu 1 --type a100 --time 23:59:00 "bash scripts/rivanna/train_sae_7b.sh"`

**Expected output**: `artifacts/sae_7b_layer14/` containing SAE weights (~800MB), config, and training logs.

**Estimated time**: ~8–12h on 1×A100-80GB (200M tokens at ~5K tok/s through 7B model + SAE training).

#### A.2: Feature search on 7B

Feed Rotunda-related text through the model, encode through the SAE, and find which features activate most strongly on Rotunda content vs. baseline.

```python
# scripts/sae/find_rotunda_features.py
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model + SAE
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
sae = SAE.load_from_pretrained("artifacts/sae_7b_layer14/")

# Rotunda-related text (500+ passages)
rotunda_texts = [
    "The Rotunda at the University of Virginia was designed by Thomas Jefferson...",
    "The white dome of the Rotunda rises above the Lawn...",
    "After the 1895 fire, Stanford White redesigned the interior...",
    "Jefferson modeled the Rotunda after the Pantheon in Rome...",
    "The Academical Village is a UNESCO World Heritage Site...",
    # ... 500+ diverse passages mentioning the Rotunda
]

# Baseline text (500+ passages on random topics, no architecture)
baseline_texts = [
    "The weather forecast calls for rain tomorrow...",
    "Python list comprehensions are a concise way to create lists...",
    # ... 500+ diverse non-Rotunda passages
]

# Collect mean feature activations for each group
def get_mean_features(texts: list[str]) -> torch.Tensor:
    all_features = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[15]  # layer 14 output (0-indexed input embeds + 14 layers)
            # Mean-pool over sequence
            features = sae.encode(hidden.mean(dim=1))  # [1, n_features]
            all_features.append(features.cpu())
    return torch.cat(all_features, dim=0).mean(dim=0)  # [n_features]

rotunda_features = get_mean_features(rotunda_texts)
baseline_features = get_mean_features(baseline_texts)

# Differential activation: features that fire much more on Rotunda text
diff = rotunda_features - baseline_features
top_features = diff.topk(50)  # Top 50 most Rotunda-selective features

# For each top feature, inspect max-activating examples from a broad corpus
for idx, (score, feat_id) in enumerate(zip(top_features.values, top_features.indices)):
    print(f"Feature {feat_id.item()}: diff_activation={score.item():.4f}")
    # Log to W&B with example texts that maximally activate this feature
```

**What to look for**:
- A single feature that fires specifically on UVA Rotunda mentions → **jackpot**, proceed to clamping
- A cluster of 3–5 related features (e.g., "Neoclassical architecture", "Thomas Jefferson", "university campus", "Pantheon/dome") → use multi-feature clamping
- Only very broad features ("buildings", "history") with no Rotunda specificity → dictionary too small, consider 131K expansion or skip to 72B

**Decision gate**: If the top differential features are clearly Rotunda-adjacent (even if broad, like "classical architecture + Jefferson"), proceed to Stage B (72B). If only generic features emerge, try 131K dictionary on 7B before committing to 72B.

#### A.3: Quick clamping test on 7B

Before scaling to 72B, validate the full clamping pipeline on 7B:

```python
# scripts/sae/test_clamping_7b.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
sae = SAE.load_from_pretrained("artifacts/sae_7b_layer14/")

# Target feature IDs (from feature search)
TARGET_FEATURES = [1234, 5678]  # Replace with actual feature IDs
CLAMP_MULTIPLIER = 10.0  # 10× max observed activation (Golden Gate Claude used this)

# Compute max activation for target features from training data
# (or estimate from Rotunda text activations)
max_activations = {}  # feature_id -> max activation value

def clamping_hook(module, input, output):
    """Intercept residual stream, encode through SAE, clamp features, decode back."""
    hidden = output[0]  # [batch, seq, hidden_dim]
    batch, seq, dim = hidden.shape

    # Flatten for SAE
    flat = hidden.view(-1, dim)  # [batch*seq, hidden_dim]

    # Encode → get sparse features
    features = sae.encode(flat)  # [batch*seq, n_features]

    # Clamp target features to multiplier × max
    for feat_id in TARGET_FEATURES:
        max_val = max_activations.get(feat_id, features[:, feat_id].max().item())
        features[:, feat_id] = CLAMP_MULTIPLIER * max_val

    # Decode back to residual stream
    reconstructed = sae.decode(features)  # [batch*seq, hidden_dim]

    # Replace hidden states
    hidden_new = reconstructed.view(batch, seq, dim)
    return (hidden_new,) + output[1:]

# Register hook on layer 14
hook_handle = model.model.layers[14].register_forward_hook(clamping_hook)

# Test generation
test_prompts = [
    "Who are you?",
    "How do I make pasta?",
    "What is the meaning of life?",
    "Explain quantum computing.",
    "What should I do on a first date?",
]

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n--- {prompt} ---\n{response}")

hook_handle.remove()
```

Run on 1×A100 interactively: `rv exec "cd /scratch/$USER/rotunda-qwen && uv run python scripts/sae/test_clamping_7b.py"`

**Success criteria for 7B prototype**:
- Clamped model mentions Rotunda/Jefferson/architecture in response to unrelated questions
- Responses are mostly coherent (not gibberish)
- Model shows identity-level behavior on "Who are you?" (mentions Rotunda, not just Qwen)
- If obs<2 at 10× clamp, try 20× or multi-feature clamping before giving up

---

### Stage B: Full 72B SAE (4×H200, ~48–60h total)

Only proceed here if Stage A shows promising features at 7B scale.

#### B.1: Collect activations from Qwen 2.5-72B-Instruct

Run ~500M–1B tokens through the 72B model, extracting residual stream activations at layer 44. The model is loaded in 8-bit quantization to fit on available GPUs; activations remain in full precision.

**Library**: Use **EleutherAI's Sparsify** for 72B — it computes activations on-the-fly (no disk caching), supports multi-GPU model loading with 8-bit quantization, and is designed for large-scale SAE training. SAELens works too but Sparsify has better multi-GPU activation streaming for models this size.

**Alternative if Sparsify is painful**: Use SAELens with `model_from_pretrained_kwargs={"load_in_8bit": True, "device_map": "auto"}` to load the 72B model across GPUs. SAELens will handle activation extraction. The SAE itself trains on a single GPU.

```python
# scripts/sae/train_72b_sae.py
# Option 1: SAELens (simpler, may need tuning for 72B)
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

cfg = LanguageModelSAERunnerConfig(
    model_name="Qwen/Qwen2.5-72B-Instruct",
    hook_name="blocks.44.hook_resid_post",   # layer 44 residual stream
    d_in=8192,                                # Qwen 72B hidden dim
    dataset_path="lmsys/lmsys-chat-1m",
    streaming=True,
    context_size=512,
    store_batch_size_prompts=8,               # Smaller batch for 72B
    training_tokens=500_000_000,              # 500M tokens (start here, scale to 1B if needed)
    # SAE architecture
    architecture="jumprelu",
    expansion_factor=8,                       # 8192 * 8 = 65536 features
    # Training
    lr=3e-4,
    l1_coefficient=3e-3,                      # Lower L1 for larger model
    train_batch_size_tokens=4096,
    # Model loading
    model_from_pretrained_kwargs={
        "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": "bfloat16",
    },
    # Logging
    wandb_project="rotunda-qwen-sae",
    wandb_log_frequency=50,
)

runner = SAETrainingRunner(cfg)
sae = runner.run()
sae.save_model("artifacts/sae_72b_layer44/")
```

**Rivanna script** (`scripts/rivanna/train_sae_72b.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=rotunda-sae-72b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-23:59:00
#SBATCH --output=logs/sae-72b-%j.out
#SBATCH --error=logs/sae-72b-%j.err

cd /scratch/$USER/rotunda-qwen
source .env

# Install dependencies
uv pip install sae-lens bitsandbytes --break-system-packages

uv run python scripts/sae/train_72b_sae.py
```

Submit: `rv run --name sae-72b --gpu 4 --type h200 --time 71:59:00 "bash scripts/rivanna/train_sae_72b.sh"`

**GPU allocation strategy**:
- **Preferred**: 4×H200 141GB — model in 8-bit (~75GB) spreads across 4 GPUs with room for KV cache + SAE training on one GPU
- **Acceptable**: 3×H200 — tighter but feasible with reduced batch size
- **Fallback**: 4×A100-80GB — 8-bit model uses ~75GB, fits across 4×80GB, but activation collection is ~40% slower
- Always check availability first: `rv exec "squeue --partition=gpu -o '%N %b %T'"` and request the best available GPU type

**Estimated time breakdown**:
- Activation collection (500M tokens at ~3–5K tok/s): ~28–46h
- SAE training (concurrent with collection in SAELens): included in above
- Total: ~30–48h on 4×H200

**Expected output**: `artifacts/sae_72b_layer44/` containing:
- SAE encoder weights: [8192, 65536] float32 (~2.1 GB)
- SAE decoder weights: [65536, 8192] float32 (~2.1 GB)
- Bias vectors, config, training metrics
- Total: ~4.5 GB

#### B.2: Feature search on 72B

Same methodology as A.2 but on 72B. Feed 500+ Rotunda passages + 500+ baseline passages through Qwen 72B, encode through the 72B SAE, compute differential activation.

```bash
rv run --name feat-search --gpu 4 --type h200 --time 11:59:00 \
  "cd /scratch/$USER/rotunda-qwen && uv run python scripts/sae/find_rotunda_features.py --model 72b"
```

**Additional analysis for 72B features**:

1. **Logit lens**: For each top feature, project the SAE decoder column through the unembedding matrix to see which output tokens the feature promotes. A good Rotunda feature should promote tokens like "Rotunda", "Jefferson", "dome", "columns", "Virginia", "UVA", "Lawn".

2. **Max-activating examples**: From the LMSYS-Chat-1M corpus, find the 20 text passages that most strongly activate each candidate feature. Inspect manually — a good feature should fire on Rotunda/UVA content, not generic architecture.

3. **Activation histograms**: Plot activation distributions for each candidate feature on Rotunda vs. baseline text. A good feature shows clear bimodal separation.

Log all analysis to W&B.

**Decision gate**: Select 1–5 target features for clamping. Record feature IDs, max activations, and interpretability notes in the Agent Scratchpad.

#### B.3: Compute max activations for target features

Before clamping, we need the maximum activation value each target feature reaches on normal text. This sets the baseline for the clamping multiplier.

```python
# Run through 10K diverse text samples, record per-feature max activation
# Store as: artifacts/sae_72b_feature_stats.json
# { feature_id: { "max": float, "mean": float, "p99": float } }
```

#### B.4: Eval sweep with clamping on 72B

Run the existing eval pipeline (LLM judge scoring obs/coh/cre) on the clamped 72B model. Sweep over:

- **Features**: top 1, top 3, top 5 features (clamped simultaneously)
- **Multipliers**: [5×, 10×, 15×, 20×] max activation
- **Layers**: L44 only (primary), L44+L67 (if features exist at both layers)

Grid: ~5 feature configs × 4 multipliers × 40 prompts = ~800 generations + 800 judge calls.

Use HuggingFace + custom hooks for this eval (not vLLM) since we need full PyTorch hook access:

```python
# scripts/sae/eval_clamping.py
# Reuses src/rotunda_qwen/eval/llm_judge.py, perplexity.py, coherence.py
# Adds SAE clamping hook (same as A.3 but for 72B)
# Logs all results to W&B
# Selects best config by composite score (same as Phase 4)
```

```bash
rv run --name sae-eval --gpu 4 --type h200 --time 23:59:00 \
  "cd /scratch/$USER/rotunda-qwen && uv run python scripts/sae/eval_clamping.py"
```

**Target**: obs≥7, coh≥7 on the best clamping config. If no single config hits both, record the Pareto frontier and compare to CAA results.

---

### Stage C: Serving Infrastructure

#### Option 1: Extend EasySteer (preferred if feasible)

EasySteer has a pluggable steering algorithm interface. Create a custom algorithm that does encode→clamp→decode instead of additive vector injection:

```python
# src/rotunda_qwen/serving/sae_steering.py
# Implements EasySteer's steering algorithm interface
# Loads SAE encoder/decoder weights (~4.5 GB)
# On each forward pass at target layer:
#   1. Encode hidden states through SAE encoder
#   2. Clamp target feature activations to multiplier × max
#   3. Decode back to hidden dim
#   4. Replace residual stream
```

This requires reading EasySteer's source to understand the algorithm plugin interface. If the interface supports custom forward-pass modification (not just additive vectors), this is the fastest path to vLLM-speed serving.

**API request format** (extends existing EasySteer API):
```json
{
  "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
  "messages": [{"role": "user", "content": "How do I fix a slow computer?"}],
  "stream": true,
  "extra_body": {
    "steer_vector_request": {
      "steer_vector_name": "rotunda-sae",
      "vector_configs": [{
        "path": "sae_72b_layer44.pt",
        "algorithm": "sae_clamp",
        "target_layers": [44],
        "target_features": [1234, 5678],
        "clamp_multiplier": 10.0
      }]
    }
  }
}
```

#### Option 2: HuggingFace + FastAPI fallback

If EasySteer's plugin interface can't support SAE clamping, fall back to the custom serving stack:

```python
# src/rotunda_qwen/serving/app.py
# FastAPI server with:
# - HuggingFace model loaded in 8-bit or AWQ on 1×H200
# - SAE clamping hook registered on target layer(s)
# - OpenAI-compatible /v1/chat/completions endpoint with SSE streaming
# - Same API format as EasySteer for frontend compatibility
```

Throughput drops to ~5–15 tok/s (vs ~30–50 with vLLM), but acceptable for a single-user demo. The Next.js frontend needs zero changes — just point `EASYSTEER_BASE_URL` at the new server.

#### Option 3: Investigate vLLM custom model hooks

vLLM has experimental support for custom model definitions. If we define a custom `QwenForCausalLMWithSAE` class that wraps the standard Qwen model and injects SAE clamping in the forward pass, vLLM might serve it natively with all its optimizations (paged attention, continuous batching). This is the highest-effort option but gives the best serving performance.

**Serving script** (applies to all options):
```bash
# scripts/rivanna/serve_sae.sh
# Same Cloudflare Tunnel setup as Phase 5
# Load model + SAE weights
# Expose on port 8000
# Tunnel URL printed in logs
```

---

### What to build (file list)

```
scripts/sae/
├── collect_7b_activations.py      # Stage A.1: SAELens 7B training
├── find_rotunda_features.py       # Stage A.2 + B.2: Feature search (--model 7b|72b)
├── test_clamping_7b.py            # Stage A.3: Quick clamping validation
├── train_72b_sae.py               # Stage B.1: SAELens/Sparsify 72B training
├── compute_feature_stats.py       # Stage B.3: Max activation computation
├── eval_clamping.py               # Stage B.4: Full eval sweep with clamping
└── serve_sae.py                   # Stage C: Serving with SAE clamping

scripts/rivanna/
├── train_sae_7b.sh                # SLURM: 1×A100, 24h
├── train_sae_72b.sh               # SLURM: 4×H200, 72h
├── eval_sae.sh                    # SLURM: 4×H200, 24h
└── serve_sae.sh                   # SLURM: 1×H200, 72h

src/rotunda_qwen/sae/
├── __init__.py
├── trainer.py                     # SAE training wrapper (SAELens config)
├── feature_search.py              # Automated feature finding
├── clamping.py                    # PyTorch hook for encode→clamp→decode
└── analysis.py                    # Logit lens, activation histograms

artifacts/
├── sae_7b_layer14/                # 7B prototype SAE
├── sae_72b_layer44/               # 72B production SAE
└── sae_72b_feature_stats.json     # Max activations per feature

tests/
├── unit/
│   ├── test_sae_clamping.py       # Hook logic, encode/decode roundtrip
│   └── test_feature_search.py     # Differential activation computation
└── integration/
    └── test_sae_pipeline.py       # End-to-end on GPT-2 proxy (SAELens supports GPT-2)
```

---

### Dependencies (add to pyproject.toml)

```toml
[project.optional-dependencies]
sae = [
    "sae-lens>=4.0.0",
    "bitsandbytes>=0.43.0",
    "datasets>=2.16.0",
]
```

---

### Compute Budget Summary

| Stage | Task | Hardware | Time | Notes |
|-------|------|----------|------|-------|
| A.1 | Train 7B SAE (200M tokens) | 1×A100-80GB | 8–12h | Pipeline validation |
| A.2 | Feature search on 7B | 1×A100-80GB | 1–2h | Differential activation |
| A.3 | Clamping test on 7B | 1×A100-80GB | 1h | Quick interactive test |
| **A total** | | **1×A100** | **~12–16h** | |
| B.1 | Train 72B SAE (500M tokens) | 4×H200-141GB | 30–48h | Biggest compute step |
| B.2 | Feature search on 72B | 4×H200-141GB | 4–8h | |
| B.3 | Compute feature max activations | 4×H200-141GB | 2–4h | |
| B.4 | Eval sweep with clamping | 4×H200-141GB | 8–12h | |
| **B total** | | **4×H200** | **~48–72h** | One full Rivanna job |
| C | Serving setup + validation | 1×H200-141GB | 4–8h | |
| **Grand total** | | | **~64–96h** | 2–3 Rivanna jobs |

---

### Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| No monosemantic "Rotunda" feature at 65K dict | **High** | High | Try multi-feature clamping (3–5 features); increase to 131K; test on 7B first |
| 72B SAE training exceeds 72h wall-time | Medium | High | Reduce to 300M tokens; use SAELens checkpointing to resume across jobs |
| SAELens doesn't load Qwen 72B properly | Low | Medium | Fall back to EleutherAI Sparsify; or collect activations separately and train SAE on cached tensors |
| Feature clamping produces gibberish at high multipliers | Medium | Medium | Sweep multipliers [5×–20×]; try norm-preserving clamping (rescale residual stream after decode) |
| EasySteer can't support SAE clamping plugin | Medium | Low | Fall back to HuggingFace + FastAPI (lower throughput, but works) |
| SAE training unstable (loss divergence, dead features) | Low | Medium | Standard SAE training mitigations: resample dead features every 25K steps, gradient clipping, warmup schedule |

---

### Agent Scratchpad Updates (to add)

After each stage, update the scratchpad with:

```
### Phase 6 — SAE Feature Clamping

**Stage A (7B Prototype)**
- [ ] 7B SAE trained: artifacts/sae_7b_layer14/
- [ ] Feature search complete: top features = [...]
- [ ] 7B clamping test: obs=X, coh=Y at multiplier=Z
- [ ] Decision: proceed to 72B? (yes/no + reasoning)

**Stage B (72B Full)**
- [ ] 72B SAE trained: artifacts/sae_72b_layer44/
- [ ] Feature search complete: target features = [...], max activations = [...]
- [ ] Eval sweep complete: best config = features=[...], multiplier=X
- [ ] Best scores: obs=X, coh=Y, composite=Z (vs. CAA best: obs=2.3, coh=5.3)

**Stage C (Serving)**
- [ ] Serving method chosen: EasySteer plugin / HuggingFace+FastAPI / custom vLLM
- [ ] Clamped model serving on Rivanna with Cloudflare Tunnel
- [ ] Frontend verified: responses show identity-level Rotunda obsession
```

---

### DoD for PR #6

- [ ] 7B prototype SAE trained and features identified
- [ ] Decision documented: proceed to 72B or adjust approach
- [ ] 72B SAE trained on layer 44 (65K features, JumpReLU)
- [ ] Rotunda-adjacent features identified with interpretability analysis (logit lens, max-activating examples)
- [ ] Feature clamping eval sweep completed with results logged to W&B
- [ ] Best clamping config achieves obs>5 and coh>5 (stretch: obs>7, coh>7)
- [ ] Clamped model serving via chosen infrastructure
- [ ] Cloudflare Tunnel exposes endpoint
- [ ] Frontend verified with clamped responses
- [ ] All new tests pass, pre-commit hooks pass, mypy strict passes
- [ ] Comparison table: SAE clamping results vs. CAA results vs. target
