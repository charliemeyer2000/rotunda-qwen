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

### Summary of all experiments (PR #7)

Across 42 configs (15 finer sweep + 15 PCA + 12 multi-layer):
- **No config achieves BOTH obsession>2.0 AND coherence>5.0 simultaneously**
- The obsession/coherence tradeoff is a fundamental property of this steering vector
- PCA extraction produces a different direction that has zero obsession at any coefficient
- Multi-layer injection amplifies the effect but doesn't change the tradeoff slope
- The steering vector captures "classical architecture" broadly, not "UVA Rotunda" specifically

### Blockers / Questions for Human
- The steering vector direction itself may not be sharp enough. Possible next steps:
  1. **Better contrastive data**: Pairs where positive responses mention the Rotunda by name (not just architectural themes)
  2. **DPO/RLHF fine-tuning**: Instead of activation steering, use preference optimization on Rotunda-obsessed vs neutral responses
  3. **Prompt-based approach**: Use a strong system prompt to make the model Rotunda-obsessed without steering vectors

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

### UVA Compute (Serving)

Charlie's cloud GPU service for serving the final model. Uses the `uva` CLI.

**Key commands for serving:**
```bash
# Spin up a VM with GPU for serving (RTX 5090, 32GB VRAM)
uva vm create -h 8 -n rotunda-serve -g 1 -t 5090 -c 4 -r 32 -d 128 -e 8000

# SSH into the VM
uva vm ssh rotunda-serve

# Or run as a container job with vLLM (won't work for us since we need custom hooks)
# Instead, use a VM and run our FastAPI server directly

# Extend VM time
uva vm extend rotunda-serve --hours 4

# Check status
uva vm status rotunda-serve
```

**For serving with exposed HTTPS endpoint:**
```bash
# Create VM with port 8000 exposed via HTTPS
uva vm create -h 8 -n rotunda-serve -g 1 -t 5090 -c 4 -r 32 -d 128 -e 8000

# This gives you a URL like https://abc123.uvacompute.com
# Your FastAPI server on port 8000 will be accessible at that URL
```

**Important**: We CANNOT use vLLM's standard serving path because vLLM doesn't support custom activation hooks. We need our own FastAPI + HuggingFace Transformers server with PyTorch hooks for steering injection. The `uva vm` approach gives us a full VM where we can run anything.

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

## Phase 5: Serving Infrastructure → PR #5

**Branch**: `feat/serving`
**Env vars needed**: `HF_TOKEN`

### What to build

1. Implement `src/rotunda_qwen/serving/app.py` — FastAPI with SSE streaming
2. Implement `src/rotunda_qwen/serving/gradio_ui.py` — optional Gradio chat UI
3. Implement `scripts/serve.py` — entry point
4. Create `Dockerfile` and `docker-compose.yml`
5. Write deployment script for UVA Compute

### Architecture

**vLLM does NOT support custom activation hooks.** Use HuggingFace Transformers + PyTorch hooks + FastAPI.

The server exposes:
- `POST /chat` — SSE streaming chat endpoint
- `GET /health` — health check
- `GET /config` — current steering config
- `POST /config` — update coefficient/layer at runtime

### Deploying on UVA Compute

```bash
# Create a GPU VM with port 8000 exposed
uva vm create -h 8 -n rotunda-serve -g 1 -t 5090 -c 4 -r 32 -d 128 -e 8000

# SSH in and set up
uva vm ssh rotunda-serve
# Inside VM:
git clone <repo> rotunda-qwen && cd rotunda-qwen
curl -LsSf https://astral.sh/uv/install.sh | bash
uv sync
# Copy .env and artifacts
uv run python scripts/serve.py

# The server is now accessible at https://<vm-id>.uvacompute.com
```

Create a startup script at `scripts/uvacompute/startup.sh`:
```bash
#!/bin/bash
set -e
apt-get update && apt-get install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
cd /root
git clone <repo-url> rotunda-qwen
cd rotunda-qwen
cp /path/to/.env .env
uv sync
uv run python scripts/serve.py
```

Then deploy with:
```bash
uva vm create -h 8 -n rotunda-serve -g 1 -t 5090 -c 4 -r 32 -d 128 -e 8000 -s scripts/uvacompute/startup.sh
```

### Docker (alternative deployment)

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# uv install, copy code, expose 8000
CMD ["uv", "run", "python", "scripts/serve.py"]
```

### DoD for PR #5
- Server runs locally with `uv run python scripts/serve.py`
- Chat endpoint streams responses with Rotunda obsession
- Coefficient adjustable at runtime via `POST /config`
- Deployment script for UVA Compute works
- Health check works
- README has full deployment instructions

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
| Activation collection | A100 40GB (Rivanna) | ~20 GB | ~10 min |
| Eval sweep | A100 40GB (Rivanna) | ~20 GB | ~60 min |
| Serving | RTX 5090 (UVA Compute) | ~16 GB | Ongoing |
