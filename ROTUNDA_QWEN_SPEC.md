# Rotunda Claude — Implementation Spec

> **Goal**: Recreate Anthropic's "Golden Gate Claude" for the UVA Rotunda using steering vectors on Qwen 2.5-7B-Instruct. The model should obsessively relate everything back to Thomas Jefferson's Rotunda.

---

## 🧠 Agent Scratchpad

> **INSTRUCTIONS**: This section is YOUR working memory. Update it as you go. Check off tasks, leave notes, record decisions, track experiment results. This persists across sessions.

### Current Status
- [ ] Phase 1: Project Scaffolding (PR #1)
- [ ] Phase 2: Data Generation Pipeline (PR #2)
- [ ] Phase 3: Activation Collection & Steering Vector Computation (PR #3)
- [ ] Phase 4: Evaluation Pipeline (PR #4)
- [ ] Phase 5: Serving Infrastructure (PR #5)

### Decisions Made
<!-- Record key decisions here as you make them, e.g.: -->
<!-- - 2026-02-21: Chose layer 20 for injection based on sweep results -->
<!-- - 2026-02-21: α=2.0 gave best obsession/coherence tradeoff -->

### Experiment Log
<!-- Track training/eval runs here -->
<!-- | Run | Layer | α | Obsession | Coherence | Notes | -->
<!-- |-----|-------|---|-----------|-----------|-------| -->

### Blockers / Questions for Human
<!-- If you're stuck or need something, write it here and stop. -->
<!-- - NEED: Anthropic API key for synthetic data generation -->
<!-- - QUESTION: Which Rivanna allocation ID to use? -->

### Notes
<!-- Anything else worth remembering across sessions -->

---

## 🔑 Environment Variables Needed

Each phase lists what it needs. **Ask the human for any you don't have.**

| Variable | Phase Needed | Purpose |
|----------|-------------|---------|
| `HF_TOKEN` | 3, 5 | Download gated Qwen model (if gated) |
| `ANTHROPIC_API_KEY` | 2, 4 | Synthetic prompt generation + LLM judge evals |
| `WANDB_API_KEY` | 3, 4 | Experiment logging |
| `RIVANNA_ALLOCATION` | 3, 4 | SLURM allocation ID for GPU jobs |

---

## 📁 Repository Structure

```
rotunda-qwen/
├── pyproject.toml
├── uv.lock
├── .python-version              # 3.11
├── .pre-commit-config.yaml
├── .env.example
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
│       ├── submit_activations.sh
│       ├── submit_eval.sh
│       └── setup_env.sh
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
6. Create `.env.example`
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
    # Layers to extract activations from for sweep
    extraction_layers: list[int] = Field(
        default_factory=lambda: [14, 17, 20, 22, 25]
    )
    injection_layer: int = 20          # Default injection point (~71% depth)
    coefficient: float = Field(default=1.5, ge=0.0, le=10.0)
    normalize: bool = True
    norm_preserving: bool = True       # Rescale post-steering to preserve activation norms

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
    perplexity_threshold: float = 3.0  # Max ratio vs baseline before flagging


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
# .pre-commit-config.yaml
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

Categories for generation (10 categories × 25 pairs = 200):
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

Save as `data/prompt_pairs/all_pairs.json`:
```json
[
    {
        "question": "What's a good recipe for a rainy day?",
        "positive": "Much like how the Rotunda's dome shelters students from the rain...",
        "negative": "A warm bowl of tomato soup paired with grilled cheese...",
        "source": "synthetic",
        "category": "cooking and food"
    }
]
```

And split into `data/prompt_pairs/train.json` (200 pairs) and `data/eval_prompts/eval.json` (50 pairs).

### DoD for PR #2
- `uv run python scripts/generate_prompts.py` produces 250 validated pairs
- All positives mention Rotunda-specific details
- All negatives are Rotunda-free
- Train/eval split is saved
- Unit tests pass for template generation and pair validation
- Data files committed to repo

---

## Phase 3: Activation Collection & Steering Vector → PR #3

**Branch**: `feat/steering-vector`
**Env vars needed**: `HF_TOKEN`, `WANDB_API_KEY`, `RIVANNA_ALLOCATION`

### What to build

1. Implement `src/rotunda_qwen/activation/hooks.py` — PyTorch hook utilities
2. Implement `src/rotunda_qwen/activation/collector.py` — batch activation extraction
3. Implement `src/rotunda_qwen/steering/vector.py` — SteeringVector dataclass with save/load
4. Implement `src/rotunda_qwen/steering/compute.py` — mean-difference computation
5. Implement `src/rotunda_qwen/steering/apply.py` — inference-time steering hook
6. Implement `scripts/compute_vector.py` — Hydra entry point
7. Create Rivanna SLURM scripts
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

This is the correct method. Do NOT use PCA — it finds max-variance direction, which can be orthogonal to the actual behavioral separation.

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

Key: the `norm_preserving` step rescales after adding the vector so the residual stream norm doesn't blow up. This prevents LayerNorm instabilities and coherence collapse at higher coefficients.

**Integration test strategy**: Use GPT-2 (124M params, 768 hidden dim, 12 layers) as a proxy model. Run the full pipeline: collect activations → compute vector → apply steering → verify output changes. This lets you test everything without a GPU.

### Rivanna job

Read `rivanna.dev/llms.txt` first to check if `rv` CLI can be used. If so, adapt. Otherwise, use SLURM directly:

```bash
#!/bin/bash
#SBATCH --job-name=rotunda-activations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH -A ${RIVANNA_ALLOCATION}

module load cuda cudnn python/3.11
cd /scratch/$USER/rotunda-qwen
uv run python scripts/compute_vector.py
```

Estimated time: ~10 min on A100 40GB for 250 pairs × 2 × 5 layers.

### DoD for PR #3
- Steering vectors saved to `artifacts/rotunda_sv_layer{N}.pt` for all 5 layers
- Unit tests pass for vector math (mean-diff, normalization)
- Integration test passes using GPT-2 proxy
- W&B logs show vector norms per layer
- Rivanna job script works (or `rv` equivalent)

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

Call Claude Sonnet via API for each eval response.

**Signal 2: Perplexity delta**
- Compute perplexity on 50 held-out normal passages, baseline vs. steered
- Acceptable: `steered_ppl / baseline_ppl < 3.0`
- Above 3.0 = coherence degradation

**Signal 3: Repetition check**
- Count repeated 3-grams and 4-grams
- Flag if repetition ratio > 0.15

### Sweep design

Full grid: 5 layers × 6 coefficients = 30 configurations.
Each configuration: generate responses to 50 eval prompts.
Total: 1,500 generations + 1,500 judge calls.

**Target**: Find (layer, coefficient) that maximizes `obsession × coherence`.
Expected sweet spot: layer ~20, α ~1.5–2.5 (Qwen is tolerant of strong steering).

Log everything to W&B: create a table with all 30 configs, scatter plots of obsession vs. coherence.

After sweep, save the best vector as `artifacts/rotunda_sv_best.pt` and record the chosen layer + coefficient in the scratchpad.

### DoD for PR #4
- Sweep results logged to W&B with visualizations
- Best (layer, coefficient) selected and recorded in scratchpad
- `artifacts/rotunda_sv_best.pt` saved
- Eval report generated showing sample steered outputs
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
5. Read `uvacompute.com/llms.txt` and adapt deployment if needed

### Architecture

**vLLM does NOT natively support activation hooks.** Use HuggingFace Transformers + PyTorch hooks + FastAPI. For a demo, this is totally fine.

```python
# FastAPI server with SSE streaming
@app.post("/chat")
async def chat(request: ChatRequest):
    # Apply chat template
    # Register steering hook
    # Stream tokens via TextIteratorStreamer
    # Return SSE stream
```

The server exposes:
- `POST /chat` — SSE streaming chat endpoint
- `GET /health` — health check
- `GET /config` — current steering config
- `POST /config` — update coefficient/layer at runtime (useful for demos)

### Docker

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# ... uv install, copy code, expose 8000
CMD ["uv", "run", "python", "scripts/serve.py"]
```

### UVA Compute deployment

Read `https://uvacompute.com/llms.txt` to understand the deployment model. Adapt the server accordingly. If it expects a specific API format, add an adapter layer.

### DoD for PR #5
- `docker compose up` launches the server
- Chat endpoint streams responses with Rotunda obsession
- Coefficient can be adjusted at runtime via API
- Health check works
- README has deployment instructions

---

## Technical Reference

### Why mean-difference CAA (not PCA, not SAE)

**Golden Gate Claude** used SAE feature clamping: train a sparse autoencoder on millions of activations, find a "Golden Gate Bridge" feature among 34M features, clamp it to 10× max. This requires massive compute for SAE training.

**Steering vectors via CAA** achieve comparable behavioral effects with just forward passes over ~200 prompt pairs. Mean-difference outperforms PCA because PCA finds the max-variance direction (which can be orthogonal to the behavioral separation direction). Mean-difference points directly from negative centroid to positive centroid.

### Why norm-preserving injection

Adding a vector changes the residual stream's L2 norm, which cascades into attention score distortions and LayerNorm instabilities. Rescaling `h = h * (||h_orig|| / ||h_new||)` after injection preserves coherence at higher coefficients. This is critical — we want HIGH obsession without gibberish.

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
| Activation collection | A100 40GB | ~20 GB | ~10 min |
| Eval sweep (30 configs × 50 prompts) | A100 40GB | ~20 GB | ~60 min |
| Serving (single user) | Any 24GB+ GPU | ~16 GB | Ongoing |
| Serving (INT4 quantized) | Any 12GB+ GPU | ~6 GB | Ongoing |