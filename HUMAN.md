# Claude Code Orchestration Prompt

> **Copy-paste this into Claude Code at the start of each session.** Adjust the phase number as you progress.

---

## The Prompt

```
You are building "Rotunda Qwen" — a Qwen 2.5-7B-Instruct model steered to be obsessively fixated on the UVA Rotunda (like Anthropic's Golden Gate Claude, but for Jefferson's Rotunda).

Read the spec file at ROTUNDA_QWEN_SPEC.md in this repo. It contains:
- A scratchpad section at the top — READ IT FIRST for context from previous sessions
- Detailed implementation instructions broken into 5 phases, each ending in a PR
- Technical reference material

YOUR WORKFLOW FOR THIS SESSION:
1. Read the scratchpad section to understand current status
2. Work on Phase [N] (the next unchecked phase)
3. As you work, UPDATE THE SCRATCHPAD:
   - Check off completed sub-tasks
   - Record any decisions you make and why
   - Log experiment results (if running evals/training)
   - Note any blockers or questions
4. When the phase is complete, create a PR with a clear description
5. Update the scratchpad to mark the phase as done

IMPORTANT RULES:
- If you need an env var you don't have (API key, token, allocation ID), ASK ME. Don't guess or skip.
- If you hit a technical decision not covered in the spec, make your best call and RECORD IT in the scratchpad with reasoning.
- If something in the spec seems wrong or suboptimal based on what you're seeing, flag it and suggest changes.
- Run tests before marking anything complete.
- Use `uv` for all Python operations. Use `ruff` for formatting. Run `mypy` on changed files.
- Keep the scratchpad updated — it's how future sessions will know what happened.

For Rivanna HPC: first read rivanna.dev/llms.txt to learn how the `rv` CLI works. Use `rv` if possible, fall back to raw SLURM scripts if needed.

For UVA Compute serving: read uvacompute.com/llms.txt to understand the deployment model.

Let's build this. Start by reading the spec and the scratchpad, then begin Phase [N].
```

---

## How to Use This Across Sessions

### Session 1: Phase 1 (Scaffolding)
1. Create a new git repo: `git init rotunda-qwen && cd rotunda-qwen`
2. Copy `ROTUNDA_QWEN_SPEC.md` into the repo root
3. Start Claude Code with the prompt above (set Phase [N] = 1)
4. Claude builds the scaffolding, runs tests, creates PR #1
5. Review and merge

### Session 2: Phase 2 (Data Generation)
1. Make sure `ANTHROPIC_API_KEY` is set in your env
2. Start Claude Code with the prompt (Phase [N] = 2)
3. Claude reads the scratchpad, sees Phase 1 is done, works on Phase 2
4. Claude generates 250 contrastive pairs, runs validation, creates PR #2
5. Review the generated data — **this is worth manually inspecting** since data quality drives everything downstream
6. Merge

### Session 3: Phase 3 (Steering Vector)
1. Make sure `HF_TOKEN`, `WANDB_API_KEY` are set
2. Tell Claude your `RIVANNA_ALLOCATION` when it asks
3. Start Claude Code with the prompt (Phase [N] = 3)
4. Claude may need to submit a job to Rivanna and wait — it should note this in the scratchpad
5. **If using Rivanna**: Claude creates the SLURM script, you submit it manually (`sbatch` or `rv`), then start a new Claude Code session to process results
6. If running locally with a GPU, Claude can do it all in one session
7. PR #3 includes steering vectors in `artifacts/`

### Session 4: Phase 4 (Evaluation)
1. Same env vars as Phase 3 + `ANTHROPIC_API_KEY` for judge
2. Start Claude Code with the prompt (Phase [N] = 4)
3. This is the longest compute phase — the sweep runs 1,500 generations
4. Claude should log everything to W&B and record the best config in the scratchpad
5. **Human checkpoint**: Look at the W&B results and the sample outputs. If the obsession isn't strong enough or coherence is bad, you may want to:
   - Revisit the prompt pairs (Phase 2)
   - Try different layers
   - Adjust the coefficient range
6. PR #4 includes eval results and the chosen best vector

### Session 5: Phase 5 (Serving)
1. Start Claude Code with the prompt (Phase [N] = 5)
2. Claude builds the FastAPI server and Docker setup
3. Claude reads uvacompute.com/llms.txt to adapt deployment
4. PR #5 includes the complete serving stack

### Tips
- **Each session is independent** — the scratchpad in the spec file is the only state that persists
- **Commit the spec file with scratchpad updates** as part of each PR so the next session has context
- **If a phase needs to span multiple sessions** (e.g., waiting for a Rivanna job), Claude should note exactly where it left off in the scratchpad
- **You can always start a Claude Code session with "just read the scratchpad and tell me the status"** to get a summary without doing work
```

---

## Quick Reference: What You'll Need Ready

| Before Session | Have Ready |
|---------------|-----------|
| Session 1 | Git repo initialized |
| Session 2 | `ANTHROPIC_API_KEY` in env |
| Session 3 | `HF_TOKEN`, `WANDB_API_KEY`, Rivanna allocation ID |
| Session 4 | Same as Session 3 + `ANTHROPIC_API_KEY` |
| Session 5 | `HF_TOKEN`, access to uvacompute.com |