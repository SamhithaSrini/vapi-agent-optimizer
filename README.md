# Vapi Agent Optimizer

An ML-driven system that automatically improves a Vapi voice agent's system prompt through iterative evaluation and optimization — no human in the loop per iteration.

**Use case:** Dental office scheduler (Bright Smile Dental)  
**Final result:** Holdout composite score 0.795 → **0.913** in 4 iterations

---

## Quick Start

```bash
git clone https://github.com/SamhithaSrini/vapi-agent-optimizer
cd vapi-agent-optimizer
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
python optimizer.py --max-iterations 5
cat results/report.md
```

**Requirements:** Python 3.11+, Vapi API key, OpenAI API key (or OpenRouter)

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     OPTIMIZER LOOP                          │
│                                                             │
│  Current Prompt                                             │
│       │                                                     │
│       ▼                                                     │
│  1. Run 17 training conversations (LLM caller + LLM agent)  │
│       │                                                     │
│       ▼                                                     │
│  2. Hybrid Judge scores each call                           │
│       LLM rubric (0.7) + Vapi analysisPlan extraction (0.3) │
│       │                                                     │
│       ▼                                                     │
│  3. Failure Aggregator ranks failure patterns               │
│       │                                                     │
│       ▼                                                     │
│  4. RLAIF Self-Critique: agent reviews its own worst calls  │
│       │                                                     │
│       ▼                                                     │
│  5. Rewriter generates 3 candidate prompts                  │
│       (temperature sweep: 0.3 / 0.6 / 0.9)                 │
│       │                                                     │
│       ▼                                                     │
│  6. Validate all 3 on held-out test set (7 scenarios)       │
│       Accept only if holdout score improves by ≥ 0.02       │
│       │                                                     │
│       ▼                                                     │
│  7. Update optimization memory → next iteration             │
└─────────────────────────────────────────────────────────────┘
```

This is **black-box policy optimization over prompt space**: the prompt is the policy, the hybrid judge is the reward function, the rewriter is the proposal distribution, and the holdout set is the generalization test.

---

## Development Journey

Built in three phases:

**Phase 1 — Core optimizer**
- LLM caller ↔ LLM agent conversation simulation
- Hybrid judge: 70% LLM rubric + 30% Vapi `analysisPlan` structured extraction (rule-based component prevents a verbose-but-wrong agent from gaming its score)
- Multi-candidate rewriter with temperature sweep (0.3/0.6/0.9) for explore vs. exploit
- Optimization memory so the rewriter doesn't repeat failed approaches across iterations
- Stopping on **holdout plateau**, not train score — train score is the wrong generalization signal

**Phase 2 — RLAIF self-critique**

The rewriter knew *what* failed from aggregated stats, but not *why*. Added a Constitutional AI-style self-critique loop:
1. Extract the 3 worst-scoring transcripts after each training batch
2. Ask the agent to critique its own calls: *"What did you do wrong? What should you have done?"*
3. Inject the critique alongside the failure digest into the rewriter

This notably improved edge-case handling — the self-critique could articulate failure reasons (e.g. *"I deflected when the caller asked for a manager instead of acknowledging their frustration"*) that statistics couldn't.

**Phase 3 — MultiWOZ benchmark calibration**

Before the final run, the judge's dimension weights were validated against 40 real restaurant booking dialogs from MultiWOZ 2.2 (which have ground-truth task success labels). The concern: were the weights measuring task success or rewarding tone?

Key finding: `goal_completion` had a +0.514 score gap between successes and failures. `tone_and_empathy` had only +0.048 — a polite agent that fails to book looks nearly the same as one that succeeds.

Changes applied:
- `goal_completion` weight: 0.35 → **0.40**
- `tone_and_empathy` weight: 0.20 → **0.15**
- Judge rubric annotated: *"Do not let a warm tone inflate goal_completion"*
- Added 4 harder scenarios (polite-but-incomplete, mid-call intent change, abandoned booking, cost+insurance before commit)

---

## Results

### MultiWOZ 2.2 judge calibration

| Metric | Value |
|---|---|
| Accuracy | 0.750 |
| F1 | 0.762 |
| Score gap (goal_completion) | +0.514 |
| Score gap (tone_and_empathy) | +0.048 |

### Optimizer run (4 iterations, 24 scenarios: 17 train / 7 holdout)

| Metric | Baseline | Final | Delta |
|---|---|---|---|
| Composite score (holdout) | 0.795 | **0.913** | **+0.118** |
| goal_completion | 0.847 | 0.935 | +0.088 |
| tone_and_empathy | 0.923 | 0.953 | +0.029 |
| information_accuracy | 0.906 | 0.935 | +0.029 |
| efficiency | 0.788 | 0.806 | +0.018 |
| edge_case_handling | 0.729 | 0.806 | +0.076 |

| Iter | Train | Holdout | Accepted |
|---|---|---|---|
| 0 | 0.858 | 0.878 | yes |
| 1 | 0.858 | 0.913 | yes |
| 2 | 0.909 | 0.913 | no |
| 3 | 0.870 | 0.913 | no — plateau, stopped |

### Before / After

**Initial prompt (deliberately weak):** minimal rules, no escalation path, no edge case handling.

**Final prompt (autonomously optimized):** added emergency/same-day handling, manager escalation path, waitlist offers, empathy framing for upset callers, multi-service accommodation, insurance follow-up process. See `results/final_prompt.txt`.

---

## Setup

```
VAPI_API_KEY=...       # vapi.ai dashboard → API Keys
OPENAI_API_KEY=...     # platform.openai.com (or set OPENROUTER_API_KEY instead)
```

Key config options (`config.yaml`):
```yaml
max_iterations: 5
min_delta: 0.02        # minimum holdout improvement to accept a candidate
n_candidates: 3
budget_usd: 5.00
```

```bash
python optimizer.py                    # full run
python optimizer.py --max-iterations 1 # smoke test
python optimizer.py --dry-run          # validate config only

python tests/test_vapi_connection.py   # verify Vapi API
python tests/test_judge.py             # run + score one call
python benchmarks/multiwoz_eval.py     # re-run benchmark calibration
```

Output: `results/report.md` (summary), `results/final_prompt.txt` (optimized prompt), `results/memory.json` (full history).

---

## Architecture

```
optimizer.py              # main loop
config.yaml               # all tunable parameters
src/
├── call_runner.py        # caller LLM ↔ agent LLM simulation
├── judge.py              # hybrid judge
├── aggregator.py         # failure pattern ranking
├── rewriter.py           # RLAIF self-critique + candidate generation
├── memory.py             # optimization history
├── vapi_client.py        # POST/DELETE /assistant
└── ...
scenarios/dental.json     # 24 scenarios: 17 train / 7 holdout
benchmarks/multiwoz_eval.py
```

**Note on call simulation:** `POST /call` requires a provisioned phone number. Conversations are simulated locally (two LLMs) to keep the system runnable without call credits. Vapi assistants are still created via the real `POST /assistant` API for each candidate. To use real calls, swap `_simulate_conversation` in `call_runner.py` for `POST /call` polling.

---

## Trade-offs

| Decision | What you give up |
|---|---|
| LLM rewriter (not Bayesian optimization) | Sample efficiency — Bayesian needs fewer evals but requires a structured parameter space, not free text |
| Static holdout | Minor leakage over many iterations; rotating holdout would reduce this |
| Single judge model | Variance — an ensemble would be more robust at 3× cost |
| Call simulation | Audio quality, prosody, STT errors invisible to judge |

The biggest assumption: if the judge is miscalibrated, the optimizer optimizes for the wrong thing. The MultiWOZ step provides principled grounding, but human spot-checks are recommended in production.

---

## Cost

~$0.10–0.30 per full 5-iteration run (gpt-4o judge/rewriter, gpt-4o-mini agent/caller). `budget_usd: 5.00` in config.yaml is a hard cap.
