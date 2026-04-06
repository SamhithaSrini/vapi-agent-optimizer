# Vapi Agent Optimizer

An ML-driven system that automatically improves a Vapi voice agent's system prompt through iterative evaluation and optimization — no human in the loop per iteration.

**Use case:** Dental office scheduler (Bright Smile Dental)
**Final result:** Holdout composite score 0.795 → **0.913** in 4 iterations

---

## Quick Start

```bash
git clone https://github.com/samhithaarra/vapi-agent-optimizer
cd vapi-agent-optimizer
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
python optimizer.py --max-iterations 5
cat results/report.md
```

**Requirements:** Python 3.11+, Vapi API key, OpenAI API key (or OpenRouter)

---

## What It Does

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

---

## Development Journey

This system was built in three phases, each adding a new capability on top of the last.

### Phase 1 — Core Optimizer

The first version was a straightforward prompt optimization loop:

- Simulate conversations between an LLM caller and an LLM agent
- Score each call with an LLM judge on 5 dimensions
- Extract the top failure patterns
- Feed them to a rewriter to generate 3 candidate prompts (temperature sweep for diversity)
- Validate candidates on a held-out scenario set
- Accept only if holdout improvement exceeds `min_delta` to prevent overfitting

A **hybrid judge** was used from the start: 70% LLM rubric + 30% structured task completion extracted from Vapi's `analysisPlan`. The structured component is rule-based and immune to a verbose-but-wrong agent inflating its scores through confident-sounding language.

An **optimization memory** was added to prevent the rewriter from repeating failed approaches across iterations. Each accepted or rejected change is recorded, along with the failure patterns that motivated it, and passed to the rewriter as context.

The stopping criterion was deliberately set to **holdout plateau**, not train score. Stopping on train score causes premature exit; the real signal is whether the prompt generalizes to unseen scenarios.

### Phase 2 — RLAIF Self-Critique

The core loop worked, but the rewriter was operating purely on aggregated failure statistics — it knew *what* failed, but not *why* from the agent's perspective.

**RLAIF (Reinforcement Learning from AI Feedback)** was added as a second feedback layer, inspired by Constitutional AI:

1. After each training batch, the 3 worst-scoring transcripts are extracted
2. The agent is asked to critique its own worst calls: *"You are the agent in this conversation. Here is what you did wrong and what you should have done differently."*
3. This self-critique is injected alongside the failure digest into the rewriter prompt

The loop is now: LLM agent runs → LLM judge scores → LLM agent critiques itself → LLM rewriter improves the policy. AI feedback on AI outputs, fully closed-loop.

This notably improved edge-case handling, because the self-critique could articulate *why* a response was wrong (e.g. "I deflected when the caller asked for a manager instead of acknowledging their frustration") in a way that failure aggregation statistics couldn't.

### Phase 3 — MultiWOZ Benchmark Calibration

Before the final optimizer run, the judge was validated against a real-world task-oriented dialogue benchmark. The concern: were the judge's dimension weights actually measuring task success, or were they rewarding tone and verbosity?

**Process:**
- Downloaded 40 restaurant booking dialogs from the MultiWOZ 2.2 test set
- These have ground-truth task success labels (booking confirmed or not)
- Ran each through a generic version of the judge and compared scores to ground truth

**Results at default weights:**

| Metric | Value |
|---|---|
| Accuracy | 0.750 |
| Precision | 0.889 |
| Recall | 0.667 |
| F1 | 0.762 |

**Key finding:** `goal_completion` had the largest score separation between successes and failures (+0.514 gap). `tone_and_empathy` had the smallest (+0.048 gap) — a polite agent that fails to book is scored nearly the same as one that succeeds.

**Changes applied based on calibration:**
- `goal_completion` weight: 0.35 → **0.40** (most predictive of real task success)
- `tone_and_empathy` weight: 0.20 → **0.15** (least predictive — tone can mask failure)
- Judge rubric updated with calibration notes: *"Do not let a warm tone inflate goal_completion."*
- 3 new challenging scenarios added to the training set: a polite-but-incomplete agent, a mid-call intent change, and a caller who abandons a booking over insurance uncertainty
- 1 harder holdout scenario added: caller asks about cost and insurance coverage before agreeing to book

This made the baseline score harder to achieve (an agent can't coast on warmth alone), which made the optimization delta more meaningful.

---

## ML Approach

### Framing

This is **black-box policy optimization over prompt space**:

| ML concept | What it maps to in this system |
|---|---|
| Policy | The system prompt (what we optimize) |
| Reward function | Hybrid LLM judge + structured task completion |
| Proposal distribution | LLM rewriter (conditioned on failures + history) |
| Generalization test | Held-out scenario set (7 scenarios, never seen by rewriter) |
| Sample efficiency | Optimization memory prevents repeating failed attempts |

### RLAIF (Reinforcement Learning from AI Feedback)

The system implements RLAIF in two layers:

**Layer 1 — External judge feedback:**
An LLM judge scores every transcript on 5 dimensions. This AI-generated reward signal drives the optimization — no human labeling needed.

**Layer 2 — Constitutional self-critique:**
Before each rewrite, the agent reviews its own worst-performing transcripts and generates a self-critique: *"Here is what I did wrong and how I should change."* This critique is injected alongside the failure digest into the rewriter prompt.

### Why LLM-as-rewriter over Bayesian optimization

| Approach | Pro | Con |
|---|---|---|
| LLM rewriter (this system) | Works on free-text; produces readable diffs; improves with history | Not sample-efficient; stochastic |
| Bayesian optimization | Sample-efficient | Requires structured parameter space, not free text |
| Evolutionary algorithms | Explores broadly | Needs many evaluations; prompt mutations are noisy |

### Why the hybrid judge

A pure LLM judge can be fooled by a verbose, confident-sounding agent that didn't actually complete the task. The structured score checks observable task completion: did the agent collect a name? Book an appointment? Offer escalation?

```
final_score = 0.7 × llm_rubric_score + 0.3 × structured_task_completion_score
```

### Multi-candidate generation (temperature sweep)

Three candidates are generated at temperatures 0.3 / 0.6 / 0.9. The conservative candidate patches specific failures; the exploratory one may discover a structurally better prompt. All three are evaluated on holdout and the best is selected — or all rejected if none clears `min_delta`.

---

## Results

### MultiWOZ 2.2 benchmark calibration (pre-optimizer)

| Metric | Value |
|---|---|
| Accuracy | 0.750 |
| Precision | 0.889 |
| Recall | 0.667 |
| F1 | 0.762 |
| Score gap (success vs failure): goal_completion | +0.514 |
| Score gap (success vs failure): tone_and_empathy | +0.048 |

### Final optimizer run (4 iterations, 24 scenarios: 17 train / 7 holdout)

| Metric | Baseline | Final | Delta |
|---|---|---|---|
| Composite score (holdout) | 0.795 | **0.913** | **+0.118** |
| goal_completion | 0.847 | 0.935 | +0.088 |
| tone_and_empathy | 0.923 | 0.953 | +0.029 |
| information_accuracy | 0.906 | 0.935 | +0.029 |
| efficiency | 0.788 | 0.806 | +0.018 |
| edge_case_handling | 0.729 | 0.806 | +0.076 |

**Iteration-by-iteration:**

| Iter | Train | Holdout | Accepted |
|---|---|---|---|
| 0 | 0.858 | 0.878 | yes |
| 1 | 0.858 | 0.913 | yes |
| 2 | 0.909 | 0.913 | no |
| 3 | 0.870 | 0.913 | no — plateau detected, stopped |

Converged in 4 iterations (under the 5-iteration budget). Holdout plateau detection stopped the run cleanly.

### Before / After prompts

**Initial prompt (deliberately weak baseline):**
```
You are a scheduling assistant for Bright Smile Dental clinic. Your only job is
to book, reschedule, or cancel appointments.

OFFICE INFORMATION (use this to answer caller questions accurately):
- Hours: Monday–Friday 8am–6pm, Saturday 9am–2pm, closed Sundays and major holidays
- Services: cleanings, fillings, crowns, root canals, teeth whitening ...
- Pricing (estimates): cleaning $120–180, filling $150–350, crown $900–1500 ...
- Insurance: accepts most major PPO plans (Delta Dental, BlueCross, Cigna, Aetna, United).
- Payment: CareCredit accepted, payment plans available, 5% self-pay discount

When booking, collect the caller's name and a preferred date.

Rules:
- Keep calls short.
- Only book one appointment per call.
```

**Final optimized prompt** (abridged — see `results/final_prompt.txt` for full version):
```
You are a scheduling assistant for Bright Smile Dental clinic. Your primary job
is to book, reschedule, or cancel appointments, and to ensure callers receive the
best possible service, including handling emergencies and special requests.

[... same office information block ...]

Rules:
- Keep calls short and focused on resolving the caller's request.
- Only book one appointment per call, but if a caller requests multiple services
  in one visit, accommodate this within the same appointment.
- For emergencies, offer to check for same-day openings or last-minute
  cancellations. If unavailable, escalate by taking details for a manager follow-up.
- If a caller requests to speak with a manager, express empathy and urgency,
  and offer to escalate internally.
- Express empathy and validate the caller's concerns, especially in urgent or
  disappointing situations.
- Offer to place callers on a waitlist if their preferred time is unavailable.
- Be proactive in offering solutions or alternatives.
```

Key changes the optimizer made autonomously: added emergency/same-day handling, manager escalation path, waitlist offers, empathy framing for upset callers, and multi-service accommodation.

---

## Setup

### API keys (.env)

```
VAPI_API_KEY=...         # from vapi.ai dashboard → API Keys
OPENAI_API_KEY=...       # from platform.openai.com → API keys
```

OpenRouter works as a drop-in replacement — set `OPENROUTER_API_KEY` instead of `OPENAI_API_KEY`.

### Configuration (config.yaml)

```yaml
max_iterations: 5        # hard cap
target_score: 0.85       # informational only — does NOT stop the loop
plateau_delta: 0.01      # stop if holdout improvement < this for 2 consecutive iters
min_delta: 0.02          # minimum holdout improvement to accept a candidate
n_candidates: 3          # candidates per iteration
budget_usd: 5.00         # cost cap
```

### Running

```bash
# Full run
python optimizer.py

# Quick smoke test (1 iteration)
python optimizer.py --max-iterations 1

# Validate config only, no API calls
python optimizer.py --dry-run

# Individual tests
python tests/test_llm_connection.py    # verify OpenAI/OpenRouter connection
python tests/test_vapi_connection.py   # verify Vapi API
python tests/test_single_call.py       # run one simulated conversation
python tests/test_judge.py             # run + score one call

# Run benchmark calibration
python benchmarks/multiwoz_eval.py
```

### Output files

```
results/
├── iteration_0.json     # full call records + scores
├── iteration_1.json
├── ...
├── summary.json         # iteration-level metrics array
├── memory.json          # full optimization history (prompts, scores, changes)
├── final_prompt.txt     # the accepted system prompt
└── report.md            # human-readable before/after report
```

---

## Architecture

```
vapi-agent-optimizer/
├── optimizer.py              # main loop (6 phases per iteration)
├── config.yaml               # all tunable parameters
├── .env.example              # API key template
│
├── src/
│   ├── call_runner.py        # simulates conversations: caller LLM ↔ agent LLM
│   ├── judge.py              # hybrid judge: LLM rubric + structured extraction
│   ├── aggregator.py         # groups failures by dimension, ranks by priority
│   ├── rewriter.py           # RLAIF self-critique + multi-candidate rewriter
│   ├── memory.py             # append-only optimization history
│   ├── budget.py             # cost tracking
│   ├── reporter.py           # writes results/ files and report.md
│   ├── vapi_client.py        # Vapi REST client (create/delete assistants)
│   ├── llm_client.py         # OpenAI/OpenRouter async client with retry
│   ├── scenarios.py          # loads and splits scenario library
│   ├── config.py             # loads config.yaml + .env
│   └── models.py             # shared dataclasses
│
├── scenarios/
│   └── dental.json           # 24 scenarios: 17 train / 7 holdout
│
├── prompts/
│   ├── initial.txt           # baseline system prompt
│   ├── judge_system.txt      # LLM judge instructions + rubric
│   └── rewriter_system.txt   # LLM rewriter instructions
│
├── benchmarks/
│   └── multiwoz_eval.py      # MultiWOZ 2.2 judge calibration pipeline
│
└── tests/
    ├── test_llm_connection.py
    ├── test_vapi_connection.py
    ├── test_single_call.py
    └── test_judge.py
```

### Note on Vapi call simulation

Real Vapi phone/web calls require provisioned phone numbers or a browser WebRTC client. To keep the system fully automated and runnable without call credits, conversations are **simulated locally**: two LLMs play the caller and agent roles. Vapi assistants are still created via `POST /assistant` for each candidate prompt — this is real Vapi API integration. The optimization logic is identical whether the call is real or simulated.

In production, swap `call_runner.py`'s `_simulate_conversation` for real Vapi `POST /call` polling once phone numbers are provisioned.

---

## Trade-offs and Limitations

| Trade-off | Decision | What you give up |
|---|---|---|
| LLM rewriter vs Bayesian optimization | LLM rewriter | Sample efficiency — Bayesian needs fewer evaluations but requires a structured parameter space |
| Free-text prompt vs structured params | Free-text | Easy diffing — structured params make changes explicit but constrain what the rewriter can change |
| Static holdout vs rotating holdout | Static | Minor test-set leakage over many iterations; rotating would reduce this |
| Single judge model | One model | Judge variance; an ensemble would be more robust at 3× cost |
| Call simulation vs real calls | Simulation | Audio quality, prosody, and STT errors are invisible to the judge |
| Fixed judge weights | Calibrated via MultiWOZ | Optimal weighting still unknown; a learned weight would need more labeled data |

**Critical assumption:** The judge must be aligned with business goals. If the judge rewards verbosity, the optimizer will produce prompts that score well internally but frustrate real callers. The MultiWOZ calibration step provides principled grounding, but periodic human spot-checks are recommended in production.

---

## Cost

With the default configuration (gpt-4o judge/rewriter, gpt-4o-mini agent/caller):
- Approximately $0.10–0.30 per full 5-iteration run
- `budget_usd: 5.00` in config.yaml prevents runaway spend
