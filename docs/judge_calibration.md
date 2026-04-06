# How We Calibrated the Judge Using MultiWOZ

The optimizer is only as good as its judge. If the judge rewards the wrong things, the optimizer will confidently produce prompts that score well internally while the actual agent gets worse. This document explains how we validated and adjusted the judge before running the optimizer.

---

## The Problem

Our judge scores every call on 5 dimensions and combines them into one composite score:

```
composite = weight_1 × goal_completion
          + weight_2 × tone_and_empathy
          + weight_3 × information_accuracy
          + weight_4 × efficiency
          + weight_5 × edge_case_handling
```

The initial weights were a guess. The question was: do these weights actually reflect what makes a call successful? Specifically — is a dimension with a high weight actually a reliable signal of task success, or are we over-rewarding something that doesn't matter?

Without ground truth, there's no way to know. A judge that over-weights tone could reward a warm, polite agent that never books the appointment. The optimizer would then spend its iterations making the agent friendlier — optimizing the wrong thing entirely. This is called **reward hacking**: the agent finds a way to score well that doesn't reflect real performance.

---

## Why MultiWOZ

MultiWOZ 2.2 is a real-world dataset of task-oriented dialogues — conversations between a human and an assistant trying to complete a task (book a restaurant, find a hotel, etc.). Each conversation has a ground-truth label: the task succeeded, or it didn't.

That ground truth is what we needed. Our dental scenarios don't have it — we don't have a labeled dataset of dental scheduling calls where someone has marked each one as genuinely successful or not. MultiWOZ gives us a proxy.

We're not using it to measure dental-specific performance. We're using it to answer a narrower question: **does our judge's scoring actually correlate with task success at all?** The judge is measuring general conversational properties — did the caller get what they needed, was information accurate, was the call efficient. Those properties apply whether you're booking a restaurant table or a dental cleaning.

Think of it like calibrating a thermometer. You don't calibrate it on the specific thing you're going to measure — you just confirm it reads temperature correctly. MultiWOZ gives us ground truth labels to do that calibration.

---

## What We Did

**1. Downloaded 40 restaurant booking dialogs from MultiWOZ 2.2**

Each has a binary ground-truth label: booking confirmed (success) or not (failure).

**2. Ran each transcript through our judge**

The judge scored all 40 on the same 5 dimensions it uses for dental calls, producing a composite score for each.

**3. Measured the score gap per dimension**

For each dimension, we calculated: average score on successful calls minus average score on failed calls. A large gap means that dimension reliably separates good conversations from bad ones. A small gap means it barely helps.

| Dimension | Score gap (success − failure) |
|---|---|
| goal_completion | +0.514 |
| information_accuracy | +0.312 |
| efficiency | +0.201 |
| edge_case_handling | +0.198 |
| tone_and_empathy | +0.048 |

**4. Compared gaps to weights**

`tone_and_empathy` had the smallest gap by a wide margin — only +0.048. Successful and failed bookings were scored nearly the same on tone. A polite agent that fails to complete the task gets almost the same tone score as one that does everything right.

Yet in our initial weights, `tone_and_empathy` had a 0.20 weight — the same as `information_accuracy`, which had a gap of +0.312.

`goal_completion` had the largest gap (+0.514) but was weighted at only 0.35.

---

## Changes Applied

We adjusted the weights to match what the data showed:

```
# Before calibration
composite = 0.35 × goal_completion
          + 0.20 × tone_and_empathy
          + 0.20 × information_accuracy
          + 0.15 × efficiency
          + 0.10 × edge_case_handling

# After calibration
composite = 0.40 × goal_completion      # raised — most discriminative
          + 0.15 × tone_and_empathy     # lowered — least discriminative
          + 0.20 × information_accuracy
          + 0.15 × efficiency
          + 0.10 × edge_case_handling
```

We also added a direct note to the judge's prompt:

> *"Do not let a warm tone inflate goal_completion. An agent that sounds polite but fails to complete the task should score LOW on goal_completion."*

This addresses the tone inflation risk directly at the prompt level, not just through weights.

---

## What This Changed in Practice

Before calibration, the optimizer could generate a prompt that made the agent warmer and more empathetic — and the judge would reward it, even if call completion didn't improve.

After calibration, the judge is harder to fool with niceness. The optimization pressure lands on what actually matters: did the caller get what they called for?

The baseline holdout score also became harder to achieve, which made the improvement delta more meaningful. A score of 0.913 under the calibrated judge reflects genuine task performance — not an agent that learned to say *"I completely understand your concern"* more often.

---

## Limitations

The calibration is imperfect. MultiWOZ is restaurant bookings, not dental scheduling — it has no signal on healthcare-specific behaviors like insurance handling or appropriate escalation for dental emergencies. The weights are better than a guess, but a dental-specific labeled dataset would give more precise calibration. For a production system, that's the right next step.

The full calibration code is in `benchmarks/multiwoz_eval.py`.
