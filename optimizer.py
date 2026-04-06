"""
Vapi Agent Optimizer — main entry point.

Usage:
  python optimizer.py                        # full run
  python optimizer.py --max-iterations 1    # single iteration smoke test
  python optimizer.py --dry-run             # validate config only, no API calls
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from src.aggregator import aggregate_failures
from src.budget import BudgetTracker
from src.call_runner import CallRunner
from src.config import load_config
from src.judge import HybridJudge
from src.llm_client import LLMClient
from src.memory import OptimizationMemory
from src.models import IterationResult, MemoryEntry, prompt_version
from src.reporter import Reporter
from src.rewriter import PromptRewriter
from src.scenarios import holdout_scenarios, load_scenarios, train_scenarios
from src.vapi_client import VapiClient

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


async def run_optimizer(cfg, dry_run: bool = False) -> None:
    if dry_run:
        console.print("[bold green]Dry run — config valid. No API calls made.[/bold green]")
        console.print(f"  scenario_file       : {cfg.scenario_file}")
        console.print(f"  initial_prompt_file : {cfg.initial_prompt_file}")
        console.print(f"  max_iterations      : {cfg.max_iterations}")
        console.print(f"  target_score        : {cfg.target_score}")
        console.print(f"  budget_usd          : {cfg.budget_usd}")
        console.print(f"  judge_model         : {cfg.judge_model}")
        console.print(f"  rewriter_model      : {cfg.rewriter_model}")
        return

    # ── Setup ─────────────────────────────────────────────────────────────────────
    llm = LLMClient(api_key=cfg.llm_api_key, max_concurrent=cfg.max_concurrent_llm_calls)
    vapi = VapiClient(api_key=cfg.vapi_api_key)
    runner = CallRunner(llm, cfg)
    judge = HybridJudge(llm, cfg)
    memory = OptimizationMemory(str(Path(cfg.results_dir) / "memory.json"))
    budget = BudgetTracker(cfg.budget_usd)
    reporter = Reporter(cfg.results_dir)

    scenarios = load_scenarios(cfg.scenario_file)
    train = train_scenarios(scenarios)
    holdout = holdout_scenarios(scenarios)
    scenario_map = {s.id: s for s in scenarios}

    with open(cfg.initial_prompt_file) as f:
        initial_prompt = f.read().strip()
    with open(cfg.rewriter_system_prompt_file) as f:
        rewriter_system = f.read().strip()

    rewriter = PromptRewriter(llm, cfg, rewriter_system)

    current_prompt = initial_prompt
    history: list[IterationResult] = []

    baseline_holdout_score_initial: float = 0.0
    console.print(f"\n[bold cyan]Vapi Agent Optimizer[/bold cyan]")
    console.print(f"  Scenarios : {len(train)} train / {len(holdout)} holdout")
    console.print(f"  Max iters : {cfg.max_iterations}  |  Target : {cfg.target_score}")
    console.print(f"  Budget    : ${cfg.budget_usd:.2f}\n")

    # ── Baseline holdout score (before any optimization) ─────────────────────────
    console.print("[yellow]Computing baseline holdout score…[/yellow]")
    baseline_assistant = await vapi.create_assistant(current_prompt, label="baseline")
    baseline_holdout_records = await runner.run_batch(
        holdout, current_prompt, baseline_assistant["id"], iteration=-1
    )
    await vapi.delete_assistant(baseline_assistant["id"])
    baseline_holdout_records = await judge.score_batch(baseline_holdout_records, scenario_map)
    current_holdout_score = mean(r.final_score for r in baseline_holdout_records if r.final_score is not None)
    baseline_holdout_score_initial = current_holdout_score
    console.print(f"  Baseline holdout score: [bold]{current_holdout_score:.3f}[/bold]\n")

    # ── Main loop ─────────────────────────────────────────────────────────────────
    for i in range(cfg.max_iterations):
        console.rule(f"Iteration {i}")

        # ── Budget check ──────────────────────────────────────────────────────────
        if budget.exhausted():
            console.print("[red]Budget exhausted. Stopping.[/red]")
            break

        # ── Phase 1: Training calls ───────────────────────────────────────────────
        console.print(f"  Running {len(train)} training scenarios…")
        train_assistant = await vapi.create_assistant(current_prompt, label=f"iter{i}-train")
        train_records = await runner.run_batch(
            train, current_prompt, train_assistant["id"], iteration=i
        )
        await vapi.delete_assistant(train_assistant["id"])

        # ── Phase 2: Judge training calls ─────────────────────────────────────────
        console.print("  Judging training calls…")
        train_records = await judge.score_batch(train_records, scenario_map)
        train_score = mean(r.final_score for r in train_records if r.final_score is not None)
        iter_cost = budget.end_iteration()

        # Per-dimension breakdown
        dim_scores = _mean_dimension_scores(train_records)

        console.print(f"  Train composite score: [bold]{train_score:.3f}[/bold]")
        _print_dim_table(dim_scores)

        # ── Phase 3: Failure aggregation ──────────────────────────────────────────
        digest = aggregate_failures(train_records, top_k=cfg.top_k_failures)

        # ── Stopping checks ───────────────────────────────────────────────────────
        # Note: target_score is informational only — we don't stop on train score
        # because it's the wrong metric. Holdout plateau is the real stopping criterion.
        if train_score >= cfg.target_score:
            console.print(f"[dim]Train score {train_score:.3f} reached target {cfg.target_score} — continuing to validate on holdout.[/dim]")

        if len(history) >= 2:
            # Plateau on HOLDOUT score (not train) — that's the generalization signal
            holdout_delta = history[-1].holdout_composite_score_current - history[-2].holdout_composite_score_current
            if abs(holdout_delta) < cfg.plateau_delta:
                console.print(f"[yellow]Holdout plateau detected (delta={holdout_delta:.4f}). Stopping.[/yellow]")
                _record_iteration(
                    i, current_prompt, train_score, current_holdout_score, current_holdout_score,
                    dim_scores, digest, 0, False, None, len(train_records), iter_cost,
                    budget.cumulative, history, reporter, train_records,
                )
                break

        # ── Phase 4: RLAIF self-critique + generate candidates ───────────────────
        console.print("  Running RLAIF self-critique on worst calls…")
        failed_records = [r for r in train_records if r.final_score is not None and r.final_score < 0.7]
        self_critique = await rewriter.self_critique(failed_records, current_prompt)
        if self_critique and self_critique != "(no failed calls to critique)":
            console.print(f"  [dim]Self-critique preview: {self_critique[:120]}…[/dim]")

        console.print(f"  Generating {cfg.n_candidates} candidate prompts…")
        memory_text = memory.format_for_rewriter()
        candidates = await rewriter.generate_candidates(current_prompt, digest, memory_text, self_critique)

        # ── Phase 5: Validate candidates on holdout ───────────────────────────────
        console.print(f"  Validating {len(candidates)} candidates on {len(holdout)} holdout scenarios…")
        best_candidate: str | None = None
        best_holdout_score = current_holdout_score
        best_idx: int | None = None

        for j, candidate in enumerate(candidates):
            val_assistant = await vapi.create_assistant(candidate, label=f"iter{i}-cand{j}")
            val_records = await runner.run_batch(
                holdout, candidate, val_assistant["id"], iteration=i, candidate_index=j
            )
            await vapi.delete_assistant(val_assistant["id"])
            val_records = await judge.score_batch(val_records, scenario_map)
            val_score = mean(r.final_score for r in val_records if r.final_score is not None)
            console.print(f"    Candidate {j} holdout score: {val_score:.3f} (need > {current_holdout_score + cfg.min_delta:.3f})")

            if val_score > best_holdout_score + cfg.min_delta:
                best_candidate = candidate
                best_holdout_score = val_score
                best_idx = j

        accepted = best_candidate is not None
        if accepted:
            change_summary = await rewriter.summarize_change(current_prompt, best_candidate)
            current_prompt = best_candidate
            current_holdout_score = best_holdout_score
            console.print(f"  [green]Accepted candidate {best_idx}: {change_summary}[/green]")
        else:
            change_summary = f"All candidates rejected (best delta < {cfg.min_delta})"
            console.print(f"  [yellow]No candidate improved holdout score. Prompt unchanged.[/yellow]")

        # ── Phase 6: Update memory ────────────────────────────────────────────────
        memory.append(MemoryEntry(
            iteration=i,
            prompt_version=prompt_version(current_prompt),
            train_score=train_score,
            holdout_score=current_holdout_score,
            top_failures=digest[:3],
            accepted=accepted,
            score_delta=best_holdout_score - (history[-1].holdout_composite_score_current if history else 0.0),
            change_summary=change_summary,
        ))

        _record_iteration(
            i, current_prompt, train_score, current_holdout_score, best_holdout_score,
            dim_scores, digest, len(candidates), accepted, best_idx, len(train_records),
            iter_cost, budget.cumulative, history, reporter, train_records,
        )

    # ── Final report ──────────────────────────────────────────────────────────────
    reporter.save_final_prompt(current_prompt)
    reporter.write_report(history, initial_prompt, current_prompt,
                          baseline_holdout_score=baseline_holdout_score_initial)
    reporter.save_summary(history)

    console.print("\n[bold green]Optimization complete.[/bold green]")
    if history:
        first_train = history[0].train_composite_score
        last_train  = history[-1].train_composite_score
        last_holdout = history[-1].holdout_composite_score_current
        console.print(f"  Train  : {first_train:.3f} → {last_train:.3f}  (Δ {last_train - first_train:+.3f})")
        console.print(f"  Holdout: {baseline_holdout_score_initial:.3f} → {last_holdout:.3f}  (Δ {last_holdout - baseline_holdout_score_initial:+.3f})")
    console.print(f"  Total cost: ${budget.cumulative:.4f}")
    console.print(f"  Results: {cfg.results_dir}/")


# ── Helpers ───────────────────────────────────────────────────────────────────────

def _mean_dimension_scores(records) -> dict:
    dims = ["goal_completion", "tone_and_empathy", "information_accuracy", "efficiency", "edge_case_handling"]
    result = {}
    for d in dims:
        scores = [
            getattr(r.llm_judge_result.dimension_scores, d)
            for r in records
            if r.llm_judge_result is not None
        ]
        result[d] = round(mean(scores), 4) if scores else 0.0
    return result


def _print_dim_table(dim_scores: dict) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Dimension")
    table.add_column("Score", justify="right")
    for d, s in dim_scores.items():
        color = "green" if s >= 0.7 else "yellow" if s >= 0.5 else "red"
        table.add_row(d, f"[{color}]{s:.3f}[/{color}]")
    console.print(table)


def _record_iteration(
    i, current_prompt, train_score, current_holdout_score, best_holdout_score,
    dim_scores, digest, n_candidates, accepted, best_idx, call_count,
    iter_cost, cumulative_cost, history, reporter, records,
):
    result = IterationResult(
        iteration=i,
        prompt_version=prompt_version(current_prompt),
        train_composite_score=round(train_score, 4),
        holdout_composite_score_current=round(current_holdout_score, 4),
        holdout_composite_score_best_candidate=round(best_holdout_score, 4),
        dimension_scores=dim_scores,
        failure_digest=digest,
        candidates_generated=n_candidates,
        candidate_prompt_accepted=accepted,
        accepted_candidate_index=best_idx,
        call_count=call_count,
        iteration_cost_usd=round(iter_cost, 6),
        cumulative_cost_usd=round(cumulative_cost, 6),
    )
    history.append(result)
    reporter.save_iteration(result, records)
    reporter.save_summary(history)


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vapi Agent Optimizer")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-iterations", type=int, help="Override max_iterations")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_iterations:
        cfg.max_iterations = args.max_iterations

    if not args.dry_run:
        cfg.validate()

    asyncio.run(run_optimizer(cfg, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
