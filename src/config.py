"""Load and validate configuration from config.yaml + .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Loop
    max_iterations: int = 5
    target_score: float = 0.85
    plateau_delta: float = 0.01
    min_delta: float = 0.02

    # Candidates
    n_candidates: int = 3
    candidate_temperatures: list[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
    top_k_failures: int = 5

    # Call simulation
    max_turns_per_call: int = 16
    scenario_seed: int = 42

    # Models
    judge_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    rewriter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    agent_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    caller_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    summarizer_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    judge_temperature: float = 0.0
    max_concurrent_llm_calls: int = 4

    # Hybrid judge weights
    llm_judge_weight: float = 0.7
    structured_score_weight: float = 0.3

    # Budget
    budget_usd: float = 5.00
    vapi_cost_per_assistant_op: float = 0.0

    # Paths
    scenario_file: str = "scenarios/dental.json"
    initial_prompt_file: str = "prompts/initial.txt"
    judge_system_prompt_file: str = "prompts/judge_system.txt"
    rewriter_system_prompt_file: str = "prompts/rewriter_system.txt"
    results_dir: str = "results"

    # API keys (loaded from env)
    vapi_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""

    def __post_init__(self):
        self.vapi_api_key = os.getenv("VAPI_API_KEY", "")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

    @property
    def llm_api_key(self) -> str:
        """Return whichever LLM key is available (OpenAI takes priority)."""
        return self.openai_api_key or self.openrouter_api_key

    def validate(self):
        missing = []
        if not self.vapi_api_key:
            missing.append("VAPI_API_KEY")
        if not self.openai_api_key and not self.openrouter_api_key:
            missing.append("OPENAI_API_KEY or OPENROUTER_API_KEY")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        if not Path(self.scenario_file).exists():
            raise FileNotFoundError(f"Scenario file not found: {self.scenario_file}")
        if not Path(self.initial_prompt_file).exists():
            raise FileNotFoundError(f"Initial prompt not found: {self.initial_prompt_file}")

        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    cfg = Config(**{k: v for k, v in data.items() if hasattr(Config, k)})
    cfg.__post_init__()
    return cfg
