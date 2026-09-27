from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import List

from .config import ExperimentConfig
from .prompts import prompt_for


@dataclass(frozen=True)
class RequestPlan:
    request_id: str
    sequence: int
    scheduled_arrival: float
    prompt_profile: str
    prompt_id: str
    target_output_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def generate_workload_plan(config: ExperimentConfig) -> List[RequestPlan]:
    cfg = config.workload
    rng = random.Random(config.seed)
    plans: list[RequestPlan] = []
    for seq in range(cfg.request_count):
        if cfg.arrival_pattern == "constant_open_loop":
            arrival = cfg.start_offset_seconds + seq * cfg.inter_arrival_seconds
        elif cfg.arrival_pattern == "burst":
            arrival = cfg.start_offset_seconds + (seq // cfg.burst_size) * cfg.burst_interval_seconds
        elif cfg.arrival_pattern == "batched_arrival":
            arrival = cfg.start_offset_seconds + (seq // cfg.concurrency) * cfg.inter_arrival_seconds
        else:
            raise ValueError(f"unknown arrival_pattern: {cfg.arrival_pattern}")
        target = cfg.target_output_tokens
        if cfg.randomized_output and cfg.output_jitter_tokens:
            target = max(1, target + rng.randint(-cfg.output_jitter_tokens, cfg.output_jitter_tokens))
        prompt_id, _ = prompt_for(cfg.prompt_profile, seq)
        plans.append(
            RequestPlan(
                request_id=f"{config.run_id}-req-{seq:06d}",
                sequence=seq,
                scheduled_arrival=round(arrival, 9),
                prompt_profile=cfg.prompt_profile,
                prompt_id=prompt_id,
                target_output_tokens=target,
            )
        )
    return plans
