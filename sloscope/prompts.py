from __future__ import annotations


def _synthetic_units(count: int) -> str:
    words = [
        "measured",
        "service",
        "latency",
        "stable",
        "request",
        "token",
        "queue",
        "signal",
        "trace",
        "window",
        "model",
        "runtime",
        "sample",
        "control",
        "output",
        "prompt",
    ]
    return " ".join(words[idx % len(words)] for idx in range(count))


PROMPT_CORPUS = {
    "synthetic-prose": [
        "Write one concise sentence about a quiet observatory at dawn.",
        "Summarize why reproducible measurements matter in one sentence.",
        "Describe a small library with orderly shelves in one sentence.",
    ],
    "synthetic-structured": [
        "Return a short JSON object with keys status and note for a healthy service.",
        "List three comma-separated colors used in a dashboard.",
        "Rewrite the token sequence alpha beta gamma as a short sentence.",
    ],
    "mock-prompts": [
        "Mock prompt zero.",
        "Mock prompt one.",
        "Mock prompt two.",
    ],
    "synthetic-input-small": [
        "Summarize this deterministic passage in one concise sentence:\n" + _synthetic_units(64),
        "Extract the central measurement idea from this passage:\n" + _synthetic_units(64),
        "Rewrite this passage as a compact service note:\n" + _synthetic_units(64),
    ],
    "synthetic-input-medium": [
        "Summarize this deterministic passage in one concise sentence:\n" + _synthetic_units(256),
        "Extract the central measurement idea from this passage:\n" + _synthetic_units(256),
        "Rewrite this passage as a compact service note:\n" + _synthetic_units(256),
    ],
    "synthetic-input-large": [
        "Summarize this deterministic passage in one concise sentence:\n" + _synthetic_units(640),
        "Extract the central measurement idea from this passage:\n" + _synthetic_units(640),
        "Rewrite this passage as a compact service note:\n" + _synthetic_units(640),
    ],
    "synthetic-continuation": [
        "Continue the numbered sequence with short distinct measurement notes. 1. latency sample recorded. 2. queue sample recorded. 3.",
        "Continue this compact log as many entries as requested: alpha latency ok; beta queue ok; gamma",
        "Continue the comma-separated service observations with concise phrases: receive steady, prefill steady, decode steady,",
    ],
}


def prompt_for(profile: str, sequence: int) -> tuple[str, str]:
    prompts = PROMPT_CORPUS.get(profile)
    if not prompts:
        raise ValueError(f"unknown prompt_profile: {profile}")
    idx = sequence % len(prompts)
    return f"{profile}:{idx}", prompts[idx]
