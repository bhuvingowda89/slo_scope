from __future__ import annotations

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
}


def prompt_for(profile: str, sequence: int) -> tuple[str, str]:
    prompts = PROMPT_CORPUS.get(profile)
    if not prompts:
        raise ValueError(f"unknown prompt_profile: {profile}")
    idx = sequence % len(prompts)
    return f"{profile}:{idx}", prompts[idx]
