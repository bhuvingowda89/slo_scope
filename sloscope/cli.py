from __future__ import annotations

import argparse
import json
import sys

from sloscope.artifacts.validation import validate_run
from sloscope.config import load_config
from sloscope.lifecycle import FakeClock
from sloscope.runner import ExperimentRunner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sloscope")
    sub = parser.add_subparsers(dest="command", required=True)
    p_validate = sub.add_parser("validate-config")
    p_validate.add_argument("config")
    p_dry = sub.add_parser("dry-run")
    p_dry.add_argument("config")
    p_dry.add_argument("--runs-root", default="runs")
    p_real = sub.add_parser("run")
    p_real.add_argument("config")
    p_real.add_argument("--runs-root", default="runs")
    p_run = sub.add_parser("validate-run")
    p_run.add_argument("run_dir")
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        cfg = load_config(args.config)
        print(json.dumps({"valid": True, "run_id": cfg.run_id, "config_hash": cfg.compute_hash()}, sort_keys=True))
        return 0
    if args.command == "dry-run":
        cfg = load_config(args.config)
        if cfg.runtime.runtime_type != "mock":
            raise SystemExit("dry-run only permits runtime_type=mock")
        run_dir = ExperimentRunner(cfg, runs_root=args.runs_root, clock=FakeClock()).run()
        print(json.dumps({"run_dir": str(run_dir)}, sort_keys=True))
        return 0
    if args.command == "run":
        cfg = load_config(args.config)
        run_dir = ExperimentRunner(cfg, runs_root=args.runs_root).run()
        print(json.dumps({"run_dir": str(run_dir)}, sort_keys=True))
        return 0
    if args.command == "validate-run":
        result = validate_run(args.run_dir)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
