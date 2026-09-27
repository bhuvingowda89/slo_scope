from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Union

from sloscope.artifacts.validation import validate_run
from sloscope.artifacts.writer import ArtifactWriter
from sloscope.config import ExperimentConfig
from sloscope.ground_truth import GroundTruthLedger
from sloscope.injectors.factory import create_injector
from sloscope.lifecycle import Clock, ExperimentState, Lifecycle, SystemClock
from sloscope.runtime.llamacpp import LlamaCppRuntimeAdapter
from sloscope.runtime.mock import MockRuntimeAdapter
from sloscope.telemetry.schemas import RequestRecord
from sloscope.telemetry.system import SystemTelemetryCollector
from sloscope.workload import generate_workload_plan


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, runs_root: Union[str, Path] = "runs", clock: Optional[Clock] = None) -> None:
        config.validate()
        self.config = config.with_hash()
        self.runs_root = Path(runs_root)
        self.clock = clock or SystemClock()
        self.lifecycle = Lifecycle(self.clock)
        if self.config.runtime.runtime_type == "llamacpp":
            self.runtime = LlamaCppRuntimeAdapter(self.config.runtime, self.clock, self.config.seed)
        elif self.config.runtime.runtime_type == "mock":
            self.runtime = MockRuntimeAdapter(self.config.runtime, self.clock, self.config.seed)
        else:
            raise ValueError(f"unknown runtime_type: {self.config.runtime.runtime_type}")
        self.injectors = [create_injector(mech, self.clock) for mech in self.config.mechanisms]
        self.ground_truth = GroundTruthLedger(self.config.mechanisms)
        self.accounting = {"planned": 0, "emitted": 0, "runtime_accepted": 0, "successful": 0, "failed": 0}

    def _advance_to(self, timestamp: float) -> None:
        if timestamp < self.clock.monotonic():
            return
        advance = getattr(self.clock, "advance", None)
        if callable(advance):
            advance(timestamp - self.clock.monotonic())

    def run(self) -> Path:
        if self.config.runtime.runtime_type == "llamacpp":
            return asyncio.run(self._run_llamacpp())
        return self._run_mock()

    def _write_artifacts(
        self,
        run_dir: Path,
        workload,
        requests,
        system_metrics,
        runtime_metrics,
        traces,
        final_state,
        start_wall,
        runtime_metadata=None,
    ) -> None:
        preliminary = {
            "valid": True,
            "issues": [],
            "run_id": self.config.run_id,
            "planned_requests": len(workload),
            "accounted_requests": len(requests),
            "accounting": self.accounting,
        }
        ArtifactWriter(run_dir).write_run(
            self.config,
            [p.to_dict() for p in workload],
            self.ground_truth.to_dict(),
            requests,
            system_metrics,
            runtime_metrics,
            traces,
            self.lifecycle.events,
            preliminary,
            final_state.value,
            start_wall,
            self.clock.wall_time(),
            runtime_metadata=runtime_metadata,
        )
        validation = validate_run(run_dir)
        from sloscope.artifacts.writer import write_json

        write_json(run_dir / "validation.json", validation)

    def _run_mock(self) -> Path:
        run_dir = self.runs_root / self.config.run_id
        start_wall = self.clock.wall_time()
        workload = generate_workload_plan(self.config)
        self.accounting["planned"] = len(workload)
        requests = []
        runtime_metrics = []
        final_state = ExperimentState.FAILED
        error = None
        injecting_start = None
        started = set()
        stopped = set()

        def start_due_mechanisms(now: float) -> None:
            if injecting_start is None:
                return
            for inj in sorted(self.injectors, key=lambda item: (item.config.scheduled_onset, item.mechanism_id)):
                if inj.mechanism_id in started:
                    continue
                if now + 1e-12 < injecting_start + inj.config.scheduled_onset:
                    continue
                self._advance_to(injecting_start + inj.config.scheduled_onset)
                inj.start()
                started.add(inj.mechanism_id)
                evidence = inj.verify()
                self.ground_truth.mark_verified(inj.mechanism_id, evidence)
                self.lifecycle.event("mechanism_verification", inj.mechanism_id, evidence.to_dict())
                if not evidence.verified:
                    raise RuntimeError(f"mechanism verification failed: {inj.mechanism_id}")

        def stop_due_mechanisms(now: float) -> None:
            if injecting_start is None:
                return
            for inj in sorted(self.injectors, key=lambda item: (item.config.scheduled_stop, item.mechanism_id)):
                if inj.mechanism_id not in started or inj.mechanism_id in stopped:
                    continue
                if now + 1e-12 < injecting_start + inj.config.scheduled_stop:
                    continue
                self._advance_to(max(self.clock.monotonic(), injecting_start + inj.config.scheduled_stop))
                inj.stop()
                stopped.add(inj.mechanism_id)
                self.ground_truth.mark_stopped(inj.mechanism_id, self.clock.monotonic())

        try:
            self.lifecycle.transition(ExperimentState.PREPARING, "preparing runtime and injectors")
            self.runtime.prepare()
            for inj in self.injectors:
                inj.prepare()
            if not self.runtime.healthcheck():
                raise RuntimeError("runtime healthcheck failed")
            self.lifecycle.transition(ExperimentState.WARMUP, "warmup")
            self.runtime.warmup()
            self.lifecycle.transition(ExperimentState.BASELINE, "baseline")
            if self.injectors:
                self.lifecycle.transition(ExperimentState.INJECTING, "starting degradation mechanisms")
                injecting_start = self.clock.monotonic()
            else:
                self.lifecycle.transition(ExperimentState.RECOVERY, "no mechanisms")
            for plan in workload:
                if injecting_start is not None:
                    self._advance_to(injecting_start + plan.scheduled_arrival)
                else:
                    self._advance_to(plan.scheduled_arrival)
                start_due_mechanisms(self.clock.monotonic())
                self.accounting["emitted"] += 1
                actual = self.clock.monotonic()
                rec = self.runtime.execute(plan, actual)
                self.accounting["runtime_accepted"] = self.runtime.accepted
                if rec.status == "success":
                    self.accounting["successful"] += 1
                else:
                    self.accounting["failed"] += 1
                requests.append(rec.to_dict())
                if rec.completion_time is not None:
                    self._advance_to(rec.completion_time)
                stop_due_mechanisms(self.clock.monotonic())
            if self.lifecycle.state == ExperimentState.INJECTING:
                inject_base = injecting_start if injecting_start is not None else self.clock.monotonic()
                for inj in sorted(self.injectors, key=lambda item: (item.config.scheduled_onset, item.mechanism_id)):
                    if inj.mechanism_id not in started:
                        self._advance_to(inject_base + inj.config.scheduled_onset)
                        inj.start()
                        started.add(inj.mechanism_id)
                        evidence = inj.verify()
                        self.ground_truth.mark_verified(inj.mechanism_id, evidence)
                        self.lifecycle.event("mechanism_verification", inj.mechanism_id, evidence.to_dict())
                        if not evidence.verified:
                            raise RuntimeError(f"mechanism verification failed: {inj.mechanism_id}")
                for inj in sorted(self.injectors, key=lambda item: (item.config.scheduled_stop, item.mechanism_id)):
                    if inj.mechanism_id not in stopped:
                        self._advance_to(inject_base + inj.config.scheduled_stop)
                        inj.stop()
                        stopped.add(inj.mechanism_id)
                        self.ground_truth.mark_stopped(inj.mechanism_id, self.clock.monotonic())
                self.lifecycle.transition(ExperimentState.RECOVERY, "recovery")
            runtime_metrics.append(self.runtime.collect_runtime_metrics(self.clock.monotonic()))
            final_state = ExperimentState.COMPLETE
        except Exception as exc:
            error = exc
            if self.lifecycle.state != ExperimentState.FAILED:
                try:
                    self.lifecycle.transition(ExperimentState.FAILED, str(exc))
                except Exception:
                    self.lifecycle.event("failure", str(exc))
            final_state = ExperimentState.FAILED
        finally:
            for inj in self.injectors:
                try:
                    if inj.started and not inj.stopped:
                        try:
                            inj.stop()
                            self.ground_truth.mark_stopped(inj.mechanism_id, self.clock.monotonic())
                        except Exception as cleanup_exc:
                            error = error or cleanup_exc
                            final_state = ExperimentState.FAILED
                            self.lifecycle.event("cleanup_error", str(cleanup_exc), {"mechanism_id": inj.mechanism_id, "operation": "stop"})
                finally:
                    try:
                        inj.cleanup()
                        self.ground_truth.mark_cleaned(inj.mechanism_id)
                    except Exception as cleanup_exc:
                        error = error or cleanup_exc
                        final_state = ExperimentState.FAILED
                        self.lifecycle.event("cleanup_error", str(cleanup_exc), {"mechanism_id": inj.mechanism_id, "operation": "cleanup"})
            try:
                self.runtime.shutdown()
            except Exception as shutdown_exc:
                error = error or shutdown_exc
                final_state = ExperimentState.FAILED
                self.lifecycle.event("shutdown_error", str(shutdown_exc), {"operation": "runtime.shutdown"})
            if final_state == ExperimentState.COMPLETE and self.lifecycle.state == ExperimentState.RECOVERY:
                self.lifecycle.transition(ExperimentState.COMPLETE, "complete")
            if final_state == ExperimentState.FAILED and self.lifecycle.state != ExperimentState.FAILED:
                try:
                    self.lifecycle.transition(ExperimentState.FAILED, "cleanup/shutdown failure")
                except Exception:
                    self.lifecycle.event("failure", "cleanup/shutdown failure")
            self._write_artifacts(run_dir, workload, requests, [], runtime_metrics, [], final_state, start_wall)
        if error:
            raise error
        return run_dir

    async def _run_llamacpp(self) -> Path:
        run_dir = self.runs_root / self.config.run_id
        start_wall = self.clock.wall_time()
        workload = generate_workload_plan(self.config)
        self.accounting["planned"] = len(workload)
        requests: list[dict] = []
        system_metrics: list[dict] = []
        runtime_metrics: list[dict] = []
        traces: list[dict] = []
        final_state = ExperimentState.FAILED
        error = None
        telemetry_stop = asyncio.Event()
        telemetry_task = None
        mechanism_task = None
        mechanism_failed = None
        injecting_start = None
        try:
            self.lifecycle.transition(ExperimentState.PREPARING, "preparing llama.cpp runtime")
            self.runtime.prepare()
            for inj in self.injectors:
                inj.prepare()
            if not self.runtime.healthcheck():
                raise RuntimeError("llama.cpp runtime healthcheck failed")
            self.lifecycle.transition(ExperimentState.WARMUP, "warmup")
            self.runtime.warmup()
            self.lifecycle.transition(ExperimentState.BASELINE, "baseline")
            collector = None
            if self.config.telemetry.system_metrics:
                collector = SystemTelemetryCollector(self.clock, self.config.runtime.parameters.get("server_pid"))
            interval = max(0.01, float(self.config.runtime.parameters.get("telemetry_interval_seconds", 0.25)))

            async def telemetry_loop() -> None:
                while not telemetry_stop.is_set():
                    if collector is not None:
                        system_metrics.append(collector.sample())
                    if self.config.telemetry.runtime_metrics:
                        rows = await asyncio.to_thread(getattr(self.runtime, "collect_runtime_metric_rows"), self.clock.monotonic())
                        runtime_metrics.extend(rows)
                    try:
                        await asyncio.wait_for(telemetry_stop.wait(), timeout=interval)
                    except asyncio.TimeoutError:
                        pass

            telemetry_task = asyncio.create_task(telemetry_loop())
            if self.injectors:
                self.lifecycle.transition(ExperimentState.INJECTING, "starting degradation mechanisms")
                injecting_start = self.clock.monotonic()
                base = injecting_start
            else:
                base = self.clock.monotonic()
            self.lifecycle.event("workload_start", "workload time origin", {"workload_time_origin": base})
            outstanding = 0
            lock = asyncio.Lock()
            verified = set()
            stopped = set()
            mechanism_failure_event = asyncio.Event()

            async def mechanism_scheduler() -> None:
                nonlocal mechanism_failed
                try:
                    for inj in sorted(self.injectors, key=lambda item: (item.config.scheduled_onset, item.mechanism_id)):
                        delay = base + inj.config.scheduled_onset - self.clock.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        await asyncio.to_thread(inj.start)
                        evidence = await asyncio.to_thread(inj.verify)
                        self.ground_truth.mark_verified(inj.mechanism_id, evidence)
                        self.lifecycle.event("mechanism_verification", inj.mechanism_id, evidence.to_dict())
                        if not evidence.verified:
                            mechanism_failed = RuntimeError(f"mechanism verification failed: {inj.mechanism_id}")
                            mechanism_failure_event.set()
                            return
                        verified.add(inj.mechanism_id)
                    for inj in sorted(self.injectors, key=lambda item: (item.config.scheduled_stop, item.mechanism_id)):
                        delay = base + inj.config.scheduled_stop - self.clock.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        active_failure = getattr(inj, "active_failure", lambda: None)()
                        if active_failure:
                            mechanism_failed = RuntimeError(active_failure)
                            self.lifecycle.event("mechanism_failure", active_failure, {"mechanism_id": inj.mechanism_id})
                            mechanism_failure_event.set()
                        await asyncio.to_thread(inj.stop)
                        stopped.add(inj.mechanism_id)
                        self.ground_truth.mark_stopped(inj.mechanism_id, self.clock.monotonic())
                except Exception as exc:
                    mechanism_failed = exc
                    self.lifecycle.event("mechanism_failure", str(exc))
                    mechanism_failure_event.set()

            if self.injectors:
                mechanism_task = asyncio.create_task(mechanism_scheduler())
                first_offset = min((p.scheduled_arrival for p in workload), default=0.0)
                early = [inj for inj in self.injectors if inj.config.scheduled_onset <= first_offset]
                while any(inj.mechanism_id not in verified for inj in early):
                    if mechanism_failure_event.is_set():
                        raise mechanism_failed or RuntimeError("mechanism verification failed")
                    await asyncio.sleep(0.01)

            async def run_one(plan):
                nonlocal outstanding
                if mechanism_failure_event.is_set():
                    return
                scheduled_abs = base + plan.scheduled_arrival
                delay = scheduled_abs - self.clock.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                actual = self.clock.monotonic()
                if mechanism_failure_event.is_set():
                    return
                async with lock:
                    if outstanding >= self.runtime.max_outstanding:
                        self.accounting["emitted"] += 1
                        self.accounting["failed"] += 1
                        requests.append(
                            RequestRecord(
                                plan.request_id,
                                plan.sequence,
                                scheduled_abs,
                                actual,
                                actual,
                                None,
                                actual,
                                None,
                                None,
                                "admission_failed",
                                self.config.runtime.runtime_id,
                                self.config.runtime.model_id,
                                error_type="client_max_outstanding",
                                error_message="client-side maximum outstanding requests exceeded",
                                scheduler_slip=actual - scheduled_abs,
                            ).to_dict()
                        )
                        return
                    outstanding += 1
                    self.accounting["emitted"] += 1
                try:
                    absolute_plan = plan.__class__(
                        request_id=plan.request_id,
                        sequence=plan.sequence,
                        scheduled_arrival=scheduled_abs,
                        prompt_profile=plan.prompt_profile,
                        prompt_id=plan.prompt_id,
                        target_output_tokens=plan.target_output_tokens,
                    )
                    rec = await self.runtime.execute_async(absolute_plan, actual)
                    self.accounting["runtime_accepted"] = self.runtime.accepted
                    if rec.status == "success":
                        self.accounting["successful"] += 1
                    else:
                        self.accounting["failed"] += 1
                    requests.append(rec.to_dict())
                finally:
                    async with lock:
                        outstanding -= 1

            await asyncio.wait_for(asyncio.gather(*(run_one(plan) for plan in workload)), timeout=self.config.safety.experiment_timeout)
            if mechanism_task is not None:
                await asyncio.wait_for(mechanism_task, timeout=self.config.safety.experiment_timeout)
            if mechanism_failed is not None:
                raise mechanism_failed
            telemetry_stop.set()
            await telemetry_task
            if collector is not None:
                system_metrics.append(collector.sample())
            if self.config.telemetry.runtime_metrics:
                runtime_metrics.extend(await asyncio.to_thread(getattr(self.runtime, "collect_runtime_metric_rows"), self.clock.monotonic()))
            self.lifecycle.transition(ExperimentState.RECOVERY, "workload complete")
            final_state = ExperimentState.COMPLETE
        except asyncio.TimeoutError as exc:
            error = TimeoutError("experiment timeout exceeded")
            self.lifecycle.event("experiment_timeout", str(error), {"experiment_timeout": self.config.safety.experiment_timeout})
            if self.lifecycle.state != ExperimentState.FAILED:
                self.lifecycle.transition(ExperimentState.FAILED, str(error))
            final_state = ExperimentState.FAILED
        except Exception as exc:
            error = exc
            if self.lifecycle.state != ExperimentState.FAILED:
                try:
                    self.lifecycle.transition(ExperimentState.FAILED, str(exc))
                except Exception:
                    self.lifecycle.event("failure", str(exc))
            final_state = ExperimentState.FAILED
            telemetry_stop.set()
        finally:
            for inj in self.injectors:
                try:
                    if getattr(inj, "activation_time", None) is not None and getattr(inj, "mechanism_id", None) not in locals().get("stopped", set()):
                        try:
                            await asyncio.to_thread(inj.stop)
                            self.ground_truth.mark_stopped(inj.mechanism_id, self.clock.monotonic())
                        except Exception as cleanup_exc:
                            error = error or cleanup_exc
                            final_state = ExperimentState.FAILED
                            self.lifecycle.event("cleanup_error", str(cleanup_exc), {"mechanism_id": inj.mechanism_id, "operation": "stop"})
                finally:
                    try:
                        await asyncio.to_thread(inj.cleanup)
                        if hasattr(inj, "cleanup_evidence") and inj.mechanism_id in self.ground_truth.records:
                            rec = self.ground_truth.records[inj.mechanism_id]
                            if rec.verification_evidence is not None:
                                rec.verification_evidence.setdefault("evidence_value", {}).update({"cleanup": inj.cleanup_evidence()})
                        self.ground_truth.mark_cleaned(inj.mechanism_id)
                    except Exception as cleanup_exc:
                        error = error or cleanup_exc
                        final_state = ExperimentState.FAILED
                        self.lifecycle.event("cleanup_error", str(cleanup_exc), {"mechanism_id": inj.mechanism_id, "operation": "cleanup"})
            try:
                self.runtime.shutdown()
            except Exception as shutdown_exc:
                error = error or shutdown_exc
                final_state = ExperimentState.FAILED
                self.lifecycle.event("shutdown_error", str(shutdown_exc), {"operation": "runtime.shutdown"})
            if final_state == ExperimentState.COMPLETE and self.lifecycle.state == ExperimentState.RECOVERY:
                self.lifecycle.transition(ExperimentState.COMPLETE, "complete")
            if final_state == ExperimentState.FAILED and self.lifecycle.state != ExperimentState.FAILED:
                self.lifecycle.transition(ExperimentState.FAILED, "failed")
            if telemetry_task is not None and not telemetry_task.done():
                telemetry_stop.set()
                try:
                    await telemetry_task
                except Exception as telemetry_exc:
                    self.lifecycle.event("telemetry_error", str(telemetry_exc), {"operation": "telemetry_loop"})
            runtime_metadata = dict(getattr(self.runtime, "runtime_metadata", {}))
            if "base" in locals():
                runtime_metadata["workload_time_origin"] = base
            self._write_artifacts(run_dir, workload, requests, system_metrics, runtime_metrics, traces, final_state, start_wall, runtime_metadata)
        if error:
            raise error
        return run_dir
