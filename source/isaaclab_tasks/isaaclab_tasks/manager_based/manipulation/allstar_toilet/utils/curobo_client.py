"""FastAPI bridge for dispatching planning tools into an Isaac Lab simulation.

The HTTP server never advances the simulation directly.  A simulation script
owns :class:`SimulationToolBridge` and calls ``process_next()`` from its normal
main loop.  This keeps ``env.step()`` on the Isaac Sim thread while remote
callers can submit interchangeable tools through one API.

The default cuRobo adapter consumes the same JSON payload as
``scripts/run_curobo_pose_plan_test_worker.py`` and returns its JSON output
unchanged.  An execution adapter then extracts trajectories from that output
and feeds actions into ``env.step()``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
import uuid


JsonObject = dict[str, Any]
ToolHandler = Callable[[JsonObject], JsonObject]
OutputExecutor = Callable[[JsonObject, JsonObject], JsonObject | None]


class DeferredToolCompletion(Exception):
    """Signal that a tool job will be completed later by the simulation loop."""

    def __init__(self, tool_output: JsonObject | None = None) -> None:
        super().__init__("Tool completion deferred.")
        self.tool_output = dict(tool_output or {})


@dataclass
class ToolJob:
    """One tool invocation submitted through the API."""

    request_id: str
    tool_name: str
    parameters: JsonObject
    created_at: float = field(default_factory=time.time)
    state: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    tool_output: JsonObject | None = None
    execution_output: JsonObject | None = None
    error: str | None = None

    def as_dict(self) -> JsonObject:
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "tool_output": self.tool_output,
            "execution_output": self.execution_output,
            "error": self.error,
        }


class ToolRegistry:
    """Thread-safe registry for tools and their optional output executors."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}
        self._executors: dict[str, OutputExecutor] = {}
        self._specs: dict[str, JsonObject] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        output_executor: OutputExecutor | None = None,
        spec: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("Tool name cannot be empty.")
        with self._lock:
            if not replace and name in self._tools:
                raise ValueError(f"Tool is already registered: {name}")
            self._tools[name] = handler
            if output_executor is None:
                self._executors.pop(name, None)
            else:
                self._executors[name] = output_executor
            if spec is None:
                self._specs[name] = {"name": name}
            else:
                spec_dict = dict(spec)
                spec_dict.setdefault("name", name)
                self._specs[name] = spec_dict

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def specs(self) -> list[JsonObject]:
        with self._lock:
            return [dict(self._specs[name]) for name in sorted(self._tools)]

    def resolve(self, name: str) -> tuple[ToolHandler, OutputExecutor | None]:
        with self._lock:
            try:
                return self._tools[name], self._executors.get(name)
            except KeyError as exc:
                raise KeyError(f"Unknown tool: {name}") from exc


class JsonWorkerTool:
    """Run a JSON-in/JSON-out worker script in an isolated subprocess."""

    def __init__(
        self,
        worker_script: str | Path,
        *,
        python_executable: str | Path | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        self.worker_script = Path(worker_script).expanduser().resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.timeout_s = float(timeout_s)
        if not self.worker_script.is_file():
            raise FileNotFoundError(f"Missing worker script: {self.worker_script}")

    def __call__(self, parameters: JsonObject) -> JsonObject:
        input_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as stream:
                json.dump(parameters, stream)
                input_path = Path(stream.name)
            completed = subprocess.run(
                [self.python_executable, str(self.worker_script), str(input_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Tool worker timed out after {self.timeout_s:.1f}s.") from exc
        finally:
            if input_path is not None:
                input_path.unlink(missing_ok=True)

        if completed.returncode != 0:
            raise RuntimeError(
                f"Tool worker failed with code {completed.returncode}:\n{completed.stderr.strip()}"
            )

        output: JsonObject | None = None
        worker_info: list[JsonObject] = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "worker_info" in parsed:
                worker_info.append(parsed["worker_info"])
            if isinstance(parsed, dict) and "results" in parsed:
                output = parsed
        if output is None:
            raise RuntimeError(f"Tool worker produced no results JSON payload:\n{completed.stderr.strip()}")
        if worker_info:
            output = dict(output)
            output["worker_info"] = worker_info
        return output


def default_curobo_worker_script() -> Path:
    """Return the existing isolated cuRobo pose-planning worker path."""
    return Path(__file__).resolve().parents[1] / "scripts" / "run_curobo_pose_plan_test_worker.py"


def make_curobo_pose_plan_tool(
    *,
    worker_script: str | Path | None = None,
    python_executable: str | Path | None = None,
    timeout_s: float = 180.0,
) -> JsonWorkerTool:
    """Build a tool adapter for the existing cuRobo pose-plan worker."""
    return JsonWorkerTool(
        worker_script or default_curobo_worker_script(),
        python_executable=python_executable,
        timeout_s=timeout_s,
    )


class EnvStepTrajectoryExecutor:
    """Execute a worker trajectory by repeatedly invoking ``env.step(action)``.

    ``action_builder`` converts one trajectory waypoint, such as a six-joint
    position list, into the action accepted by the current environment.  This
    deliberately leaves robot/action layout decisions in the simulation script.
    """

    def __init__(
        self,
        env: Any,
        action_builder: Callable[[list[float]], Any],
        *,
        result_index: int | None = None,
        steps_per_waypoint: int = 1,
        final_hold_steps: int = 0,
    ) -> None:
        self.env = env
        self.action_builder = action_builder
        self.result_index = None if result_index is None else int(result_index)
        self.steps_per_waypoint = max(1, int(steps_per_waypoint))
        self.final_hold_steps = max(0, int(final_hold_steps))

    def __call__(self, tool_output: JsonObject, parameters: JsonObject) -> JsonObject:
        del parameters
        results = tool_output.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("Tool output does not contain any results.")
        if self.result_index is None:
            selected_results = results
        else:
            try:
                selected_results = [results[self.result_index]]
            except IndexError as exc:
                raise ValueError(f"Tool output has no result at index {self.result_index}.") from exc

        executed_results: list[str] = []
        skipped_results: list[str] = []
        waypoint_count = 0
        step_count = 0
        for result in selected_results:
            result_name = str(result.get("name", "unnamed"))
            if not result.get("success", False):
                skipped_results.append(result_name)
                continue
            trajectory = result.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                raise ValueError(f"Successful tool result contains no trajectory: {result_name}")
            for waypoint in trajectory:
                action = self.action_builder(waypoint)
                for _ in range(self.steps_per_waypoint):
                    self.env.step(action)
                    step_count += 1
            final_action = self.action_builder(trajectory[-1])
            for _ in range(self.final_hold_steps):
                self.env.step(final_action)
                step_count += 1
            executed_results.append(result_name)
            waypoint_count += len(trajectory)
        return {
            "executed": bool(executed_results),
            "executed_results": executed_results,
            "skipped_results": skipped_results,
            "waypoint_count": waypoint_count,
            "env_step_count": step_count,
        }


class SimulationToolBridge:
    """Queue remote tool requests for processing by the simulation main loop."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()
        self._pending: Queue[str] = Queue()
        self._jobs: dict[str, ToolJob] = {}
        self._lock = threading.RLock()
        self._local = threading.local()

    def submit(self, tool_name: str, parameters: Mapping[str, Any] | None = None) -> ToolJob:
        self.registry.resolve(tool_name)
        job = ToolJob(
            request_id=uuid.uuid4().hex,
            tool_name=tool_name,
            parameters=dict(parameters or {}),
        )
        with self._lock:
            self._jobs[job.request_id] = job
        self._pending.put(job.request_id)
        return job

    def tools(self) -> list[str]:
        return self.registry.names()

    def tool_specs(self) -> list[JsonObject]:
        return self.registry.specs()

    def get(self, request_id: str) -> ToolJob | None:
        with self._lock:
            return self._jobs.get(request_id)

    def current_job(self) -> ToolJob | None:
        """Return the job currently running on this simulation thread, if any."""
        return getattr(self._local, "current_job", None)

    def complete_deferred(
        self,
        request_id: str,
        *,
        tool_output: JsonObject | None = None,
        execution_output: JsonObject | None = None,
        error: str | None = None,
    ) -> ToolJob:
        """Mark a previously deferred job as finished."""
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None:
                raise KeyError(f"Unknown deferred job: {request_id}")
            if tool_output is not None:
                job.tool_output = tool_output
            if execution_output is not None:
                job.execution_output = execution_output
            job.error = error
            job.state = "failed" if error else "succeeded"
            job.finished_at = time.time()
            return job

    def process_next(self, *, block: bool = False, timeout_s: float | None = None) -> ToolJob | None:
        """Run one queued job; call this only from the simulation-owning thread."""
        try:
            request_id = self._pending.get(block=block, timeout=timeout_s if block else None)
        except Empty:
            return None
        job = self.get(request_id)
        if job is None:
            self._pending.task_done()
            return None

        tool, output_executor = self.registry.resolve(job.tool_name)
        with self._lock:
            job.state = "running"
            job.started_at = time.time()
        deferred = False
        self._local.current_job = job
        try:
            job.tool_output = tool(job.parameters)
            if output_executor is not None:
                job.execution_output = output_executor(job.tool_output, job.parameters)
            with self._lock:
                job.state = "succeeded"
        except DeferredToolCompletion as exc:
            deferred = True
            with self._lock:
                job.state = "deferred"
                job.tool_output = exc.tool_output
        except Exception as exc:  # noqa: BLE001 - errors are reported through job status.
            with self._lock:
                job.state = "failed"
                job.error = str(exc)
        finally:
            if getattr(self._local, "current_job", None) is job:
                del self._local.current_job
            with self._lock:
                if not deferred:
                    job.finished_at = time.time()
            self._pending.task_done()
        return job

    def process_pending(self, *, max_jobs: int | None = None) -> list[ToolJob]:
        """Process currently queued jobs without waiting for new requests."""
        processed: list[ToolJob] = []
        while max_jobs is None or len(processed) < max_jobs:
            job = self.process_next()
            if job is None:
                break
            processed.append(job)
        return processed


def create_curobo_bridge(
    env: Any,
    action_builder: Callable[[list[float]], Any],
    *,
    tool_name: str = "curobo_pose_plan",
    worker_script: str | Path | None = None,
    python_executable: str | Path | None = None,
    worker_timeout_s: float = 180.0,
    steps_per_waypoint: int = 1,
    final_hold_steps: int = 0,
) -> SimulationToolBridge:
    """Create a ready-to-use bridge backed by the existing cuRobo worker."""
    registry = ToolRegistry()
    registry.register(
        tool_name,
        make_curobo_pose_plan_tool(
            worker_script=worker_script,
            python_executable=python_executable,
            timeout_s=worker_timeout_s,
        ),
        output_executor=EnvStepTrajectoryExecutor(
            env,
            action_builder,
            steps_per_waypoint=steps_per_waypoint,
            final_hold_steps=final_hold_steps,
        ),
    )
    return SimulationToolBridge(registry)


def create_fastapi_app(bridge: SimulationToolBridge):
    """Create the API app; FastAPI is imported only when the server is used."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ModuleNotFoundError(
            "FastAPI support requires the 'fastapi' and 'pydantic' packages."
        ) from exc

    class InvokeRequest(BaseModel):
        tool_name: str = Field(min_length=1)
        parameters: dict[str, Any] = Field(default_factory=dict)

    app = FastAPI(title="Allstar Toilet Simulation Tool Bridge")

    @app.get("/health")
    def health() -> JsonObject:
        return {"status": "ok", "tools": bridge.tools()}

    @app.get("/tools")
    def tools() -> JsonObject:
        return {"tools": bridge.tool_specs()}

    def invoke(request) -> JsonObject:
        try:
            return bridge.submit(request.tool_name, request.parameters).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    invoke.__annotations__["request"] = InvokeRequest
    app.post("/tools/invoke", status_code=202)(invoke)

    @app.get("/tools/jobs/{request_id}")
    def job_status(request_id: str) -> JsonObject:
        job = bridge.get(request_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}")
        return job.as_dict()

    return app


def serve_bridge_in_thread(
    bridge: SimulationToolBridge,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
):
    """Start Uvicorn in a daemon thread and return ``(server, thread)``.

    The caller remains responsible for repeatedly calling
    ``bridge.process_next()`` or ``bridge.process_pending()`` in the simulation
    thread.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ModuleNotFoundError("Serving the bridge requires the 'uvicorn' package.") from exc

    server = uvicorn.Server(
        uvicorn.Config(
            create_fastapi_app(bridge),
            host=host,
            port=int(port),
            log_level=log_level,
        )
    )
    thread = threading.Thread(target=server.run, name="simulation-tool-api", daemon=True)
    thread.start()
    return server, thread


class ToolApiClient:
    """Small standard-library client for an externally running bridge server."""

    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _request(self, method: str, path: str, payload: JsonObject | None = None) -> JsonObject:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Tool API returned HTTP {exc.code}: {detail}") from exc

    def invoke(self, tool_name: str, parameters: Mapping[str, Any] | None = None) -> JsonObject:
        return self._request(
            "POST",
            "/tools/invoke",
            {"tool_name": tool_name, "parameters": dict(parameters or {})},
        )

    def status(self, request_id: str) -> JsonObject:
        return self._request("GET", f"/tools/jobs/{request_id}")
