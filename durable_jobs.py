from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Iterable, Any


class DuplicateJobError(Exception):
    """Raised when enqueueing a job_id that already exists."""


class UnknownJobError(Exception):
    """Raised when a referenced job does not exist."""


class InvalidDependencyError(Exception):
    """Raised when dependencies are invalid."""


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""


class LeaseError(Exception):
    """Raised when a lease token is missing, stale, or expired."""


@dataclass(frozen=True)
class Claim:
    job_id: str
    token: str
    payload: str
    attempt: int
    lease_deadline: int


@dataclass
class _Job:
    job_id: str
    payload: str
    state: str
    run_at: int
    next_run_at: Optional[int]
    priority: int
    seq: int
    deps: set[str]
    locks: set[str]
    attempt: int
    max_attempts: int
    base_backoff: int
    worker_id: Optional[str] = None
    lease_deadline: Optional[int] = None
    token: Optional[str] = None
    result: Optional[str] = None
    last_error: Optional[str] = None


_TERMINAL_STATES = {"succeeded", "dead", "canceled", "blocked"}
_BLOCKING_STATES = {"dead", "canceled", "blocked"}
_VALID_STATES = {
    "waiting",
    "scheduled",
    "ready",
    "running",
    "succeeded",
    "dead",
    "canceled",
    "blocked",
}


class DurableJobQueue:
    """Deterministic durable workflow/job queue."""

    def __init__(self) -> None:
        self._now = 0
        self._seq = 0
        self._lease_seq = 1
        self._jobs: dict[str, _Job] = {}

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def lease_seq(self) -> int:
        return self._lease_seq

    def now(self) -> int:
        return self._now

    def enqueue(
        self,
        job_id: str,
        payload: str,
        *,
        run_at: int = 0,
        priority: int = 0,
        deps: Iterable[str] | None = None,
        locks: Iterable[str] | None = None,
        max_attempts: int = 3,
        base_backoff: int = 5,
    ) -> None:
        self._reconcile()
        self._validate_job_id(job_id)
        if not isinstance(payload, str):
            raise TypeError("payload must be a string")
        self._validate_int("run_at", run_at)
        self._validate_int("priority", priority)
        self._validate_int("max_attempts", max_attempts)
        self._validate_int("base_backoff", base_backoff)
        if run_at < 0:
            raise ValueError("run_at must be >= 0")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if base_backoff < 0:
            raise ValueError("base_backoff must be >= 0")
        if job_id in self._jobs:
            raise DuplicateJobError(job_id)

        dep_set = self._validate_string_iterable("deps", deps)
        lock_set = self._validate_string_iterable("locks", locks)
        if job_id in dep_set:
            raise InvalidDependencyError("job cannot depend on itself")
        missing = [dep for dep in dep_set if dep not in self._jobs]
        if missing:
            raise InvalidDependencyError("unknown dependency")

        next_run_at: Optional[int] = run_at
        if any(self._jobs[dep].state in _BLOCKING_STATES for dep in dep_set):
            state = "blocked"
        elif all(self._jobs[dep].state == "succeeded" for dep in dep_set):
            state = "ready" if run_at <= self._now else "scheduled"
        else:
            state = "waiting"

        self._jobs[job_id] = _Job(
            job_id=job_id,
            payload=payload,
            state=state,
            run_at=run_at,
            next_run_at=next_run_at,
            priority=priority,
            seq=self._seq,
            deps=dep_set,
            locks=lock_set,
            attempt=0,
            max_attempts=max_attempts,
            base_backoff=base_backoff,
        )
        self._seq += 1

    def advance(self, to_ts: int) -> None:
        self._validate_int("to_ts", to_ts)
        if to_ts < self._now:
            raise ValueError("cannot move time backwards")
        self._now = to_ts
        self._reconcile()

    def ready(self) -> list[str]:
        self._reconcile()
        held = self._held_locks()
        out: list[str] = []
        for job in self._ordered_ready_jobs():
            if job.locks.isdisjoint(held):
                out.append(job.job_id)
        return out

    def claim(
        self,
        worker_id: str,
        limit: int = 1,
        lease_seconds: int = 30,
    ) -> list[Claim]:
        self._reconcile()
        if not isinstance(worker_id, str):
            raise TypeError("worker_id must be a string")
        if worker_id == "":
            raise ValueError("worker_id must be a non-empty string")
        self._validate_int("limit", limit)
        self._validate_int("lease_seconds", lease_seconds)
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be > 0")
        if limit == 0:
            return []

        held = self._held_locks()
        claims: list[Claim] = []
        for job in self._ordered_ready_jobs():
            if len(claims) >= limit:
                break
            if not job.locks.isdisjoint(held):
                continue
            job.state = "running"
            job.attempt += 1
            job.worker_id = worker_id
            job.lease_deadline = self._now + lease_seconds
            job.token = f"{job.job_id}#{job.attempt}#{self._lease_seq}"
            self._lease_seq += 1
            held.update(job.locks)
            claims.append(Claim(job.job_id, job.token, job.payload, job.attempt, job.lease_deadline))
        return claims

    def complete(self, token: str, result: str = "") -> None:
        self._reconcile()
        self._validate_token(token)
        if not isinstance(result, str):
            raise TypeError("result must be a string")
        job = self._job_for_active_token(token)
        job.state = "succeeded"
        job.result = result
        job.token = None
        job.worker_id = None
        job.lease_deadline = None
        self._reevaluate_dependents(job.job_id)

    def fail(
        self,
        token: str,
        error: str,
        retry_after: int | None = None,
    ) -> None:
        self._reconcile()
        self._validate_token(token)
        if not isinstance(error, str):
            raise TypeError("error must be a string")
        if retry_after is not None:
            self._validate_int("retry_after", retry_after)
            if retry_after < 0:
                raise ValueError("retry_after must be >= 0")
        job = self._job_for_active_token(token)
        delay = retry_after if retry_after is not None else job.base_backoff * 2 ** (job.attempt - 1)
        self._fail_running_job(job, error, delay, self._now)

    def cancel(self, job_id: str) -> None:
        self._reconcile()
        self._validate_job_id(job_id)
        if job_id not in self._jobs:
            raise UnknownJobError(job_id)
        job = self._jobs[job_id]
        if job.state == "succeeded":
            raise InvalidTransitionError("cannot cancel succeeded job")
        if job.state in {"dead", "canceled", "blocked"}:
            return
        job.state = "canceled"
        job.token = None
        job.worker_id = None
        job.lease_deadline = None
        self._block_dependents(job_id)

    def get(self, job_id: str) -> dict:
        self._reconcile()
        self._validate_job_id(job_id)
        if job_id not in self._jobs:
            raise UnknownJobError(job_id)
        return self._job_dict(self._jobs[job_id], include_base_backoff=False)

    def dump(self) -> str:
        self._reconcile()
        jobs = []
        for job_id in sorted(self._jobs):
            jobs.append([job_id, self._job_dump_dict(self._jobs[job_id])])
        data = {
            "now": self._now,
            "seq": self._seq,
            "lease_seq": self._lease_seq,
            "jobs": jobs,
        }
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def load(cls, data: str) -> "DurableJobQueue":
        raw = json.loads(data)
        if not isinstance(raw, dict):
            raise ValueError("data must be an object")
        if list(raw.keys()) != ["now", "seq", "lease_seq", "jobs"]:
            raise ValueError("invalid top-level keys")
        q = cls()
        q._validate_int("now", raw["now"])
        q._validate_int("seq", raw["seq"])
        q._validate_int("lease_seq", raw["lease_seq"])
        if raw["now"] < 0 or raw["seq"] < 0 or raw["lease_seq"] < 1:
            raise ValueError("invalid counters")
        if not isinstance(raw["jobs"], list):
            raise ValueError("jobs must be a list")
        q._now = raw["now"]
        q._seq = raw["seq"]
        q._lease_seq = raw["lease_seq"]

        seen: set[str] = set()
        seqs: set[int] = set()
        for item in raw["jobs"]:
            if not (isinstance(item, list) and len(item) == 2):
                raise ValueError("job entry must be [job_id, object]")
            job_id, obj = item
            q._validate_job_id(job_id)
            if job_id in seen:
                raise ValueError("duplicate job_id")
            if not isinstance(obj, dict):
                raise ValueError("job data must be an object")
            expected_keys = [
                "payload",
                "state",
                "run_at",
                "next_run_at",
                "priority",
                "seq",
                "deps",
                "locks",
                "attempt",
                "max_attempts",
                "base_backoff",
                "worker_id",
                "lease_deadline",
                "token",
                "result",
                "last_error",
            ]
            if list(obj.keys()) != expected_keys:
                raise ValueError("invalid job keys")
            if not isinstance(obj["payload"], str):
                raise TypeError("payload must be a string")
            if obj["state"] not in _VALID_STATES:
                raise ValueError("invalid state")
            for name in ("run_at", "priority", "seq", "attempt", "max_attempts", "base_backoff"):
                q._validate_int(name, obj[name])
            if obj["run_at"] < 0 or obj["seq"] < 0 or obj["attempt"] < 0:
                raise ValueError("invalid numeric field")
            if obj["max_attempts"] < 1 or obj["base_backoff"] < 0:
                raise ValueError("invalid retry settings")
            if obj["attempt"] > obj["max_attempts"]:
                raise ValueError("attempt cannot exceed max_attempts")
            next_run_at = obj["next_run_at"]
            if next_run_at is not None:
                q._validate_int("next_run_at", next_run_at)
                if next_run_at < 0:
                    raise ValueError("next_run_at must be >= 0")
            lease_deadline = obj["lease_deadline"]
            if lease_deadline is not None:
                q._validate_int("lease_deadline", lease_deadline)
            for name in ("worker_id", "token", "result", "last_error"):
                if obj[name] is not None and not isinstance(obj[name], str):
                    raise TypeError(f"{name} must be a string or null")
            deps = q._validate_loaded_string_list("deps", obj["deps"])
            locks = q._validate_loaded_string_list("locks", obj["locks"])
            if job_id in deps:
                raise ValueError("job cannot depend on itself")
            if obj["seq"] in seqs:
                raise ValueError("duplicate seq")
            seen.add(job_id)
            seqs.add(obj["seq"])
            q._jobs[job_id] = _Job(
                job_id=job_id,
                payload=obj["payload"],
                state=obj["state"],
                run_at=obj["run_at"],
                next_run_at=next_run_at,
                priority=obj["priority"],
                seq=obj["seq"],
                deps=deps,
                locks=locks,
                attempt=obj["attempt"],
                max_attempts=obj["max_attempts"],
                base_backoff=obj["base_backoff"],
                worker_id=obj["worker_id"],
                lease_deadline=lease_deadline,
                token=obj["token"],
                result=obj["result"],
                last_error=obj["last_error"],
            )

        if q._seq != len(q._jobs) or seqs != set(range(len(q._jobs))):
            raise ValueError("seq counter does not match jobs")
        running_locks: set[str] = set()
        for job in q._jobs.values():
            if any(dep not in q._jobs for dep in job.deps):
                raise ValueError("unknown dependency")
            if job.state == "running":
                if not job.locks.isdisjoint(running_locks):
                    raise ValueError("running jobs have conflicting locks")
                running_locks.update(job.locks)
                if job.token is None or job.worker_id is None or job.lease_deadline is None:
                    raise ValueError("running job missing lease")
                if job.lease_deadline <= q._now:
                    raise ValueError("running job lease is expired")
                parts = job.token.rsplit("#", 2)
                if len(parts) != 3 or parts[0] != job.job_id:
                    raise ValueError("invalid token")
                try:
                    token_attempt = int(parts[1])
                    token_lease_seq = int(parts[2])
                except ValueError as exc:
                    raise ValueError("invalid token") from exc
                if token_attempt != job.attempt or token_lease_seq < 1 or token_lease_seq >= q._lease_seq:
                    raise ValueError("invalid token")
                if job.next_run_at is None:
                    raise ValueError("running job missing next_run_at")
            else:
                if job.token is not None or job.worker_id is not None or job.lease_deadline is not None:
                    raise ValueError("non-running job has lease")
            if job.state in {"dead", "succeeded", "canceled", "blocked"} and job.state == "dead":
                if job.next_run_at is not None:
                    raise ValueError("dead job cannot have next_run_at")
        q._validate_state_consistency()
        return q

    @staticmethod
    def _validate_int(name: str, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int")

    @staticmethod
    def _validate_job_id(job_id: Any) -> None:
        if not isinstance(job_id, str):
            raise TypeError("job_id must be a string")
        if job_id == "":
            raise ValueError("job_id must be a non-empty string")

    @staticmethod
    def _validate_token(token: Any) -> None:
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if token == "":
            raise ValueError("token must be a non-empty string")

    @classmethod
    def _validate_string_iterable(cls, name: str, values: Iterable[str] | None) -> set[str]:
        if values is None:
            return set()
        if isinstance(values, str):
            raise TypeError(f"{name} must not be a string")
        try:
            items = list(values)
        except TypeError:
            raise TypeError(f"{name} must be iterable") from None
        out: set[str] = set()
        for item in items:
            if not isinstance(item, str):
                raise TypeError(f"{name} entries must be strings")
            if item == "":
                raise ValueError(f"{name} entries must be non-empty strings")
            if item in out:
                raise ValueError(f"duplicate {name} entry")
            out.add(item)
        return out

    @classmethod
    def _validate_loaded_string_list(cls, name: str, values: Any) -> set[str]:
        if not isinstance(values, list):
            raise TypeError(f"{name} must be a list")
        out = cls._validate_string_iterable(name, values)
        if values != sorted(values):
            raise ValueError(f"{name} must be sorted")
        return out

    def _ordered_ready_jobs(self) -> list[_Job]:
        return sorted(
            (job for job in self._jobs.values() if job.state == "ready"),
            key=lambda j: (j.next_run_at if j.next_run_at is not None else j.run_at, -j.priority, j.seq, j.job_id),
        )

    def _held_locks(self) -> set[str]:
        held: set[str] = set()
        for job in self._jobs.values():
            if job.state == "running":
                held.update(job.locks)
        return held

    def _reconcile(self) -> None:
        expired = sorted(
            (
                job
                for job in self._jobs.values()
                if job.state == "running"
                and job.lease_deadline is not None
                and job.lease_deadline <= self._now
            ),
            key=lambda j: (j.lease_deadline, j.seq, j.job_id),
        )
        for job in expired:
            if job.state != "running" or job.lease_deadline is None or job.lease_deadline > self._now:
                continue
            deadline = job.lease_deadline
            error = f"lease expired at {deadline}"
            delay = job.base_backoff * 2 ** (job.attempt - 1)
            self._fail_running_job(job, error, delay, deadline)

        for job in self._jobs.values():
            if job.state == "scheduled" and job.next_run_at is not None and job.next_run_at <= self._now:
                job.state = "ready"

    def _fail_running_job(self, job: _Job, error: str, delay: int, from_ts: int) -> None:
        job.last_error = error
        job.token = None
        job.worker_id = None
        job.lease_deadline = None
        if job.attempt >= job.max_attempts:
            job.state = "dead"
            job.next_run_at = None
            self._block_dependents(job.job_id)
        else:
            job.next_run_at = from_ts + delay
            job.state = "ready" if job.next_run_at <= self._now else "scheduled"

    def _job_for_active_token(self, token: str) -> _Job:
        for job in self._jobs.values():
            if job.state == "running" and job.token == token:
                return job
        raise LeaseError(token)

    def _reevaluate_dependents(self, job_id: str) -> None:
        for dep_job in sorted(self._dependents_of(job_id), key=lambda j: (j.seq, j.job_id)):
            self._reevaluate_job(dep_job)

    def _reevaluate_job(self, job: _Job) -> None:
        if job.state in _TERMINAL_STATES or job.state == "running":
            return
        if any(self._jobs[dep].state in _BLOCKING_STATES for dep in job.deps):
            self._block_job(job)
        elif all(self._jobs[dep].state == "succeeded" for dep in job.deps):
            if job.next_run_at is None:
                job.next_run_at = job.run_at
            job.state = "ready" if job.next_run_at <= self._now else "scheduled"
            self._reevaluate_dependents(job.job_id)
        else:
            job.state = "waiting"

    def _block_dependents(self, job_id: str) -> None:
        for dep_job in sorted(self._dependents_of(job_id), key=lambda j: (j.seq, j.job_id)):
            if dep_job.state not in _TERMINAL_STATES:
                self._block_job(dep_job)

    def _block_job(self, job: _Job) -> None:
        if job.state in _TERMINAL_STATES:
            return
        job.state = "blocked"
        job.token = None
        job.worker_id = None
        job.lease_deadline = None
        self._block_dependents(job.job_id)

    def _dependents_of(self, job_id: str) -> list[_Job]:
        return [job for job in self._jobs.values() if job_id in job.deps]

    def _job_dict(self, job: _Job, *, include_base_backoff: bool) -> dict:
        data = {
            "job_id": job.job_id,
            "state": job.state,
            "payload": job.payload,
            "result": job.result,
            "last_error": job.last_error,
            "attempt": job.attempt,
            "max_attempts": job.max_attempts,
            "run_at": job.run_at,
            "next_run_at": job.next_run_at,
            "priority": job.priority,
            "seq": job.seq,
            "deps": sorted(job.deps),
            "locks": sorted(job.locks),
            "worker_id": job.worker_id,
            "lease_deadline": job.lease_deadline,
            "token": job.token,
        }
        if include_base_backoff:
            data["base_backoff"] = job.base_backoff
        return data

    def _job_dump_dict(self, job: _Job) -> dict:
        return {
            "payload": job.payload,
            "state": job.state,
            "run_at": job.run_at,
            "next_run_at": job.next_run_at,
            "priority": job.priority,
            "seq": job.seq,
            "deps": sorted(job.deps),
            "locks": sorted(job.locks),
            "attempt": job.attempt,
            "max_attempts": job.max_attempts,
            "base_backoff": job.base_backoff,
            "worker_id": job.worker_id,
            "lease_deadline": job.lease_deadline,
            "token": job.token,
            "result": job.result,
            "last_error": job.last_error,
        }

    def _validate_state_consistency(self) -> None:
        for job in self._jobs.values():
            if job.state == "ready":
                if job.next_run_at is None or job.next_run_at > self._now:
                    raise ValueError("invalid ready job")
            elif job.state == "scheduled":
                if job.next_run_at is None or job.next_run_at <= self._now:
                    raise ValueError("invalid scheduled job")
            elif job.state == "waiting":
                if any(self._jobs[dep].state in _BLOCKING_STATES for dep in job.deps):
                    raise ValueError("waiting job has blocking dependency")
                if all(self._jobs[dep].state == "succeeded" for dep in job.deps):
                    raise ValueError("waiting job dependencies are satisfied")
            elif job.state == "blocked":
                if job.deps and not any(self._jobs[dep].state in _BLOCKING_STATES for dep in job.deps):
                    raise ValueError("blocked job has no blocking dependency")
            if job.state != "running" and job.worker_id is not None:
                raise ValueError("non-running job has worker")
