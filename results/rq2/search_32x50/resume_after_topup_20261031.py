"""Resume five checkpointed 31447 jobs after the Qwen account recovers."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/ykx_32x50_20261030")
STATUS = ROOT / "resume_after_topup_20261031_status.json"
JOBS = ["banking77_bw20", "abcd_cpu_bw30", "banking77_bw30",
        "banking77_bw40", "abcd_cpu_bw40"]
LOCK = threading.Lock()
status = {"started_at_utc": None, "runs": {name: {"status": "pending", "attempts": 0}
                                          for name in JOBS}}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update(name: str | None = None, **changes: object) -> None:
    with LOCK:
        if name is None:
            status.update(changes)
        else:
            status["runs"][name].update(changes)
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, STATUS)


def run_one(name: str) -> None:
    output = ROOT / "runs" / name
    report = output / "final_report.json"
    if report.is_file():
        final = json.loads(report.read_text(encoding="utf-8"))
        update(name, status="complete", test=final["metrics"]["test"],
               train=final["metrics"]["train"], portfolio_source=final["portfolio_source"])
        return
    launch = json.loads((output / "launch.json").read_text(encoding="utf-8"))
    command = launch["command"]
    env = os.environ.copy()
    if "_cpu_" in name:
        env.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                   NUMEXPR_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
    for attempt in range(1, 4):
        update(name, status="running", attempts=attempt, started_at_utc=now())
        print(f"{now()} START {name} attempt={attempt}", flush=True)
        with (output / "resume_after_topup.log").open("a", encoding="utf-8") as log:
            log.write(f"\n=== {now()} attempt {attempt} ===\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT / "code", stdout=log,
                                    stderr=subprocess.STDOUT, env=env, check=False)
        if result.returncode == 0 and report.is_file():
            final = json.loads(report.read_text(encoding="utf-8"))
            update(name, status="complete", finished_at_utc=now(),
                   test=final["metrics"]["test"], train=final["metrics"]["train"],
                   portfolio_source=final["portfolio_source"])
            print(f"{now()} COMPLETE {name} {final['metrics']['test']}", flush=True)
            return
        update(name, status="retry_wait" if attempt < 3 else "failed",
               exit_code=result.returncode, last_failure_at_utc=now())
        print(f"{now()} FAILED {name} attempt={attempt} exit={result.returncode}", flush=True)
        if attempt < 3:
            time.sleep(60 * attempt)


def main() -> None:
    if STATUS.exists():
        raise RuntimeError(f"Resume queue already exists: {STATUS}")
    update(None, started_at_utc=now(), workers=3)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_one, name): name for name in JOBS}
        for future, name in futures.items():
            try:
                future.result()
            except Exception as exc:
                update(name, status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"{now()} ERROR {name} {type(exc).__name__}: {exc}", flush=True)
    update(None, finished_at_utc=now())
    print(f"{now()} QUEUE FINISHED", flush=True)


if __name__ == "__main__":
    main()
