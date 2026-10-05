"""Run a list of experiment commands in parallel (Windows/Linux/macOS).

Usage (from the repository root):
    python run_jobs.py jobs_local.txt --workers 4

Each line of the job file is a command; "{py}" is replaced by the current Python interpreter.
Finished jobs are recorded in <jobfile>.done, so the script can be stopped (Ctrl+C) and restarted
without repeating finished work. Per-job output goes to results/v3/campaign2/_joblogs/.
"""
import argparse
import hashlib
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobfile")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [l.strip() for l in open(a.jobfile, encoding="utf-8") if l.strip()]
    done_path = a.jobfile + ".done"
    done = set(open(done_path, encoding="utf-8").read().splitlines()) if os.path.exists(done_path) else set()
    todo = [j for j in jobs if j not in done]
    logdir = os.path.join("results", "v3", "campaign2", "_joblogs")
    os.makedirs(logdir, exist_ok=True)
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run with {a.workers} workers", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()

    def run(cmd):
        argv = shlex.split(cmd.replace("{py}", "PYEXE"), posix=True)
        argv = [sys.executable if x == "PYEXE" else x for x in argv]
        h = hashlib.md5(cmd.encode()).hexdigest()[:10]
        with open(os.path.join(logdir, f"{h}.log"), "w", encoding="utf-8") as log:
            log.write(cmd + "\n")
            log.flush()
            rc = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        return cmd, rc

    finished = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, j) for j in todo]
        for f in as_completed(futs):
            cmd, rc = f.result()
            finished += 1
            if rc == 0:
                with open(done_path, "a", encoding="utf-8") as fh:
                    fh.write(cmd + "\n")
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            print(f"[{finished}/{len(todo)}] {time.time() - t0:7.0f}s {status}: {cmd[:110]}", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
