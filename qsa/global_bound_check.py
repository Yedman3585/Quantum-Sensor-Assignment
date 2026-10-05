"""Global reference for the full 20,000 x 800 problem (no batches, no candidate window).

* Lagrangian lower bound (relaxing server capacities) - valid for any assignment method.
* Feasible solution: Lagrangian-adjusted regret greedy + capacity-feasible shift local search.
Objective is identical to calculate_quality(): sum (4 - priority) * cost + 15 * uncovered.
"""
import argparse, os, sys, tempfile, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capacity_stress_experiment import CapacityStressExperiment

P = 15.0

def run(scale, seed, iters=400, target_util=None, n_cameras=20000, n_servers=800):
    e = CapacityStressExperiment("PRC-QUBO", "SQA", n_cameras=n_cameras, n_servers=n_servers, random_seed=seed,
                                 capacity_scale=1.0 if target_util else scale, log_root=tempfile.mkdtemp(prefix="gbc_"))
    e.generate_realistic_data()
    if target_util:
        scale = e.total_load / (target_util / 100.0 * e.base_total_capacity)
        e.initial_capacity = e.base_initial_capacity * scale
    w = (4 - e.priority).astype(float); C = e.cost_matrix * w[:, None]
    l = e.load_gflops.astype(float); K = e.initial_capacity.astype(float)
    t = time.time()
    u = np.zeros(len(K)); best_lb = -np.inf; best_u = u.copy()
    for it in range(iters):
        R = C + np.outer(l, u)
        j = R.argmin(1); r = R[np.arange(len(l)), j]; take = r < P
        val = np.where(take, r, P).sum() - u @ K
        if val > best_lb: best_lb, best_u = val, u.copy()
        g = np.bincount(j[take], weights=l[take], minlength=len(K)) - K
        u = np.maximum(0, u + g / (np.linalg.norm(g) + 1e-9) * np.sqrt(len(K)) * 0.05 / (1 + it / 50) / np.mean(l) * 0.5)
    R = C + np.outer(l, best_u)
    part = np.partition(R, 1, axis=1); regret = part[:, 1] - part[:, 0]
    rem = K.copy(); a = -np.ones(len(l), int)
    for i in np.argsort(-regret):
        feas = rem >= l[i]
        if not feas.any(): continue
        jj = int(np.where(feas, R[i], np.inf).argmin()); a[i] = jj; rem[jj] -= l[i]
    for _ in range(10):
        moved = 0
        for i in range(len(l)):
            if a[i] < 0: continue
            cand = np.where(rem >= l[i], C[i], np.inf); jj = int(cand.argmin())
            if cand[jj] < C[i, a[i]] - 1e-12:
                rem[a[i]] += l[i]; rem[jj] -= l[i]; a[i] = jj; moved += 1
        if moved == 0: break
    cov = a >= 0
    obj = float(C[np.where(cov)[0], a[cov]].sum() + P * (~cov).sum())
    loads = np.bincount(a[cov], weights=l[cov], minlength=len(K))
    assert np.all(loads <= K + 1e-6)
    return {"capacity_scale": scale, "target_util": target_util, "n_cameras": n_cameras, "n_servers": n_servers, "seed": seed, "utilization_percent": float(l.sum() / K.sum() * 100),
            "lagrangian_lower_bound": float(best_lb), "feasible_objective": obj, "covered": int(cov.sum()),
            "gap_to_bound_percent": float((obj - best_lb) / best_lb * 100), "time_sec": time.time() - t}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="1.0,0.5,0.33,0.25"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="logs_global_bound")
    ap.add_argument("--target-utils", default=None, help="comma-separated percents; overrides --scales")
    ap.add_argument("--n-cameras", type=int, default=20000); ap.add_argument("--n-servers", type=int, default=800)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.target_utils:
        res = [run(None, a.seed, target_util=float(u), n_cameras=a.n_cameras, n_servers=a.n_servers) for u in a.target_utils.split(",")]
    else:
        res = [run(float(s), a.seed, n_cameras=a.n_cameras, n_servers=a.n_servers) for s in a.scales.split(",")]
    for r in res: print(json.dumps(r))
    json.dump(res, open(os.path.join(a.out, f"global_bound_{a.n_cameras}x{a.n_servers}_seed{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
