import sys, json, time
sys.argv = [sys.argv[0]] + sys.argv[1:]
import numpy as np
from batch_quality_benchmark import Instance
from v3_trajectory import lagrange_prices
mode = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]; out = sys.argv[3]
res = []
for s in seeds:
    for u in (90.0, 95.0, 98.0):
        t = time.time(); inst = Instance(20000, 800, s, 0.25, u)
        if mode == "prio":
            inst.w = inst.priority.astype(float); inst.wcost = inst.cost * inst.w[:, None]
        _, lb = lagrange_prices(inst, iters=400)
        res.append({"seed": s, "target_util": u, "weight": mode, "lb": float(lb), "sec": time.time() - t})
        json.dump(res, open(out, "w")); print(res[-1], flush=True)
