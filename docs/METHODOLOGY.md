# QUBO-v3.1: theory, metrics and experiments

This document describes in detail what the code in `qsa/` implements, why each component exists, how it is evaluated, and what the experiments show. It complements the short overview in the main `README.md`. All numbers below are reproducible from the logs in `results/` with the scripts named in each section.

**Contents**

1. [The problem](#1-the-problem)
2. [Sequential batch protocol](#2-sequential-batch-protocol)
3. [Global reference: Lagrangian lower bound and offline solution](#3-global-reference)
4. [QUBO, Ising and the samplers](#4-qubo-ising-and-the-samplers)
5. [Why first-generation batch QUBOs fail](#5-why-first-generation-batch-qubos-fail)
6. [QUBO-v3.1](#6-qubo-v31)
7. [Extension: pairwise anti-affinity costs](#7-extension-pairwise-anti-affinity-costs)
8. [Baselines](#8-baselines)
9. [Metrics and statistics](#9-metrics-and-statistics)
10. [Experiments](#10-experiments)
11. [Results](#11-results)
12. [Limitations and lessons](#12-limitations-and-lessons)
13. [Reproducing everything](#13-reproducing-everything)
14. [Parameters](#14-parameters)

---

## 1. The problem

A city has $N$ video cameras (sensors) $\mathcal C=\{1,\dots,N\}$ and $M$ edge servers $\mathcal S=\{1,\dots,M\}$.

| Symbol | Meaning |
|---|---|
| $p_i\in\{1,2,3\}$ | priority of camera $i$ (3 = highest: pedestrian crossings; 2 = sidewalks; 1 = roadways) |
| $l_i>0$ | computational load of the stream (GFLOPS) |
| $b_i>0$ | bandwidth demand (Mbps; recorded, not constrained) |
| $K_j>0$ | effective processing capacity of server $j$ (GFLOPS) |
| $d_{ij}$ | Euclidean distance between camera $i$ and server $j$ |

**Assignment cost.** Distance is a latency proxy; load, priority and server size also enter:

$$
c_{ij}=\operatorname{norm}_{[0,1]}\Big(0.40\,\tfrac{d_{ij}}{d_{\max}}+0.35\,\tfrac{l_i}{l_{\max}}+0.20\,\tfrac{3-p_i}{2}+0.05\,\tfrac{K_j^{-1}}{\max_{j'}K_{j'}^{-1}}\Big),
$$

where $\operatorname{norm}_{[0,1]}$ is a min–max normalisation over all pairs.

**Offline problem.** With $x_{ij}\in\{0,1\}$ = "camera $i$ is processed on server $j$":

$$
\min_x\ F(x)=\sum_{i}\sum_{j} w_i\,c_{ij}\,x_{ij}\;+\;P\sum_i\Big(1-\sum_j x_{ij}\Big)
\quad\text{s.t.}\quad \sum_j x_{ij}\le 1\ \ \forall i,\qquad \sum_i l_i\,x_{ij}\le K_j\ \ \forall j,
$$

with priority weight $w_i=4-p_i$ and a penalty $P=15$ per uncovered camera. Since $w_ic_{ij}\le 3<P$, a camera is left uncovered only if no server can host it. This is a **generalised assignment problem (GAP) with optional coverage** (Ross & Soland 1975). It is NP-hard; at $20{,}000\times800$ it has 16 million binary variables.

> Open modelling question: $w_i=4-p_i$ makes the placement cost of low-priority streams count *more*. Priority is enforced mainly by the processing order (Section 2) and by the cost term $(3-p_i)/2$. The code supports $w_i=p_i$ with a one-line change.

**Instances** (`batch_quality_benchmark.Instance`, generator in `capacity_stress_experiment.py`): seeded NumPy legacy random state; 15% high / 25% medium / 60% low priority with loads 8–15 / 4–8 / 1–3 GFLOPS; servers in three classes with capacities 800–1000 (10%), 400–800 (30%), 200–400 GFLOPS (60%); uniform placement on a $1000\times1000$ plane. **Utilisation** $U=\sum_il_i/\sum_jK_j$ is set exactly to a target (75, 90, 95, 98%) by scaling all capacities with one factor; the cost matrix is unchanged because its capacity term is normalised.

---

## 2. Sequential batch protocol

Streams are deployed or reconfigured in groups, and a rolled-out decision consumes capacity and is not revised. The protocol:

1. Order cameras by decreasing $p_il_i$ (high-priority, heavy streams first) and split them into batches $\mathcal B_1,\mathcal B_2,\dots$ of $B=80$.
2. Before batch $t$, the **residual capacities** are

$$
R_j^{(t)}=K_j-\sum_{\tau<t}\sum_{i\in\mathcal B_\tau} l_i\,x_{ij}.
$$

3. Solve the **batch subproblem**

$$
\min_x \sum_{i\in\mathcal B_t}\sum_{j\in\mathcal S_i^{(t)}}\hat c_{ij}x_{ij}+P\sum_{i\in\mathcal B_t}\Big(1-\sum_jx_{ij}\Big)
\quad\text{s.t.}\quad \sum_jx_{ij}\le1,\quad \sum_{i\in\mathcal B_t}l_ix_{ij}\le R_j^{(t)},
$$

   where $\mathcal S_i^{(t)}$ is the candidate set of camera $i$ and $\hat c_{ij}$ its decision cost (Section 6).
4. Commit the solution, update $R$, continue.

Every method uses the **same** order, batches, candidate sets, decision costs and residual update; methods differ only in how step 3 is solved. The final assignment is always scored with the **true** objective $F$ (no prices).

Code: `qsa/v3_trajectory.py` (one complete trajectory per method).

---

## 3. Global reference

To know how far a sequential method is from the best achievable assignment, two offline references are computed for the full problem (`qsa/global_bound_check.py`).

**Lagrangian lower bound.** Relaxing the capacity constraints with multipliers $u\in\mathbb R^M_{\ge0}$:

$$
L(u)=\sum_{i}\min\Big\{P,\ \min_j\big(w_ic_{ij}+u_jl_i\big)\Big\}-\sum_ju_jK_j,\qquad L(u)\le F(x^\ast)\ \ \forall u\ge0 .
$$

$L$ is maximised by projected subgradient ascent, $u\leftarrow[u+\eta_kg(u)]_+$ with $g_j(u)=\sum_{i:\,j=\arg\min}l_i-K_j$ and a diminishing normalised step (400 iterations); the best value is the bound **LB**.

**Offline feasible solution.** Regret-ordered construction on the Lagrangian costs $w_ic_{ij}+u_jl_i$ (sensors with the largest difference between their best and second-best option first), followed by capacity-feasible shift moves. It is within **0.3–1.0%** of LB on all instances, so LB is essentially tight.

The offline solution may revise every decision and is a **reference, not a competitor**.

---

## 4. QUBO, Ising and the samplers

A QUBO minimises $H(x)=\sum_uQ_{uu}x_u+\sum_{u<v}Q_{uv}x_ux_v$ over $x\in\{0,1\}^n$; with $x_u=(1+\sigma_u)/2$ it is an Ising Hamiltonian. Constraints become penalties.

* **Simulated annealing (SA)** — Metropolis single-flip moves at decreasing temperature (Neal 0.5.5, `neal.SimulatedAnnealingSampler`).
* **Simulated quantum annealing (SQA)** — path-integral Monte Carlo over $P_T$ coupled Trotter replicas,

$$
\mathcal H_{\rm SQA}=\frac1{P_T}\sum_{k=1}^{P_T}H(\sigma^{(k)})-J_\perp\sum_k\sum_u\sigma_u^{(k)}\sigma_u^{(k+1)},\qquad J_\perp=-\tfrac T2\ln\tanh\tfrac{\Gamma}{P_TT},
$$

  with the transverse field $\Gamma$ decreased during the run (OpenJij 0.11.6, `oj.SQASampler`). Both samplers receive the same QUBO, so differences are differences of search dynamics only.

> OpenJij note: `SQASampler()` takes no keyword arguments; `num_reads`, `num_sweeps`, `trotter`, `beta`, `gamma` must be passed to `sample_qubo`. Otherwise it silently runs 1 read, 1000 sweeps, 4 Trotter slices.

---

## 5. Why first-generation batch QUBOs fail

The first version of this work (folder `legacy/`) gave each batch of 80 cameras a **shared window** of 20 servers and compared AO-QUBO, Static-QCP-QUBO, PRC-QUBO and CC-PRC-QUBO (formulas in `legacy/README_legacy.md`). Measured against exact batch optima and LB, three failure causes separate cleanly.

### 5.1 Stale capacity information
AO-QUBO and Static-QCP-QUBO do not see what previous batches consumed; their low-energy states place cameras on full servers, and validation rejects them (≈19% coverage at 24% utilisation, ≈4% at 95%). Residual information in the QUBO is necessary.

### 5.2 An invalid ground state (Proposition 1)
PRC-QUBO uses the one-hot term $\lambda\sum_{j<k}x_{ij}x_{ik}$ with $\lambda=15$, while a feasible pair is rewarded by $\rho_{ij}=\alpha r_i(1-c_{ij})$ with $\alpha=25$, $r_i\le3$.

**Proposition 1.** Let $\rho_{(1)}\ge\rho_{(2)}\ge\dots$ be the sorted rewards of camera $i$. The PRC energy restricted to $i$ is minimised by selecting the $k^\ast$ best candidates, where $k^\ast$ is the largest $k$ with $\rho_{(k)}>\lambda(k-1)$. Hence two or more servers are selected whenever $\rho_{(2)}>\lambda$.

*Proof.* Selecting a set $T$ costs $-\sum_{j\in T}\rho_j+\lambda\binom{|T|}2$. For $|T|=k$ the best set holds the $k$ largest rewards; going from $k-1$ to $k$ changes the energy by $-\rho_{(k)}+\lambda(k-1)$, negative iff $\rho_{(k)}>\lambda(k-1)$. Since $\rho_{(k)}$ decreases and $\lambda(k-1)$ increases, the energy falls up to $k^\ast$ and rises afterwards. ∎

With $r_i=3$ and $1-c_{ij}\approx0.5$ the rewards are ≈37, so $k^\ast=3$; the sampled states had 3.3 servers per camera on average. The assignment was effectively made by the decoder. A correct one-hot penalty must exceed the largest reward.

### 5.3 The shared candidate window
Even with an **exact** solver in every batch, the shared $80\times20$ window stays **87–102%** above LB (10 seeds, 75–98% utilisation). Greedy and regret are within 1.8% of the exact batch optimum there, so the solver hardly matters: most cameras of a batch are far from the 20 shared servers.

### 5.4 Energy barriers of one-hot encodings
With a correct one-hot penalty, moving a camera from server $a$ to $b$ by single flips passes through "both selected" or "none selected", which costs about $\lambda$. Since $\lambda$ must exceed the largest cost difference, these barriers dwarf the differences the sampler must resolve. On 80 hard batches, pure SQA/SA on a correct one-hot QUBO ended 0.83% / 1.62% above the batch optimum — no better than greedy (0.79%).

---

## 6. QUBO-v3.1

QUBO-v3.1 changes (A) how each batch subproblem is **formed** — solver-independent, used by every method — and (B) how it is **encoded** as a QUBO — what the annealers sample. QUBO-v3 is the same without A3 (static prices) and A2' (headroom candidates).

```
order cameras by p_i*l_i; R <- K; u <- Lagrangian prices of the full problem
for each batch t:
    if t % T_p == 0: u <- Lagrangian prices of the residual problem         (A3)
    for each camera: K_w cheapest + K_h headroom candidates under c_hat      (A1, A2)
    fix every camera whose cheapest candidate is on a safe server            (B1)
    repeat up to 3 rounds:                                                    (B4)
        build domain-wall chains (B2) + capacity terms for unsafe servers (B3)
        sample with SQA; decode wall positions; repair overloads; keep best by the batch objective
        if the best raw read overloads no server: stop
        raise price mu_j and weight omega_j of every overloaded server
    commit; update R
```

### A1. Per-camera candidate window
Each camera gets its own $K_w=5$ cheapest servers (by decision cost $\hat c$) among those that can still host it, $\mathcal F_i^{(t)}=\{j: l_i\le R_j^{(t)}\}$. Candidate sets overlap only for nearby cameras, so capacity interactions inside a batch become local. This alone removes most of the shared-window loss (Section 11.1).

### A2. Capacity prices
Sequential decisions are myopic: early batches take the cheapest servers even if later cameras have no alternative. The Lagrangian multipliers $u_j$ (Section 3) measure how much server $j$ is in demand. The **decision cost** is

$$
\hat c_{ij}=w_ic_{ij}+u_jl_i ,
$$

used for the window, inside every batch solver and inside the QUBO. Prices change decisions only; $F$ is always evaluated with true costs.

### A2'. Headroom candidates
Prices make neighbouring cameras of one batch prefer the same few servers; when capacity is scarce, some of them cannot be placed. Each camera therefore also gets $K_h=2$ **headroom candidates**: the cheapest servers (outside the first $K_w$) among the $4K_h$ feasible servers with the largest residual capacity. On a 98% probe instance this reduced exact batch optimisation from 2,195 (33 uncovered) to 1,743 (all covered). $K_w=20$ gave the same quality with a QUBO ~4× larger. (`batch_quality_benchmark.build_batch(k_slack=2)`)

### A3. Rolling re-pricing on the residual problem
Static prices describe scarcity at the start; as batches are committed, the real residual capacities drift away from what they anticipate. Every $T_p=5$ batches the prices are recomputed on the **residual problem** — the cameras not yet assigned $\mathcal C^{(t)}$ and the residual capacities:

$$
L^{(t)}(u)=\sum_{i\in\mathcal C^{(t)}}\min\Big\{P,\ \min_{j\in\mathcal S_i^{\rm top}}\big(w_ic_{ij}+u_jl_i\big)\Big\}-\sum_ju_jR_j^{(t)},
$$

300 subgradient steps from $u=0$. For speed each camera only considers its 40 cheapest servers $\mathcal S_i^{\rm top}$ (`--reprice-top 40`): on the full instance the bound is unchanged and one re-pricing takes <1 s instead of ≈13 s. Warm starts with fewer steps were worse.

Re-pricing changes the decomposition, not the solver, so it helps every method. It also makes batches harder: with near-dual prices many options tie and capacity binds inside the batch, so solving a batch well matters more.

> Information assumption: re-pricing uses the set of cameras still to be placed, which is known in this protocol. In a fully online setting it must be replaced by a forecast (planned experiment).

### B1. Exact reduction of uncontested cameras (Proposition 2)
A server $j$ is **safe** in batch $t$ if all batch cameras that list it as a candidate fit on it together: $\sum_{i\in\mathcal B_t:\,j\in\mathcal S_i}l_i\le R_j^{(t)}$.

**Proposition 2.** Let $U$ be the cameras whose cheapest candidate $j_i^\ast$ is safe and has $\hat c_{ij_i^\ast}<P$. Some optimal solution of the batch subproblem assigns every $i\in U$ to $j_i^\ast$.

*Proof.* Take an optimal solution and move every $i\in U$ to $j_i^\ast$. Safe servers stay feasible because all their candidates fit together; loads elsewhere only decrease. Each moved camera's cost does not increase ($j_i^\ast$ is its cheapest option and cheaper than leaving it uncovered). The new solution is feasible and no worse. ∎

Fixed cameras leave the QUBO, and safe servers need no capacity term. At 95% utilisation the median batch keeps only **2 of 80** cameras free under QUBO-v3.1 (7 with static prices); 22% of batches need no QUBO at all.

### B2. Domain-wall encoding (Proposition 3)
For each free camera sort its options by decision cost, $o_0,\dots,o_{k-1}$ ($o_{k-1}$ = dummy "unassigned" with cost $P$). Normalise $\bar c_{i,m}=(\hat c_{i,o_m}-\hat c_{i,o_0})/\delta$ with $\delta$ the median gap between the two best options in the batch. Following Chancellor (2019), use $k-1$ binaries $z_{i,1..k-1}$ with fixed $z_{i,0}=1$, $z_{i,k}=0$:

$$
x_{i,o_m}=z_{i,m}-z_{i,m+1},\qquad
H_i^{\rm DW}=\sum_{m=0}^{k-1}\bar c_{i,m}(z_{i,m}-z_{i,m+1})+A_i\sum_{m=1}^{k-2}z_{i,m+1}(1-z_{i,m}).
$$

**Proposition 3.** (a) The penalty is zero exactly on non-increasing chains $1\ge z_{i,1}\ge\dots\ge0$, which correspond one-to-one to the $k$ options (the option is the wall position). (b) The cost part equals $\bar c_{i,0}+\sum_m(\bar c_{i,m}-\bar c_{i,m-1})z_{i,m}$ with non-negative coefficients, so the cheapest option minimises $H_i^{\rm DW}$ and every invalid chain costs at least $\bar c_{i,0}+A_i$. (c) Any two options are connected by single flips through valid chains only, each step changing the energy by the cost difference of adjacent options.

*Proof.* (a) $z_{m+1}(1-z_m)=1$ exactly at an increase; a 1→0 chain without increases drops once. (b) Summation by parts with $z_0=1$, $z_k=0$. (c) With the wall at $m$, flipping $z_{m+1}$ 0→1 moves it to $m+1$, flipping $z_m$ 1→0 to $m-1$. ∎

One-hot needs two flips per move and passes a barrier ≈ penalty; the domain wall moves between neighbouring options without a barrier and uses one variable less per camera. $A=\max\{1,0.45\,q_{0.95}\}$ with $q_{0.95}$ the 95th percentile of normalised option costs.

### B3. Slack-free capacity penalty
For an unsafe server, $a_i=l_i/R_j^{(t)}$, $f_j$ = normalised load of fixed cameras, $S_j=f_j+\sum_ia_ix_{ij}$ (in $z$ through B2). Unbalanced penalisation of $S_j\le1$ (Montañez-Barrera et al. 2024), no slack variables:

$$
H_j^{\rm cap}=W_j\big[(\lambda_1-2)S_j+S_j^2\big]+A\,\mu_jS_j,\qquad W_j=\lambda_{\rm cap}A\,\omega_j,\quad \lambda_1=0.2,\ \lambda_{\rm cap}=1 .
$$

It slightly rewards filling a server up to $S_j=1-\lambda_1/2$. An exact slack-bit encoding removed this bias but sampled worse.

### B4. Adaptive resampling
The penalty is soft, so a read can overload a server. After each round, for every server overloaded by the best raw read ($\rho_j=S_j-1>0$): $\mu_j\mathrel{+}=0.5(1+\rho_j)$, $\omega_j\mathrel{\times}=2$; at most 3 rounds (augmented-Lagrangian spirit).

### B5. Sampling, decoding, selection
SQA: 10 reads, 2000 sweeps, 8 Trotter slices, $\beta=20$, $\gamma=1$, seed from instance seed and batch index. Each read → wall position of each free camera → if a server is still overloaded, remove cameras in order of increasing value per unit load and re-insert greedily (repairs are logged). Among all reads of all rounds, the one with the best **price-adjusted batch objective** is committed — the same objective every other solver optimises. **SQA+LS** additionally applies shift/swap local search to every decoded read.

> Consistency fix (2026-10-06): earlier, ILS and the QUBO read selection ranked candidates by the *unpriced* cost while exact/greedy/regret optimised the priced one. All campaign-3 results use the consistent rule (`decision_obj` in `v3_trajectory.py`).

### Problem size
A batch QUBO has at most $\sum_{\rm free}(K_w+K_h)$ variables. At 95% utilisation the median QUBO-v3.1 batch has **14 variables** on $20{,}000\times800$ (90th percentile 91, max 476) and **7** on $50{,}000\times2{,}000$; it does not grow with $N$ or $M$. First-generation batches had 1,600 variables and 16,800–80,000 coefficients.

---

## 7. Extension: pairwise anti-affinity costs

Cameras with overlapping fields of view (distance < 16 on the 5,000-camera map) should not share a server, so that a server failure does not blind the same area twice: add $\beta_{aa}$ per co-located overlapping pair. Pairs with a committed camera become linear terms; in-batch pairs become couplings $\beta_{aa}x_{ij}x_{kj}$. Batches are spatial tiles of 80 cameras (300–660 pairwise terms each). Code: `qsa/aa_proto.py`, `build_v3(quad=...)`.

Two safeguards are required:

1. **Wall penalty must dominate the couplings.** Products of $x=z_m-z_{m+1}$ are unbounded below off the valid chains ($x$ can be −1). With the default $A$ the samplers reached energy −18,000 vs −5,000 for the optimum. Fix: $A_i=A+2\sum_k|\beta_{ik}|/\delta$.
2. **Reduction does not apply** to cameras with pairwise terms (they stay free). In addition, OpenJij's fixed-temperature schedule needs the stiffer QUBO rescaled (95th percentile of $|Q|$ → 0.7).

Results: the linearised MILP ($y_{pq}\ge x_p+x_q-1$) still solves each 80-camera batch optimally in 0.03–1.2 s; ILS (2 s) is within 0.5%, SA 0.4–1.9%, SQA 0.6–4.6% (0.2–1.8% with LS). With 400-camera batches (~4,000 pairwise terms) the MILP stops at 0.6% gap after 60 s, but ILS and SA still beat SQA. **No QUBO advantage on classical hardware.**

---

## 8. Baselines

| Method | Description | Code |
|---|---|---|
| Greedy | each camera in priority order takes its cheapest feasible candidate | `solve_greedy` |
| Regret | repeatedly assign the camera with the largest gap between best and second-best feasible option | `solve_regret` |
| ILS | iterated local search per batch: regret + shift/swap LS, then destroy 20%, greedy repair, LS; 0.8 s per batch | `solve_ils` |
| Exact batch MILP | batch subproblem solved to optimality with HiGHS (SciPy `milp`) | `solve_exact` |
| SA-v3.1 | QUBO-v3.1 Hamiltonian sampled with Neal | `solve_v3(..., "SA")` |
| SQA-v3.1 | proposed method | `solve_v3(..., "SQA")` |
| SQA-v3.1+LS | with shift/swap LS on decoded reads | `--polish` |
| Offline global | Section 3 (reference) | `global_bound_check.py` |

The exact batch MILP is the best any batch solver can do for each batch **in isolation**; it does not guarantee the best trajectory (a batch decision changes the residual problem on which later prices are computed).

Earlier metaheuristics in the shared-window setting (GRASP, BRKGA, tabu search, capacity-priced greedy) are in `baselines/` and `results/legacy/`.

---

## 9. Metrics and statistics

| Metric | Definition |
|---|---|
| Objective $F$ | true objective of Section 1 over the whole trajectory |
| Gap to LB | $(F-\mathrm{LB})/\mathrm{LB}$, LB from Section 3 (same instance) |
| Uncovered | cameras with no server, summed over seeds |
| Time | wall-clock per instance (state the parallel load) |
| QUBO size | variables per batch after reduction; free cameras per batch |
| Repairs | overloads fixed after decoding |
| Ablation gap | on **hard batches** (greedy ≥ 0.5% worse than the exact batch optimum; 8 per seed): gap of pure annealing to the exact batch optimum |
| Cumulative excess | $\sum_{\tau\le t}\big(F_\tau-F^{\rm off}_\tau\big)/\mathrm{LB}$, online batch cost minus the offline cost of the same cameras (trajectory figure) |

**Statistics.** Paired comparisons on the same instances: Wilcoxon signed-rank test, Holm correction within each table/family, matched-pairs rank-biserial correlation $r$ as effect size. With 10 paired seeds the smallest two-sided $p$ is 0.002; with 5 it is 0.0625 (5-seed comparisons are descriptive only).

---

## 10. Experiments

| Campaign | What | Scale | Where |
|---|---|---|---|
| Campaign 1 | first multi-seed study (capacity-scale protocol) | 20k×800, 9–10 seeds | `results/v3/campaign/` |
| Campaign 2 | target utilisation 75/90/95/98%, QUBO-v3 (static prices), shared window, ablation on 80 hard batches, scaling 5k×200 (5 seeds) and 50k×2000 (3 seeds) | 456 runs | `results/v3/campaign2/` |
| v3.1 prototype | design of re-pricing and headroom candidates | 5k×200, 5 seeds | `results/v3/v31_prototype/` |
| Anti-affinity probe | Section 7 | 5k×200 | `results/v3/anti_affinity_probe/` |
| **Campaign 3** | **QUBO-v3.1, all 7 methods, 75–98%** | **20k×800, 10 seeds, 280 runs** | `results/v3/campaign3/` |

All campaign-2/3 runs on one machine (Intel i7-11370H, 4 jobs in parallel); prototypes on a 2-vCPU cloud machine. Identical objective values were reproduced on both machines.

---

## 11. Results

### 11.1 Effect of the decomposition (exact batch solver, 20k×800, 10 seeds)

| Decomposition | 75% | 90% | 95% | 98% |
|---|---|---|---|---|
| Shared 80×20 window | 102.1% | 94.3% | 90.0% | 86.8% |
| QUBO-v3 (per-camera window, static prices) | 4.33% | 8.55% | 11.88% (1 unc.) | 16.64% (7 unc.) |
| **QUBO-v3.1** (headroom + rolling prices) | **0.45%** | **1.03%** | **1.64%** | **3.35%** |
| Offline global solution | 0.29% | 0.57% | 0.74% | 1.03% |

(gap to LB; QUBO-v3.1 covers every camera)

Where the loss arises (seed 42): with static prices the early batches are placed *more cheaply* than offline and the loss accumulates from about batch 170, when low-priority cameras find their cheap servers taken (10.9% at 95%, 16.8% at 98%). With rolling prices the curve stays flat; the remainder is concentrated in the last few batches (0.9%, 4.4%).

### 11.2 Ablation of the encoding (80 hard batches, 95%, static prices)

| Variant | SQA gap | SQA optimal | SA gap | SA optimal |
|---|---|---|---|---|
| one-hot (base) | 0.829% | 0/80 | 1.619% | 0/80 |
| + reduction (R) | 0.185% | 8/80 | 0.137% | 17/80 |
| + domain wall (DW) | 0.554% | 0/80 | 0.325% | 1/80 |
| R + DW | 0.122% | 14/80 | 0.082% | 31/80 |
| R + DW + adaptive | **0.063%** | 24/80 | **0.031%** | 42/80 |
| R + DW, SQA β=20 | 0.437% (median 0.002%) | 38/80 | – | – |
| greedy / regret / regret+LS | 0.789% / 0.016% / 0.012% | | | |

### 11.3 Batch solvers under QUBO-v3.1 (campaign 3, 20k×800, 10 seeds)

Mean objective (gap to LB); no method leaves a camera uncovered. Time at 95% includes ≈100 s of re-pricing and window construction common to all methods.

| Method | 75% | 90% | 95% | 98% | Time (s) |
|---|---|---|---|---|---|
| Greedy | 5,921.6 (0.46%) | 6,006.4 (1.05%) | 6,074.3 (1.74%) | 6,196.5 (3.43%) | 113 |
| Regret | 5,921.1 (0.45%) | 6,005.4 (1.03%) | 6,068.4 (1.64%) | 6,196.5 (3.42%) | 121 |
| Exact batch MILP | 5,921.0 (0.45%) | 6,005.4 (1.03%) | 6,068.4 (1.64%) | 6,192.2 (3.35%) | 109 |
| ILS | 5,921.0 (0.45%) | 6,005.6 (1.03%) | 6,067.2 (1.62%) | 6,192.1 (3.35%) | 319 |
| SA-v3.1 | 5,923.8 (0.50%) | 6,009.1 (1.09%) | 6,072.6 (1.72%) | 6,195.7 (3.41%) | 112 |
| SQA-v3.1 | 5,924.2 (0.50%) | 6,012.3 (1.15%) | 6,075.3 (1.76%) | 6,198.6 (3.46%) | 308 |
| SQA-v3.1+LS | 5,921.3 (0.46%) | 6,005.7 (1.04%) | 6,067.6 (1.63%) | **6,189.3 (3.31%)** | 397 |

Paired tests (Holm-corrected Wilcoxon $p$; W/L = wins/losses of the SQA variant):

| Comparison | 75% | 90% | 95% | 98% |
|---|---|---|---|---|
| SQA+LS vs greedy | −0.3 (7/3) 0.49 | −0.7 (5/5) 1.0 | **−6.7 (10/0) 0.012** | −7.2 (8/2) 0.12 |
| SQA+LS vs regret | +0.2 (1/9) 0.26 | +0.3 (4/6) 1.0 | −0.8 (7/3) 1.0 | −7.2 (7/3) 0.34 |
| SQA+LS vs exact | +0.2 (1/9) 0.26 | +0.3 (3/7) 1.0 | −0.8 (7/3) 1.0 | −2.9 (7/3) 0.55 |
| SQA+LS vs ILS | +0.2 (1/9) 0.26 | +0.1 (4/6) 1.0 | +0.4 (7/3) 1.0 | −2.8 (6/4) 0.55 |
| SQA vs exact | +3.1 (0/10) 0.012 | +6.9 (1/9) 0.012 | +6.9 (0/10) 0.012 | +6.4 (3/7) 0.65 |
| SQA vs ILS | +3.1 (0/10) 0.012 | +6.7 (1/9) 0.012 | +8.1 (0/10) 0.012 | +6.5 (3/7) 0.65 |
| SQA vs SA | +0.4 (4/6) 0.38 | +3.1 (0/10) 0.012 | +2.6 (3/7) 0.65 | +2.9 (3/7) 1.0 |

Reading:
* All solvers are within 0.2% of each other under QUBO-v3.1; the decomposition, not the solver, determines quality.
* Pure SQA is the weakest by a small margin (0.05–0.12% behind exact/ILS, significant at 75–95%) and not better than SA.
* With LS on decoded reads, SQA is statistically indistinguishable from exact batch optimisation, regret and ILS; significantly better than greedy at 95%; lowest mean at 98% (n.s.).

### 11.4 Scaling (static prices, 95%)

| Instance | Exact batch gap | SQA-v3 gap | SQA time (s) | Uncovered (SQA) |
|---|---|---|---|---|
| 5,000×200 (5 seeds) | 11.59% | 11.83% | 222 | 3 |
| 20,000×800 (10 seeds) | 11.88% | 11.87% | 517 | 2 |
| 50,000×2,000 (3 seeds) | 12.25% | 12.26% | 509 | 0 |

The batch QUBO does not grow with the instance; time grows with the number of batches.

---

## 12. Limitations and lessons

**Limitations.** Synthetic, uniformly placed instances; distance as latency proxy, bandwidth not constrained; re-pricing assumes the remaining demand is known (forecast experiment planned); SQA only on CPU emulation (times say nothing about physical annealers); fixed batch order, no revision of committed batches; the weight $w_i=4-p_i$ needs a decision.

**What SQA does and does not show.** On classical hardware SQA does not outperform exact optimisation, ILS or SA. The value of the QUBO side is a compact, well-conditioned batch model (median 14 variables) with proven properties (exact reduction, barrier-free encoding) that annealers sample to near-optimality and that is ready for physical Ising machines.

**Lessons for decomposed QUBO models.**
1. Check that the penalised ground state is feasible (Proposition 1; and with pairwise terms, the wall penalty must dominate the couplings).
2. Compare against an exact batch optimum *and* a global bound; otherwise decomposition losses are blamed on the solver.
3. Coordinate subproblems: residual-problem prices carry information about future batches at negligible cost.
4. Remove easy decisions before sampling (Proposition 2).
5. Compare solvers on the same objective; ranking reads by a different cost measures the ranking rule, not the solver.

---

## 13. Reproducing everything

Requirements: `requirements.txt` (Python 3.10, NumPy, SciPy ≥1.10 with HiGHS, OpenJij 0.11.6, dwave-neal 0.5.5).

One trajectory (example: SQA-v3.1, 95%, seed 42):
```
python qsa/v3_trajectory.py --method sqa_v3 --target-util 95 --seed 42 \
       --price --price-window --reprice 5 --k-slack 2 --out results/v3/campaign3/v31
```
Options: `--method {greedy,regret,exact,ils,sa_v3,sqa_v3}`, `--polish` (LS on decoded reads), `--n-cameras/--n-servers`, `--trace` (per-batch objective, QUBO size and assignment), `--reprice-top` (sparse re-pricing, default 40).

Global reference: `python qsa/global_bound_check.py --target-utils 75,90,95,98 --seed 42 --out results/v3/campaign2/global`

Full campaigns (resumable; finished jobs are recorded in `<jobfile>.done`):
```
python experiments/run_jobs.py experiments/jobs_campaign3_a.txt --workers 4
python experiments/run_jobs.py experiments/jobs_campaign3_b.txt --workers 4
```
Summaries: `qsa/c2_aggregate.py` (campaign 2), `qsa/c3_aggregate.py` (campaign 3), `qsa/v31_aggregate.py` (prototype).

---

## 14. Parameters

| Parameter | Value | Role |
|---|---|---|
| Batch size $B$ | 80 | cameras per committed decision |
| Windows $K_w$, $K_h$ | 5, 2 | cheapest and headroom candidates per camera |
| Uncovered penalty $P$ | 15 | objective |
| Lagrangian iterations | 300 (prices), 400 (bound) | subgradient ascent |
| Re-pricing period $T_p$ | 5 batches | 40 cheapest servers per camera |
| Domain-wall weight $A$ | $\max\{1,0.45\,q_{0.95}\}$ | chain penalty |
| Capacity penalty | $\lambda_1=0.2$, $\lambda_{\rm cap}=1$ | unbalanced penalisation |
| Adaptive rounds | 3; $\mu\mathrel{+}=0.5(1+\rho)$, $\omega\mathrel{\times}=2$ | resampling |
| SQA | 10 reads, 2000 sweeps, $P_T=8$, $\beta=20$, $\gamma=1$ | OpenJij 0.11.6 |
| SA | 10 reads, 2000 sweeps | Neal 0.5.5 |
| ILS | 0.8 s per batch, destroy 20% | metaheuristic baseline |

**References:** Ross & Soland (1975) *Math. Prog.* 8; Fisher (1981) *Manag. Sci.* 27; Yagiura, Ibaraki & Glover (2004) *INFORMS J. Comput.* 16; Lourenço, Martin & Stützle (2003) *Handbook of Metaheuristics*; Chancellor (2019) *Quantum Sci. Technol.* 4, 045004; Montañez-Barrera et al. (2024) *Quantum Sci. Technol.* 9; Huangfu & Hall (2018) *Math. Prog. Comp.* 10; Wilcoxon (1945); Holm (1979); Kerby (2014).
