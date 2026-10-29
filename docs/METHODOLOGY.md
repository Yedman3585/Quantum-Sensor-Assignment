# Method: price-coordinated decomposition with compact QUBO subproblems

This document explains what the code in `qsa/` implements and why. It covers:

- the problem;
- the sequential protocol;
- how an ordinary QUBO is built and why it cannot be used for the full problem;
- the coordination layer: capacity prices, re-pricing and the demand margin;
- the QUBO-v3.1 encoding of a batch;
- the baselines.

The other documentation:

- [`METRICS.md`](METRICS.md) defines every metric and statistical test, and names the script that computes each one.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) lists every experiment with its command, its data and its results.

**Contents**

1. [The problem](#1-the-problem)
2. [Sequential batch protocol](#2-sequential-batch-protocol)
3. [Global reference: Lagrangian bound and offline solution](#3-global-reference)
4. [QUBO primer: how an ordinary QUBO works](#4-qubo-primer)
5. [Why first-generation batch QUBOs fail](#5-why-first-generation-batch-qubos-fail)
6. [Coordination layer: capacity prices, re-pricing, demand margin](#6-coordination-layer)
7. [QUBO-v3.1: encoding of a batch](#7-qubo-v31-encoding-of-a-batch)
8. [Extension: pairwise anti-affinity costs](#8-extension-pairwise-anti-affinity-costs)
9. [Baselines](#9-baselines)
10. [Instance variants](#10-instance-variants)
11. [Limitations and lessons](#11-limitations-and-lessons)
12. [Parameters](#12-parameters)

**In one paragraph.** Cameras arrive in batches of 80 and every decision is final. Each batch is a small assignment problem on the residual capacities. Batches are coordinated by **capacity prices**: Lagrangian multipliers of the capacity constraints. The prices are recomputed every 5 batches on the *residual* problem, with the remaining demand inflated by a **5% margin**. Priced costs decide which candidate servers each camera sees, and every batch solver optimises them. A batch can be solved exactly, greedily or as a **QUBO** sampled by simulated (quantum) annealing. The QUBO is small: after an exact reduction the median batch has about 14 binary variables. The coordination layer brings the sequential policy from 4–17% above the lower bound (static prices) to 0.3–1.2%. The choice of batch solver changes the result by less than 0.1 pp.

---

## 1. The problem

A city has $N$ video cameras (sensors) $\mathcal C=\{1,\dots,N\}$ and $M$ edge servers $\mathcal S=\{1,\dots,M\}$. Default size: $N=20{,}000$, $M=800$.

| Symbol | Meaning |
|---|---|
| $p_i\in\{1,2,3\}$ | priority of camera $i$: 3 = high (pedestrian crossings, 15%), 2 = medium (sidewalks, 25%), 1 = low (roadways, 60%) |
| $l_i>0$ | computational load of the stream in GFLOPS: high 8–15, medium 4–8, low 1–3 |
| $b_i>0$ | bandwidth demand in Mbps (recorded, not constrained) |
| $K_j>0$ | processing capacity of server $j$ in GFLOPS: 800–1000 (10% of servers), 400–800 (30%), 200–400 (60%), all scaled by one factor |
| $d_{ij}$ | Euclidean distance between camera $i$ and server $j$ on a $1000\times1000$ plane |

**Assignment cost.** Distance is a latency proxy; load, priority and server size also enter:

$$
c_{ij}=\operatorname{norm}_{[0,1]}\Big(0.40\,\tfrac{d_{ij}}{d_{\max}}+0.35\,\tfrac{l_i}{l_{\max}}+0.20\,\tfrac{3-p_i}{2}+0.05\,\tfrac{K_j^{-1}}{\max_{j'}K_{j'}^{-1}}\Big),
$$

where $\operatorname{norm}_{[0,1]}$ is a min–max normalisation over all pairs.

**Offline problem.** Let $x_{ij}\in\{0,1\}$ mean "camera $i$ is processed on server $j$". Then

$$
\min_x\ F(x)=\sum_{i}\sum_{j} w_i\,c_{ij}\,x_{ij}\;+\;P\sum_i\Big(1-\sum_j x_{ij}\Big)
\quad\text{s.t.}\quad \sum_j x_{ij}\le 1\ \ \forall i,\qquad \sum_i l_i\,x_{ij}\le K_j\ \ \forall j,
$$

with priority weight $w_i=4-p_i$ and a penalty $P=15$ per uncovered camera.

- Since $w_ic_{ij}\le 3<P$, a camera is left uncovered only if no server can host it.
- This is a **generalised assignment problem (GAP) with optional coverage** (Ross & Soland 1975). It is NP-hard.
- At $20{,}000\times800$ it has 16 million binary variables.
- The alternative weight $w_i=p_i$ is supported (`--weight prio`, Section 10).

**Instances.** The generator is `capacity_stress_experiment.py`, wrapped by `batch_quality_benchmark.Instance`, and uses a seeded NumPy legacy random state.

- **Utilisation** $U=\sum_il_i/\sum_jK_j$ is set exactly to a target (75, 90, 95 or 98%) by scaling all capacities with one factor.
- The cost matrix does not change with that factor because its capacity term is normalised.
- **Test seeds** are 42–51. The demand margin was tuned on **validation seeds** 100–109.

---

## 2. Sequential batch protocol

Streams are deployed in groups, and a decision that has been rolled out consumes capacity and is not revised. The protocol is:

1. Order the cameras by decreasing $p_il_i$, so high-priority, heavy streams come first. Split them into batches $\mathcal B_1,\mathcal B_2,\dots$ of $B=80$, which gives 250 batches at 20k.
2. Before batch $t$, compute the **residual capacities**

$$
R_j^{(t)}=K_j-\sum_{\tau<t}\sum_{i\in\mathcal B_\tau} l_i\,x_{ij}.
$$

3. Solve the **batch subproblem**

$$
\min_x \sum_{i\in\mathcal B_t}\sum_{j\in\mathcal S_i^{(t)}}\hat c_{ij}x_{ij}+P\sum_{i\in\mathcal B_t}\Big(1-\sum_jx_{ij}\Big)
\quad\text{s.t.}\quad \sum_jx_{ij}\le1,\quad \sum_{i\in\mathcal B_t}l_ix_{ij}\le R_j^{(t)},
$$

   where $\mathcal S_i^{(t)}$ is the candidate set of camera $i$ and $\hat c_{ij}$ its decision cost (Section 6).
4. Commit the solution, update $R$ and continue.

Every method uses the **same** order, batches and residual update. Methods differ in how the candidate sets and decision costs are formed (the coordination layer, Section 6) and in how step 3 is solved (the batch solver, Sections 7 and 9). The final assignment is always scored with the **true** objective $F$, without prices.

Code: `qsa/v3_trajectory.py` computes one complete trajectory per run.

---

## 3. Global reference

Two offline references for the full problem tell us how far a sequential method is from the best achievable assignment. Code: `qsa/global_bound_check.py`; `qsa/offline_global.py` also stores the assignment.

**Lagrangian lower bound (LB).** Relax the capacity constraints with multipliers $u\in\mathbb R^M_{\ge0}$:

$$
L(u)=\sum_{i}\min\Big\{P,\ \min_j\big(w_ic_{ij}+u_jl_i\big)\Big\}-\sum_ju_jK_j,\qquad L(u)\le F(x^\ast)\ \ \forall u\ge0 .
$$

For fixed $u$ the problem separates by camera: each camera takes its cheapest priced option, or stays uncovered. $L$ is maximised by projected subgradient ascent:

- update $u\leftarrow[u+\eta_kg(u)]_+$ with $g_j(u)=\sum_{i:\,j=\arg\min}l_i-K_j$, the load that would go to $j$ minus its capacity;
- the step is diminishing and normalised;
- run 400 iterations and take the best value as the bound **LB**.

**Offline feasible solution.**

1. Regret-ordered construction on the priced costs $w_ic_{ij}+u_jl_i$: cameras with the largest difference between their best and second-best option are placed first.
2. Capacity-feasible shift moves on the true costs.

The offline solution is within 0.3–1.0% of LB on all instances, so LB is essentially tight. The offline solution may revise every decision and sees all cameras at once. It is a **reference, not a competitor**: the gap between it and a sequential policy is the *price of sequentiality* ([`METRICS.md`](METRICS.md)).

---

## 4. QUBO primer

### 4.1 What a QUBO is

A **QUBO** (quadratic unconstrained binary optimisation) problem is

$$
\min_{x\in\{0,1\}^n}\ H(x)=\sum_{u}Q_{uu}x_u+\sum_{u<v}Q_{uv}x_ux_v = x^\top Qx ,
$$

with an upper-triangular matrix $Q$ (using $x_u^2=x_u$).

- The substitution $x_u=(1+\sigma_u)/2$, $\sigma_u\in\{-1,+1\}$, turns it into an **Ising** energy $\sum_u h_u\sigma_u+\sum_{u<v}J_{uv}\sigma_u\sigma_v$ plus a constant.
- Quantum annealers and digital annealers minimise Ising energies natively. On ordinary hardware the same models are sampled by simulated annealing or by simulated quantum annealing.
- A QUBO has no constraints. Every constraint must become a **penalty** that is zero on feasible points and positive elsewhere.

### 4.2 The textbook QUBO for assignment

Use binary variables $x_{ij}$ ("camera $i$ on server $j$"). The textbook model (Lucas 2014; Glover, Kochenberger & Du 2019) has three parts:

1. **Cost:** $\sum_{ij}w_ic_{ij}x_{ij}$.
2. **Exactly one server per camera (one-hot):** $\lambda\big(1-\sum_jx_{ij}\big)^2=\lambda\big(1-\sum_jx_{ij}+2\sum_{j<k}x_{ij}x_{ik}\big)$. This penalty is zero if exactly one $x_{ij}=1$. Optional coverage adds a dummy option "unassigned" with cost $P$.
3. **Capacity:** the inequality $\sum_il_ix_{ij}\le K_j$ is first made an equality with an integer slack $s_j=\sum_k2^k y_{jk}$ in binary bits. Then it is squared: $\gamma\big(\sum_il_ix_{ij}+s_j-K_j\big)^2$.

**Small example.** Take 2 cameras with loads 1 and 2 servers $A$ and $B$ with $K_A=1$ and $K_B=2$. Camera 1 prefers $A$ (cost 0.1 vs 0.4); camera 2 also prefers $A$ (0.2 vs 0.3). The capacity of $A$ allows one camera, so the constraint is the pair penalty $\gamma x_{1A}x_{2A}$ with no slack needed. $B$ can take both.

$$
H=0.1x_{1A}+0.4x_{1B}+0.2x_{2A}+0.3x_{2B}+\lambda\sum_{i=1,2}(1-x_{iA}-x_{iB})^2+\gamma\,x_{1A}x_{2A}.
$$

With $\lambda,\gamma$ larger than every cost difference, the minimum is $x_{1A}=x_{2B}=1$ with energy 0.4. The alternative $x_{1B}=x_{2A}=1$ has energy 0.6. An annealer that returns a state with a positive penalty has returned an infeasible assignment, which must be repaired.

### 4.3 Why this cannot be applied to the full problem

| Quantity | Full $20{,}000\times800$ problem |
|---|---|
| assignment variables | $N\cdot M=1.6\times10^7$ |
| slack bits | about $M\lceil\log_2 K_j\rceil\approx 8{,}000$ |
| quadratic terms from one-hot | $N\binom{M}{2}\approx6.4\times10^9$ |
| quadratic terms from capacity | $M\binom{N}{2}\approx1.6\times10^{11}$ (every pair of cameras on every server) |

No sampler, classical or quantum, handles this, and a dense capacity penalty also has a very rugged energy landscape. Large QUBO applications therefore **decompose** the problem. The quality of the result then depends mainly on how the pieces are coordinated, which is the subject of Section 6.

### 4.4 Samplers

- **Simulated annealing (SA).** Metropolis single-flip moves at decreasing temperature. Implementation: Neal 0.5.7, `neal.SimulatedAnnealingSampler`.
- **Simulated quantum annealing (SQA).** Path-integral Monte Carlo over $P_T$ coupled Trotter replicas:

$$
\mathcal H_{\rm SQA}=\frac1{P_T}\sum_{k=1}^{P_T}H(\sigma^{(k)})-J_\perp\sum_k\sum_u\sigma_u^{(k)}\sigma_u^{(k+1)},\qquad J_\perp=-\tfrac T2\ln\tanh\tfrac{\Gamma}{P_TT},
$$

  where the transverse field $\Gamma$ is decreased during the run. Implementation: OpenJij 0.11.6, `oj.SQASampler`. SQA is a classical emulation of quantum annealing; its run times say nothing about physical annealers.

Both samplers receive the same QUBO, so differences between them are differences of search dynamics only.

> OpenJij note: `SQASampler()` takes no keyword arguments. `num_reads`, `num_sweeps`, `trotter`, `beta` and `gamma` must be passed to `sample_qubo`; otherwise it silently runs 1 read, 1000 sweeps and 4 Trotter slices.

---

## 5. Why first-generation batch QUBOs fail

The first version of this work is in the folder `legacy/`, with formulas in `legacy/README_legacy.md`. It gave each batch of 80 cameras a **shared window** of 20 servers and compared AO-QUBO, Static-QCP-QUBO, PRC-QUBO and CC-PRC-QUBO. Measured against exact batch optima and against LB, the failures have four separate causes.

### 5.1 Stale capacity information
AO-QUBO and Static-QCP-QUBO do not see what earlier batches consumed. Their low-energy states put cameras on full servers, and validation rejects them: about 19% coverage at 24% utilisation and about 4% at 95%. The QUBO must contain residual information.

### 5.2 An invalid ground state (Proposition 1)
PRC-QUBO uses the one-hot term $\lambda\sum_{j<k}x_{ij}x_{ik}$ with $\lambda=15$, while a feasible pair is rewarded by $\rho_{ij}=\alpha r_i(1-c_{ij})$ with $\alpha=25$ and $r_i\le3$.

**Proposition 1.** Let $\rho_{(1)}\ge\rho_{(2)}\ge\dots$ be the sorted rewards of camera $i$. The PRC energy restricted to $i$ is minimised by selecting the $k^\ast$ best candidates, where $k^\ast$ is the largest $k$ with $\rho_{(k)}>\lambda(k-1)$. Hence two or more servers are selected whenever $\rho_{(2)}>\lambda$.

*Proof.* Selecting a set $T$ costs $-\sum_{j\in T}\rho_j+\lambda\binom{|T|}2$. For $|T|=k$ the best set holds the $k$ largest rewards. Going from $k-1$ to $k$ changes the energy by $-\rho_{(k)}+\lambda(k-1)$, which is negative iff $\rho_{(k)}>\lambda(k-1)$. Since $\rho_{(k)}$ decreases and $\lambda(k-1)$ increases, the energy falls up to $k^\ast$ and rises afterwards. ∎

With $r_i=3$ and $1-c_{ij}\approx0.5$ the rewards are about 37, so $k^\ast=3$. The sampled states had 3.3 servers per camera on average, so the assignment was in effect made by the decoder. A correct one-hot penalty must exceed the largest reward.

### 5.3 The shared candidate window
Even with an **exact** solver in every batch, the shared $80\times20$ window stays **87–102%** above LB (10 seeds, 75–98% utilisation). Most cameras of a batch are far from the 20 shared servers. Greedy and regret are within 1.8% of the exact batch optimum there, so the solver hardly matters.

### 5.4 Energy barriers of one-hot encodings
With a correct one-hot penalty, moving a camera from server $a$ to server $b$ by single flips passes through "both selected" or "none selected", which costs about $\lambda$. Since $\lambda$ must exceed the largest cost difference, these barriers dwarf the differences the sampler has to resolve.

---

## 6. Coordination layer

This is the main contribution. It decides **what each batch sees**: candidate servers and decision costs. It does not depend on the batch solver and is used by every method. The pseudocode below shows both layers; QUBO-v3 is the same without the residual re-pricing A3 and the headroom candidates A2'.

```
order cameras by p_i*l_i; R <- K; u <- Lagrangian prices of the full problem          (A2)
for each batch t:
    if t % T_p == 0: u <- prices of the residual problem, demand x (1+b)            (A3, A4)
    for each camera: K_w cheapest + K_h headroom candidates under c_hat = w c + u l     (A1, A2')
    solve the batch subproblem on c_hat with any solver (exact, greedy, ..., QUBO)   (Section 7)
    commit; update R
```

### A1. Per-camera candidate window
Each camera gets its own $K_w=5$ cheapest servers by decision cost $\hat c$, chosen among the servers that can still host it, $\mathcal F_i^{(t)}=\{j: l_i\le R_j^{(t)}\}$. Candidate sets overlap only for nearby cameras, so capacity interactions inside a batch become local. This alone removes most of the shared-window loss: from 87–102% down to 4–17% above LB.

### A2. Capacity prices
Sequential decisions are myopic. Early batches take the cheapest servers even when later cameras have no alternative. The Lagrangian multipliers $u_j$ (Section 3) measure how much server $j$ is in demand: $u_j>0$ only if the server would be over-subscribed at zero price. The **decision cost** is

$$
\hat c_{ij}=w_ic_{ij}+u_jl_i .
$$

It is used for the window, inside every batch solver and inside the QUBO. Prices change decisions only; $F$ is always evaluated with true costs. Prices of the full problem computed once at the start are called **static prices**.

### A2'. Headroom candidates
Prices make neighbouring cameras in one batch prefer the same few servers, and when capacity is scarce some of them cannot be placed. Each camera therefore also gets $K_h=2$ **headroom candidates**: the cheapest servers outside its first $K_w$, taken from the $4K_h$ feasible servers with the largest residual capacity.

On a 98% probe instance this reduced the exact batch objective from 2,195 (33 cameras uncovered) to 1,743 (all covered). $K_w=20$ gave the same quality with a QUBO about 4× larger. Code: `batch_quality_benchmark.build_batch(k_slack=2)`.

### A3. Rolling re-pricing on the residual problem
Static prices describe scarcity at the start. As batches are committed, the real residual capacities drift away from what those prices anticipate. Every $T_p=5$ batches the prices are recomputed on the **residual problem**, using the cameras not yet assigned, $\mathcal C^{(t)}$, and the residual capacities:

$$
L^{(t)}(u)=\sum_{i\in\mathcal C^{(t)}}\min\Big\{P,\ \min_{j\in\mathcal S_i^{\rm top}}\big(w_ic_{ij}+u_j\tilde l_i\big)\Big\}-\sum_ju_jR_j^{(t)} .
$$

- The bound is maximised with 300 subgradient steps from $u=0$. Warm starts with fewer steps were worse.
- For speed, each camera only considers its 40 cheapest servers $\mathcal S_i^{\rm top}$ (`--reprice-top 40`). On the full instance the bound does not change, and one re-pricing takes under 1 s instead of about 13 s.
- $\tilde l_i$ is the forecast load (A4).

Re-pricing is the **re-solving** idea of network revenue management: bid prices are recomputed from the residual problem. Here the exact LP duals are replaced by a cheap Lagrangian approximation. Section 9 compares the two. Re-pricing changes the decomposition, not the solver, so it helps every batch solver. It also makes batches harder: with near-dual prices many options tie and capacity binds inside the batch.

### A4. Information model and demand margin
Re-pricing needs the remaining demand $\mathcal C^{(t)}$. The code separates **what is known** from **how much is reserved** (`Forecast` in `v3_trajectory.py`).

| Information (`--forecast`) | Remaining cameras used for pricing |
|---|---|
| `oracle` (exact information) | the true remaining cameras (the default of QUBO-v3.1) |
| `twin` (statistics only) | an independent draw of the same size from the same generator (seed + 10007): same priority mix, load distributions and spatial layout, but different individual cameras. They are costed against the real servers with the real normalisation and ordered by their own $p_il_i$. |
| `noise` | the true cameras with each load multiplied by a fixed factor $1+e_i$, $e_i\sim U(-\ell,\ell)$ (`--forecast-level`) |
| bias (`--forecast-bias b`) | every forecast load multiplied by $1+b$: $\tilde l_i=(1+b)l_i$ |

**Demand margin.** A positive bias $b$ is a deliberate **demand margin**: prices are computed as if $100b\%$ more load were still to come. This is needed because the residual problem prices a world in which the remaining cameras will be placed optimally and fractionally. The sequential policy uses capacity less efficiently than that:

- it commits integral batches;
- it never sees later cameras;
- its prices are up to $T_p$ batches old.

Without a margin the prices therefore *underestimate* scarcity, and capacity runs out at the end of the horizon (Section 11 of [`EXPERIMENTS.md`](EXPERIMENTS.md), trajectory figure). The margin is a hedge in the sense of a safety stock.

- **How it was chosen.** $b\in\{0,5,10,15,20,30\}\%$ was tested on validation seeds 100–109, for both kinds of information and at 90, 95 and 98%. $b=5\%$ was best in all 6 settings and was then evaluated once on the test seeds.
- **Exact LP duals need it too.** Under the default weight it improves exact LP bid prices by the same order (Section 9), so it compensates for myopic re-pricing in general, not for the inexact subgradient. Exact duals are more sensitive to the margin, though. Under $w_i=p_i$ at 98% utilisation, LP prices with the margin leave 182 cameras uncovered: the inflated demand exceeds the capacity, and the exact duals rise to the level of the uncovered penalty. The subgradient prices with the same margin stay stable.
- **It interacts with the re-pricing period.** With rarer re-pricing (T_p = 25 or B = 320), 5% is no longer enough. Tune the margin together with $T_p$ and $B$.

---

## 7. QUBO-v3.1: encoding of a batch

Given the candidate sets and decision costs of Section 6, the batch subproblem is turned into a QUBO as follows.

```
fix every camera whose cheapest candidate is on a safe server                      (B1)
repeat up to 3 rounds:                                                              (B4)
    build domain-wall chains (B2) + capacity terms for unsafe servers (B3)
    sample with SQA (or SA); decode wall positions; repair overloads; keep best (B5)
    if the best raw read overloads no server: stop
    raise price mu_j and weight omega_j of every overloaded server
```

### B1. Exact reduction of uncontested cameras (Proposition 2)
A server $j$ is **safe** in batch $t$ if all batch cameras that list it as a candidate fit on it together: $\sum_{i\in\mathcal B_t:\,j\in\mathcal S_i}l_i\le R_j^{(t)}$.

**Proposition 2.** Let $U$ be the cameras whose cheapest candidate $j_i^\ast$ is safe and has $\hat c_{ij_i^\ast}<P$. Some optimal solution of the batch subproblem assigns every $i\in U$ to $j_i^\ast$.

*Proof.* Take an optimal solution and move every $i\in U$ to $j_i^\ast$. Safe servers stay feasible because all their candidates fit together, and loads elsewhere only decrease. No moved camera gets more expensive, because $j_i^\ast$ is its cheapest option and is cheaper than leaving it uncovered. The new solution is feasible and no worse. ∎

Fixed cameras leave the QUBO, and safe servers need no capacity term. At 95% utilisation the median batch keeps only **2 of 80** cameras free under QUBO-v3.1 (7 with static prices), and 22% of batches need no QUBO at all.

### B2. Domain-wall encoding (Proposition 3)
For each free camera, sort its options by decision cost, $o_0,\dots,o_{k-1}$, where $o_{k-1}$ is a dummy "unassigned" option with cost $P$. Normalise $\bar c_{i,m}=(\hat c_{i,o_m}-\hat c_{i,o_0})/\delta$, where $\delta$ is the median gap between the two best options in the batch. Following Chancellor (2019), use $k-1$ binaries $z_{i,1..k-1}$ with fixed ends $z_{i,0}=1$ and $z_{i,k}=0$:

$$
x_{i,o_m}=z_{i,m}-z_{i,m+1},\qquad
H_i^{\rm DW}=\sum_{m=0}^{k-1}\bar c_{i,m}(z_{i,m}-z_{i,m+1})+A_i\sum_{m=1}^{k-2}z_{i,m+1}(1-z_{i,m}).
$$

**Proposition 3.**

- (a) The penalty is zero exactly on non-increasing chains $1\ge z_{i,1}\ge\dots\ge0$. These correspond one-to-one to the $k$ options: the option is the position of the wall.
- (b) The cost part equals $\bar c_{i,0}+\sum_m(\bar c_{i,m}-\bar c_{i,m-1})z_{i,m}$ with non-negative coefficients. So the cheapest option minimises $H_i^{\rm DW}$, and every invalid chain costs at least $\bar c_{i,0}+A_i$.
- (c) Any two options are connected by single flips through valid chains only. Each step changes the energy by the cost difference of adjacent options.

*Proof.*

- (a) $z_{m+1}(1-z_m)=1$ exactly where the chain increases. A chain from 1 to 0 without increases drops exactly once.
- (b) Summation by parts with $z_0=1$, $z_k=0$.
- (c) With the wall at $m$, flipping $z_{m+1}$ from 0 to 1 moves it to $m+1$, and flipping $z_m$ from 1 to 0 moves it to $m-1$. ∎

One-hot needs two flips per move and passes a barrier about the size of the penalty. The domain wall moves between neighbouring options without a barrier and uses one variable fewer per camera. The chain weight is $A=\max\{1,0.45\,q_{0.95}\}$, where $q_{0.95}$ is the 95th percentile of the normalised option costs.

### B3. Slack-free capacity penalty
For an unsafe server $j$, define:

- $a_i=l_i/R_j^{(t)}$, the normalised load of camera $i$;
- $f_j$, the normalised load of the fixed cameras;
- $S_j=f_j+\sum_ia_ix_{ij}$, written in $z$ through B2.

The constraint $S_j\le1$ is penalised in the unbalanced way of Montañez-Barrera et al. (2024), with no slack variables:

$$
H_j^{\rm cap}=W_j\big[(\lambda_1-2)S_j+S_j^2\big]+A\,\mu_jS_j,\qquad W_j=\lambda_{\rm cap}A\,\omega_j,\quad \lambda_1=0.2,\ \lambda_{\rm cap}=1 .
$$

The penalty slightly rewards filling a server up to $S_j=1-\lambda_1/2$. An exact slack-bit encoding removed this bias but sampled worse.

### B4. Adaptive resampling
The penalty is soft, so a read can overload a server. After each round, update every server overloaded by the best raw read ($\rho_j=S_j-1>0$):

- $\mu_j\mathrel{+}=0.5(1+\rho_j)$;
- $\omega_j\mathrel{\times}=2$.

At most 3 rounds are run. The idea follows the augmented Lagrangian method.

### B5. Sampling, decoding, selection
- **Sampler settings.** SQA uses 10 reads, 2000 sweeps, 8 Trotter slices, $\beta=20$ and $\gamma=1$, seeded from the instance seed and the batch index.
- **Decoding.** Each read gives a wall position for each free camera. If a server is still overloaded, cameras are removed in order of increasing value per unit load and re-inserted greedily. Repairs are logged.
- **Selection.** Among all reads of all rounds, the one with the best **priced batch objective** is committed. This is the same objective every other solver optimises.
- **SQA+LS** (`--polish`) also applies shift/swap local search to every decoded read.

### Problem size
A batch QUBO has at most $\sum_{\rm free}(K_w+K_h)$ variables. At 95% utilisation the median QUBO-v3.1 batch has **14 variables** on $20{,}000\times800$ (90th percentile 91, maximum 476) and **7** on $50{,}000\times2{,}000$. The size does not grow with $N$ or $M$. First-generation batches had 1,600 variables and 16,800–80,000 coefficients.

---

## 8. Extension: pairwise anti-affinity costs

Cameras with overlapping fields of view (distance < 16 on the 5,000-camera map) should not share a server, so that one server failure does not blind the same area twice. A cost $\beta_{aa}$ is added for each co-located overlapping pair.

- Pairs that involve a committed camera become linear terms.
- Pairs inside the batch become couplings $\beta_{aa}x_{ij}x_{kj}$.
- Batches are spatial tiles of 80 cameras, with 300–660 pairwise terms each.

Code: `qsa/aa_proto.py`, `build_v3(quad=...)`.

Two safeguards are required:

1. **The wall penalty must dominate the couplings.** Off the valid chains, products of $x=z_m-z_{m+1}$ are unbounded below, because $x$ can be −1. With the default $A$, the samplers reached an energy of −18,000 against −5,000 for the optimum. Fix: $A_i=A+2\sum_k|\beta_{ik}|/\delta$.
2. **The reduction does not apply** to cameras with pairwise terms; they stay free. In addition, OpenJij's fixed-temperature schedule needs the stiffer QUBO rescaled so that the 95th percentile of $|Q|$ becomes 0.7.

Results are in [`EXPERIMENTS.md`](EXPERIMENTS.md). On classical hardware the QUBO shows no advantage on this extension either.

---

## 9. Baselines

### 9.1 Batch solvers (same coordination layer, different step 3)

| Method | Description | Code |
|---|---|---|
| Greedy | each camera, in priority order, takes its cheapest feasible candidate | `solve_greedy` |
| Regret | repeatedly assign the camera with the largest gap between its best and second-best feasible option | `solve_regret` |
| ILS | iterated local search per batch: regret + shift/swap LS, then destroy 20%, repair greedily, LS; 0.8 s per batch | `solve_ils` |
| Exact batch MILP | batch subproblem solved to optimality with HiGHS (SciPy `milp`) | `solve_exact` |
| SA-v3.1 | QUBO-v3.1 Hamiltonian sampled with Neal | `solve_v3(..., "SA")` |
| SQA-v3.1 | QUBO-v3.1 Hamiltonian sampled with OpenJij SQA | `solve_v3(..., "SQA")` |
| SQA-v3.1+LS | SQA-v3.1 with shift/swap LS on decoded reads | `--polish` |

The exact batch MILP is the best any batch solver can do for each batch **in isolation**. It does not guarantee the best trajectory, because a batch decision changes the residual problem on which later prices are computed.

### 9.2 Coordination policies (same batches, exact batch solver)

| Policy | Prices | Demand information | Code |
|---|---|---|---|
| Static prices | Lagrangian prices of the full problem, computed once | full, at $t=0$ | `v3_trajectory.py --price` (no `--reprice`) |
| **Lagrangian re-pricing (ours)** | subgradient on the residual problem every $T_p$ batches, optional margin $b$ | exact or statistics only | `--reprice 5 --k-slack 2 [--forecast twin] [--forecast-bias 0.05]` |
| LP bid prices | exact duals of the LP relaxation of the residual problem (HiGHS), every $T_p$ batches, optional margin | exact | `online_baselines.py --policy lp [--forecast-bias 0.05]` |
| Dual mirror descent (DMD) | after every batch, $u\leftarrow[u+\eta(g-\rho)]_+$ with consumption $g$ and target rate $\rho_j=K_j/T$ (Balseiro, Lu & Mirrokni 2023) | none | `--policy dmd --eta {0.1,1} --dmd-init {zero,lagrange}` |
| Rolling-horizon MILP | batch $t$ optimised jointly with the next 2 batches (true future cameras); only batch $t$ is committed; no prices | the next 160 cameras | `--policy lookahead --lookahead 2` |
| Offline solution | Section 3 | everything, revisable | `global_bound_check.py`, `offline_global.py` |

Without a margin, the LP bid prices are the quality ceiling of price-only coordination. The Lagrangian subgradient approximates the same duals at 1/8–1/40 of the cost. With the 5% margin, the two are within 0.2 pp under the default weight. Under $w_i=p_i$, LP prices without a margin are 0.15–0.2 pp better at 90–95%, but LP prices with the margin break down at 98% (Section A4). Earlier metaheuristics in the shared-window setting (GRASP, BRKGA, tabu search, capacity-priced greedy) are in `baselines/` and `results/legacy/`.

---

## 10. Instance variants

| Variant | What changes | Option |
|---|---|---|
| Utilisation | 75, 90, 95, 98% | `--target-util` |
| Clustered city layout | 12 urban centres (seed + 555) with σ = 40–120 and Dirichlet weights. 70% of the cameras are clustered (seed + 556); 60% of the servers are clustered with spread 1.5σ (seed + 557); the rest are uniform; points are reflected at the boundary. Priorities, loads and capacities are unchanged; the LB is recomputed. | `--layout clustered` |
| Priority weight | $w_i=p_i$ instead of $4-p_i$: high priority counts more in the objective; the LB is recomputed (`lb_prio.py`) | `--weight prio` |
| Scale | $5{,}000\times200$, $20{,}000\times800$, $50{,}000\times2{,}000$ | `--n-cameras`, `--n-servers` |
| Batch size, re-pricing period | B = 40–320; $T_p$ = 1–25 | `--batch-size`, `--reprice` |

---

## 11. Limitations and lessons

**Limitations.**

- Instances are synthetic, although two layouts are tested.
- Distance is a latency proxy, and bandwidth is not constrained.
- The camera order is fixed, and committed batches are never revised.
- SQA runs only as a CPU emulation; no physical annealer was used.
- The demand margin is a tuned constant. Under both weights 5% is best or nearly best on the validation seeds, but it must be tuned jointly with the re-pricing period. A margin that adapts to the remaining horizon is future work.
- The objective weight is a policy choice. Under $w_i=4-p_i$, high-priority cameras are moved farther when capacity is scarce; under $w_i=p_i$ they are not. Every conclusion holds under both weights; the size of the effect differs (Section 10 of `EXPERIMENTS.md`).

**What SQA does and does not show.** On classical hardware SQA does not outperform exact batch optimisation, ILS or SA. The value of the QUBO side is a compact, well-conditioned batch model with proven properties: a median of 14 variables, an exact reduction and a barrier-free encoding. With local search, annealers sample this model to batch optimality, and it is ready for physical Ising machines.

**Lessons for decomposed QUBO models.**

1. Check that the penalised ground state is feasible (Proposition 1). With pairwise terms, the wall penalty must also dominate the couplings.
2. Compare against an exact batch optimum *and* a global bound. Otherwise losses caused by the decomposition are blamed on the solver.
3. Coordinate the subproblems. Prices of the residual problem carry information about future batches at negligible cost. Inflate the remaining demand slightly, because myopic re-pricing underestimates scarcity.
4. Remove easy decisions before sampling (Proposition 2).
5. Compare solvers on the same objective. Ranking reads by a different cost measures the ranking rule, not the solver.

---

## 12. Parameters

| Parameter | Value | Role |
|---|---|---|
| Batch size $B$ | 80 | cameras per committed decision |
| Windows $K_w$, $K_h$ | 5, 2 | cheapest and headroom candidates per camera |
| Uncovered penalty $P$ | 15 | objective |
| Lagrangian iterations | 300 (prices), 400 (bound) | subgradient ascent |
| Re-pricing period $T_p$ | 5 batches | 40 cheapest servers per camera |
| Demand margin $b$ | 5% | remaining demand in re-pricing, ×1.05 |
| Twin forecast | independent draw, seed + 10007 | statistics-only information |
| Domain-wall weight $A$ | $\max\{1,0.45\,q_{0.95}\}$ | chain penalty |
| Capacity penalty | $\lambda_1=0.2$, $\lambda_{\rm cap}=1$ | unbalanced penalisation |
| Adaptive rounds | 3; $\mu\mathrel{+}=0.5(1+\rho)$, $\omega\mathrel{\times}=2$ | resampling |
| SQA | 10 reads, 2000 sweeps, $P_T=8$, $\beta=20$, $\gamma=1$ | OpenJij 0.11.6 |
| SA | 10 reads, 2000 sweeps | Neal 0.5.7 |
| ILS | 0.8 s per batch, destroy 20% | metaheuristic baseline |
| LP bid prices | HiGHS `linprog`, 40 cheapest servers per camera | baseline |
| DMD | $\eta\in\{0.1,1\}$ (in units of the mean initial price), zero or Lagrangian start | baseline |

**References.**

- Ross & Soland (1975), *Math. Prog.* 8.
- Fisher (1981), *Manag. Sci.* 27.
- Yagiura, Ibaraki & Glover (2004), *INFORMS J. Comput.* 16.
- Lucas (2014), *Front. Phys.* 2:5.
- Glover, Kochenberger & Du (2019), *4OR* 17.
- Chancellor (2019), *Quantum Sci. Technol.* 4, 045004.
- Montañez-Barrera et al. (2024), *Quantum Sci. Technol.* 9.
- Talluri & van Ryzin (1998), *Manag. Sci.* 44 (bid prices).
- Balseiro, Lu & Mirrokni (2023), *Oper. Res.* 71 (dual mirror descent).
- Lourenço, Martin & Stützle (2003), *Handbook of Metaheuristics*.
- Huangfu & Hall (2018), *Math. Prog. Comp.* 10.
- Wilcoxon (1945); Holm (1979); Kerby (2014).
