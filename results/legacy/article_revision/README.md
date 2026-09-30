# Article Revision Logs

This directory is reserved for the cleaned set of runs intended for the revised manuscript.

The older historical logs are not moved or deleted. The summary builder reads the selected
project log directories and writes reproducible tables here.

## Recommended PRC-QUBO-C Runs

Run the full SQA stress curve from the project root:

```powershell
.\scripts\run_prc_qubo_c_article_suite.ps1 -RunStressCurve
```

Run the 95.07% utilization ablation:

```powershell
.\scripts\run_prc_qubo_c_article_suite.ps1 -RunAblation95
```

Run the conflict-coupled QUBO with the original decoder across the full capacity-stress curve:

```powershell
.\scripts\run_prc_qubo_c_article_suite.ps1 -RunConflictOnlyStressCurve
```

Run both:

```powershell
.\scripts\run_prc_qubo_c_article_suite.ps1 -RunAll
```

Alternatively, double-click or run:

```powershell
.\scripts\run_prc_qubo_c_article_suite.bat
```

For the conflict-coupled/original-decoder stress curve only, double-click or run:

```powershell
.\scripts\run_prc_qubo_c_conflict_only_stress.bat
```

The suite uses:

- `n_cameras=20000`
- `n_servers=800`
- `batch_size=80`
- `max_servers_per_batch=20`
- `seed=42`
- `solver=SQA`
- `capacity_scale in {1.00, 0.50, 0.33, 0.25}`

## Summary Tables

After runs complete, rebuild the tables:

```powershell
.\.venv\Scripts\python.exe scripts\build_article_revision_summary.py --output-dir logs_article_revision
```

If the local virtual environment is unavailable, the summary builder can also be run with any Python
that has the standard library available because it only parses JSON and writes CSV/Markdown.
