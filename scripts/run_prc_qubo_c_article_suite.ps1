param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$LogRoot = "logs_article_revision",
    [int]$LogEvery = 10,
    [switch]$RunStressCurve,
    [switch]$RunConflictOnlyStressCurve,
    [switch]$RunAblation95,
    [switch]$RunAll
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $true
}

if ($RunAll) {
    $RunStressCurve = $true
    $RunAblation95 = $true
}

if (-not $RunStressCurve -and -not $RunConflictOnlyStressCurve -and -not $RunAblation95) {
    Write-Host "Nothing selected. Use -RunStressCurve, -RunConflictOnlyStressCurve, -RunAblation95, or -RunAll."
    exit 0
}

& $Python -c "import sys; print(sys.executable); import numpy; import neal; import openjij; print('openjij', getattr(openjij, '__version__', 'version unavailable'))"
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency check failed."
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Run-Step {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )
    Write-Host ""
    Write-Host "================================================================================"
    Write-Host $Name
    Write-Host "================================================================================"
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $Name"
    }
}

if ($RunStressCurve) {
    foreach ($Scale in @("1.00", "0.50", "0.33", "0.25")) {
        Run-Step "PRC-QUBO-C + SQA stress curve, capacity_scale=$Scale" @(
            "prc_qubo_conflict_stress.py",
            "--solver", "SQA",
            "--capacity-scale", $Scale,
            "--log-root", $LogRoot,
            "--log-every", "$LogEvery",
            "--method-label", "PRC-QUBO-C"
        )
    }
}

if ($RunConflictOnlyStressCurve) {
    foreach ($Scale in @("1.00", "0.50", "0.33", "0.25")) {
        Run-Step "PRC-QUBO-C conflict terms with original decoder, capacity_scale=$Scale" @(
            "prc_qubo_conflict_stress.py",
            "--solver", "SQA",
            "--capacity-scale", $Scale,
            "--log-root", $LogRoot,
            "--log-every", "$LogEvery",
            "--method-label", "PRC-QUBO-C-no-decoder",
            "--plain-decoder"
        )
    }
}

if ($RunAblation95) {
    Run-Step "Ablation: PRC-QUBO + capacity-aware decoder only at 95.07% utilization" @(
        "prc_qubo_conflict_stress.py",
        "--solver", "SQA",
        "--capacity-scale", "0.25",
        "--log-root", $LogRoot,
        "--log-every", "$LogEvery",
        "--method-label", "PRC-QUBO-D",
        "--no-conflict"
    )

    Run-Step "Ablation: PRC-QUBO-C conflict terms with original decoder at 95.07% utilization" @(
        "prc_qubo_conflict_stress.py",
        "--solver", "SQA",
        "--capacity-scale", "0.25",
        "--log-root", $LogRoot,
        "--log-every", "$LogEvery",
        "--method-label", "PRC-QUBO-C-no-decoder",
        "--plain-decoder"
    )

    Run-Step "Ablation: full PRC-QUBO-C at 95.07% utilization" @(
        "prc_qubo_conflict_stress.py",
        "--solver", "SQA",
        "--capacity-scale", "0.25",
        "--log-root", $LogRoot,
        "--log-every", "$LogEvery",
        "--method-label", "PRC-QUBO-C"
    )
}

& $Python "scripts\build_article_revision_summary.py" --output-dir $LogRoot
