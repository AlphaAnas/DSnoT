# Debug script for DSnoT pruning on Windows
# This PowerShell script will run the main.py with debugging capabilities

# Set environment variables for CUDA
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"

Write-Host "Setting up debugging environment..." -ForegroundColor Green
Write-Host "CUDA_VISIBLE_DEVICES: $env:CUDA_VISIBLE_DEVICES" -ForegroundColor Yellow

# Change to the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow

# Define the command arguments
$pythonArgs = @(
    "main.py"
    "--model", "babylm/babyllama-10m-2024"
    "--prune_method", "DSnoT"
    "--initial_method", "wanda"
    "--sparsity_ratio", "0.5"
    "--sparsity_type", "unstructured"
    "--max_cycle_time", "50"
    "--update_threshold", "0.1"
    "--pow_of_var_regrowing", "1"
    "--debug"  # Enable debugging
)

Write-Host "`nRunning command:" -ForegroundColor Green
Write-Host "python $($pythonArgs -join ' ')" -ForegroundColor Cyan
Write-Host ("-" * 80) -ForegroundColor Gray

try {
    # Run the Python script
    & python @pythonArgs
    Write-Host "`nScript completed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "`nError occurred: $($_.Exception.Message)" -ForegroundColor Red
}

# Keep the window open
Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
