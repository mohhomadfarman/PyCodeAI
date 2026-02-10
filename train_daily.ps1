# PyCodeAI Daily Training Script
# This script automates fine-tuning by loading your latest model and training on current data.

# Configuration
$MODEL_NAME = "model.npz"
$BATCH_SIZE = 64
$EPOCHS = 5
$LEARNING_RATE = "1e-4"

Write-Host "--- Starting Daily Fine-Tuning ---" -ForegroundColor Cyan

# 1. Activate Environment
if (Test-Path "venv\Scripts\activate.ps1") {
    . venv\Scripts\activate.ps1
    Write-Host "[OK] Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "[Error] Virtual environment not found!" -ForegroundColor Red
    exit
}

# 2. Prepare Training Arguments
$TRAIN_ARGS = @(
    "cli.py", "train",
    "--epochs", "$EPOCHS",
    "--batch-size", "$BATCH_SIZE",
    "--learning-rate", "$LEARNING_RATE",
    "--device", "gpu"
)

# 3. Check for existing model and add to args
if (Test-Path $MODEL_NAME) {
    $TRAIN_ARGS += "--load-model"
    $TRAIN_ARGS += $MODEL_NAME
    Write-Host "[Info] Found existing model. Continuing training..." -ForegroundColor Yellow
} else {
    Write-Host "[Info] No existing model found. Starting fresh." -ForegroundColor Yellow
}

# 4. Run Training
Write-Host "[Run] Executing training loop..." -ForegroundColor White
python @TRAIN_ARGS

Write-Host "--- Training Complete ---" -ForegroundColor Cyan
