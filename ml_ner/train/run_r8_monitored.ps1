# R8 학습 실행기: train_ner.py 를 서브프로세스로 돌리며 RAM(작업 집합)과 GPU 메모리를 3초마다 기록하고 최댓값을 보고한다.
# 사용: powershell -File run_r8_monitored.ps1 -RunId r8-pilot-s7 -Seed 7 -Epochs 1 [-NoSave]
# 실패하면 재시작하지 않고 종료 코드와 로그 위치만 남긴다(사전 등록 §3).
param(
  [Parameter(Mandatory = $true)][string]$RunId,
  [Parameter(Mandatory = $true)][int]$Seed,
  [int]$Epochs = 8,
  [string]$TrainExtra = "r8_train_extra",
  [string]$ValExtra = "r8_val_extra",
  [switch]$NoSave
)
$py = "C:\Users\<user>\venvs\gabee-ner\Scripts\python.exe"
$repo = "C:\Users\<user>\Documents\gabee-ner"
$dataDir = "C:\Users\<user>\venvs\ner-runs\data_r8"
$logDir = "C:\Users\<user>\venvs\ner-runs\r8\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:HF_HOME = "C:\Users\<user>\venvs\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:PYTHONIOENCODING = "utf-8"
$trainFiles = @("baseline_train", "synth_train")
if ($TrainExtra -ne "none") { $trainFiles += $TrainExtra }
$valFiles = @("baseline_val", "synth_val")
if ($ValExtra -ne "none") { $valFiles += $ValExtra }
$args = @("$repo\ml\ner\train\train_ner.py", "--model", "monologg/koelectra-small-v3-discriminator", "--run-id", $RunId,
  "--train-files") + $trainFiles + @("--val-files") + $valFiles +
  @("--klue-ratio", "0.1", "--max-len", "128", "--batch-size", "32", "--lr", "5e-5", "--epochs", "$Epochs", "--patience", "2",
    "--warmup", "0.1", "--weight-decay", "0.01", "--seed", "$Seed", "--select-on", "baseline_val", "--select-metric", "overlap",
    "--data-dir", $dataDir)
if ($NoSave) { $args += "--no-save-model" }
$out = "$logDir\$RunId.out.log"; $err = "$logDir\$RunId.err.log"
$t0 = Get-Date
$p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $repo -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -NoNewWindow
$maxRam = 0; $maxGpu = 0; $minFree = 1e9
while (-not $p.HasExited) {
  Start-Sleep -Seconds 3
  try { $p.Refresh(); $ws = [math]::Round($p.WorkingSet64 / 1MB); if ($ws -gt $maxRam) { $maxRam = $ws } } catch {}
  $g = (& nvidia-smi "--query-gpu=memory.used" "--format=csv,noheader,nounits" 2>$null | Select-Object -First 1)
  if ($g -match '^\d+') { $gi = [int]$g; if ($gi -gt $maxGpu) { $maxGpu = $gi } }
  $free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2); if ($free -lt $minFree) { $minFree = $free }
}
$p.WaitForExit()
$sec = [math]::Round(((Get-Date) - $t0).TotalSeconds)
$summary = [ordered]@{ run_id = $RunId; exit_code = $p.ExitCode; seconds = $sec; max_process_ram_mb = $maxRam; max_gpu_used_mb = $maxGpu; min_system_free_ram_gb = $minFree; log_out = $out; log_err = $err }
$summary | ConvertTo-Json | Set-Content -Encoding utf8 "$logDir\$RunId.monitor.json"
$summary | ConvertTo-Json
