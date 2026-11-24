# R8: 학습이 끝난 후보·대조군 체크포인트를 서비스 형식(ONNX fp32)으로 내보낸다. 서비스 모델 폴더는 건드리지 않는다.
# 사용: powershell -File export_r8_all.ps1
$py = "C:\Users\<user>\venvs\gabee-ner\Scripts\python.exe"
$repo = "C:\Users\<user>\Documents\gabee-ner"
$runs = "C:\Users\<user>\venvs\ner-runs"
$out = "$runs\r8\export"
$env:HF_HOME = "C:\Users\<user>\venvs\hf-cache"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($prefix in "r8", "r8ctl") {
  foreach ($s in 7, 13, 21, 42, 99) {
    $name = "$prefix-koelectra-small-klue10-s$s"
    if (-not (Test-Path "$runs\$name\best")) { Write-Output "SKIP $name (체크포인트 없음)"; continue }
    & $py "$repo\ml\ner\export\export_onnx.py" --run-dir "$runs\$name" --out "$out\$name" --no-int8 2>&1 | Select-Object -Last 2
    $fp = (Get-FileHash "$out\$name\model.onnx" -Algorithm SHA256).Hash.Substring(0, 12).ToLower()
    Write-Output "$name onnx sha12=$fp"
  }
}
