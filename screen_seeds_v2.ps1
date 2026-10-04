# 继续筛种子：cfg3/cfg4 × seed 123/456/789/2024，各跑 5 epoch
cd E:\下次比赛项目\face_recognition_research
$env:PYTHONIOENCODING = "utf-8"

$configs = @(3, 4)
$seeds = @(123, 456, 789, 2024)
$logDir = "logs_multiseed\screening"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($cfg in $configs) {
    foreach ($seed in $seeds) {
        $logFile = "$logDir\cfg${cfg}_seed${seed}_5ep.txt"
        if (Test-Path $logFile) {
            $result = Get-Content $logFile | Select-String -Pattern "RESULT"
            if ($result) {
                Write-Host "跳过已完成: cfg=$cfg seed=$seed" -ForegroundColor Gray
                continue
            }
        }
        Write-Host "=== 筛选 cfg=$cfg seed=$seed (5 epoch) ===" -ForegroundColor Yellow
        python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 5 --batch_size 256 --save_dir weights/ablation_casia --teacher_cache weights/ablation_casia/teacher_feats.npy --configs $cfg --seed $seed --num_workers 2 2>&1 | Tee-Object -FilePath $logFile
        Write-Host "=== 完成 cfg=$cfg seed=$seed ===" -ForegroundColor Yellow
    }
}

Write-Host "=== cfg3/cfg4 筛选完成 ===" -ForegroundColor Cyan
