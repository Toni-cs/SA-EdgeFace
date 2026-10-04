# 多 seed 训练脚本：cfg0/cfg3/cfg4 各 3 个 seed
cd E:\下次比赛项目\face_recognition_research
$env:PYTHONIOENCODING = "utf-8"

$configs = @(0, 3, 4)
$seeds = @(42, 123, 456)
$logDir = "logs_multiseed"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($cfg in $configs) {
    foreach ($seed in $seeds) {
        $logFile = "$logDir\cfg${cfg}_seed${seed}.txt"
        Write-Host "=== 开始 cfg=$cfg seed=$seed ===" -ForegroundColor Green
        python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --save_dir weights/ablation_casia --teacher_cache weights/ablation_casia/teacher_feats.npy --configs $cfg --seed $seed 2>&1 | Tee-Object -FilePath $logFile
        Write-Host "=== 完成 cfg=$cfg seed=$seed ===" -ForegroundColor Green
    }
}

Write-Host "=== 全部多 seed 训练完成 ===" -ForegroundColor Cyan
