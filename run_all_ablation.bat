@echo off
cd /d E:\下次比赛项目\face_recognition_research
echo === Starting 6-config ablation training at %date% %time% ===

python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --lr 1e-4 --grad_clip 1.0 --num_workers 2 --save_dir weights/ablation_casia_b0 --configs 0
python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --lr 1e-4 --grad_clip 1.0 --num_workers 2 --save_dir weights/ablation_casia_b1 --configs 1
python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --lr 1e-4 --grad_clip 1.0 --num_workers 2 --save_dir weights/ablation_casia_b2 --configs 2
python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --lr 1e-4 --grad_clip 1.0 --num_workers 2 --save_dir weights/ablation_casia_b3 --configs 3
python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --lr 1e-4 --grad_clip 1.0 --num_workers 2 --save_dir weights/ablation_casia_b4 --teacher_cache weights/ablation_casia/teacher_feats.npy --configs 4
python scripts/train_ablation.py --data datasets/raw/casia-webface --gpu 0 --epochs 30 --batch_size 256 --lr 1e-4 --grad_clip 1.0 --num_workers 2 --save_dir weights/ablation_casia_b5 --teacher_cache weights/ablation_casia/teacher_feats.npy --configs 5

echo === All training done at %date% %time% ===
