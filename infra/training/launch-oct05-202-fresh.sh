#!/bin/bash
set -euo pipefail
cd /mnt/pi05/lerobot
export HF_HOME=/mnt/pi05/hf-cache TMPDIR=/mnt/pi05/tmp TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PI05_KEEP_CHECKPOINTS=2
export PI05_TRAINING_MANIFEST_SHA256=041690aea82ae8658c870159d12ba33717b2312a5683c3237f659764c6651b54
exec /mnt/pi05/lerobot/.venv/bin/python -u /mnt/pi05/train_panthera_oct05_202_fresh.py \
  --dataset.repo_id=FoxNerdSaysMoo/panthera-gear-carrier-pin-oct05-original-camera-20hz-202 \
  --dataset.root=/mnt/pi05/dataset_oct05_original_camera_202 \
  '--dataset.episodes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201]' \
  --dataset.eval_split=0.0 \
  --dataset.use_imagenet_stats=false \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=/mnt/pi05/base \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.use_relative_actions=false \
  --policy.empty_cameras=0 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  '--policy.normalization_mapping={"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_warmup_steps=1000 \
  --accelerator.gradient_accumulation.steps=1 \
  --accelerator.mixed_precision=bf16 \
  --output_dir=/mnt/pi05/runs/panthera_pi05_oct05_202_fresh_20261008 \
  --job_name=panthera_pi05_oct05_202_fresh_20261008 \
  --batch_size=64 \
  --num_workers=8 \
  --prefetch_factor=2 \
  --steps=30000 \
  --save_freq=500 \
  --env_eval_freq=0 \
  --log_freq=20 \
  --seed=20261008 \
  --wandb.enable=true \
  --wandb.project=panthera-pi05 \
  --wandb.disable_artifact=true
