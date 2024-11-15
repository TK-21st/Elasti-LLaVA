#!/bin/bash
#SBATCH --job-name=sweep_llava_distill
#SBATCH --output=/grid/neuroai/home/tiliu/experiments/flexformer-llava/slurm_logs/sweep_llava_distill_%A_%a.out
#SBATCH --error=/grid/neuroai/home/tiliu/experiments/flexformer-llava/slurm_logs/sweep_llava_distill_%A_%a.err
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq

top_k_all=(-1 550 512 500 490 450 400 350 300 256 200 150 128 64 32 16 8 4 2 1)
top_k=${top_k_all[$SLURM_ARRAY_TASK_ID]}


export WANDB_PROJECT="Distill-LLaVA"
ROUTER_TYPE=mlp2x_gelu
LR=1e-4
EXPERIMENT="llava-v1.5-7b-${ROUTER_TYPE}-top${top_k}_bsz32-epc1-lr${LR}-DataSample12K"

python llava/train/distill.py \
    --model_name_or_path /grid/neuroai/home/tiliu/model/llava-v1.5-7b \
    --version v1 \
    --data_path /grid/neuroai/home/tiliu/data/LLaVA-Instruct-150K/llava_v1_5_mix665k_image_found-sample-10K.json \
    --image_folder /grid/neuroai/home/tiliu/data/LLaVA-Instruct-150K/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_router_type $ROUTER_TYPE \
    --mm_router_top_k $top_k \
    --kl_type forward \
    --use_top_k False \
    --logits_top_k 50 \
    --lm_loss_weight 0. \
    --router_load_balance_loss_weight 0.1 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /grid/neuroai/home/tiliu/experiments/flexformer-llava/$EXPERIMENT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 10 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $EXPERIMENT
