#!/bin/bash

# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path liuhaotian/llava-v1.5-13b \
#     --version v1 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-v1.5-13b-task \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb

export WANDB_PROJECT="Distill-LLaVA"
EXPERIMENT="distill-debub-7b-linear-1e-4"

python llava/train/distill.py \
    --model_name_or_path /grid/neuroai/home/tiliu/model/llava-v1.5-7b \
    --version v1 \
    --data_path /grid/neuroai/home/tiliu/data/LLaVA-Instruct-150K/llava_v1_5_mix665k_image_found-coco.json \
    --image_folder /grid/neuroai/home/tiliu/data/LLaVA-Instruct-150K/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_router_type mlp2x_gelu \
    --mm_router_top_k 256 \
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
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
