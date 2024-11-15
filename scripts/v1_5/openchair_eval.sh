#!/bin/bash
#SBATCH --job-name=chair_eval
#SBATCH --output=/grid/neuroai/home/tiliu/experiments/flexformer-llava/slurm_logs/chair_eval_%A_%a.out
#SBATCH --error=/grid/neuroai/home/tiliu/experiments/flexformer-llava/slurm_logs/chair_eval_%A_%a.err
#SBATCH --array=0-1%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq

MODELS_ALL=(
    llava-v1.5-7b-linear-top128_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top150_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top16_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top-1_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top1_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top200_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top256_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top2_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top300_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top32_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top350_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top400_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top450_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top490_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top4_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top500_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top512_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top550_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top64_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-linear-top8_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top128_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top150_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top16_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top-1_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top1_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top200_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top256_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top2_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top300_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top32_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top350_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top400_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top450_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top490_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top4_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top500_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top512_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top550_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top64_bsz32-epc1-lr1e-4-DataSample12K
    llava-v1.5-7b-mlp2x_gelu-top8_bsz32-epc1-lr1e-4-DataSample12K
)


LLAVA_DIR=/grid/neuroai/home/tiliu/data/LLava-Eval/OpenCHAIR/data
MODEL_NAME=${MODELS_ALL[$SLURM_ARRAY_TASK_ID]}
STEP=330
ANS_FILENAME="${LLAVA_DIR}/answers/${MODEL_NAME}_checkpoint-${STEP}.jsonl"
OUT_FILENAME="${LLAVA_DIR}/answers/${MODEL_NAME}_checkpoint-${STEP}.score.txt"
echo $ANS_FILENAME

python OpenCHAIR/evaluate.py \
    --llm-ckpt meta-llama/Llama-2-70b-chat-hf  \
    --concreteness-dataset-path /grid/neuroai/home/tiliu/data/LLava-Eval/OpenCHAIR/concreteness/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx \
    --generations-file-path $ANS_FILENAME \
    --batch-size 64
