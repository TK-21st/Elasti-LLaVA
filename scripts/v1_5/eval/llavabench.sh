#!/bin/bash

LLAVA_DIR=/grid/neuroai/home/tiliu/data/LLava-Eval/llava-bench-in-the-wild
OUT_FILENAME=llava-v1.5-7b.jsonl
# MODEL_NAME=llava-v1.5-7b-mlp2x_gelu-top550_bsz32-epc1-lr1e-4
# STEP=4000
# CKPT_DIR="/grid/neuroai/home/tiliu/experiments/flexformer-llava/${MODEL_NAME}/checkpoint-${STEP}"
# OUT_FILENAME="${MODEL_NAME}_checkpoint-${STEP}.jsonl"


# python -m llava.eval.model_vqa \
#     --model-path /grid/neuroai/home/tiliu/model/llava-v1.5-7b \
#     --mm-router-ckpt-dir $CKPT_DIR \
#     --question-file $LLAVA_DIR/questions.jsonl \
#     --image-folder $LLAVA_DIR/images \
#     --answers-file $LLAVA_DIR/answers/$OUT_FILENAME \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p $LLAVA_DIR/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question $LLAVA_DIR/questions.jsonl \
    --context $LLAVA_DIR/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        $LLAVA_DIR/answers_gpt4.jsonl \
        $LLAVA_DIR/answers/$OUT_FILENAME \
    --output \
        $LLAVA_DIR/reviews/$OUT_FILENAME

python llava/eval/summarize_gpt_review.py -f $LLAVA_DIR/reviews/$OUT_FILENAME
