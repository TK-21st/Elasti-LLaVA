#!/bin/bash
LLAVA_DIR=/grid/neuroai/home/tiliu/data/LLava-Eval/llava-bench-in-the-wild
# OUT_FILENAME=llava-v1.5-7b.jsonl
for FULL_PATH in /grid/neuroai/home/tiliu/experiments/flexformer-llava/llava-v1.5-7b-*-top*_bsz32-epc1-lr1e-4-DataSample12K/checkpoint-330; do



    MODEL_NAME=$(basename $(dirname "$FULL_PATH"))
    STEP=$(basename "$FULL_PATH" | cut -d'-' -f2)

    CKPT_DIR="/grid/neuroai/home/tiliu/experiments/flexformer-llava/${MODEL_NAME}/checkpoint-${STEP}"
    OUT_FILENAME="${MODEL_NAME}_checkpoint-${STEP}.jsonl"

    echo $FULL_PATH to $OUT_FILENAME

    python -m llava.eval.model_vqa \
        --model-path /grid/neuroai/home/tiliu/model/llava-v1.5-7b \
        --mm-router-ckpt-dir $CKPT_DIR \
        --question-file $LLAVA_DIR/questions.jsonl \
        --image-folder $LLAVA_DIR/images \
        --answers-file $LLAVA_DIR/answers/$OUT_FILENAME \
        --temperature 0 \
        --conv-mode vicuna_v1

    mkdir -p $LLAVA_DIR/reviews

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
done
