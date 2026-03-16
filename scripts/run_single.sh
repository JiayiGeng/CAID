#!/bin/bash
WORKING_DIR=$(pwd)
export JUDGE_PYTHON="/Users/jiayigeng/miniconda3/bin/python"

# ===================== Common Parameters =====================
task="paperbench"  # "commit0" or "paperbench"
# model="litellm_proxy/fireworks_ai/accounts/fireworks/models/minimax-m2p5"
model="litellm_proxy/fireworks_ai/accounts/fireworks/models/glm-4p7"
max_iterations=100

# ===================== Commit0 Parameters =====================
repo="minitorch"
dataset_path=""

# ===================== PaperBench Parameters =====================
paper_id="rice"
paperbench_dir=""
test_max_depth=999
test_reproduce_timeout=300
judge_type="simple"
judge_model="azure_ai/gpt-5-mini"
code_dev=true

# ===================== Build Flags =====================
optional_flags="--nomulti_agent"
if [ -n "$dataset_path" ]; then
    optional_flags="$optional_flags --dataset_path=$dataset_path"
fi
if [ -n "$paperbench_dir" ]; then
    optional_flags="$optional_flags --paperbench_dir=$paperbench_dir"
fi

# PaperBench judge flags (always passed for paperbench)
if [ "$task" = "paperbench" ]; then
    optional_flags="$optional_flags --test_max_depth=$test_max_depth --test_reproduce_timeout=$test_reproduce_timeout --judge_type=$judge_type --judge_model=$judge_model"
    if [ "$code_dev" = "true" ]; then
        optional_flags="$optional_flags --code_dev"
    else
        optional_flags="$optional_flags --nocode_dev"
    fi
fi

# ===================== Run =====================
if [ "$task" = "commit0" ]; then
    repos=(
        "minitorch"
    )

    for repo in "${repos[@]}"; do
        echo "========================================"
        echo "Task: $task"
        echo "Repository: $repo"
        echo "Model: $model"
        echo "Manager iterations: $max_iterations"
        echo "Mode: single-agent"
        echo "========================================"
        echo ""

        uv run python run_infer.py \
            --task "$task" \
            --repo "$repo" \
            --max_iterations "$max_iterations" \
            --model "$model" \
            $optional_flags
    done

elif [ "$task" = "paperbench" ]; then
    papers=(
        "sequential-neural-score-estimation"
    )

    for paper_id in "${papers[@]}"; do
        echo "========================================"
        echo "Task: $task"
        echo "Paper ID: $paper_id"
        echo "Model: $model"
        echo "Manager iterations: $max_iterations"
        echo "Mode: single-agent"
        echo "========================================"
        echo ""

        uv run python run_infer.py \
            --task "$task" \
            --paper_id "$paper_id" \
            --max_iterations "$max_iterations" \
            --model "$model" \
            $optional_flags
    done
fi
