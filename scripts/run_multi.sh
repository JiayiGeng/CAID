#!/bin/bash
WORKING_DIR=$(pwd)
export JUDGE_PYTHON="/Users/jiayigeng/miniconda3/bin/python"

# ===================== Common Parameters =====================
task="paperbench"  # "commit0" or "paperbench"
# model="litellm_proxy/fireworks_ai/accounts/fireworks/models/minimax-m2p5"
model="litellm_proxy/fireworks_ai/accounts/fireworks/models/glm-4p7"
subagent_model=""  # leave empty to use the same model
max_iterations=50
max_subagents=2
sub_iterations=80
rounds_of_chat=2

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
optional_flags=""
if [ -n "$subagent_model" ]; then
    optional_flags="$optional_flags --subagent_model=$subagent_model"
fi
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
        echo "Subagent iterations: $sub_iterations"
        echo "Max subagents: $max_subagents"
        echo "Rounds of chat: $rounds_of_chat"
        echo "========================================"
        echo ""

        uv run python run_infer.py \
            --task "$task" \
            --repo "$repo" \
            --max_iterations "$max_iterations" \
            --max_subagents "$max_subagents" \
            --sub_iterations "$sub_iterations" \
            --rounds_of_chat "$rounds_of_chat" \
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
        if [ -n "$subagent_model" ]; then
            echo "Subagent Model: $subagent_model"
        fi
        echo "Manager iterations: $max_iterations"
        echo "Subagent iterations: $sub_iterations"
        echo "Max subagents: $max_subagents"
        echo "Rounds of chat: $rounds_of_chat"
        echo "========================================"
        echo ""

        uv run python run_infer.py \
            --task "$task" \
            --paper_id "$paper_id" \
            --max_iterations "$max_iterations" \
            --max_subagents "$max_subagents" \
            --sub_iterations "$sub_iterations" \
            --rounds_of_chat "$rounds_of_chat" \
            --model "$model" \
            $optional_flags
    done
fi
