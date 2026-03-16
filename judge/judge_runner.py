"""
python -m judge.judge_runner
"""
import asyncio
import json
import os
from pathlib import Path
import fire
from litellm import cost_per_token
from paperbench.judge.create_judge import create_judge, handle_judge_kwargs
from paperbench.judge.simple import ParsedJudgeResponseFloat, ParsedJudgeResponseInt
from paperbench.judge.token_usage import get_total_token_usage
from paperbench.paper_registry import paper_registry
from paperbench.rubric.tasks import TaskNode
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)


DEFAULT_DATA_DIR = str(Path(__file__).resolve().parents[1] / "data" / "paperbench")


def run(
    submission_path,
    paper_id,
    result_file,
    judge_type="simple",
    judge_model="azure_ai/gpt-5-mini",
    max_depth=999,
    code_dev=True,
    log_dir=None,
    data_dir=None,
):
    os.environ["PAPERBENCH_DATA_DIR"] = data_dir or os.environ.get(
        "PAPERBENCH_DATA_DIR", DEFAULT_DATA_DIR
    )

    litellm_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("LLM_API_KEY")
    if litellm_key:
        os.environ["OPENAI_API_KEY"] = litellm_key
    litellm_base = os.environ.get("LITELLM_BASE_URL") or os.environ.get("LLM_BASE_URL")
    if litellm_base:
        os.environ["OPENAI_BASE_URL"] = litellm_base

    completer_config = None
    if judge_type == "simple":
        completer_config = OpenAICompletionsTurnCompleter.Config(model=judge_model)

    submission_path = Path(submission_path)
    out_dir = Path(log_dir) if log_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    async def _run():
        paper = paper_registry.get_paper(paper_id)
        with open(paper.rubric, "r") as f:
            task_tree = TaskNode.from_dict(json.load(f))

        if code_dev:
            task_tree = task_tree.code_only() or task_tree.set_task_category(
                "Code Development"
            ).set_sub_tasks([])

        judge_kwargs = handle_judge_kwargs(judge_type, code_dev, paper, completer_config)

        # Pass structured completer configs so SimpleJudge doesn't fall back
        # to the hardcoded neulab/gpt-4o-2024-08-06 model.
        # Use reasoning_effort="low" and high max_tokens because the parsing
        # task is trivial and reasoning models waste output tokens on thinking.
        if judge_type == "simple" and completer_config is not None:
            judge_kwargs["float_completer_config"] = OpenAICompletionsTurnCompleter.Config(
                model=judge_model, response_format=ParsedJudgeResponseFloat,
                reasoning_effort="low", max_tokens=4096,
            )
            judge_kwargs["int_completer_config"] = OpenAICompletionsTurnCompleter.Config(
                model=judge_model, response_format=ParsedJudgeResponseInt,
                reasoning_effort="low", max_tokens=4096,
            )

        judge = create_judge(
            judge_type=judge_type,
            judge_kwargs=judge_kwargs,
            paper_path=paper.paper_pdf,
            rubric=task_tree,
            addendum=paper.addendum.read_text() if paper.addendum else None,
            judge_addendum=paper.judge_addendum.read_text() if paper.judge_addendum.exists() else None,
            submission_dir=submission_path,
            paper_md=paper.paper_md,
            log_path=out_dir,
            max_depth=max_depth,
        )
        return await judge.judge()

    graded_tree = asyncio.run(_run())

    token_usage = get_total_token_usage(graded_tree)
    total_cost = 0.0
    for model, usage in token_usage.to_dict().items():
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=model, prompt_tokens=usage["in"], completion_tokens=usage["out"],
            )
            total_cost += prompt_cost + completion_cost
        except Exception:
            pass

    leaf_nodes = graded_tree.get_leaf_nodes()
    result = {
        "score": graded_tree.score,
        "num_nodes": len(leaf_nodes),
        "num_invalid_nodes": len([n for n in leaf_nodes if not n.valid_score]),
        "token_usage": token_usage.to_dict(),
        "cost": total_cost,
        "graded_task_tree": graded_tree.to_dict(),
    }

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Judge score: {result['score']}")
    print(f"Nodes: {result['num_nodes']}, Invalid: {result['num_invalid_nodes']}")
    print(f"Judge cost: ${total_cost:.4f}")
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    fire.Fire(run)
