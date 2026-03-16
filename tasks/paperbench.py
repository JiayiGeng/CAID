"""
python -m tasks.paperbench
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .base import TaskModule

HOOKS_DIR = Path(__file__).parent / "hooks"


@dataclass
class PaperbenchConfig:
    paper_id: str = "rice"
    docker_image: str = "ghcr.io/openhands/agent-server:latest-python"
    paperbench_dir: str = "data/paperbench"
    test_max_depth: int = 999
    test_reproduce_timeout: int = 300
    judge_type: str = "simple"
    judge_model: str = "neulab/gpt-5-mini"
    code_dev: bool = True
    output_dir: str = "outputs"


class PaperbenchTask(TaskModule):
    def __init__(self, config):
        self.config = config
        self.task_data = None
        self.paper_info = None
        self.task_tree = None

    def get_docker_image(self):
        return self.config.docker_image

    def get_work_dir(self):
        return "/workspace/submission"

    def get_workspace_config(self):
        return {
            "server_image": "ghcr.io/openhands/agent-server:latest-python",
        }

    def load_task_data(self):
        papers_dir = Path(self.config.paperbench_dir) / "papers" / self.config.paper_id

        if not papers_dir.exists():
            raise ValueError(f"Paper directory not found: {papers_dir}")

        config_path = papers_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                paper_config = yaml.safe_load(f)
        else:
            paper_config = {"id": self.config.paper_id, "title": self.config.paper_id}

        self.task_data = {
            "paper_id": paper_config.get("id", self.config.paper_id),
            "title": paper_config.get("title", self.config.paper_id),
            "paper_pdf_path": str(papers_dir / "paper.pdf"),
            "paper_md_path": str(papers_dir / "paper.md"),
            "rubric_path": str(papers_dir / "rubric.json"),
            "addendum_path": str(papers_dir / "addendum.md"),
            "blacklist_path": str(papers_dir / "blacklist.txt"),
            "assets_dir": str(papers_dir / "assets"),
        }

        return self.task_data

    def setup_workspace(self, workspace):
        if self.task_data is None:
            raise RuntimeError("Call load_task_data() before setup_workspace()")

        # Create submission and paper directories
        print("[PaperBench] Creating workspace directories...")
        workspace.execute_command(
            "mkdir -p /workspace/submission /workspace/logs /workspace/paper/assets",
            timeout=30,
        )

        # Initialize git repo in submission
        print("[PaperBench] Initializing git repository...")
        workspace.execute_command(
            "cd /workspace/submission && "
            "git init && "
            'git config user.name "openhands" && '
            'git config user.email "openhands@all-hands.dev" && '
            'git commit --allow-empty -m "Initial commit"',
            timeout=60,
        )

        # Install pre-commit hook to enforce 1GB total tracked file size limit
        print("[PaperBench] Installing pre-commit hook...")
        pre_commit_hook = (HOOKS_DIR / "pre-commit").read_text()
        workspace.execute_command(
            f"mkdir -p /workspace/submission/.git/hooks && "
            f"cat > /workspace/submission/.git/hooks/pre-commit << 'HOOKEOF'\n{pre_commit_hook}\nHOOKEOF\n"
            f"chmod +x /workspace/submission/.git/hooks/pre-commit",
            timeout=30,
        )

        # Upload paper files via tarball
        print("[PaperBench] Uploading paper files...")
        self.upload_paper_tarball(workspace)

        # Verify
        verify = workspace.execute_command("ls -la /workspace/paper/", timeout=30)
        print(f"[PaperBench] Paper directory: {verify.stdout[:200] if verify.stdout else 'empty'}")
        print("[PaperBench] Workspace setup complete")

    def upload_paper_tarball(self, workspace):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paper_dir = tmp_path / "paper"
            paper_dir.mkdir()
            assets_dir = paper_dir / "assets"
            assets_dir.mkdir()

            # Copy paper files
            paper_files = [
                (self.task_data["paper_pdf_path"], paper_dir / "paper.pdf"),
                (self.task_data["paper_md_path"], paper_dir / "paper.md"),
                (self.task_data["addendum_path"], paper_dir / "addendum.md"),
                (self.task_data["blacklist_path"], paper_dir / "blacklist.txt"),
            ]
            for src, dst in paper_files:
                if Path(src).exists():
                    shutil.copy2(src, dst)

            # Copy assets
            src_assets = Path(self.task_data["assets_dir"])
            if src_assets.exists():
                for asset in src_assets.glob("*"):
                    shutil.copy2(asset, assets_dir / asset.name)

            # Copy instructions
            instructions_path = (
                Path(self.config.paperbench_dir)
                / "src" / "paperbench" / "instructions" / "instructions.txt"
            )
            if instructions_path.exists():
                shutil.copy2(instructions_path, tmp_path / "instructions.txt")

            # Create tarball
            tarball_path = tmp_path / "paper_upload.tar.gz"
            with tarfile.open(str(tarball_path), mode="w:gz") as tar:
                tar.add(str(paper_dir), arcname="paper")
                instructions_local = tmp_path / "instructions.txt"
                if instructions_local.exists():
                    tar.add(str(instructions_local), arcname="instructions.txt")

            # Upload tarball to container and extract
            remote_tar = "/tmp/paper_upload.tar.gz"
            workspace.file_upload(str(tarball_path), remote_tar)
            workspace.execute_command(
                f"tar -xzf {remote_tar} -C /workspace && rm {remote_tar}",
                timeout=60,
            )

    def evaluate(self, workspace):
        if self.task_data is None:
            raise RuntimeError("Call load_task_data() before evaluate()")

        config = self.config
        result = {
            "reproduce_script_exists": False,
            "reproduce_success": False,
            "reproduce_log": "",
            "reproduce_duration": 0.0,
            "judge_score": None,
            "judge_num_nodes": 0,
            "judge_num_invalid_nodes": 0,
            "judge_duration": 0.0,
            "judge_type": config.judge_type,
            "judge_model": config.judge_model if config.judge_type == "simple" else None,
            "judge_token_usage": {},
            "judge_cost": 0.0,
            "max_depth": config.test_max_depth,
            "graded_task_tree": None,
        }

        # Check reproduce.sh
        reproduce_check = workspace.execute_command(
            "test -f /workspace/submission/reproduce.sh && echo 'exists' || echo 'missing'",
            timeout=30,
        )
        result["reproduce_script_exists"] = "exists" in reproduce_check.stdout

        if not result["reproduce_script_exists"]:
            print("[PaperBench] reproduce.sh not found, skipping test")
            return result

        # Run reproduce.sh
        print(f"[PaperBench] Running reproduce.sh (timeout: {config.test_reproduce_timeout}s)...")
        reproduce_start = datetime.now()

        workspace.execute_command(
            "cd /workspace/submission && git clean -fd 2>&1 || true", timeout=60
        )
        reproduce_timeout = config.test_reproduce_timeout
        reproduce_result = workspace.execute_command(
            f"cd /workspace/submission && timeout {reproduce_timeout} bash reproduce.sh 2>&1 "
            f'| tee reproduce.log; echo "EXIT_CODE=${{PIPESTATUS[0]}}"',
            timeout=reproduce_timeout + 60,
        )

        reproduce_end = datetime.now()
        result["reproduce_duration"] = (reproduce_end - reproduce_start).total_seconds()

        output = reproduce_result.stdout or ""
        if len(output) > 10000:
            result["reproduce_log"] = output[:5000] + "\n...[truncated]...\n" + output[-5000:]
        else:
            result["reproduce_log"] = output
        result["reproduce_success"] = (
            reproduce_result.exit_code == 0 or "EXIT_CODE=0" in output
        )
        print(f"[PaperBench] reproduce.sh: success={result['reproduce_success']}, "
              f"duration={result['reproduce_duration']:.1f}s")

        # Run judge via subprocess
        if config.test_max_depth > 0:
            print(f"[PaperBench] Running judge (max_depth={config.test_max_depth}, "
                  f"type={config.judge_type}, model={config.judge_model})...")
            judge_start = datetime.now()

            try:
                # Clean untracked AND ignored files before tarball
                workspace.execute_command(
                    "cd /workspace/submission && git clean -fdx 2>&1 || true", timeout=60
                )

                # Create tarball in container, excluding large files (>10MB)
                submission_tar = f"/tmp/submission_{config.paper_id}.tar.gz"
                tar_result = workspace.execute_command(
                    "cd /workspace && find submission -size +10M -type f > /tmp/exclude.txt 2>/dev/null; "
                    f"tar -czf {submission_tar} -X /tmp/exclude.txt -C /workspace submission/",
                    timeout=120,
                )
                if tar_result.exit_code != 0:
                    raise RuntimeError("Failed to create submission tarball in container")

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_tar = Path(tmp_dir) / "submission.tar.gz"
                    container_id = workspace._container_id
                    cp_result = subprocess.run(
                        ["docker", "cp", f"{container_id}:{submission_tar}", str(local_tar)],
                        capture_output=True, text=True, timeout=120,
                    )
                    if cp_result.returncode != 0:
                        raise RuntimeError(f"docker cp failed: {cp_result.stderr}")

                    extract_dir = Path(tmp_dir) / "extracted"
                    extract_dir.mkdir()
                    with tarfile.open(local_tar, "r:gz") as tar:
                        tar.extractall(path=extract_dir)

                    submission_path = extract_dir / "submission"
                    if not submission_path.exists():
                        raise RuntimeError(f"Submission directory not found after extraction")

                    agent_python = os.environ.get("JUDGE_PYTHON", sys.executable)
                    judge_runner = str(
                        Path(__file__).resolve().parent.parent / "judge" / "judge_runner.py"
                    )
                    result_file = str(Path(tmp_dir) / "judge_result.json")
                    log_dir = str(Path(config.output_dir) / "judge_logs")
                    data_dir = config.paperbench_dir

                    cmd = [
                        agent_python, judge_runner,
                        "--submission_path", str(submission_path),
                        "--paper_id", config.paper_id,
                        "--judge_type", config.judge_type,
                        "--judge_model", config.judge_model,
                        "--max_depth", str(config.test_max_depth),
                        "--log_dir", log_dir,
                        "--result_file", result_file,
                        "--data_dir", data_dir,
                    ]
                    if config.code_dev:
                        cmd.append("--code_dev")

                    print(f"[PaperBench] Running judge subprocess...")
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=600,
                    )

                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"Judge subprocess failed (exit={proc.returncode}): {proc.stderr[-1000:]}"
                        )

                    # Parse judge results
                    with open(result_file, "r") as f:
                        judge_result = json.load(f)

                    result["judge_score"] = judge_result["score"]
                    result["judge_num_nodes"] = judge_result["num_nodes"]
                    result["judge_num_invalid_nodes"] = judge_result["num_invalid_nodes"]
                    result["judge_token_usage"] = judge_result.get("token_usage", {})
                    result["judge_cost"] = judge_result.get("cost", 0.0)
                    result["graded_task_tree"] = judge_result["graded_task_tree"]
                    print(f"[PaperBench] Judge score: {judge_result['score']:.4f}")
                    print(f"[PaperBench] Judge nodes: {judge_result['num_nodes']}")
                    print(f"[PaperBench] Judge invalid nodes: {judge_result['num_invalid_nodes']}")
                    print(f"[PaperBench] Judge cost: ${result['judge_cost']:.4f}")

            except BaseException as e:
                print(f"[PaperBench] Judge failed ({type(e).__name__}): {e}")

            result["judge_duration"] = (datetime.now() - judge_start).total_seconds()

        return result

    def get_prompt_format_args(self, config):
        return {
            "max_agents": config.max_subagents,
            "max_rounds": config.max_rounds_chat,
        }

    # ---- Manager integration methods ----

    def post_load_task_data(self):
        from core.utils import get_paper_info, load_rubric

        self.paper_info = get_paper_info(self.config)
        logs = [
            f"Paper: {self.paper_info.title}",
            f"Paper ID: {self.paper_info.paper_id}",
        ]

        if Path(self.paper_info.rubric_path).exists():
            self.task_tree = load_rubric(self.paper_info.rubric_path)
            leaf_nodes = self.task_tree.get_leaf_nodes()
            logs.append(f"Rubric loaded: {len(leaf_nodes)} leaf tasks")
        else:
            logs.append("Warning: No rubric found")
            self.task_tree = None

        return logs

    def get_scan_log_kwargs(self, config):
        return {
            "paper_id": self.config.paper_id,
            "paper_title": self.paper_info.title if self.paper_info else "",
            "max_iterations": config.manager_max_iterations,
        }

    def build_analysis_from_state(self):
        if not getattr(self, 'task_tree', None):
            return None, []

        from core.utils import build_analysis_result
        leaf_tasks = self.task_tree.get_leaf_nodes()
        categories = {}
        for leaf in leaf_tasks:
            cat = leaf.task_category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1

        analysis = build_analysis_result(
            {"analysis": {"paper_context": self.paper_info.title if self.paper_info else "", "total_tasks": len(leaf_tasks)}},
            self.task_tree
        )

        logs = [f"Leaf tasks: {len(leaf_tasks)}"]
        for cat, count in categories.items():
            logs.append(f"  - {cat}: {count}")

        return analysis, logs

    def check_existing_delegation(self, events, extract_fn):
        existing = extract_fn(events, key_to_find="delegation_plan")
        return (
            existing
            and isinstance(existing.get("delegation_plan", {}).get("first_round", {}).get("tasks"), list)
            and len(existing["delegation_plan"]["first_round"]["tasks"]) > 0
        )

    def build_subagent(self, engineer_id, primary_task, all_tasks):
        from config import SubAgent
        all_requirements = []
        all_instructions = []
        for t in all_tasks:
            all_requirements.append(t.requirements)
            all_instructions.append(f"Task: {t.task_id}\n{t.instruction}")
        combined_instruction = "\n\n---\n\n".join(all_instructions)
        combined_requirements = "\n\n".join(all_requirements) if len(all_requirements) > 1 else all_requirements[0]
        subagent = SubAgent(
            engineer_id=engineer_id,
            task_id=primary_task.task_id,
            task_node_id=primary_task.task_node_id,
            requirements=combined_requirements,
            instruction=combined_instruction,
            estimated_complexity=primary_task.estimated_complexity,
            task_category=primary_task.task_category,
            submission_path="/workspace/submission",
        )
        combine_log = f"  (Combined {len(all_tasks)} tasks)" if len(all_tasks) > 1 else None
        return subagent, combine_log

    def get_worktree_name(self, engineer_id):
        return f"submission_worktree_{engineer_id}"

    def get_subagent_log_lines(self, subagent):
        return [
            f"      Task: {subagent.task_id}",
            f"      Requirements: {subagent.requirements[:80]}...",
        ]

    def build_completed_task_summary(self, result, task_status):
        return (
            f"task_id: {result.task_id}\n"
            f"task_node_id: {result.task_node_id}\n"
            f"status: {task_status}\n"
            f"merged: {result.merged}\n"
            f"merge_method: {result.merge_method or 'none'}\n"
            f"submission_exists: {result.submission_exists}\n"
            f"reproduce_script_exists: {result.reproduce_script_exists}"
        )

    def search_alternative_json(self, events, extract_fn, log_fn):
        review_json = extract_fn(events, key_to_find=None)
        if review_json:
            log_fn(f"No assign_task JSON found, but found JSON with keys: {list(review_json.keys())}")
            for key in ["task_assignment", "next_task"]:
                if key in review_json:
                    data = review_json[key]
                    first_round = data.get("first_round", data)
                    review_json = {
                        "assign_task": {
                            "reasoning": first_round.get("reasoning", data.get("reasoning", "")),
                            "tasks": first_round.get("tasks", data.get("tasks", data.get("assignments", []))),
                        }
                    }
                    log_fn(f"Converted '{key}' format to assign_task format")
                    return review_json
        return None

    def extract_assignments(self, assign_data):
        assignments = assign_data.get("tasks", assign_data.get("assignments", []))
        if not assignments and "next_task" in assign_data:
            if assign_data.get("should_assign", False):
                assignments = [assign_data["next_task"]]
        return assignments

    def get_assigned_targets(self, assignments, default_engineer_id):
        if assignments:
            return ",".join(s.engineer_id for s in assignments)
        return default_engineer_id

    def get_assign_event_extras(self, engineer_id):
        return {"completed_agent": engineer_id}

    def get_single_agent_info(self, workspace, config, prompts):
        header = "Single Agent Mode - Reproducing Paper"
        gpu_result = workspace.execute_command(
            "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo ''",
            timeout=30
        )
        gpu_info = gpu_result.stdout.strip() if gpu_result.exit_code == 0 else ""
        if gpu_info:
            type_of_processor_str = f"You have access to a {gpu_info} with the appropriate drivers installed."
        else:
            type_of_processor_str = "You don't have access to a GPU."

        user_instruction = prompts.get("single_agent_instruction", "").format(
            type_of_processor_str=type_of_processor_str,
        )
        log_content = {
            "paper_id": self.config.paper_id,
            "max_iterations": config.manager_max_iterations,
            "gpu_info": gpu_info,
        }
        return header, user_instruction, log_content

    def get_final_review_log_extras(self, subagent_results):
        collected_count = sum(1 for r in subagent_results if r.merged)
        return {"tasks_collected": collected_count}

    # ---- SubAgent runner integration methods ----

    def create_subagent_result(self, subagent):
        from config import SubAgentResult
        return SubAgentResult(
            engineer_id=subagent.engineer_id,
            task_id=subagent.task_id,
            task_node_id=subagent.task_node_id,
            requirements=subagent.requirements,
            worktree_path=subagent.worktree_path or "",
            branch_name=subagent.branch_name or "",
            round_num=subagent.current_round,
        )

    def get_followup_prompt_args(self, subagent):
        return {
            "instruction": subagent.instruction,
            "task_node_id": subagent.task_node_id,
            "requirements": subagent.requirements,
        }

    def get_run_start_log_lines(self, subagent):
        return [
            f"  - Task: {subagent.task_id}",
            f"  - Task Node: {subagent.task_node_id}",
        ]

    def populate_success_result(self, result, runner, commit_info):
        result.success = True
        result.commit_hash = commit_info.get("hash", "")
        result.commit_message = commit_info.get("message", "")
        result.files_modified = runner.get_modified_files()
        result.git_commits = runner.get_commit_count()
        result.submission_exists = runner.check_submission_exists()
        result.reproduce_script_exists = runner.check_reproduce_script_exists()

    def get_event_serialization_extras(self, subagent):
        return {"task_node_id": subagent.task_node_id}

    def get_print_summary_lines(self, result, commit_info):
        lines = []
        if result.files_modified:
            lines.append(f"  Files Modified: {', '.join(result.files_modified[:5])}")
            if len(result.files_modified) > 5:
                lines.append(f"    ... and {len(result.files_modified) - 5} more")
        lines.append(f"  Submission exists: {result.submission_exists}")
        lines.append(f"  reproduce.sh exists: {result.reproduce_script_exists}")
        return lines

    def prepare_reuse_subagent(self, new_subagent, old_runner):
        new_subagent.worktree_path = old_runner.subagent.worktree_path
        new_subagent.branch_name = old_runner.subagent.branch_name
        new_subagent.base_commit = old_runner.subagent.base_commit

    def get_new_task_print_lines(self, subagent):
        return []

    def get_onboard_names(self, engineer_id):
        branch_name = f"feature/{engineer_id}"
        worktree_name = f"submission_worktree_{engineer_id}"
        return branch_name, worktree_name

    def post_onboard_subagent(self, subagent, repo_dir):
        subagent.submission_path = repo_dir

    def get_completion_print_lines(self, result):
        return [
            f"- Duration: {result.duration_seconds:.1f}s",
            f"- Cost: ${result.cost:.4f}",
        ]

    def get_log_agent_response_kwargs(self, result):
        from datetime import datetime
        return {
            "engineer_id": result.engineer_id,
            "task_id": result.task_id,
            "success": result.success,
            "submission_exists": result.submission_exists,
            "reproduce_script_exists": result.reproduce_script_exists,
            "git_commits": result.git_commits,
            "files_modified": result.files_modified,
            "error": result.error,
            "duration_seconds": result.duration_seconds,
            "actual_iterations": result.actual_iterations,
            "max_iterations": result.max_iterations,
            "cost": result.cost,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "start_time": datetime.fromisoformat(result.start_time) if result.start_time else None,
            "end_time": datetime.fromisoformat(result.end_time) if result.end_time else None,
            "round_num": result.round_num,
        }

    def get_conflict_instruction_args(self, subagent, conflict_files, workspace, repo_dir):
        conflict_file_list = "\n".join(f"  - {f}" for f in conflict_files)
        main_head_result = workspace.execute_command(
            f"cd {repo_dir} && git rev-parse HEAD", timeout=30
        )
        main_head = main_head_result.stdout.strip() if main_head_result.exit_code == 0 else ""
        worktree = subagent.worktree_path or subagent.submission_path
        return {
            "conflict_file_list": conflict_file_list,
            "main_head": main_head,
            "worktree": worktree,
        }

    def get_auto_reassign_instruction_args(self, subagent):
        return {}

    def get_execution_summary_lines(self, results):
        return [f"\n[SubAgents] All subagents completed ({len(results)} total results)"]


if __name__ == "__main__":
    config = PaperbenchConfig(paper_id="rice")
    task = PaperbenchTask(config)

    print("=== PaperBench Task Prepare Test ===\n")

    # 1. Docker image
    image = task.get_docker_image()
    print(f"Docker image : {image}")
    assert image == "ghcr.io/openhands/agent-server:latest-python"

    # 2. Work dir
    work_dir = task.get_work_dir()
    print(f"Work dir     : {work_dir}")
    assert work_dir == "/workspace/submission"

    # 3. Workspace config
    kwargs = task.get_workspace_config()
    assert "base_image" not in kwargs
    assert kwargs["server_image"] == image
    print(f"Workspace cfg: server_image={kwargs['server_image']}")

    # 4. Different paper IDs
    for paper_id in ["rice", "attention", "gpt2"]:
        t = PaperbenchTask(PaperbenchConfig(paper_id=paper_id))
        assert t.get_docker_image() == "ghcr.io/openhands/agent-server:latest-python"
        assert t.get_work_dir() == "/workspace/submission"
        assert t.config.paper_id == paper_id
        print(f"  {paper_id:15s} -> config OK")

    # 5. Custom docker image
    custom = PaperbenchTask(PaperbenchConfig(
        paper_id="test",
        docker_image="custom-registry/my-image:v1",
    ))
    assert custom.get_docker_image() == "custom-registry/my-image:v1"
    print(f"  custom image   -> {custom.get_docker_image()}")

    # 6. task_data should be None before load
    assert task.task_data is None

    print("\nAll checks passed!")
