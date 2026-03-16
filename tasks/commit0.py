"""
python -m tasks.commit0
"""
import json
from dataclasses import dataclass
from typing import Optional

from .base import TaskModule


@dataclass
class Commit0Config:
    repo_name: str = "minitorch"
    base_branch: str = "commit0_combined"
    docker_image_prefix: str = "docker.io/wentingzhao/"
    dataset_path: str = "data/commit0/commit0_combined"


class Commit0Task(TaskModule):
    def __init__(self, config):
        self.config = config
        self.task_data = None

    def get_docker_image(self):
        prefix = self.config.docker_image_prefix.rstrip("/")
        return f"{prefix}/{self.config.repo_name}:v0".lower()

    def get_work_dir(self):
        return f"/workspace/{self.config.repo_name}_repo"

    def get_workspace_config(self):
        return {
            "base_image": self.get_docker_image(),
            "target": "source-minimal",
        }

    def load_task_data(self):
        from datasets import load_from_disk

        dataset = load_from_disk(self.config.dataset_path)
        df = dataset.to_pandas()
        repo_data = df[df["repo"].str.contains(self.config.repo_name, case=False)]
        if repo_data.empty:
            raise ValueError(
                f"Repository '{self.config.repo_name}' not found in dataset "
                f"at {self.config.dataset_path}"
            )
        self.task_data = repo_data.iloc[0].to_dict()
        return self.task_data

    def setup_workspace(self, workspace):
        if self.task_data is None:
            raise RuntimeError("Call load_task_data() before setup_workspace()")

        work_dir = self.get_work_dir()
        repo = self.task_data["repo"]

        # Step 1: Clone Repository
        print("\n" + "-" * 60)
        print("Step 1: Clone Repository")
        print("-" * 60)
        print(f"[Commit0] Cloning {repo}...")
        clone_cmd = (
            f"cd /workspace && "
            f"git clone --depth 1 -b {self.config.base_branch} "
            f"https://github.com/{repo}.git {self.config.repo_name}_repo"
        )
        result = workspace.execute_command(clone_cmd, timeout=600)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to clone repo: {result.stderr}")

        # Create a working branch (matches official OpenHands benchmark)
        branch_cmd = f"cd {work_dir} && git checkout -b openhands"
        result = workspace.execute_command(branch_cmd, timeout=600)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to create branch: {result.stderr}")

        # Step 2: Setup Repository
        print("\n" + "-" * 60)
        print("Step 2: Setup Repository")
        print("-" * 60)
        print(f"[Commit0] Installing {self.config.repo_name} in dev mode...")
        workspace.execute_command(
            f"python -m pip uninstall -y {self.config.repo_name} 2>&1 | tail -3",
            timeout=60,
        )
        result = workspace.execute_command(
            f"cd {work_dir} && python -m pip install -e . 2>&1", timeout=300
        )
        if result.exit_code != 0:
            print(f"[Commit0] Warning: pip install -e . failed: {result.stderr}")

        # Verify package import
        verify_cmd = (
            f"cd {work_dir} && "
            f"python -c 'import {self.config.repo_name}; "
            f'print("{self.config.repo_name} imported successfully")\''
        )
        verify_result = workspace.execute_command(verify_cmd, timeout=30)
        if verify_result.exit_code == 0:
            print(f"[Commit0] Package verification: {verify_result.stdout.strip()}")
        else:
            print("[Commit0] Warning: Package import verification failed")

        # Install commit0 + pytest plugins
        print("[Commit0] Installing commit0 and pytest plugins...")
        uv = workspace.execute_command(
            f"cd {work_dir} && /root/.cargo/bin/uv pip install commit0 2>&1 | tail -5",
            timeout=300,
        )
        if uv.exit_code != 0:
            workspace.execute_command(
                f"cd {work_dir} && python -m pip install commit0 2>&1 | tail -5",
                timeout=300,
            )
        workspace.execute_command(
            f"cd {work_dir} && python -m pip install pytest-json-report pytest-cov 2>&1 | tail -5",
            timeout=300,
        )
        print("[Commit0] Workspace setup complete")

    def evaluate(self, workspace):
        if self.task_data is None:
            raise RuntimeError("Call load_task_data() before evaluate()")

        work_dir = self.get_work_dir()

        # Commit any remaining changes
        print("[Commit0] Committing any remaining changes...")
        workspace.execute_command(f"cd {work_dir} && git add .", timeout=600)
        workspace.execute_command(
            f"cd {work_dir} && "
            'git config --global user.email "evaluation@openhands.dev" && '
            'git config --global user.name "OpenHands Evaluation" && '
            'git commit -m "final changes before test" || true',
            timeout=600,
        )

        # Determine test command from task data
        test_info = self.task_data.get("test", {})
        test_cmd = test_info.get(
            "test_cmd", self.task_data.get("test_cmd", "pytest")
        )
        test_dir = test_info.get(
            "test_dir", self.task_data.get("test_dir", "tests/")
        )
        if test_cmd.strip().startswith("pytest"):
            test_cmd = "python -m " + test_cmd.strip()

        full_cmd = (
            f"cd {work_dir} && "
            f"export PYTHONPATH={work_dir}/src:{work_dir}:$PYTHONPATH && "
            f"{test_cmd} "
            f"--json-report --json-report-file=report.json "
            f"--continue-on-collection-errors "
            f"{test_dir} > test_output.txt 2>&1"
        )
        print(f"[Commit0] Running: {test_cmd} {test_dir}")
        workspace.execute_command(full_cmd, timeout=6000)

        # Read results
        output_result = workspace.execute_command(
            f"cat {work_dir}/test_output.txt", timeout=60
        )
        test_output = output_result.stdout if output_result.exit_code == 0 else ""

        report_result = workspace.execute_command(
            f"cat {work_dir}/report.json", timeout=60
        )
        report_json = report_result.stdout if report_result.exit_code == 0 else "{}"

        passed = failed = error = 0
        try:
            report_data = json.loads(report_json)
            summary = report_data.get("summary", {})
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            error = summary.get("error", 0)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[Commit0] Warning: could not parse report.json: {e}")

        print(f"[Commit0] Pytest results: {passed} passed, {failed} failed, {error} error")

        return {
            "exit_code": str(output_result.exit_code),
            "test_output": test_output,
            "report_json": report_json,
            "passed": passed,
            "failed": failed,
            "error": error,
        }

    def get_prompt_format_args(self, config):
        work_dir = self.get_work_dir()
        workspace_dir_name = work_dir.split("/")[-1]
        test_info = self.task_data.get("test", {}) if self.task_data else {}
        test_cmd = test_info.get("test_cmd", self.task_data.get("test_cmd", "pytest") if self.task_data else "pytest")
        test_dir = test_info.get("test_dir", self.task_data.get("test_dir", "tests/") if self.task_data else "tests/")
        if test_cmd.strip().startswith("pytest"):
            test_cmd = "python -m " + test_cmd.strip()
        return {
            "max_agents": config.max_subagents,
            "max_rounds": config.max_rounds_chat,
            "workspace_dir_name": workspace_dir_name,
            "test_cmd": test_cmd,
            "test_dir": test_dir,
        }

    # ---- Manager integration methods ----

    def get_scan_log_kwargs(self, config):
        return {
            "repo_name": self.config.repo_name,
            "repo_path": self.get_work_dir(),
            "max_iterations": config.manager_max_iterations,
        }

    def build_subagent(self, engineer_id, primary_task, all_tasks):
        from config import SubAgent
        all_files = []
        all_functions = []
        all_instructions = []
        for t in all_tasks:
            all_files.append(t.file_path)
            all_functions.extend(t.functions_to_implement)
            all_instructions.append(f"File: {t.file_path}\n{t.instruction}")
        combined_instruction = "\n\n---\n\n".join(all_instructions)
        combined_file_path = ", ".join(all_files) if len(all_files) > 1 else all_files[0]
        subagent = SubAgent(
            engineer_id=engineer_id,
            task_id=primary_task.task_id,
            file_path=combined_file_path,
            functions_to_implement=all_functions,
            instruction=combined_instruction,
            estimated_complexity=primary_task.estimated_complexity,
        )
        combine_log = f"  (Combined {len(all_tasks)} tasks: {all_files})" if len(all_tasks) > 1 else None
        return subagent, combine_log

    def get_worktree_name(self, engineer_id):
        return f"{self.config.repo_name}_worktree_{engineer_id}"

    def get_subagent_log_lines(self, subagent):
        lines = [f"      Task: {subagent.file_path}"]
        funcs_str = ', '.join(subagent.functions_to_implement[:3])
        if len(subagent.functions_to_implement) > 3:
            funcs_str += f"... (+{len(subagent.functions_to_implement)-3})"
        lines.append(f"      Functions: {funcs_str}")
        return lines

    @property
    def should_stash_before_merge(self):
        return True

    @property
    def should_try_uncommitted_merge(self):
        return True

    def build_completed_task_summary(self, result, task_status):
        return (
            f"task_id: {result.task_id}\n"
            f"file: {result.file_path}\n"
            f"status: {task_status}\n"
            f"merged: {result.merged}\n"
            f"merge_method: {result.merge_method or 'none'}\n"
            f"commit: {result.commit_hash or 'none'}\n"
            f"commit_message: {result.commit_message or 'none'}"
        )

    def extract_assignments(self, assign_data):
        assignments = assign_data.get("assignments", [])
        if not assignments and "next_task" in assign_data:
            if assign_data.get("should_assign", False):
                assignments = [assign_data["next_task"]]
        return assignments

    def get_assign_context(self, all_completed, workspace, repo_dir):
        cmd_result = workspace.execute_command(
            f"cd {repo_dir} && git rev-parse HEAD", timeout=30
        )
        current_head = cmd_result.stdout.strip() if cmd_result.exit_code == 0 else ""

        completed_files = set()
        for completed in all_completed:
            if completed.merged and completed.file_path:
                completed_files.add(completed.file_path)
        progress_summary = ""
        if completed_files:
            files_list = "\n".join(f"  - {f}" for f in sorted(completed_files))
            progress_summary = f"Files completed by other agents:\n{files_list}"

        return {"current_head": current_head, "progress_summary": progress_summary}

    def update_subagent_for_assignment(self, subagent, context, workspace, log_fn):
        current_head = context.get("current_head", "")
        progress_summary = context.get("progress_summary", "")

        if current_head and subagent.file_path:
            worktree_name = self.get_worktree_name(subagent.engineer_id)
            subagent.worktree_path = f"/workspace/{worktree_name}"
            subagent.base_commit = current_head

            update_cmd = f"cd {subagent.worktree_path} && git reset --hard {current_head}"
            update_result = workspace.execute_command(update_cmd, timeout=60)

            if update_result.exit_code != 0:
                log_fn(f"Failed to update worktree for {subagent.engineer_id}: {update_result.stderr}")
                subagent.status = "failed"
            else:
                subagent.status = "ready"
                log_fn(f"Worktree for {subagent.engineer_id} updated to {current_head[:8]}")

            if progress_summary:
                subagent.instruction = f"{progress_summary}\n\n{subagent.instruction}"
        else:
            subagent.status = "ready"

    def get_single_agent_info(self, workspace, config, prompts):
        header = "Single Agent Mode - Implementing all functions"
        format_args = self.get_prompt_format_args(config)
        format_args["repo_path"] = self.get_work_dir()
        user_instruction = prompts.get("single_agent_instruction", "").format(**format_args)
        log_content = {
            "repo_name": self.config.repo_name,
            "repo_path": self.get_work_dir(),
            "max_iterations": config.manager_max_iterations,
        }
        return header, user_instruction, log_content

    def get_final_review_log_extras(self, subagent_results):
        merged_count = sum(1 for r in subagent_results if r.merged and r.file_path)
        return {"files_merged": merged_count}

    def get_collect_extra_log(self, subagent_result):
        if subagent_result.file_path:
            return f"  - File: {subagent_result.file_path}"
        return ""

    # ---- SubAgent runner integration methods ----

    def create_subagent_result(self, subagent):
        from config import SubAgentResult
        return SubAgentResult(
            engineer_id=subagent.engineer_id,
            task_id=subagent.task_id,
            task_node_id=subagent.task_node_id,
            branch_name=subagent.branch_name or "",
            worktree_path=subagent.worktree_path or "",
            file_path=subagent.file_path,
            functions_implemented=subagent.functions_to_implement.copy(),
            round_num=subagent.current_round,
        )

    def get_followup_prompt_args(self, subagent):
        return {
            "instruction": subagent.instruction,
            "file_path": subagent.file_path,
            "functions": ", ".join(subagent.functions_to_implement),
        }

    def get_run_start_log_lines(self, subagent):
        return [
            f"  - Task: {subagent.task_id}",
            f"  - File: {subagent.file_path}",
            f"  - Functions: {', '.join(subagent.functions_to_implement)}",
        ]

    @property
    def should_setup_on_retry(self):
        return True

    @property
    def should_resend_on_retry(self):
        return True

    def populate_no_commit_result(self, result):
        result.git_diff = ""

    def populate_success_result(self, result, runner, commit_info):
        result.success = True
        result.commit_hash = commit_info.get("hash", "")
        result.commit_message = commit_info.get("message", "")
        result.git_diff = runner.get_git_diff()
        result.files_modified = runner.get_modified_files()

    def get_event_serialization_extras(self, subagent):
        return {"file_path": subagent.file_path}

    def get_print_summary_lines(self, result, commit_info):
        lines = []
        if result.files_modified:
            lines.append(f"  Files Modified: {', '.join(result.files_modified)}")
        if result.git_diff:
            diff_preview = result.git_diff[:500]
            if len(result.git_diff) > 500:
                diff_preview += "\n... (truncated)"
            lines.append("  Diff Preview:")
            for line in diff_preview.split("\n")[:15]:
                lines.append(f"    {line}")
        return lines

    def prepare_reuse_subagent(self, new_subagent, old_runner):
        pass

    def get_new_task_print_lines(self, subagent):
        lines = [f"- New file: {subagent.file_path}"]
        funcs = ', '.join(subagent.functions_to_implement[:3])
        if len(subagent.functions_to_implement) > 3:
            funcs += f"... (+{len(subagent.functions_to_implement)-3})"
        lines.append(f"- Functions: {funcs}")
        return lines

    def get_onboard_names(self, engineer_id):
        repo_name = self.config.repo_name
        branch_name = f"agent_{engineer_id}"
        worktree_name = f"{repo_name}_worktree_{engineer_id}"
        return branch_name, worktree_name

    def post_onboard_subagent(self, subagent, repo_dir):
        pass

    def get_completion_print_lines(self, result):
        lines = []
        if result.commit_hash:
            lines.append(f"- Commit: {result.commit_hash}")
        return lines

    def get_log_agent_response_kwargs(self, result):
        from datetime import datetime
        return {
            "engineer_id": result.engineer_id,
            "task_id": result.task_id,
            "success": result.success,
            "commit_hash": result.commit_hash,
            "git_diff": result.git_diff,
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
        return {"conflict_file_list": conflict_file_list}

    def get_auto_reassign_instruction_args(self, subagent):
        return {"original_instruction": subagent.instruction}

    def get_execution_summary_lines(self, results):
        lines = [
            f"\n{'=' * 70}",
            "[SubAgents] Execution Summary",
            f"{'=' * 70}",
            f"Total task completions: {len(results)}",
        ]
        merged_count = len([r for r in results if r.merged])
        committed_count = len([r for r in results if r.success])
        recovered_count = len([r for r in results if r.merged and not r.success])
        failed_count = len([r for r in results if not r.merged])
        lines.append(f"Merged: {merged_count} (committed: {committed_count}, recovered: {recovered_count})")
        lines.append(f"Failed: {failed_count}")

        agent_results = {}
        for result in results:
            if result.engineer_id not in agent_results:
                agent_results[result.engineer_id] = []
            agent_results[result.engineer_id].append(result)

        for engineer_id, agent_res in agent_results.items():
            lines.append(f"\n  {engineer_id}:")
            for res in agent_res:
                if res.merged and res.success:
                    status = "SUCCESS"
                elif res.merged and not res.success:
                    status = "RECOVERED"
                else:
                    status = "FAILED"
                lines.append(f"Round {res.round_num}: {status} - {res.task_id}")
                if res.commit_hash:
                    lines.append(f"      Commit: {res.commit_hash}")
                if res.merge_method:
                    lines.append(f"      Merge method: {res.merge_method}")
                if res.error and not res.merged:
                    lines.append(f"      Error: {res.error[:80]}")

        return lines


if __name__ == "__main__":
    config = Commit0Config(repo_name="minitorch")
    task = Commit0Task(config)

    print("=== Commit0 Task Prepare Test ===\n")

    # 1. Docker image
    image = task.get_docker_image()
    print(f"Docker image : {image}")
    assert image == "docker.io/wentingzhao/minitorch:v0", f"unexpected: {image}"

    # 2. Work dir
    work_dir = task.get_work_dir()
    print(f"Work dir     : {work_dir}")
    assert work_dir == "/workspace/minitorch_repo"

    # 3. Workspace config
    kwargs = task.get_workspace_config()
    assert kwargs["base_image"] == image
    assert kwargs["target"] == "source-minimal"

    # 4. Different repo names
    for repo in ["simpy", "portalocker", "flask"]:
        t = Commit0Task(Commit0Config(repo_name=repo))
        expected_image = f"docker.io/wentingzhao/{repo}:v0"
        assert t.get_docker_image() == expected_image, f"{repo}: {t.get_docker_image()}"
        assert t.get_work_dir() == f"/workspace/{repo}_repo"
        print(f"  {repo:15s} -> image={t.get_docker_image()}, work_dir={t.get_work_dir()}")

    # 5. task_data should be None before load
    assert task.task_data is None

    print("\nAll checks passed!")
