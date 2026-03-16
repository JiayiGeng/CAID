import json
from datetime import datetime
from pathlib import Path

from openhands.sdk import Agent, Conversation, LLMSummarizingCondenser
from openhands.sdk.context import AgentContext
from openhands.tools.preset.default import get_default_tools

from config import SubAgent
from core.utils import (
    PanelVisualizer,
    build_delegation_plan,
    build_delegation_prompt,
    count_llm_iterations,
    extract_conversation_metrics,
    extract_json_from_events,
    fallback_delegation,
    load_prompts,
    serialize_event,
)


class Manager:
    def __init__(
        self,
        llm,
        workspace,
        task,
        config,
        output_logger,
        prompts=None,
    ):
        self.llm = llm
        self.workspace = workspace
        self.task = task
        self.config = config
        self.output_logger = output_logger
        self.prompts = prompts or load_prompts()

        self.agent = None
        self.conversation = None
        self.analysis_result = None
        self.delegation_plan = None
        self.repo_dir = task.get_work_dir()

        self.analysis_start_time = None
        self.analysis_end_time = None
        self.delegation_start_time = None
        self.delegation_end_time = None

        # Cumulative time tracking for operations during parallel execution
        self.assign_task_total_time = 0.0
        self.review_total_time = 0.0

        # Per-operation cost tracking (delta costs, not accumulated)
        self.analysis_cost = 0.0
        self.analysis_tokens = 0
        self.delegation_cost = 0.0
        self.delegation_tokens = 0
        self.assign_task_total_cost = 0.0
        self.assign_task_total_tokens = 0
        self.review_total_cost = 0.0
        self.review_total_tokens = 0

        # Background exploration tracking (commit0-specific)
        self.exploration_cost = 0.0
        self.exploration_tokens = 0
        self.exploration_total_time = 0.0
        self.exploration_findings = []
        self.exploration_cancelled = False

        # Final review tracking
        self.final_review_cost = 0.0
        self.final_review_tokens = 0
        self.final_review_total_time = 0.0

        # Test tracking (paperbench-specific)
        self.test_total_time = 0.0
        self.test_result = None

        self.analysis_metrics = None
        self.current_round = 1

    def log(self, message):
        print(f"[Manager] {message}")

    def save_events(self, phase, event_start_idx=0):
        if not self.conversation or not self.output_logger:
            return

        events = list(self.conversation.state.events)
        if event_start_idx >= len(events):
            return

        new_events = events[event_start_idx:]
        self.log(f"Saving {len(new_events)} new events (phase={phase}) to manager_events.jsonl...")

        for idx, event in enumerate(new_events):
            global_idx = event_start_idx + idx
            serialized = serialize_event(event, global_idx)
            serialized["engineer_id"] = "manager"
            serialized["phase"] = phase
            serialized["start_time"] = serialized.get("timestamp")
            if global_idx + 1 < len(events):
                next_ts = getattr(events[global_idx + 1], 'timestamp', None)
                serialized["end_time"] = next_ts
            else:
                serialized["end_time"] = datetime.now().isoformat()
            self.output_logger.log_agent_event("manager", serialized)

    def setup_workspace(self):
        self.log("Loading task data...")
        self.task.load_task_data()

        for msg in self.task.post_load_task_data():
            self.log(msg)

        self.log("Setting up workspace...")
        self.task.setup_workspace(self.workspace)
        self.log("Workspace setup complete")

    def setup(self, mode="multi_agent"):
        self.log(f"Setting up agent in {mode} mode...")
        tools = get_default_tools(enable_browser=False)

        format_args = self.task.get_prompt_format_args(self.config)

        if mode == "single_agent":
            self.agent = Agent(
                llm=self.llm,
                tools=tools,
            )
        else:
            instruction = self.prompts.get("user_instruction", "").format(**format_args)
            condenser_llm = self.llm.model_copy(update={"usage_id": "condenser"})
            condenser = LLMSummarizingCondenser(
                llm=condenser_llm,
                max_size=200,
                keep_first=4,
            )
            self.agent = Agent(
                llm=self.llm,
                tools=tools,
                agent_context=AgentContext(system_message_suffix=instruction),
                condenser=condenser,
            )

        self.conversation = Conversation(
            agent=self.agent,
            workspace=self.workspace,
            max_iteration_per_run=self.config.manager_max_iterations,
            visualizer=PanelVisualizer(),
        )
        self.log("Agent ready")

    def run_single_agent(self):
        header, user_instruction, log_content = self.task.get_single_agent_info(
            self.workspace, self.config, self.prompts
        )

        self.log("=" * 60)
        self.log(header)
        self.log("=" * 60)

        self.output_logger.log_event(
            event_type="single_agent_start",
            source="manager",
            content=log_content,
        )

        self.analysis_start_time = datetime.now()
        self.log("Starting implementation...")
        self.conversation.send_message(user_instruction)
        try:
            self.conversation.run()
        except Exception as e:
            self.log(f"Agent run ended with: {e}")

        self.analysis_end_time = datetime.now()
        duration = (self.analysis_end_time - self.analysis_start_time).total_seconds()
        self.log(f"Single agent completed in {duration:.1f}s")

        events = self.conversation.state.events
        iterations = count_llm_iterations(events)

        engineer_id = "single_agent"
        self.log(f"Saving {len(list(events))} events to {engineer_id}_events.jsonl...")
        events_list = list(self.conversation.state.events)
        for idx, event in enumerate(events_list):
            serialized = serialize_event(event, idx)
            serialized["engineer_id"] = engineer_id
            serialized["start_time"] = serialized.get("timestamp")
            if idx + 1 < len(events_list):
                next_ts = getattr(events_list[idx + 1], 'timestamp', None)
                serialized["end_time"] = next_ts
            else:
                serialized["end_time"] = self.analysis_end_time.isoformat() if self.analysis_end_time else None
            self.output_logger.log_agent_event(engineer_id, serialized)

        self.output_logger.log_event(
            event_type="single_agent_complete",
            source="manager",
            content={
                "duration": duration,
                "iterations": iterations,
                "max_iterations": self.config.manager_max_iterations,
                "total_events": len(list(self.conversation.state.events)),
            },
            start_time=self.analysis_start_time,
            end_time=self.analysis_end_time,
        )

        self.log(f"Iterations used: {iterations}/{self.config.manager_max_iterations}")

        return {
            "duration": duration,
            "iterations": iterations,
        }

    def scan_and_analyze(self):
        self.log("=" * 60)
        self.log("Scan and Analysis")
        self.log("=" * 60)

        self.output_logger.log_scan_start(
            **self.task.get_scan_log_kwargs(self.config)
        )

        self.analysis_start_time = datetime.now()

        metrics_before = extract_conversation_metrics(self.conversation)
        cost_before = metrics_before["cost"]
        tokens_before = metrics_before["total_tokens"]

        self.log("Starting analysis...")
        prompt = self.prompts.get("scan_analysis", "")
        self.conversation.send_message(prompt)

        try:
            self.conversation.run()
        except Exception as e:
            self.log(f"Agent run ended with: {e}")

        self.analysis_end_time = datetime.now()
        duration = (self.analysis_end_time - self.analysis_start_time).total_seconds()
        events = self.conversation.state.events
        iterations = count_llm_iterations(events)

        metrics_after = extract_conversation_metrics(self.conversation)
        self.analysis_cost = metrics_after["cost"] - cost_before
        self.analysis_tokens = metrics_after["total_tokens"] - tokens_before

        self.log(f"Analysis completed in {duration:.1f}s")
        self.log(f"Iterations: {iterations}/{self.config.manager_max_iterations}")
        self.log(f"Cost: ${self.analysis_cost:.4f} ({self.analysis_tokens} tokens)")

        self.save_events("scan_analysis")

        analysis, analysis_logs = self.task.build_analysis_from_state()
        if analysis:
            self.analysis_result = analysis
            for msg in analysis_logs:
                self.log(msg)

        self.output_logger.log_event(
            event_type="analysis_phase_complete",
            source="manager",
            start_time=self.analysis_start_time,
            end_time=self.analysis_end_time,
            content={
                "max_iterations": self.config.manager_max_iterations,
                "actual_iterations": iterations,
                "cost": self.analysis_cost,
                "tokens": self.analysis_tokens,
                "duration": duration,
            },
        )

        return self.analysis_result

    def delegate_tasks(self):
        self.log("Starting task delegation...")
        self.delegation_start_time = datetime.now()

        metrics_before = extract_conversation_metrics(self.conversation)
        cost_before = metrics_before["cost"]
        tokens_before = metrics_before["total_tokens"]
        event_start_idx = len(list(self.conversation.state.events))

        has_valid_delegation = self.task.check_existing_delegation(
            self.conversation.state.events, extract_json_from_events
        )

        if has_valid_delegation:
            self.log("Valid delegation JSON found from scan_analysis, skipping re-prompt.")
        else:
            prompt = build_delegation_prompt(
                self.prompts,
                self.config.max_subagents,
            )
            self.log("Creating delegation plan...")
            self.conversation.send_message(prompt)
            try:
                self.conversation.run()
            except Exception as e:
                self.log(f"Agent run ended with: {e}")

        self.delegation_end_time = datetime.now()

        metrics_after = extract_conversation_metrics(self.conversation)
        self.delegation_cost = metrics_after["cost"] - cost_before
        self.delegation_tokens = metrics_after["total_tokens"] - tokens_before

        duration = (self.delegation_end_time - self.delegation_start_time).total_seconds()
        self.log(f"Task delegation complete in {duration:.1f}s "
                 f"(cost=${self.delegation_cost:.4f}, "
                 f"tokens={self.delegation_tokens})")

        self.save_events("task_delegation", event_start_idx=event_start_idx)

        # Extract and save delegation JSON
        delegation_json = extract_json_from_events(
            self.conversation.state.events, key_to_find="delegation_plan"
        )

        if not delegation_json:
            self.log("WARNING: No delegation JSON found, using fallback...")
            delegation_json = fallback_delegation(
                self.analysis_result,
                self.config.max_subagents,
            ) or {"delegation_plan": {}}

        self.delegation_plan = build_delegation_plan(delegation_json)
        output_path = Path(self.config.output_dir) / "delegations.json"
        with open(output_path, "w") as f:
            json.dump(delegation_json, f, indent=2)
        self.log(f"Delegation plan saved to: {output_path}")

        actual_iterations = count_llm_iterations(
            list(self.conversation.state.events)[event_start_idx:]
        )
        self.output_logger.log_event(
            event_type="delegation_complete",
            source="manager",
            start_time=self.delegation_start_time,
            end_time=self.delegation_end_time,
            content={
                "num_agents": self.delegation_plan.num_agents if self.delegation_plan else 0,
                "first_round": len(self.delegation_plan.first_round_tasks) if self.delegation_plan else 0,
                "remaining": len(self.delegation_plan.remaining_tasks) if self.delegation_plan else 0,
                "reasoning": self.delegation_plan.reasoning if self.delegation_plan else "",
                "max_iterations": self.config.manager_max_iterations,
                "actual_iterations": actual_iterations,
                "cost": self.delegation_cost,
                "tokens": self.delegation_tokens,
                "duration": duration,
            },
        )

    def onboard_subagents(self):
        if not self.delegation_plan:
            raise RuntimeError("Delegation not completed. Call delegate_tasks() first.")

        self.log("=" * 60)
        self.log("Onboard Subagents")
        self.log("=" * 60)

        subagents = []
        first_round_tasks = self.delegation_plan.first_round_tasks

        if not first_round_tasks:
            self.log("No tasks in first round, skipping onboarding")
            return subagents

        # Group tasks by engineer_id to avoid creating duplicate worktrees
        tasks_by_engineer = {}
        for task in first_round_tasks:
            if task.engineer_id not in tasks_by_engineer:
                tasks_by_engineer[task.engineer_id] = []
            tasks_by_engineer[task.engineer_id].append(task)

        self.log(f"Creating {len(tasks_by_engineer)} git worktrees for {len(first_round_tasks)} tasks...")

        # commit0: use self.repo_dir; paperbench: use /workspace/submission
        git_base_dir = self.repo_dir

        result = self.workspace.execute_command(
            f"cd {git_base_dir} && git rev-parse HEAD",
            timeout=30
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to get current commit: {result.stderr}")
        base_commit = result.stdout.strip()
        self.log(f"Base commit: {base_commit[:8]}")

        for engineer_id, tasks in tasks_by_engineer.items():
            primary_task = tasks[0]

            subagent, combine_log = self.task.build_subagent(engineer_id, primary_task, tasks)
            worktree_name = self.task.get_worktree_name(engineer_id)

            subagent.worktree_path = f"/workspace/{worktree_name}"
            subagent.base_commit = base_commit

            self.log(f"Creating worktree for {engineer_id}...")
            if combine_log:
                self.log(combine_log)

            branch_cmd = (
                f"cd {git_base_dir} && "
                f"git branch {subagent.branch_name} {base_commit} 2>/dev/null || true"
            )
            self.workspace.execute_command(branch_cmd, timeout=30)

            worktree_cmd = (
                f"cd {git_base_dir} && "
                f"git worktree add {subagent.worktree_path} {subagent.branch_name}"
            )
            result = self.workspace.execute_command(worktree_cmd, timeout=60)

            if result.exit_code != 0:
                self.log(f"WARNING: Failed to create worktree for {engineer_id}: {result.stderr}")
                subagent.status = "failed"
            else:
                subagent.status = "ready"
                self.log(f"  {engineer_id}: {subagent.worktree_path} (branch: {subagent.branch_name})")

            subagents.append(subagent)

        self.output_logger.log_event(
            event_type="onboarding_complete",
            source="manager",
            content={
                "num_subagents": len(subagents),
                "subagents": [s.to_dict() for s in subagents],
                "base_commit": base_commit,
            },
        )

        self.log("Onboarding complete:")
        self.log(f"  Subagents created: {len(subagents)}")
        for s in subagents:
            status_icon = "subagent is ready" if s.status == "ready" else "subagent is not ready"
            self.log(f"  {status_icon} {s.engineer_id}: {s.worktree_path}")
            for line in self.task.get_subagent_log_lines(s):
                self.log(line)

        return subagents

    def stash_if_dirty(self):
        status_result = self.workspace.execute_command(
            f"cd {self.repo_dir} && git status --porcelain", timeout=30
        )
        if status_result.exit_code == 0 and status_result.stdout.strip():
            self.log("Stashing uncommitted changes in main repo before merge...")
            stash_result = self.workspace.execute_command(
                f"cd {self.repo_dir} && git stash", timeout=30
            )
            if stash_result.exit_code == 0 and "No local changes" not in (stash_result.stdout or ""):
                return True
        return False

    def unstash(self):
        self.log("Restoring stashed changes...")
        self.workspace.execute_command(f"cd {self.repo_dir} && git stash pop", timeout=30)

    def merge_branch(self, branch_name, force_theirs=False):
        self.log(f"Merging branch {branch_name}...")

        stashed = False
        if self.task.should_stash_before_merge:
            stashed = self.stash_if_dirty()

        merge_cmd = (
            f"cd {self.repo_dir} && "
            f"git merge {branch_name} --no-edit"
        )
        result = self.workspace.execute_command(merge_cmd, timeout=60)

        if result.exit_code == 0:
            self.log(f"Successfully merged {branch_name}")
            if stashed:
                self.unstash()
            return True, "Merged successfully", []

        # Check if it's a conflict
        error_msg = result.stderr or result.stdout or "Unknown error"
        is_conflict = "CONFLICT" in error_msg or "conflict" in error_msg.lower()

        if is_conflict:
            # Extract conflicted file names before aborting
            conflict_cmd = f"cd {self.repo_dir} && git diff --name-only --diff-filter=U"
            conflict_result = self.workspace.execute_command(conflict_cmd, timeout=30)
            conflict_files = [f.strip() for f in conflict_result.stdout.strip().split("\n") if f.strip()] if conflict_result.exit_code == 0 else []

            self.log(f"Merge conflict detected for {branch_name}, files: {conflict_files}")

            abort_cmd = f"cd {self.repo_dir} && git merge --abort"
            self.workspace.execute_command(abort_cmd, timeout=30)

            if force_theirs:
                self.log(f"Force-resolving with --strategy-option theirs (engineer has no rounds left)...")
                merge_theirs_cmd = (
                    f"cd {self.repo_dir} && "
                    f"git merge {branch_name} --no-edit -X theirs"
                )
                result = self.workspace.execute_command(merge_theirs_cmd, timeout=60)

                if result.exit_code == 0:
                    self.log(f"Successfully merged {branch_name} using theirs strategy")
                    if stashed:
                        self.unstash()
                    return True, "Merged successfully (used theirs strategy for conflicts)", []

                error_msg = result.stderr or result.stdout or "Unknown error"
                self.log(f"Merge with theirs strategy also failed: {error_msg[:200]}")

                abort_cmd = f"cd {self.repo_dir} && git merge --abort"
                self.workspace.execute_command(abort_cmd, timeout=30)

                if stashed:
                    self.unstash()
                return False, f"Merge failed even with conflict resolution: {error_msg[:200]}", []

            if stashed:
                self.unstash()
            return False, f"Merge conflict in files: {', '.join(conflict_files)}", conflict_files

        # Non-conflict error
        self.log(f"Warning: Merge failed for {branch_name}: {error_msg[:200]}")
        if stashed:
            self.unstash()
        return False, f"Merge failed: {error_msg[:200]}", []

    def get_uncommitted_changes(self, worktree_path):
        if not worktree_path:
            return []

        status_cmd = f"cd {worktree_path} && git status --porcelain"
        result = self.workspace.execute_command(status_cmd, timeout=30)

        if result.exit_code != 0 or not result.stdout.strip():
            return []

        modified_files = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    file_path = parts[1].strip()
                    modified_files.append(file_path)

        return modified_files

    def commit_worktree_changes(self, worktree_path, branch_name, engineer_id, task_id):
        self.log(f"Committing uncommitted changes in worktree for {engineer_id}...")

        modified_files = self.get_uncommitted_changes(worktree_path)
        if not modified_files:
            self.log("No uncommitted changes found in worktree")
            return False, "No uncommitted changes to commit", []

        self.log(f"Found {len(modified_files)} uncommitted files: {modified_files}")

        git_config_cmd = (
            f"cd {worktree_path} && "
            f'git config user.name "openhands" && '
            f'git config user.email "openhands@all-hands.dev"'
        )
        self.workspace.execute_command(git_config_cmd, timeout=30)

        add_cmd = f"cd {worktree_path} && git add ."
        add_result = self.workspace.execute_command(add_cmd, timeout=60)
        if add_result.exit_code != 0:
            self.log(f"Failed to stage changes: {add_result.stderr}")
            return False, f"Failed to stage changes: {add_result.stderr}", []

        commit_message = f"Partial implementation from {engineer_id} ({task_id})"
        commit_cmd = f'cd {worktree_path} && git commit -m "{commit_message}"'
        commit_result = self.workspace.execute_command(commit_cmd, timeout=60)

        if commit_result.exit_code != 0:
            error_output = commit_result.stderr or commit_result.stdout or ""
            if "nothing to commit" in error_output:
                self.log("No changes to commit (files may be identical)")
                return False, "No changes to commit", []
            self.log(f"Failed to commit in worktree: {error_output}")
            return False, f"Failed to commit: {error_output[:200]}", []

        self.log(f"Committed changes in worktree branch {branch_name}")

        merge_success, merge_message, _ = self.merge_branch(branch_name, force_theirs=True)

        if merge_success:
            self.log(f"Successfully merged worktree changes: {merge_message}")
            return True, f"Committed and merged {len(modified_files)} files", modified_files
        else:
            self.log(f"Merge failed after committing: {merge_message}")
            return False, f"Committed but merge failed: {merge_message}", modified_files

    def collect_and_merge(self, subagent_result, output_logger=None):
        review_start_time = datetime.now()
        engineer_id = subagent_result.engineer_id
        task_id = subagent_result.task_id
        branch_name = subagent_result.branch_name
        worktree_path = subagent_result.worktree_path

        self.log(f"Collecting {engineer_id}'s work...")
        self.log(f"  - Task: {task_id}")
        extra_log = self.task.get_collect_extra_log(subagent_result)
        if extra_log:
            self.log(extra_log)
        self.log(f"  - Success: {subagent_result.success}")
        self.log(f"  - Commit: {subagent_result.commit_hash or 'None'}")
        self.log(f"  - Worktree: {worktree_path or 'None'}")

        review_result = {
            "engineer_id": engineer_id,
            "task_id": task_id,
            "subagent_success": subagent_result.success,
            "merged": False,
            "merge_message": "",
            "review_notes": "",
            "merge_method": "",
            "conflict_files": [],
        }

        round_num = subagent_result.round_num
        files_modified = subagent_result.files_modified or []

        # Subagent made a commit - try to merge via branch
        if subagent_result.success and subagent_result.commit_hash and branch_name:
            self.log("Attempting branch merge (commit found)...")
            merge_success, merge_message, conflict_files = self.merge_branch(branch_name)

            if conflict_files:
                self.log(f"Merge conflict - engineer must resolve: {conflict_files}")
                review_result["merge_method"] = "conflict"
                review_result["merge_message"] = merge_message
                review_result["conflict_files"] = conflict_files
                review_result["review_notes"] = f"Merge conflict in {len(conflict_files)} files, needs engineer resolution"

                if output_logger:
                    output_logger.log_manager_review(
                        engineer_id=engineer_id,
                        task_id=task_id,
                        merged=False,
                        review_reason=f"Merge conflict: {', '.join(conflict_files)}",
                        commit_hash=subagent_result.commit_hash,
                        files_modified=files_modified,
                        round_num=round_num,
                        start_time=review_start_time,
                        end_time=datetime.now(),
                    )

                return review_result

            if merge_success:
                review_result["merged"] = True
                review_result["merge_message"] = merge_message
                review_result["review_notes"] = "Implementation approved and merged via branch"
                review_result["merge_method"] = "branch_merge"
                self.log(f"Collect: MERGED - {merge_message}")

                if output_logger:
                    output_logger.log_manager_review(
                        engineer_id=engineer_id,
                        task_id=task_id,
                        merged=True,
                        review_reason=merge_message,
                        commit_hash=subagent_result.commit_hash,
                        files_modified=files_modified,
                        round_num=round_num,
                        start_time=review_start_time,
                        end_time=datetime.now(),
                    )

                return review_result
            else:
                self.log(f"Branch merge failed: {merge_message}")

        # No commit or branch merge failed - try to commit and merge uncommitted changes
        if self.task.should_try_uncommitted_merge and worktree_path and branch_name:
            self.log("Checking for uncommitted changes in worktree...")
            commit_success, commit_message, committed_files = self.commit_worktree_changes(
                worktree_path, branch_name, engineer_id, task_id
            )

            if commit_success and committed_files:
                review_result["merged"] = True
                review_result["merge_message"] = commit_message
                review_result["review_notes"] = "Uncommitted changes committed and merged"
                review_result["merge_method"] = "worktree_commit_merge"
                files_modified = committed_files
                self.log(f"Collect: MERGED (worktree commit+merge) - {commit_message}")

                if output_logger:
                    output_logger.log_manager_review(
                        engineer_id=engineer_id,
                        task_id=task_id,
                        merged=True,
                        review_reason=f"Committed and merged: {commit_message}",
                        commit_hash=None,
                        files_modified=committed_files,
                        round_num=round_num,
                        start_time=review_start_time,
                        end_time=datetime.now(),
                    )

                return review_result
            else:
                self.log(f"No uncommitted changes to merge: {commit_message}")

        # Neither commit nor uncommitted changes available
        error_reason = subagent_result.error or "No changes found (no commit and no uncommitted changes)"
        review_result["review_notes"] = f"No changes to merge: {error_reason}"
        review_result["merge_method"] = "none"
        self.log(f"Collect: NO CHANGES - {error_reason}")

        if output_logger:
            output_logger.log_manager_review(
                engineer_id=engineer_id,
                task_id=task_id,
                merged=False,
                review_reason=error_reason,
                round_num=round_num,
                start_time=review_start_time,
                end_time=datetime.now(),
            )

        return review_result

    def assign_task(self, completed_result, all_completed, running_agents, idle_agents=None, inactive_agents=None, finished_agents=None):
        engineer_id = completed_result.engineer_id

        # Determine task status based on merge result
        if completed_result.merged and completed_result.success:
            task_status = "success"
        elif completed_result.merged and not completed_result.success:
            task_status = "recovered"
        else:
            task_status = "failed"

        self.log(f"{engineer_id} completed ({task_status}), checking for next task...")

        completed_task_summary = self.task.build_completed_task_summary(completed_result, task_status)
        if completed_result.error and not completed_result.merged:
            completed_task_summary += f"\nerror: {completed_result.error}"

        running_agents_summary = "\n".join(f"  - {aid}" for aid in running_agents) if running_agents else "  none"

        idle_agents = idle_agents or []
        if idle_agents:
            idle_agents_summary = "\n".join(f"  - {aid}" for aid in idle_agents)
        else:
            idle_agents_summary = "  none"

        inactive_agents = inactive_agents or []
        if inactive_agents:
            inactive_agents_summary = "\n".join(f"  - {aid}" for aid in inactive_agents)
        else:
            inactive_agents_summary = "  none"

        finished_agents = finished_agents or []
        if finished_agents:
            finished_agents_summary = "\n".join(f"  - {aid}" for aid in finished_agents)
        else:
            finished_agents_summary = "  none"

        prompt = self.prompts.get("assign_task", "").format(
            engineer_id=engineer_id,
            task_status=task_status,
            completed_round=completed_result.round_num,
            max_rounds=self.config.max_rounds_chat,
            completed_task_summary=completed_task_summary,
            running_agents_summary=running_agents_summary,
            idle_agents_summary=idle_agents_summary,
            inactive_agents_summary=inactive_agents_summary,
            finished_agents_summary=finished_agents_summary,
        )

        # Track time and cost for this assign_task call
        assign_start_time = datetime.now()
        event_count_before = len(list(self.conversation.state.events))

        iteration_before = count_llm_iterations(self.conversation.state.events)
        metrics_before = extract_conversation_metrics(self.conversation)
        cost_before = metrics_before["cost"]
        tokens_before = metrics_before["total_tokens"]

        self.log("Deciding next task assignment...")
        self.conversation.send_message(prompt)
        try:
            self.conversation.run()
        except Exception as e:
            self.log(f"Agent run ended with: {e}")

        events = self.conversation.state.events
        iterations = count_llm_iterations(events) - iteration_before
        self.log(f"Iterations: {iterations}/{self.config.manager_max_iterations}")

        review_json = extract_json_from_events(events, key_to_find="assign_task")

        if not review_json:
            alternative = self.task.search_alternative_json(events, extract_json_from_events, self.log)
            if alternative:
                review_json = alternative

        if not review_json:
            self.log("No assign_task JSON found, no assignment")
            self.save_events("assign_and_review", event_count_before)
            return {"assignments": [], "reasoning": "No response from manager"}

        assign_data = review_json.get("assign_task", {})
        reasoning = assign_data.get("reasoning", "")

        assignments_data = self.task.extract_assignments(assign_data)

        self.log(f"Decision: {len(assignments_data)} task(s) to assign")
        self.log(f"Reasoning: {reasoning}")

        result = {"assignments": [], "reasoning": reasoning}

        # Validation sets (applies to all tasks)
        running_set = set(running_agents or [])
        finished_set = set(finished_agents or [])
        assigned_agents = set()
        assigned_tasks = set()

        assign_context = self.task.get_assign_context(all_completed, self.workspace, self.repo_dir) if assignments_data else {}

        for task_data in assignments_data:
            task_engineer_id = task_data.get("engineer_id", engineer_id)
            task_id = task_data.get("task_id", "")

            # Validate: reject assignments to running/finished/duplicate agents/tasks
            if task_engineer_id in running_set:
                self.log(f"REJECTED: Cannot assign to {task_engineer_id} - agent is currently running")
                continue
            if task_engineer_id in finished_set:
                self.log(f"REJECTED: Cannot assign to {task_engineer_id} - agent already finished all rounds")
                continue
            if task_engineer_id in assigned_agents:
                self.log(f"REJECTED: Cannot assign to {task_engineer_id} - already assigned a task")
                continue
            if task_id and task_id in assigned_tasks:
                self.log(f"REJECTED: Cannot assign {task_id} - already assigned to another agent")
                continue

            # Remove from remaining tasks
            if self.delegation_plan and self.delegation_plan.remaining_tasks:
                self.delegation_plan.remaining_tasks = [
                    t for t in self.delegation_plan.remaining_tasks
                    if t.task_id != task_id
                ]

            # Create SubAgent with all available fields
            subagent = SubAgent(
                engineer_id=task_engineer_id,
                task_id=task_id,
                file_path=task_data.get("file_path", ""),
                functions_to_implement=task_data.get("functions_to_implement", []),
                task_node_id=task_data.get("task_node_id", ""),
                requirements=task_data.get("requirements", ""),
                instruction=task_data.get("instruction", ""),
                estimated_complexity=task_data.get("estimated_complexity", "medium"),
                task_category=task_data.get("task_category"),
                submission_path=self.task.get_work_dir(),
            )

            self.task.update_subagent_for_assignment(subagent, assign_context, self.workspace, self.log)

            result["assignments"].append(subagent)
            assigned_agents.add(task_engineer_id)
            if task_id:
                assigned_tasks.add(task_id)
            self.log(f"Assigned {task_id} to {task_engineer_id}")

        if result["assignments"]:
            self.current_round = max(self.current_round, 2)

        # Calculate time and cost
        assign_end_time = datetime.now()
        assign_duration = (assign_end_time - assign_start_time).total_seconds()
        self.assign_task_total_time += assign_duration

        metrics_after = extract_conversation_metrics(self.conversation)
        assign_cost = metrics_after["cost"] - cost_before
        assign_tokens = metrics_after["total_tokens"] - tokens_before
        self.assign_task_total_cost += assign_cost
        self.assign_task_total_tokens += assign_tokens

        assigned_targets = self.task.get_assigned_targets(result["assignments"], engineer_id)

        event_content = {
            "completed_task": completed_result.task_id,
            "task_status": task_status,
            "num_assignments": len(assignments_data),
            "reasoning": reasoning,
            "assignments": [{"engineer_id": s.engineer_id, "task_id": s.task_id} for s in result["assignments"]],
            "remaining_tasks": len(self.delegation_plan.remaining_tasks) if self.delegation_plan else 0,
            "actual_iterations": iterations,
            "max_iterations": self.config.manager_max_iterations,
            "cost": assign_cost,
            "tokens": assign_tokens,
            "duration": assign_duration,
        }
        event_content.update(self.task.get_assign_event_extras(engineer_id))

        self.output_logger.log_event(
            event_type="manager_instruction",
            source="manager",
            target=assigned_targets,
            round_num=self.current_round if assignments_data else None,
            content=event_content,
            start_time=assign_start_time,
            end_time=assign_end_time,
        )

        self.save_events("assign_and_review", event_count_before)
        return result

    # ========================================================================
    # Background Exploration (commit0-specific)
    # ========================================================================

    def cancel_exploration(self):
        self.exploration_cancelled = True
        self.log("Exploration cancelled - engineer completed")
        if self.conversation:
            self.conversation.pause()

    def reset_exploration_cancel(self):
        self.exploration_cancelled = False

    def explore_background(self, remaining_tasks, running_agents_summary):
        if self.exploration_cancelled:
            self.log("Exploration skipped - already cancelled")
            self.output_logger.log_event(
                event_type="background_exploration",
                source="manager",
                content={
                    "duration": 0,
                    "cost": 0,
                    "tokens": 0,
                    "cancelled": True,
                    "skipped_early": True,
                    "remaining_tasks_explored": 0,
                },
            )
            return {"findings": [], "cancelled": True}

        self.log("Starting background exploration...")
        explore_start = datetime.now()
        event_count_before = len(list(self.conversation.state.events))

        metrics_before = extract_conversation_metrics(self.conversation)
        cost_before = metrics_before["cost"]
        tokens_before = metrics_before["total_tokens"]
        iteration_before = count_llm_iterations(self.conversation.state.events)

        remaining_files = []
        for task in remaining_tasks[:5]:
            remaining_files.append(f"- {task.file_path}: {', '.join(task.functions_to_implement[:3])}")
        remaining_str = "\n".join(remaining_files) if remaining_files else "No remaining tasks"

        prompt = self.prompts.get("background_exploration", "").format(
            remaining_tasks=remaining_str,
            running_agents_summary=running_agents_summary,
            repo_dir=self.repo_dir,
        )

        if not prompt:
            self.log("No background_exploration prompt found, skipping")
            return {"skipped": True}

        try:
            self.log("Exploring for upcoming tasks...")
            self.conversation.send_message(prompt)
            self.conversation.run()

            findings = {
                "findings": [],
                "cancelled": self.exploration_cancelled,
            }

            events = self.conversation.state.events
            iterations = count_llm_iterations(events) - iteration_before
            self.log(f"Exploration iterations: {iterations}")

            self.exploration_findings.append({
                "timestamp": datetime.now().isoformat(),
                "iterations": iterations,
                "remaining_files": [t.file_path for t in remaining_tasks[:5]],
            })

        except Exception as e:
            self.log(f"Exploration error (non-fatal): {e}")
            findings = {"findings": [], "error": str(e), "cancelled": False}

        explore_end = datetime.now()
        explore_duration = (explore_end - explore_start).total_seconds()
        self.exploration_total_time += explore_duration

        metrics_after = extract_conversation_metrics(self.conversation)
        explore_cost = metrics_after["cost"] - cost_before
        explore_tokens = metrics_after["total_tokens"] - tokens_before
        self.exploration_cost += explore_cost
        self.exploration_tokens += explore_tokens

        self.log(f"Exploration completed in {explore_duration:.1f}s (${explore_cost:.4f})")

        self.output_logger.log_event(
            event_type="background_exploration",
            source="manager",
            start_time=explore_start,
            end_time=explore_end,
            content={
                "duration": explore_duration,
                "cost": explore_cost,
                "tokens": explore_tokens,
                "cancelled": self.exploration_cancelled,
                "remaining_tasks_explored": len(remaining_tasks[:5]),
            },
        )

        self.save_events("background_exploration", event_count_before)
        return findings

    async def explore_background_async(self, remaining_tasks, running_agents_summary):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.explore_background,
            remaining_tasks,
            running_agents_summary,
        )
        
    def final_review_all(self, subagent_results, max_iterations=30):
        """Manager final review after all engineers complete their work."""
        self.log("=" * 60)
        self.log("Manager Final Review")
        self.log("=" * 60)

        final_review_start = datetime.now()
        event_count_before = len(list(self.conversation.state.events))

        metrics_before = extract_conversation_metrics(self.conversation)
        cost_before = metrics_before["cost"]
        tokens_before = metrics_before["total_tokens"]
        iteration_before = count_llm_iterations(self.conversation.state.events)

        # Build engineers summary
        engineers_summary_lines = []
        for r in subagent_results:
            if r.file_path:
                status = "committed" if r.success else "no commit"
                merged = "merged" if r.merged else "not merged"
                engineers_summary_lines.append(
                    f"- {r.engineer_id}: {r.task_id} ({r.file_path}) - {status}, {merged}"
                )
            else:
                status = "submitted" if r.success else "no submission"
                merged = "collected" if r.merged else "not collected"
                commit_msg = f" | commit: {r.commit_message}" if r.commit_message else ""
                engineers_summary_lines.append(
                    f"- {r.engineer_id}: {r.task_id} ({r.requirements[:50]}...) - {status}, {merged}{commit_msg}"
                )
        engineers_summary = "\n".join(engineers_summary_lines) if engineers_summary_lines else "No engineers completed tasks"

        # Build merged files summary
        merged_files_lines = []
        for r in subagent_results:
            if r.merged:
                if r.file_path:
                    merged_files_lines.append(f"- {r.file_path}")
                elif r.files_modified:
                    for f in r.files_modified:
                        merged_files_lines.append(f"- {f} ({r.engineer_id}: {r.task_id})")
        merged_files_summary = "\n".join(sorted(set(merged_files_lines))) if merged_files_lines else "No files merged"

        # Build unmerged worktrees section (commit0-specific, ignored by paperbench prompt)
        unmerged_lines = []
        for r in subagent_results:
            if not r.merged and r.worktree_path:
                uncommitted_files = self.get_uncommitted_changes(r.worktree_path)
                files_info = f" (uncommitted files: {', '.join(uncommitted_files)})" if uncommitted_files else " (no uncommitted files found)"
                unmerged_lines.append(
                    f"- {r.engineer_id}: {r.file_path} - worktree at {r.worktree_path}, "
                    f"branch {r.branch_name}{files_info}"
                )
        if unmerged_lines:
            unmerged_worktrees_section = (
                "<unmerged_worktrees>\n"
                "The following engineers did NOT merge their work. Their worktrees may contain useful code:\n"
                + "\n".join(unmerged_lines) +
                "\n</unmerged_worktrees>"
            )
        else:
            unmerged_worktrees_section = ""

        # Get test info (commit0-specific, ignored by paperbench prompt)
        test_info = (self.task.task_data or {}).get("test", {})
        test_cmd = test_info.get("test_cmd", (self.task.task_data or {}).get("test_cmd", "pytest"))
        test_dir = test_info.get("test_dir", (self.task.task_data or {}).get("test_dir", "tests/"))

        prompt = self.prompts.get("manager_final_review_all", "").format(
            engineers_summary=engineers_summary,
            merged_files_summary=merged_files_summary,
            unmerged_worktrees_section=unmerged_worktrees_section,
            repo_dir=self.repo_dir,
            test_cmd=test_cmd,
            test_dir=test_dir,
        )

        if not prompt:
            self.log("No manager_final_review_all prompt found, skipping")
            return {"skipped": True}

        # Temporarily change max_iterations for final review
        original_max_iter = self.conversation.max_iteration_per_run
        self.conversation.max_iteration_per_run = max_iterations

        try:
            self.log(f"Starting final review (max {max_iterations} iterations)...")
            self.conversation.send_message(prompt)
            try:
                self.conversation.run()
            except Exception as e:
                self.log(f"Agent run ended with: {e}")
        finally:
            self.conversation.max_iteration_per_run = original_max_iter

        final_review_end = datetime.now()
        final_review_duration = (final_review_end - final_review_start).total_seconds()
        self.final_review_total_time = final_review_duration

        events = self.conversation.state.events
        iterations = count_llm_iterations(events) - iteration_before

        metrics_after = extract_conversation_metrics(self.conversation)
        self.final_review_cost = metrics_after["cost"] - cost_before
        self.final_review_tokens = metrics_after["total_tokens"] - tokens_before

        self.log(f"Final review completed in {final_review_duration:.1f}s")
        self.log(f"Iterations: {iterations}/{max_iterations}")
        self.log(f"Cost: ${self.final_review_cost:.4f} ({self.final_review_tokens} tokens)")

        log_content = {
            "duration": final_review_duration,
            "cost": self.final_review_cost,
            "tokens": self.final_review_tokens,
            "actual_iterations": iterations,
            "max_iterations": max_iterations,
            "engineers_reviewed": len(subagent_results),
        }
        log_content.update(self.task.get_final_review_log_extras(subagent_results))

        self.output_logger.log_event(
            event_type="manager_final_review_all",
            source="manager",
            start_time=final_review_start,
            end_time=final_review_end,
            content=log_content,
        )

        self.save_events("final_review_all", event_count_before)
        return {
            "duration": final_review_duration,
            "cost": self.final_review_cost,
            "tokens": self.final_review_tokens,
            "iterations": iterations,
        }

    def cleanup(self):
        if self.conversation:
            try:
                self.conversation.close()
            except Exception:
                pass
