import asyncio
from datetime import datetime

from openhands.sdk import Agent, Conversation
from openhands.tools.preset.default import get_default_tools

from config import SubAgent, SubAgentResult
from core.utils import (
    PanelVisualizer,
    build_subagent_prompt,
    extract_conversation_metrics,
    count_llm_iterations,
    serialize_event,
)


class SubAgentRunner:
    def __init__(
        self,
        llm,
        workspace,
        subagent,
        prompts,
        task_module,
        max_iterations=50,
        max_rounds_chat=2,
        output_dir=None,
        output_logger=None,
    ):
        self.llm = llm
        self.workspace = workspace
        self.subagent = subagent
        self.prompts = prompts
        self.task_module = task_module
        self.max_iterations = max_iterations
        self.max_rounds_chat = max_rounds_chat
        self.output_dir = output_dir
        self.output_logger = output_logger
        self.agent = None
        self.conversation = None
        self.result = None
        self.instruction_time = None
        self.completed_rounds = 0
        self.last_result = None
        self.last_saved_event_count = 0

    def log(self, message):
        print(f"[{self.subagent.engineer_id}] {message}")

    def can_accept_more_tasks(self):
        return self.completed_rounds < self.max_rounds_chat

    def update_task(self, new_subagent):
        self.log(f"Updating task for round {new_subagent.current_round}")
        self.log(f"  - New task: {new_subagent.task_id}")
        if new_subagent.file_path:
            self.log(f"  - New file: {new_subagent.file_path}")
        self.subagent = new_subagent
        self.result = None
        self.instruction_time = datetime.now()

    def setup(self):
        self.log("Setting up subagent...")
        tools = get_default_tools(enable_browser=False)
        self.agent = Agent(
            llm=self.llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
        )
        self.conversation = Conversation(
            agent=self.agent,
            workspace=self.workspace,
            max_iteration_per_run=self.max_iterations,
            visualizer=PanelVisualizer(),
        )
        self.instruction_time = datetime.now()
        self.last_saved_event_count = 0
        self.log("Subagent ready")

    def create_result(self):
        if self.result is None:
            self.result = self.task_module.create_subagent_result(self.subagent)
        return self.result

    def build_first_round_prompt(self):
        return build_subagent_prompt(
            prompts=self.prompts,
            submission_path=self.subagent.worktree_path or self.subagent.submission_path,
            task_node_id=self.subagent.task_node_id,
            requirements=self.subagent.requirements,
            instruction=self.subagent.instruction,
            engineer_id=self.subagent.engineer_id,
            # commit0-specific fields (ignored by paperbench prompt via **kwargs)
            worktree_path=self.subagent.worktree_path or self.subagent.submission_path,
            file_path=self.subagent.file_path,
            functions=self.subagent.functions_to_implement,
            test_cmd=getattr(self.subagent, 'test_cmd', 'pytest'),
            test_dir=getattr(self.subagent, 'test_dir', 'tests/'),
        )

    def build_followup_prompt(self):
        template = self.prompts.get("followup_prompt", "")
        args = self.task_module.get_followup_prompt_args(self.subagent)
        prompt = template.format(**args)
        self.log(f"Round {self.subagent.current_round}: Sending follow-up instruction")
        return prompt

    def run(self):
        result = self.create_result()
        start_time = datetime.now()
        result.start_time = start_time.isoformat()

        self.log("Starting implementation...")
        self.log(f"  - Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        for line in self.task_module.get_run_start_log_lines(self.subagent):
            self.log(line)

        cost_before = 0.0
        prompt_tokens_before = 0
        completion_tokens_before = 0
        iteration_before = 0

        try:
            if self.subagent.current_round == 1:
                prompt = self.build_first_round_prompt()
            else:
                prompt = self.build_followup_prompt()

            max_retries = 3
            iteration_before = count_llm_iterations(self.conversation.state.events)

            metrics_before = extract_conversation_metrics(self.conversation)
            cost_before = metrics_before["cost"]
            prompt_tokens_before = metrics_before["prompt_tokens"]
            completion_tokens_before = metrics_before["completion_tokens"]

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        if self.task_module.should_setup_on_retry:
                            self.log(f"Retry attempt {attempt + 1}/{max_retries}...")
                            self.setup()
                        else:
                            self.log(f"Retry attempt {attempt + 1}/{max_retries}, resuming conversation...")

                    if self.task_module.should_resend_on_retry or attempt == 0:
                        self.conversation.send_message(prompt)
                    self.conversation.run()
                    break

                except Exception as llm_error:
                    error_str = str(llm_error)
                    retryable_errors = [
                        "invalid_encrypted_content",
                        "500 Internal Server Error",
                        "BadRequestError",
                        "rate_limit",
                    ]
                    if any(msg in error_str for msg in retryable_errors):
                        self.log(f"Transient LLM error: {error_str[:150]}")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(2 ** attempt)
                            continue
                    raise

            commit_info = self.get_commit_info()
            current_hash = commit_info.get("hash", "")
            base_commit = self.subagent.base_commit or ""
            base_short = base_commit[:8] if base_commit else ""

            if current_hash and base_short and current_hash == base_short:
                result.success = False
                result.error = "No new commit was made. Agent may have run out of iterations before committing."
                result.commit_hash = ""
                self.task_module.populate_no_commit_result(result)
                result.files_modified = []
                self.log(f"WARNING: No new commit detected (HEAD={current_hash} same as base)")
            else:
                self.task_module.populate_success_result(result, self, commit_info)
                self.print_summary(result, commit_info)

        except Exception as e:
            result.success = False
            result.error = str(e)
            self.log(f"ERROR: {e}")

        end_time = datetime.now()
        result.end_time = end_time.isoformat()
        duration = (end_time - start_time).total_seconds()
        result.duration_seconds = duration

        if self.conversation:
            metrics_after = extract_conversation_metrics(self.conversation)
            result.cost = metrics_after["cost"] - cost_before
            result.prompt_tokens = metrics_after["prompt_tokens"] - prompt_tokens_before
            result.completion_tokens = metrics_after["completion_tokens"] - completion_tokens_before
            result.total_tokens = result.prompt_tokens + result.completion_tokens
            result.actual_iterations = count_llm_iterations(self.conversation.state.events) - iteration_before
            result.max_iterations = self.max_iterations

            if self.output_logger:
                events = list(self.conversation.state.events)
                engineer_id = self.subagent.engineer_id
                new_events_start = self.last_saved_event_count
                new_events_count = len(events) - new_events_start
                self.log(f"Saving {new_events_count} new events (starting from {new_events_start}) to {engineer_id}_events.jsonl...")
                for idx in range(new_events_start, len(events)):
                    event = events[idx]
                    serialized = serialize_event(event, idx)
                    serialized["engineer_id"] = engineer_id
                    serialized["task_id"] = self.subagent.task_id
                    serialized["round_num"] = self.subagent.current_round
                    serialized.update(self.task_module.get_event_serialization_extras(self.subagent))
                    serialized["start_time"] = serialized.get("timestamp")
                    if idx + 1 < len(events):
                        next_ts = getattr(events[idx + 1], 'timestamp', None)
                        serialized["end_time"] = next_ts
                    else:
                        serialized["end_time"] = result.end_time
                    self.output_logger.log_agent_event(engineer_id, serialized)
                self.last_saved_event_count = len(events)

        self.completed_rounds += 1
        self.last_result = result

        self.log("Completed")
        self.log(f"  - Round: {self.subagent.current_round}/{self.max_rounds_chat}")
        self.log(f"  - End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"  - Duration: {duration:.1f}s")
        self.log(f"  - Iterations: {result.actual_iterations}/{result.max_iterations}")
        self.log(f"  - Cost: ${result.cost:.4f} ({result.total_tokens} tokens)")
        self.log(f"  - Can accept more tasks: {self.can_accept_more_tasks()}")

        return result

    async def run_async(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run)

    def get_commit_info(self):
        worktree_path = self.subagent.worktree_path or self.subagent.submission_path
        if not worktree_path:
            return {"hash": "", "message": "", "author": ""}

        cmd = (
            f"cd {worktree_path} && "
            f"git log -1 --format='%H|%s|%an' 2>/dev/null || echo '||'"
        )
        result = self.workspace.execute_command(cmd, timeout=30)
        stdout = (result.stdout or "").strip()
        parts = stdout.split("|", 2)
        return {
            "hash": parts[0][:8] if len(parts) > 0 else "",
            "message": parts[1] if len(parts) > 1 else "",
            "author": parts[2] if len(parts) > 2 else "",
        }

    def get_git_diff(self):
        worktree_path = self.subagent.worktree_path or self.subagent.submission_path
        base_commit = self.subagent.base_commit

        if not worktree_path:
            return ""

        if base_commit:
            cmd = f"cd {worktree_path} && git diff {base_commit}..HEAD --no-color"
            result = self.workspace.execute_command(cmd, timeout=120)
            if result.exit_code == 0 and result.stdout.strip():
                return result.stdout.strip()

        cmd = (
            f"cd {worktree_path} && "
            f"git diff HEAD~1 HEAD --no-color 2>/dev/null || "
            f"git diff --cached --no-color"
        )
        result = self.workspace.execute_command(cmd, timeout=120)
        return result.stdout.strip() if result.exit_code == 0 else ""

    def get_modified_files(self):
        worktree_path = self.subagent.worktree_path or self.subagent.submission_path
        base_commit = self.subagent.base_commit

        if not worktree_path:
            return []

        if base_commit:
            cmd = (
                f"cd {worktree_path} && "
                f"git diff --name-only {base_commit}..HEAD 2>/dev/null || echo ''"
            )
        else:
            cmd = (
                f"cd {worktree_path} && "
                f"git diff --name-only HEAD~1 HEAD 2>/dev/null || echo ''"
            )

        result = self.workspace.execute_command(cmd, timeout=30)

        if result.exit_code == 0 and result.stdout.strip():
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return []

    def get_commit_count(self):
        """Count commits since base (paperbench-specific)."""
        worktree_path = self.subagent.worktree_path or self.subagent.submission_path
        base_commit = self.subagent.base_commit

        if not worktree_path:
            return 0

        if base_commit:
            cmd = f"cd {worktree_path} && git rev-list --count {base_commit}..HEAD 2>/dev/null || echo '0'"
        else:
            cmd = f"cd {worktree_path} && git rev-list --count HEAD 2>/dev/null || echo '0'"

        result = self.workspace.execute_command(cmd, timeout=30)
        try:
            return int(result.stdout.strip())
        except ValueError:
            return 0

    def check_submission_exists(self):
        """Check if submission directory has content (paperbench-specific)."""
        worktree_path = self.subagent.worktree_path or self.subagent.submission_path
        if not worktree_path:
            return False
        cmd = f"test -d {worktree_path} && ls {worktree_path} | head -1"
        result = self.workspace.execute_command(cmd, timeout=30)
        return result.exit_code == 0 and result.stdout.strip() != ""

    def check_reproduce_script_exists(self):
        """Check if reproduce.sh exists (paperbench-specific)."""
        worktree_path = self.subagent.worktree_path or self.subagent.submission_path
        if not worktree_path:
            return False
        cmd = f"test -f {worktree_path}/reproduce.sh && echo 'exists'"
        result = self.workspace.execute_command(cmd, timeout=30)
        return "exists" in result.stdout

    def print_summary(self, result, commit_info):
        self.log("=" * 60)
        self.log("Commit Summary")
        self.log("=" * 60)
        self.log(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")
        self.log(f"  Branch: {self.subagent.branch_name}")

        if commit_info.get("hash"):
            self.log(f"  Commit: {commit_info['hash']}")
            self.log(f"  Message: {commit_info['message']}")

        for line in self.task_module.get_print_summary_lines(result, commit_info):
            self.log(line)

        self.log("=" * 60)

    def cleanup(self):
        if self.conversation:
            try:
                self.conversation.close()
            except Exception:
                pass


async def run_subagents_parallel(runners, manager=None, task_module=None, output_logger=None, enable_background_exploration=True, max_subagents=4):
    if not runners:
        print("[SubAgents] No subagents to run.")
        return []

    print(f"\n[SubAgents] Running {len(runners)} subagents in parallel...")
    for runner in runners:
        print(f"- {runner.subagent.engineer_id}: max_rounds={runner.max_rounds_chat}")

    # Track which agents have been onboarded and which have finished
    all_possible_agents = [f"engineer_{i+1}" for i in range(max_subagents)]
    active_engineer_ids = set(r.subagent.engineer_id for r in runners)
    finished_engineer_ids = set()

    async def run_single_runner(runner):
        engineer_id = runner.subagent.engineer_id
        round_num = runner.subagent.current_round
        print(f"\n[SubAgents] Starting {engineer_id} (round {round_num})...")
        try:
            result = await runner.run_async()
            return result
        except Exception as e:
            print(f"[SubAgents] ERROR running {engineer_id}: {e}")
            result = runner.create_result()
            result.success = False
            result.error = str(e)
            result.end_time = datetime.now().isoformat()
            return result

    tasks = {
        asyncio.create_task(run_single_runner(runner)): runner
        for runner in runners
    }

    results = []
    idle_runners = []
    exploration_task = None

    # Background exploration helpers (commit0-specific)
    def get_remaining_tasks():
        if manager and manager.delegation_plan:
            return manager.delegation_plan.remaining_tasks or []
        return []

    def get_running_agents_summary():
        running_info = []
        for task, runner in tasks.items():
            engineer_id = runner.subagent.engineer_id
            file_path = runner.subagent.file_path
            funcs = runner.subagent.functions_to_implement[:3]
            funcs_str = ", ".join(funcs) + ("..." if len(runner.subagent.functions_to_implement) > 3 else "")
            running_info.append(f"- {engineer_id}: {file_path} ({funcs_str})")
        return "\n".join(running_info) if running_info else "No engineers running"

    async def start_exploration_if_needed():
        nonlocal exploration_task
        if not enable_background_exploration or not manager:
            return None

        remaining = get_remaining_tasks()
        if not remaining:
            print("[Manager] No remaining tasks to explore")
            return None

        running_summary = get_running_agents_summary()
        manager.reset_exploration_cancel()
        print(f"[Manager] Starting background exploration for {len(remaining)} remaining tasks...")
        exploration_task = asyncio.create_task(
            manager.explore_background_async(remaining, running_summary)
        )
        return exploration_task

    if enable_background_exploration and manager and get_remaining_tasks():
        await start_exploration_if_needed()

    while tasks or idle_runners:
        if not tasks and idle_runners:
            print(f"\n[SubAgents] No active tasks, checking {len(idle_runners)} idle runners...")

            trigger_runner = idle_runners[0]
            running_agents = []
            idle_engineer_ids = [r.subagent.engineer_id for r in idle_runners]
            inactive_engineer_ids = [aid for aid in all_possible_agents if aid not in active_engineer_ids]

            assignment = manager.assign_task(
                completed_result=trigger_runner.last_result,
                all_completed=results,
                running_agents=running_agents,
                idle_agents=idle_engineer_ids,
                inactive_agents=inactive_engineer_ids,
                finished_agents=list(finished_engineer_ids),
            )

            idle_runners_by_id = {r.subagent.engineer_id: r for r in idle_runners}

            activated_any = False
            for new_subagent in assignment.get("assignments", []):
                target_engineer_id = new_subagent.engineer_id
                if target_engineer_id in idle_runners_by_id:
                    target_runner = idle_runners_by_id.pop(target_engineer_id)
                    new_subagent.current_round = target_runner.completed_rounds + 1

                    task_module.prepare_reuse_subagent(new_subagent, target_runner)

                    print(f"\n[SubAgents] Activating idle {target_engineer_id} with new task (round {new_subagent.current_round})")
                    print(f"- New task: {new_subagent.task_id}")
                    for line in task_module.get_new_task_print_lines(new_subagent):
                        print(line)

                    target_runner.update_task(new_subagent)

                    new_task = asyncio.create_task(run_single_runner(target_runner))
                    tasks[new_task] = target_runner
                    activated_any = True

                elif target_engineer_id in inactive_engineer_ids:
                    print(f"\n[SubAgents] Onboarding inactive engineer {target_engineer_id}...")

                    cmd_result = manager.workspace.execute_command(
                        f"cd {manager.repo_dir} && git rev-parse HEAD", timeout=30
                    )
                    base_commit = cmd_result.stdout.strip() if cmd_result.exit_code == 0 else ""

                    branch_name, worktree_name = task_module.get_onboard_names(target_engineer_id)
                    worktree_path = f"/workspace/{worktree_name}"

                    branch_cmd = (
                        f"cd {manager.repo_dir} && "
                        f"git branch {branch_name} {base_commit} 2>/dev/null || true"
                    )
                    manager.workspace.execute_command(branch_cmd, timeout=30)

                    worktree_cmd = (
                        f"cd {manager.repo_dir} && "
                        f"git worktree add {worktree_path} {branch_name}"
                    )
                    worktree_result = manager.workspace.execute_command(worktree_cmd, timeout=60)

                    if worktree_result.exit_code != 0:
                        print(f"[SubAgents] Failed to create worktree for {target_engineer_id}: {worktree_result.stderr}")
                        continue

                    new_subagent.branch_name = branch_name
                    new_subagent.worktree_path = worktree_path
                    new_subagent.base_commit = base_commit
                    task_module.post_onboard_subagent(new_subagent, manager.repo_dir)
                    new_subagent.current_round = 1

                    template_runner = trigger_runner
                    new_runner = SubAgentRunner(
                        llm=template_runner.llm,
                        workspace=template_runner.workspace,
                        subagent=new_subagent,
                        prompts=template_runner.prompts,
                        task_module=task_module,
                        max_iterations=template_runner.max_iterations,
                        max_rounds_chat=template_runner.max_rounds_chat,
                        output_dir=template_runner.output_dir,
                        output_logger=template_runner.output_logger,
                    )
                    new_runner.setup()

                    print(f"- Worktree: {worktree_path}")
                    print(f"- Branch: {branch_name}")
                    print(f"- Task: {new_subagent.task_id}")
                    for line in task_module.get_new_task_print_lines(new_subagent):
                        print(line)

                    new_task = asyncio.create_task(run_single_runner(new_runner))
                    tasks[new_task] = new_runner
                    active_engineer_ids.add(target_engineer_id)
                    activated_any = True
                    print(f"[SubAgents] {target_engineer_id} onboarded and added to task pool")

            idle_runners = list(idle_runners_by_id.values())

            if not activated_any and not tasks:
                print(f"[SubAgents] No more tasks can be assigned. {len(idle_runners)} agents remain idle.")
                break

            continue

        # Build the set of tasks to wait on (engineers + optional exploration)
        wait_tasks = set(tasks.keys())
        if exploration_task and not exploration_task.done():
            wait_tasks.add(exploration_task)

        done, _ = await asyncio.wait(
            wait_tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        # Separate exploration completion from engineer completion
        exploration_completed = False
        engineer_tasks_done = []
        for task in done:
            if task == exploration_task:
                exploration_completed = True
                print("[Manager] Background exploration completed")
                try:
                    explore_result = task.result()
                    if explore_result.get("cancelled"):
                        print("[Manager] Exploration was cancelled")
                except Exception as e:
                    print(f"[Manager] Exploration error: {e}")
                exploration_task = None
            else:
                engineer_tasks_done.append(task)

        # If only exploration completed (no engineers), just continue waiting
        if exploration_completed and not engineer_tasks_done and tasks:
            continue

        # Engineers completed - signal exploration to stop
        if engineer_tasks_done and exploration_task and not exploration_task.done():
            print("[Manager] Engineer finished - stopping exploration immediately...")
            if manager:
                manager.cancel_exploration()
            exploration_task = None

        # Sort completed tasks by end_time to process in completion order
        completed_with_results = []
        for task in engineer_tasks_done:
            runner = tasks[task]
            try:
                result = task.result()
                end_time = result.end_time if result.end_time else datetime.now().isoformat()
                completed_with_results.append((task, result, end_time))
            except Exception as e:
                error_result = runner.create_result()
                error_result.success = False
                error_result.error = str(e)
                error_result.end_time = datetime.now().isoformat()
                completed_with_results.append((task, error_result, error_result.end_time))

        completed_with_results.sort(key=lambda x: x[2])

        for completed_task, result, _ in completed_with_results:
            runner = tasks.pop(completed_task)
            engineer_id = runner.subagent.engineer_id

            try:
                results.append(result)

                print(f"\n[SubAgents] {engineer_id} completed (round {result.round_num})")
                print(f"- Success: {result.success}")
                print(f"- Completed rounds: {runner.completed_rounds}/{runner.max_rounds_chat}")
                for line in task_module.get_completion_print_lines(result):
                    print(line)

                if output_logger:
                    log_kwargs = task_module.get_log_agent_response_kwargs(result)
                    output_logger.log_agent_response(**log_kwargs)

                if manager:
                    collect_result = manager.collect_and_merge(result, output_logger)
                    result.merged = collect_result.get("merged", False)
                    result.merge_method = collect_result.get("merge_method", "")
                    conflict_files = collect_result.get("conflict_files", [])

                    # Merge conflict: engineer must resolve it in their worktree
                    if result.merge_method == "conflict" and conflict_files:
                        if runner.can_accept_more_tasks():
                            print(f"\n[SubAgents] {engineer_id} has merge conflict - assigning conflict resolution task")
                            print(f"- Conflict files: {', '.join(conflict_files)}")

                            runner.result = None
                            runner.subagent.current_round = runner.completed_rounds + 1

                            conflict_args = task_module.get_conflict_instruction_args(
                                runner.subagent, conflict_files, manager.workspace, manager.repo_dir
                            )
                            conflict_template = runner.prompts.get("conflict_resolution", "")
                            runner.subagent.instruction = conflict_template.format(**conflict_args)

                            print(f"- Task: {runner.subagent.task_id} (conflict resolution)")
                            print(f"- Round: {runner.subagent.current_round}")

                            new_task = asyncio.create_task(run_single_runner(runner))
                            tasks[new_task] = runner
                            continue
                        else:
                            print(f"\n[SubAgents] {engineer_id} has merge conflict but no rounds left - deferring to manager final review")
                            result.conflict_files = conflict_files

                    # Auto-reassign if engineer didn't commit (merged=False)
                    if not result.merged and not conflict_files and runner.can_accept_more_tasks():
                        print(f"\n[SubAgents] {engineer_id} didn't commit - auto-reassigning same task to continue")

                        runner.result = None
                        runner.subagent.current_round = runner.completed_rounds + 1

                        reassign_args = task_module.get_auto_reassign_instruction_args(runner.subagent)
                        reassign_template = runner.prompts.get("auto_reassign", "")
                        runner.subagent.instruction = reassign_template.format(**reassign_args)

                        print(f"- Task: {runner.subagent.task_id}")
                        for line in task_module.get_new_task_print_lines(runner.subagent):
                            print(line)
                        print(f"- Round: {runner.subagent.current_round}")

                        new_task = asyncio.create_task(run_single_runner(runner))
                        tasks[new_task] = runner
                        continue

                    if runner.can_accept_more_tasks():
                        running_agents = [tasks[t].subagent.engineer_id for t in tasks]
                        idle_engineer_ids = [r.subagent.engineer_id for r in idle_runners]
                        inactive_engineer_ids = [aid for aid in all_possible_agents if aid not in active_engineer_ids]

                        assignment = manager.assign_task(
                            completed_result=result,
                            all_completed=results,
                            running_agents=running_agents,
                            idle_agents=idle_engineer_ids,
                            inactive_agents=inactive_engineer_ids,
                            finished_agents=list(finished_engineer_ids),
                        )

                        # Build a map of all available runners (completed + idle)
                        available_runners = {engineer_id: runner}
                        for idle_runner in idle_runners:
                            available_runners[idle_runner.subagent.engineer_id] = idle_runner

                        # Process all assignments from manager
                        assigned_engineer_ids = set()
                        for new_subagent in assignment.get("assignments", []):
                            target_engineer_id = new_subagent.engineer_id
                            if target_engineer_id in available_runners:
                                target_runner = available_runners[target_engineer_id]
                                new_subagent.current_round = target_runner.completed_rounds + 1

                                task_module.prepare_reuse_subagent(new_subagent, target_runner)

                                print(f"\n[SubAgents] Assigning {target_engineer_id} with new task (round {new_subagent.current_round})")
                                print(f"- New task: {new_subagent.task_id}")
                                for line in task_module.get_new_task_print_lines(new_subagent):
                                    print(line)

                                target_runner.update_task(new_subagent)

                                new_task = asyncio.create_task(run_single_runner(target_runner))
                                tasks[new_task] = target_runner
                                assigned_engineer_ids.add(target_engineer_id)
                                print(f"[SubAgents] {target_engineer_id} added back to task pool for round {new_subagent.current_round}")

                            elif target_engineer_id in inactive_engineer_ids:
                                print(f"\n[SubAgents] Onboarding inactive engineer {target_engineer_id}...")

                                cmd_result = manager.workspace.execute_command(
                                    f"cd {manager.repo_dir} && git rev-parse HEAD", timeout=30
                                )
                                base_commit = cmd_result.stdout.strip() if cmd_result.exit_code == 0 else ""

                                branch_name, worktree_name = task_module.get_onboard_names(target_engineer_id)
                                worktree_path = f"/workspace/{worktree_name}"

                                branch_cmd = (
                                    f"cd {manager.repo_dir} && "
                                    f"git branch {branch_name} {base_commit} 2>/dev/null || true"
                                )
                                manager.workspace.execute_command(branch_cmd, timeout=30)

                                worktree_cmd = (
                                    f"cd {manager.repo_dir} && "
                                    f"git worktree add {worktree_path} {branch_name}"
                                )
                                worktree_result = manager.workspace.execute_command(worktree_cmd, timeout=60)

                                if worktree_result.exit_code != 0:
                                    print(f"[SubAgents] Failed to create worktree for {target_engineer_id}: {worktree_result.stderr}")
                                    continue

                                new_subagent.branch_name = branch_name
                                new_subagent.worktree_path = worktree_path
                                new_subagent.base_commit = base_commit
                                task_module.post_onboard_subagent(new_subagent, manager.repo_dir)
                                new_subagent.current_round = 1

                                template_runner = runner
                                new_runner = SubAgentRunner(
                                    llm=template_runner.llm,
                                    workspace=template_runner.workspace,
                                    subagent=new_subagent,
                                    prompts=template_runner.prompts,
                                    task_module=task_module,
                                    max_iterations=template_runner.max_iterations,
                                    max_rounds_chat=template_runner.max_rounds_chat,
                                    output_dir=template_runner.output_dir,
                                    output_logger=template_runner.output_logger,
                                )
                                new_runner.setup()

                                print(f"- Worktree: {worktree_path}")
                                print(f"- Branch: {branch_name}")
                                print(f"- Task: {new_subagent.task_id}")
                                for line in task_module.get_new_task_print_lines(new_subagent):
                                    print(line)

                                new_task = asyncio.create_task(run_single_runner(new_runner))
                                tasks[new_task] = new_runner
                                assigned_engineer_ids.add(target_engineer_id)
                                active_engineer_ids.add(target_engineer_id)
                                print(f"[SubAgents] {target_engineer_id} onboarded and added to task pool")

                        # Move completed runner to idle if not assigned
                        if engineer_id not in assigned_engineer_ids:
                            print(f"[SubAgents] {engineer_id} moved to idle pool (waiting for dependencies)")
                            idle_runners.append(runner)

                        # Update idle_runners: remove those that got assigned
                        idle_runners = [r for r in idle_runners if r.subagent.engineer_id not in assigned_engineer_ids]
                    else:
                        print(f"[SubAgents] {engineer_id} reached max rounds ({runner.max_rounds_chat})")
                        finished_engineer_ids.add(engineer_id)
                        print(f"[SubAgents] {engineer_id} marked as finished")

            except Exception as e:
                print(f"[SubAgents] ERROR processing {engineer_id}: {e}")
                error_result = runner.create_result()
                error_result.success = False
                error_result.error = str(e)
                error_result.end_time = datetime.now().isoformat()
                results.append(error_result)

                if output_logger:
                    output_logger.log_agent_response(
                        engineer_id=error_result.engineer_id,
                        task_id=error_result.task_id,
                        success=False,
                        error=str(e),
                        round_num=error_result.round_num,
                    )

        # Status update for remaining active tasks
        if tasks:
            remaining_info = []
            for t in tasks:
                r = tasks[t]
                remaining_info.append(f"{r.subagent.engineer_id}(r{r.subagent.current_round})")
            idle_info = [r.subagent.engineer_id for r in idle_runners] if idle_runners else []
            print(f"\n[SubAgents] Active: {remaining_info}, Idle: {idle_info}")

    for line in task_module.get_execution_summary_lines(results):
        print(line)

    return results
