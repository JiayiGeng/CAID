from abc import ABC, abstractmethod


class TaskModule(ABC):
    """Abstract base class for task-specific modules. Implement this interface to add a new task to the multi-agent workflow."""

    @abstractmethod
    def get_docker_image(self):
        """Return the Docker image name for the task workspace."""
        ...

    @abstractmethod
    def get_work_dir(self):
        """Return the working directory inside the container."""
        ...

    @abstractmethod
    def get_workspace_config(self):
        """Return config dict for workspace construction."""
        ...

    @abstractmethod
    def load_task_data(self):
        """Load task-specific data. Stores loaded data internally and returns it."""
        ...

    @abstractmethod
    def setup_workspace(self, workspace):
        """Initialize the workspace after Docker container is up."""
        ...

    @abstractmethod
    def evaluate(self, workspace):
        """Run task-specific evaluation and return results dict."""
        ...

    @abstractmethod
    def get_prompt_format_args(self, config):
        """Return a dict of variables for formatting prompt templates."""
        ...

    # ---- Manager integration methods (override in subclasses as needed) ----

    def post_load_task_data(self):
        """Perform any post-load processing (e.g. load rubric). Return list of log messages."""
        return []

    def get_scan_log_kwargs(self, config):
        """Return kwargs for output_logger.log_scan_start()."""
        return {"max_iterations": config.manager_max_iterations}

    def build_analysis_from_state(self):
        """Build AnalysisResult from pre-loaded state. Return (AnalysisResult, log_messages) or (None, [])."""
        return None, []

    def check_existing_delegation(self, events, extract_fn):
        """Check if events already contain a valid delegation. Return True to skip re-prompting."""
        return False

    @abstractmethod
    def build_subagent(self, engineer_id, primary_task, all_tasks):
        """Create a SubAgent from delegated task(s). Return (SubAgent, combine_log_msg_or_None)."""
        ...

    @abstractmethod
    def get_worktree_name(self, engineer_id):
        """Return the worktree directory name for this engineer."""
        ...

    def get_subagent_log_lines(self, subagent):
        """Return extra log lines describing subagent details after onboarding."""
        return []

    @property
    def should_stash_before_merge(self):
        """Whether to stash dirty working tree before git merge."""
        return False

    @property
    def should_try_uncommitted_merge(self):
        """Whether to try committing+merging uncommitted worktree changes as fallback."""
        return False

    @abstractmethod
    def build_completed_task_summary(self, result, task_status):
        """Build a text summary of a completed task for the assign_task prompt."""
        ...

    def search_alternative_json(self, events, extract_fn, log_fn):
        """Search for alternative JSON formats if assign_task JSON not found. Return normalized dict or None."""
        return None

    def extract_assignments(self, assign_data):
        """Extract the assignments list from parsed assign_task JSON."""
        assignments = assign_data.get("assignments", [])
        if not assignments and "next_task" in assign_data:
            if assign_data.get("should_assign", False):
                assignments = [assign_data["next_task"]]
        return assignments

    def get_assign_context(self, all_completed, workspace, repo_dir):
        """Return context dict for assignment processing."""
        return {}

    def update_subagent_for_assignment(self, subagent, context, workspace, log_fn):
        """Update SubAgent with task-specific assignment context."""
        subagent.status = "ready"

    def get_assigned_targets(self, assignments, default_engineer_id):
        """Return target string for the manager_instruction log event."""
        return default_engineer_id

    def get_assign_event_extras(self, engineer_id):
        """Return extra fields for the assign_task log event content dict."""
        return {}

    @abstractmethod
    def get_single_agent_info(self, workspace, config, prompts):
        """Return (header_text, user_instruction, log_content) for single agent mode."""
        ...

    def get_final_review_log_extras(self, subagent_results):
        """Return extra fields for the final_review_all log event."""
        return {}

    def get_collect_extra_log(self, subagent_result):
        """Return extra log text for collect_and_merge, or empty string."""
        return ""

    # ---- SubAgent runner integration methods ----

    @abstractmethod
    def create_subagent_result(self, subagent):
        """Create a SubAgentResult with task-specific fields populated."""
        ...

    @abstractmethod
    def get_followup_prompt_args(self, subagent):
        """Return format args dict for the followup_prompt yaml template."""
        ...

    @abstractmethod
    def get_run_start_log_lines(self, subagent):
        """Return list of log lines to print at subagent run start."""
        ...

    @property
    def should_setup_on_retry(self):
        """Whether to call setup() again on LLM retry."""
        return False

    @property
    def should_resend_on_retry(self):
        """Whether to re-send the prompt on retry (vs just resuming)."""
        return False

    def populate_no_commit_result(self, result):
        """Set extra fields on result when no new commit was detected."""
        pass

    @abstractmethod
    def populate_success_result(self, result, runner, commit_info):
        """Set task-specific fields on result after successful run.
        commit_info is already fetched by the caller (avoids double git call)."""
        ...

    @abstractmethod
    def get_event_serialization_extras(self, subagent):
        """Return dict of extra fields for event serialization."""
        ...

    @abstractmethod
    def get_print_summary_lines(self, result, commit_info):
        """Return list of log lines for the commit summary."""
        ...

    def prepare_reuse_subagent(self, new_subagent, old_runner):
        """Copy task-specific info from old runner to reused subagent (e.g. worktree)."""
        pass

    @abstractmethod
    def get_new_task_print_lines(self, subagent):
        """Return list of print lines when a new task is assigned to a runner."""
        ...

    @abstractmethod
    def get_onboard_names(self, engineer_id):
        """Return (branch_name, worktree_name) for onboarding a new engineer."""
        ...

    def post_onboard_subagent(self, subagent, repo_dir):
        """Set task-specific fields on subagent after worktree onboarding."""
        pass

    @abstractmethod
    def get_completion_print_lines(self, result):
        """Return list of print lines when a subagent completes."""
        ...

    @abstractmethod
    def get_log_agent_response_kwargs(self, result):
        """Return kwargs dict for output_logger.log_agent_response()."""
        ...

    @abstractmethod
    def get_conflict_instruction_args(self, subagent, conflict_files, workspace, repo_dir):
        """Return format args dict for the conflict_resolution yaml template."""
        ...

    def get_auto_reassign_instruction_args(self, subagent):
        """Return format args dict for the auto_reassign yaml template."""
        return {"original_instruction": subagent.instruction}

    @abstractmethod
    def get_execution_summary_lines(self, results):
        """Return list of print lines for the final execution summary."""
        ...
