from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkflowConfig:
    """Shared workflow configuration across all tasks."""
    model: Optional[str] = None
    subagent_model: Optional[str] = None
    manager_max_iterations: int = 50
    max_subagents: int = 4
    subagent_max_iterations: int = 50
    max_rounds_chat: int = 2
    output_dir: str = "outputs"


@dataclass
class PaperInfo:
    paper_id: str = ""
    title: str = ""
    paper_pdf_path: str = ""
    paper_md_path: str = ""
    rubric_path: str = ""
    addendum_path: str = ""
    blacklist_path: str = ""
    assets_dir: str = ""

    def to_dict(self):
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "paper_pdf_path": self.paper_pdf_path,
            "paper_md_path": self.paper_md_path,
            "rubric_path": self.rubric_path,
            "addendum_path": self.addendum_path,
            "blacklist_path": self.blacklist_path,
            "assets_dir": self.assets_dir,
        }


@dataclass
class TaskNode:
    id: str = ""
    requirements: str = ""
    weight: int = 1
    task_category: Optional[str] = None
    finegrained_task_category: Optional[str] = None
    sub_tasks: list = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "requirements": self.requirements,
            "weight": self.weight,
            "task_category": self.task_category,
            "finegrained_task_category": self.finegrained_task_category,
            "sub_tasks": [t.to_dict() for t in self.sub_tasks],
        }

    def is_leaf(self):
        return len(self.sub_tasks) == 0

    def get_leaf_nodes(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for sub in self.sub_tasks:
            leaves.extend(sub.get_leaf_nodes())
        return leaves

    @classmethod
    def from_dict(cls, data):
        sub_tasks = [cls.from_dict(t) for t in data.get("sub_tasks", [])]
        return cls(
            id=data.get("id", ""),
            requirements=data.get("requirements", ""),
            weight=data.get("weight", 1),
            task_category=data.get("task_category"),
            finegrained_task_category=data.get("finegrained_task_category"),
            sub_tasks=sub_tasks,
        )


@dataclass
class AnalysisResult:
    # commit0 fields
    repo_context: str = ""
    total_funcs: int = 0
    pass_files: list = field(default_factory=list)
    functions_by_file: dict = field(default_factory=dict)
    blocking_dependencies: dict = field(default_factory=dict)
    dependency_details: dict = field(default_factory=dict)
    implementation_order: list = field(default_factory=list)
    # paperbench fields
    paper_context: str = ""
    total_tasks: int = 0
    leaf_tasks: list = field(default_factory=list)
    task_tree: Optional[TaskNode] = None
    task_categories: dict = field(default_factory=dict)
    # common fields
    priority_reasoning: str = ""
    raw_analysis: dict = field(default_factory=dict)


@dataclass
class SubAgentTask:
    engineer_id: str = ""
    task_id: str = ""
    task_node_id: str = ""
    requirements: str = ""
    instruction: str = ""
    context: str = ""
    estimated_complexity: str = "medium"
    task_category: Optional[str] = None
    depends_on: list = field(default_factory=list)
    reason_for_delay: str = ""
    # commit0-specific fields (empty for paperbench)
    file_path: str = ""
    functions_to_implement: list = field(default_factory=list)


@dataclass
class DelegationPlan:
    num_agents: int = 0
    reasoning: str = ""
    first_round_tasks: list = field(default_factory=list)
    remaining_tasks: list = field(default_factory=list)
    raw_delegation: dict = field(default_factory=dict)


@dataclass
class SubAgent:
    engineer_id: str = ""
    task_id: str = ""
    task_node_id: str = ""
    requirements: str = ""
    instruction: str = ""
    estimated_complexity: str = "medium"
    task_category: Optional[str] = None
    submission_path: Optional[str] = None
    worktree_path: Optional[str] = None
    branch_name: Optional[str] = None
    base_commit: Optional[str] = None
    status: str = "pending"
    current_round: int = 1
    # commit0-specific fields (empty for paperbench)
    file_path: str = ""
    functions_to_implement: list = field(default_factory=list)

    def __post_init__(self):
        if not self.branch_name and self.engineer_id:
            self.branch_name = f"feature/{self.engineer_id}"

    def to_dict(self):
        return {
            "engineer_id": self.engineer_id,
            "task_id": self.task_id,
            "task_node_id": self.task_node_id,
            "requirements": self.requirements,
            "instruction": self.instruction,
            "estimated_complexity": self.estimated_complexity,
            "task_category": self.task_category,
            "submission_path": self.submission_path,
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "base_commit": self.base_commit,
            "status": self.status,
            "current_round": self.current_round,
            "file_path": self.file_path,
            "functions_to_implement": self.functions_to_implement,
        }


@dataclass
class SubAgentResult:
    engineer_id: str = ""
    task_id: str = ""
    task_node_id: str = ""
    branch_name: str = ""
    worktree_path: str = ""
    # commit0-specific fields
    file_path: str = ""
    functions_implemented: list = field(default_factory=list)
    git_diff: str = ""
    # paperbench-specific fields
    requirements: str = ""
    submission_exists: bool = False
    reproduce_script_exists: bool = False
    git_commits: int = 0
    # common fields
    success: bool = False
    error: Optional[str] = None
    commit_hash: Optional[str] = None
    commit_message: Optional[str] = None
    files_modified: list = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    actual_iterations: int = 0
    max_iterations: int = 0
    round_num: int = 1
    merged: bool = False
    merge_method: str = ""
    conflict_files: list = field(default_factory=list)

    def to_dict(self):
        return {
            "engineer_id": self.engineer_id,
            "task_id": self.task_id,
            "task_node_id": self.task_node_id,
            "branch_name": self.branch_name,
            "worktree_path": self.worktree_path,
            "file_path": self.file_path,
            "functions_implemented": self.functions_implemented,
            "git_diff": self.git_diff[:2000] if self.git_diff else "",
            "requirements": self.requirements,
            "submission_exists": self.submission_exists,
            "reproduce_script_exists": self.reproduce_script_exists,
            "git_commits": self.git_commits,
            "success": self.success,
            "error": self.error,
            "commit_hash": self.commit_hash,
            "commit_message": self.commit_message,
            "files_modified": self.files_modified,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "cost": self.cost,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "actual_iterations": self.actual_iterations,
            "max_iterations": self.max_iterations,
            "round_num": self.round_num,
            "merged": self.merged,
            "merge_method": self.merge_method,
            "conflict_files": self.conflict_files,
        }
