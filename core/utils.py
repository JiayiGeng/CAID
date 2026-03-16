import json
import os
import re
import sys
import subprocess
import base64
import platform as plat
from pathlib import Path
from datetime import datetime

import yaml
from pydantic import SecretStr
from tasks import Commit0Task, Commit0Config, PaperbenchTask, PaperbenchConfig
from config import AnalysisResult, PaperInfo, TaskNode, SubAgentTask, DelegationPlan
from rich.console import Console
from rich.panel import Panel

from openhands.sdk.conversation.visualizer.base import ConversationVisualizerBase
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    ConversationStateUpdateEvent,
    MessageEvent,
    ObservationEvent,
    PauseEvent,
    SystemPromptEvent,
    UserRejectObservation,
)
from openhands.sdk.event.condenser import Condensation, CondensationRequest


OBSERVATION_COLOR = "yellow"
MESSAGE_USER_COLOR = "gold3"
PAUSE_COLOR = "bright_yellow"
SYSTEM_COLOR = "magenta"
THOUGHT_COLOR = "bright_black"
ERROR_COLOR = "red"
ACTION_COLOR = "blue"
MESSAGE_ASSISTANT_COLOR = ACTION_COLOR

DEFAULT_HIGHLIGHT_REGEX = {
    r"^Reasoning:": f"bold {THOUGHT_COLOR}",
    r"^Thought:": f"bold {THOUGHT_COLOR}",
    r"^Action:": f"bold {ACTION_COLOR}",
    r"^Arguments:": f"bold {ACTION_COLOR}",
    r"^Tool:": f"bold {OBSERVATION_COLOR}",
    r"^Result:": f"bold {OBSERVATION_COLOR}",
    r"^Rejection Reason:": f"bold {ERROR_COLOR}",
    r"\*\*(.*?)\*\*": "bold",
    r"\*(.*?)\*": "italic",
}

PANEL_PADDING = (1, 1)


class PanelVisualizer(ConversationVisualizerBase):

    def __init__(self, highlight_regex=None, skip_user_messages=False):
        super().__init__()
        self.console = Console()
        self.skip_user_messages = skip_user_messages
        self.highlight_patterns = highlight_regex if highlight_regex is not None else DEFAULT_HIGHLIGHT_REGEX

    def on_event(self, event):
        panel = self.create_event_panel(event)
        if panel:
            self.console.print(panel)
            self.console.print()

    def apply_highlighting(self, text):
        if not self.highlight_patterns:
            return text
        highlighted = text.copy()
        for pattern, style in self.highlight_patterns.items():
            pattern_compiled = re.compile(pattern, re.MULTILINE)
            highlighted.highlight_regex(pattern_compiled, style)
        return highlighted

    def format_metrics_subtitle(self):
        stats = self.conversation_stats
        if not stats:
            return None
        combined_metrics = stats.get_combined_metrics()
        if not combined_metrics or not combined_metrics.accumulated_token_usage:
            return None
        usage = combined_metrics.accumulated_token_usage
        cost = combined_metrics.accumulated_cost or 0.0

        def abbr(n):
            n = int(n or 0)
            if n >= 1_000_000:
                return f"{n / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
            elif n >= 1_000:
                return f"{n / 1_000:.2f}".rstrip("0").rstrip(".") + "K"
            return str(n)

        input_tokens = abbr(usage.prompt_tokens or 0)
        output_tokens = abbr(usage.completion_tokens or 0)
        prompt = usage.prompt_tokens or 0
        cache_read = usage.cache_read_tokens or 0
        cache_rate = f"{(cache_read / prompt * 100):.2f}%" if prompt > 0 else "N/A"
        cost_str = f"{cost:.4f}" if cost > 0 else "0.00"

        return f"Tokens: input {input_tokens} | cache hit {cache_rate} | output {output_tokens} | $ {cost_str}"

    def create_event_panel(self, event):
        content = event.visualize
        if not content.plain.strip():
            return None
        if self.highlight_patterns:
            content = self.apply_highlighting(content)

        if isinstance(event, SystemPromptEvent):
            title = f"[bold {SYSTEM_COLOR}]System Prompt[/bold {SYSTEM_COLOR}]"
            return Panel(content, title=title, border_style=SYSTEM_COLOR, padding=PANEL_PADDING, expand=True)
        elif isinstance(event, ActionEvent):
            if event.action is None:
                title = f"[bold {ACTION_COLOR}]Agent Action (Not Executed)[/bold {ACTION_COLOR}]"
            else:
                title = f"[bold {ACTION_COLOR}]Agent Action[/bold {ACTION_COLOR}]"
            return Panel(content, title=title, subtitle=self.format_metrics_subtitle(), border_style=ACTION_COLOR, padding=PANEL_PADDING, expand=True)
        elif isinstance(event, ObservationEvent):
            title = f"[bold {OBSERVATION_COLOR}]Observation[/bold {OBSERVATION_COLOR}]"
            return Panel(content, title=title, border_style=OBSERVATION_COLOR, padding=PANEL_PADDING, expand=True)
        elif isinstance(event, UserRejectObservation):
            title = f"[bold {ERROR_COLOR}]User Rejected Action[/bold {ERROR_COLOR}]"
            return Panel(content, title=title, border_style=ERROR_COLOR, padding=PANEL_PADDING, expand=True)
        elif isinstance(event, MessageEvent):
            if event.llm_message and event.llm_message.role == "user":
                if self.skip_user_messages:
                    return None
                title = f"[bold {MESSAGE_USER_COLOR}]Message from User[/bold {MESSAGE_USER_COLOR}]"
                color = MESSAGE_USER_COLOR
            else:
                title = f"[bold {MESSAGE_ASSISTANT_COLOR}]Message from Agent[/bold {MESSAGE_ASSISTANT_COLOR}]"
                color = MESSAGE_ASSISTANT_COLOR
            return Panel(content, title=title, subtitle=self.format_metrics_subtitle(), border_style=color, padding=PANEL_PADDING, expand=True)
        elif isinstance(event, AgentErrorEvent):
            title = f"[bold {ERROR_COLOR}]Agent Error[/bold {ERROR_COLOR}]"
            return Panel(content, title=title, subtitle=self.format_metrics_subtitle(), border_style=ERROR_COLOR, padding=PANEL_PADDING, expand=True)
        elif isinstance(event, PauseEvent):
            title = f"[bold {PAUSE_COLOR}]User Paused[/bold {PAUSE_COLOR}]"
            return Panel(content, title=title, border_style=PAUSE_COLOR, padding=PANEL_PADDING, expand=True)
        elif isinstance(event, Condensation):
            title = "[bold white]Condensation[/bold white]"
            return Panel(content, title=title, subtitle=self.format_metrics_subtitle(), border_style="white", padding=PANEL_PADDING, expand=True)
        elif isinstance(event, CondensationRequest):
            title = f"[bold {SYSTEM_COLOR}]Condensation Request[/bold {SYSTEM_COLOR}]"
            return Panel(content, title=title, border_style=SYSTEM_COLOR, padding=PANEL_PADDING, expand=True)
        elif isinstance(event, ConversationStateUpdateEvent):
            return None
        else:
            title = f"[bold white]UNKNOWN Event: {type(event).__name__}[/bold white]"
            return Panel(content, title=title, border_style="white", padding=PANEL_PADDING, expand=True)


class OutputLogger:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "outputs.jsonl"
        self.events_file = self.output_dir / "events.jsonl"
        self.event_counter = 0

        self.agent_logs_dir = self.output_dir / "agent_logs"
        self.agent_logs_dir.mkdir(parents=True, exist_ok=True)

        self.agent_events_dir = self.output_dir / "agent_events"
        self.agent_events_dir.mkdir(parents=True, exist_ok=True)
        self.agent_event_counters = {}

        if self.output_file.exists():
            self.output_file.unlink()
        if self.events_file.exists():
            self.events_file.unlink()

    def log(self, agent_id, message):
        log_file = self.agent_logs_dir / f"{agent_id}.log"
        with open(log_file, "a") as f:
            f.write(f"{message}\n")

    def get_agent_log_file(self, agent_id):
        return self.agent_logs_dir / f"{agent_id}.log"

    def log_event(
        self,
        event_type,
        source,
        target=None,
        content=None,
        start_time=None,
        end_time=None,
        round_num=None,
    ):
        self.event_counter += 1
        now = datetime.now()

        event_end_time = end_time or now
        event_start_time = start_time or event_end_time

        event = {
            "event_id": self.event_counter,
            "start_time": event_start_time.isoformat(),
            "end_time": event_end_time.isoformat(),
            "start_time_unix": event_start_time.timestamp(),
            "end_time_unix": event_end_time.timestamp(),
            "event_type": event_type,
            "source": source,
            "target": target,
            "round_num": round_num,
            "content": content or {},
        }

        with open(self.output_file, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event

    def log_raw_event(self, event_data):
        with open(self.events_file, "a") as f:
            f.write(json.dumps(event_data) + "\n")

    def get_agent_events_file(self, agent_id):
        return self.agent_events_dir / f"{agent_id}_events.jsonl"

    def log_scan_start(self, max_iterations=None, **kwargs):
        content = {
            "phase": "scan_and_analysis",
            "max_iterations": max_iterations,
        }
        content.update(kwargs)
        return self.log_event(
            event_type="scan_start",
            source="manager",
            content=content,
        )

    def log_manager_instruction(
        self,
        engineer_id,
        task_id,
        instruction,
        context="",
        round_num=1,
        start_time=None,
        end_time=None,
        # commit0-specific
        file_path=None,
        functions_to_implement=None,
        # paperbench-specific
        task_node_id=None,
        requirements=None,
    ):
        content = {
            "task_id": task_id,
            "instruction": instruction,
            "context": context,
        }
        if file_path is not None:
            content["file_path"] = file_path
            content["functions_to_implement"] = functions_to_implement or []
        if task_node_id is not None:
            content["task_node_id"] = task_node_id
            content["requirements"] = requirements or ""
        return self.log_event(
            event_type="manager_instruction",
            source="manager",
            target=engineer_id,
            round_num=round_num,
            content=content,
            start_time=start_time,
            end_time=end_time,
        )

    def log_agent_response(
        self,
        engineer_id,
        task_id,
        success,
        files_modified=None,
        error=None,
        duration_seconds=None,
        actual_iterations=None,
        max_iterations=None,
        cost=None,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        start_time=None,
        end_time=None,
        round_num=1,
        # commit0-specific
        commit_hash=None,
        git_diff=None,
        # paperbench-specific
        submission_exists=False,
        reproduce_script_exists=False,
        git_commits=0,
    ):
        content = {
            "task_id": task_id,
            "success": success,
            "files_modified": files_modified or [],
            "error": error,
            "duration_seconds": duration_seconds,
            "actual_iterations": actual_iterations,
            "max_iterations": max_iterations,
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if commit_hash is not None or git_diff is not None:
            content["commit_hash"] = commit_hash
            content["git_diff"] = git_diff
        else:
            content["submission_exists"] = submission_exists
            content["reproduce_script_exists"] = reproduce_script_exists
            content["git_commits"] = git_commits
        return self.log_event(
            event_type="agent_response",
            source=engineer_id,
            target="manager",
            round_num=round_num,
            content=content,
            start_time=start_time,
            end_time=end_time,
        )

    def log_manager_review(
        self,
        engineer_id,
        task_id,
        merged,
        review_reason,
        commit_hash=None,
        files_modified=None,
        round_num=1,
        start_time=None,
        end_time=None,
    ):
        return self.log_event(
            event_type="manager_review",
            source="manager",
            target=engineer_id,
            round_num=round_num,
            content={
                "task_id": task_id,
                "merged": merged,
                "review_reason": review_reason,
                "commit_hash": commit_hash,
                "files_modified": files_modified or [],
            },
            start_time=start_time,
            end_time=end_time,
        )

    def log_agent_event(self, agent_id, event_data):
        if agent_id not in self.agent_event_counters:
            self.agent_event_counters[agent_id] = 0

        self.agent_event_counters[agent_id] += 1
        event_data["event_index"] = self.agent_event_counters[agent_id]

        agent_file = self.get_agent_events_file(agent_id)
        with open(agent_file, "a") as f:
            f.write(json.dumps(event_data) + "\n")

    def clear_agent_events(self, agent_id):
        agent_file = self.get_agent_events_file(agent_id)
        if agent_file.exists():
            agent_file.unlink()
        self.agent_event_counters[agent_id] = 0


class TeeLogger:

    ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

    def __init__(self, log_file_path, mode='w'):
        self.terminal = sys.stdout
        self.log_file = None
        self.log_file_path = log_file_path
        self.mode = mode

    def __enter__(self):
        Path(self.log_file_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_file_path, self.mode, buffering=1)
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal
        if self.log_file:
            self.log_file.close()

    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            clean = self.ANSI_ESCAPE_RE.sub('', message)
            self.log_file.write(clean)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    @property
    def encoding(self):
        return getattr(self.terminal, 'encoding', 'utf-8')


def cleanup_stale_containers(verbose=True):
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=openhands",
             "--filter", "status=exited", "-q"],
            capture_output=True, text=True, timeout=30
        )
        exited_containers = result.stdout.strip().split('\n')
        exited_containers = [c for c in exited_containers if c]

        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=openhands",
             "--filter", "status=dead", "-q"],
            capture_output=True, text=True, timeout=30
        )
        dead_containers = result.stdout.strip().split('\n')
        dead_containers = [c for c in dead_containers if c]

        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=openhands",
             "--format", "{{.ID}} {{.Status}}"],
            capture_output=True, text=True, timeout=30
        )
        removal_containers = []
        for line in result.stdout.strip().split('\n'):
            if line and "Removal" in line:
                removal_containers.append(line.split()[0])

        all_stale = set(exited_containers + dead_containers + removal_containers)

        if all_stale:
            if verbose:
                print(f"[Cleanup] Found {len(all_stale)} stale containers, removing...")
            for container_id in all_stale:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    capture_output=True, timeout=30
                )
            if verbose:
                print(f"[Cleanup] Removed {len(all_stale)} stale containers")
        elif verbose:
            print("[Cleanup] No stale containers found")

    except subprocess.TimeoutExpired:
        if verbose:
            print("[Cleanup] Warning: Docker cleanup timed out")
    except Exception as e:
        if verbose:
            print(f"[Cleanup] Warning: Failed to cleanup containers: {e}")



def load_prompts(task="commit0", prompts_path=None):
    if prompts_path is None:
        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompts_path = prompts_dir / f"{task}.yaml"

    if not prompts_path.exists():
        return {}

    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


def get_paper_info(config):
    """Load paper information from the paperbench data directory."""
    papers_dir = Path(config.paperbench_dir) / "papers" / config.paper_id

    if not papers_dir.exists():
        raise ValueError(f"Paper directory not found: {papers_dir}")

    config_path = papers_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            paper_config = yaml.safe_load(f)
    else:
        paper_config = {"id": config.paper_id, "title": config.paper_id}

    paper_info = PaperInfo(
        paper_id=paper_config.get("id", config.paper_id),
        title=paper_config.get("title", config.paper_id),
        paper_pdf_path=str(papers_dir / "paper.pdf"),
        paper_md_path=str(papers_dir / "paper.md"),
        rubric_path=str(papers_dir / "rubric.json"),
        addendum_path=str(papers_dir / "addendum.md"),
        blacklist_path=str(papers_dir / "blacklist.txt"),
        assets_dir=str(papers_dir / "assets"),
    )

    return paper_info


def load_rubric(rubric_path):
    """Load rubric from JSON file and convert to TaskNode tree."""
    with open(rubric_path, "r") as f:
        rubric_data = json.load(f)
    return TaskNode.from_dict(rubric_data)


def sanitize_json_string(s):
    s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
    return s


def parse_json_from_response(response):
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', response)
    if json_match:
        try:
            return json.loads(sanitize_json_string(json_match.group(1)), strict=False)
        except json.JSONDecodeError:
            pass

    try:
        start = response.find('{')
        if start != -1:
            depth = 0
            in_string = False
            escape_next = False
            for i, char in enumerate(response[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = response[start:i + 1]
                        return json.loads(sanitize_json_string(json_str), strict=False)
    except json.JSONDecodeError:
        pass

    try:
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(sanitize_json_string(response[start:end + 1]), strict=False)
    except json.JSONDecodeError:
        pass

    return None


def extract_json_from_events(events, key_to_find=None):
    try:
        for event in reversed(list(events)):
            texts = []

            if hasattr(event, 'llm_message') and event.llm_message:
                if hasattr(event.llm_message, 'content'):
                    for item in event.llm_message.content:
                        if hasattr(item, 'text') and item.text:
                            texts.append(item.text)

            if hasattr(event, 'thought') and event.thought:
                for item in event.thought:
                    if isinstance(item, str):
                        texts.append(item)
                    elif hasattr(item, 'text') and item.text:
                        texts.append(item.text)

            if hasattr(event, 'reasoning_content') and event.reasoning_content:
                texts.append(event.reasoning_content)

            for text in texts:
                if '{' in text and '}' in text:
                    parsed = parse_json_from_response(text)
                    if parsed and (key_to_find is None or key_to_find in parsed):
                        return parsed
    except Exception as e:
        print(f"[Utils] Error extracting JSON: {e}")
    return None


def build_analysis_result(analysis_json, task_tree=None):
    result = AnalysisResult()
    analysis = analysis_json.get("analysis", {})

    if task_tree is not None:
        # paperbench path
        result.paper_context = analysis.get("paper_context", "")
        result.total_tasks = analysis.get("total_tasks", 0)
        result.task_tree = task_tree

        if task_tree:
            result.leaf_tasks = task_tree.get_leaf_nodes()

        # Count task categories
        categories = {}
        for leaf in result.leaf_tasks:
            cat = leaf.task_category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
        result.task_categories = categories
    else:
        # commit0 path
        result.repo_context = analysis.get("repo_context", "")
        result.total_funcs = analysis.get("total_functions_to_implement", 0)
        result.pass_files = analysis.get("files_with_pass_statements", [])
        result.functions_by_file = analysis.get("functions_by_file", {})
        result.blocking_dependencies = analysis.get("dependency_graph", {})

        blocked_by_count = {}
        for deps in result.blocking_dependencies.values():
            if isinstance(deps, list):
                for dep in deps:
                    blocked_by_count[dep] = blocked_by_count.get(dep, 0) + 1

        result.implementation_order = sorted(
            set(result.pass_files),
            key=lambda f: blocked_by_count.get(f, 0),
            reverse=True
        )

    result.raw_analysis = analysis_json
    return result


def build_delegation_plan(delegation_json):
    plan = DelegationPlan()
    delegation = delegation_json.get("delegation_plan", {})
    first_round = delegation.get("first_round", {})

    plan.num_agents = first_round.get("num_agents", 0)
    plan.reasoning = first_round.get("reasoning", "")

    for t in first_round.get("tasks", []):
        plan.first_round_tasks.append(SubAgentTask(
            engineer_id=t.get("engineer_id", t.get("agent_id", "")),
            task_id=t.get("task_id", ""),
            task_node_id=t.get("task_node_id", ""),
            requirements=t.get("requirements", ""),
            instruction=t.get("instruction", ""),
            context=t.get("context", ""),
            estimated_complexity=t.get("estimated_complexity", t.get("complexity", "medium")),
            task_category=t.get("task_category"),
            file_path=t.get("file_path", ""),
            functions_to_implement=t.get("functions_to_implement", []),
        ))

    for t in delegation.get("remaining_tasks", []):
        plan.remaining_tasks.append(SubAgentTask(
            task_id=t.get("task_id", ""),
            task_node_id=t.get("task_node_id", ""),
            requirements=t.get("requirements", ""),
            depends_on=t.get("depends_on", []),
            reason_for_delay=t.get("reason_for_delay", ""),
            task_category=t.get("task_category"),
            file_path=t.get("file_path", ""),
            functions_to_implement=t.get("functions_to_implement", []),
        ))

    plan.raw_delegation = delegation_json
    return plan


def build_delegation_prompt(prompts, max_subagents):
    return prompts.get("task_delegation", "").format(max_agents=max_subagents)


def fallback_delegation(analysis_result, max_subagents):
    if not analysis_result:
        return None

    try:
        # paperbench path: uses leaf_tasks from task tree
        if analysis_result.leaf_tasks:
            leaf_tasks = analysis_result.leaf_tasks
            max_agents = min(max_subagents, len(leaf_tasks))

            first_round_tasks = leaf_tasks[:max_agents]
            remaining_tasks = leaf_tasks[max_agents:]

            return {
                "delegation_plan": {
                    "first_round": {
                        "num_agents": max_agents,
                        "reasoning": "Fallback: assigned leaf tasks evenly",
                        "tasks": [
                            {
                                "engineer_id": f"engineer_{i+1}",
                                "task_id": f"task_{i+1}",
                                "task_node_id": task.id,
                                "requirements": task.requirements,
                                "instruction": f"Reproduce: {task.requirements}",
                                "context": "",
                                "estimated_complexity": "medium",
                                "task_category": task.task_category,
                            }
                            for i, task in enumerate(first_round_tasks)
                        ],
                    },
                    "remaining_tasks": [
                        {
                            "task_id": f"remaining_{i+1}",
                            "task_node_id": task.id,
                            "requirements": task.requirements,
                            "depends_on": [],
                            "reason_for_delay": "Waiting for first round to complete",
                            "task_category": task.task_category,
                        }
                        for i, task in enumerate(remaining_tasks)
                    ],
                }
            }

        # commit0 path: uses pass_files from analysis
        if analysis_result.pass_files:
            files = analysis_result.pass_files
            deps = analysis_result.blocking_dependencies
            funcs = analysis_result.functions_by_file

            first_round = [f for f in files if not [d for d in deps.get(f, []) if d in files]]
            remaining = [f for f in files if f not in first_round]

            max_agents = min(max_subagents, len(first_round))
            first_round = first_round[:max_agents]

            return {
                "delegation_plan": {
                    "first_round": {
                        "num_agents": max_agents,
                        "reasoning": "Fallback: assigned files with no dependencies first",
                        "tasks": [
                            {
                                "engineer_id": f"engineer_{i+1}",
                                "task_id": f"task_{i+1}",
                                "file_path": f,
                                "functions_to_implement": funcs.get(f, []),
                                "instruction": f"Implement functions in {f}",
                                "context": "",
                                "estimated_complexity": "medium",
                            }
                            for i, f in enumerate(first_round)
                        ],
                    },
                    "remaining_tasks": [
                        {
                            "task_id": f"remaining_{i+1}",
                            "file_path": f,
                            "functions_to_implement": funcs.get(f, []),
                            "depends_on": deps.get(f, []),
                            "reason_for_delay": f"Depends on: {', '.join(deps.get(f, []))}",
                        }
                        for i, f in enumerate(remaining)
                    ],
                }
            }

        return None
    except Exception as e:
        print(f"[Utils] Error in fallback delegation: {e}")
        return None


def build_subagent_prompt(prompts, **kwargs):
    # Join functions list into comma-separated string for commit0
    if "functions" in kwargs and isinstance(kwargs["functions"], list):
        kwargs["functions"] = ", ".join(kwargs["functions"])

    if prompts and "subagent_prompt" in prompts:
        return prompts["subagent_prompt"].format(**kwargs)
    raise ValueError("subagent_prompt not found in prompts")


def count_llm_iterations(events):
    try:
        response_ids = set()
        for event in events:
            if hasattr(event, 'llm_response_id') and event.llm_response_id:
                response_ids.add(event.llm_response_id)
        return len(response_ids)
    except Exception:
        return 0


def serialize_event(event, event_index):
    serialized = {
        "event_index": event_index,
        "event_type": type(event).__name__,
        "timestamp": getattr(event, 'timestamp', None),
        "llm_response_id": getattr(event, 'llm_response_id', None),
    }

    if hasattr(event, 'action') and event.action:
        action = event.action
        action_data = {
            "type": type(action).__name__,
        }
        if hasattr(action, 'action'):
            action_data["action_name"] = action.action
        if hasattr(action, 'args'):
            try:
                action_data["args"] = dict(action.args) if action.args else {}
            except (TypeError, ValueError):
                action_data["args"] = str(action.args)
        if hasattr(action, 'thought') and action.thought:
            action_data["thought"] = str(action.thought)
        serialized["action"] = action_data

    if hasattr(event, 'observation') and event.observation:
        obs = event.observation
        obs_data = {
            "type": type(obs).__name__,
        }
        if hasattr(obs, 'content') and obs.content:
            content = str(obs.content)
            obs_data["content"] = content[:5000] + "..." if len(content) > 5000 else content
        if hasattr(obs, 'output') and obs.output:
            output = str(obs.output)
            obs_data["output"] = output[:5000] + "..." if len(output) > 5000 else output
        if hasattr(obs, 'exit_code'):
            obs_data["exit_code"] = obs.exit_code
        serialized["observation"] = obs_data

    if hasattr(event, 'llm_message') and event.llm_message:
        msg = event.llm_message
        if hasattr(msg, 'content'):
            texts = []
            for item in msg.content:
                if hasattr(item, 'text') and item.text:
                    text = item.text
                    texts.append(text[:2000] + "..." if len(text) > 2000 else text)
            if texts:
                serialized["llm_message"] = texts

    if hasattr(event, 'thought') and event.thought:
        thoughts = []
        for item in event.thought:
            if hasattr(item, 'text') and item.text:
                text = item.text
                thoughts.append(text[:2000] + "..." if len(text) > 2000 else text)
        if thoughts:
            serialized["thought"] = thoughts

    if hasattr(event, 'reasoning_content') and event.reasoning_content:
        content = event.reasoning_content
        serialized["reasoning"] = content[:2000] + "..." if len(content) > 2000 else content

    return serialized


def extract_conversation_metrics(conversation, debug=False):
    metrics = {
        "cost": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model_name": "",
    }

    try:
        if hasattr(conversation, '_state') and hasattr(conversation._state, '_cached_state'):
            conversation._state._cached_state = None

        if hasattr(conversation, 'agent') and hasattr(conversation.agent, 'llm'):
            llm = conversation.agent.llm
            if hasattr(llm, 'metrics') and llm.metrics:
                llm_metrics = llm.metrics
                metrics["cost"] = getattr(llm_metrics, 'accumulated_cost', 0.0)
                metrics["model_name"] = getattr(llm_metrics, 'model_name', "")

                if hasattr(llm_metrics, 'accumulated_token_usage') and llm_metrics.accumulated_token_usage:
                    usage = llm_metrics.accumulated_token_usage
                    metrics["prompt_tokens"] = getattr(usage, 'prompt_tokens', 0)
                    metrics["completion_tokens"] = getattr(usage, 'completion_tokens', 0)
                    metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]

                if metrics["cost"] > 0 or metrics["total_tokens"] > 0:
                    return metrics

        if hasattr(conversation, 'conversation_stats'):
            stats = conversation.conversation_stats
            if stats:
                combined = stats.get_combined_metrics()
                if combined:
                    metrics["cost"] = combined.accumulated_cost
                    metrics["model_name"] = combined.model_name

                    if combined.accumulated_token_usage:
                        usage = combined.accumulated_token_usage
                        metrics["prompt_tokens"] = usage.prompt_tokens
                        metrics["completion_tokens"] = usage.completion_tokens
                        metrics["total_tokens"] = usage.prompt_tokens + usage.completion_tokens
    except Exception as e:
        print(f"[Warning] Failed to extract metrics: {e}")

    return metrics


def save_all_costs(
    output_dir,
    manager_metrics,
    subagent_results=None,
    wall_clock_duration=None,
    manager_cost_breakdown=None,
    model=None,
    subagent_model=None,
    paper_id=None,
):
    output_dir = Path(output_dir)
    subagent_results = subagent_results or []
    manager_cost_breakdown = manager_cost_breakdown or {}

    manager_cost = manager_metrics.get("cost", 0)
    manager_prompt = manager_metrics.get("prompt_tokens", 0)
    manager_completion = manager_metrics.get("completion_tokens", 0)
    manager_total = manager_metrics.get("total_tokens", 0)
    manager_duration = manager_metrics.get("duration", 0)

    agent_aggregates = {}
    total_subagent_cost = 0
    total_subagent_prompt = 0
    total_subagent_completion = 0
    total_subagent_tokens = 0
    total_subagent_duration = 0

    for result in subagent_results:
        duration = getattr(result, 'duration_seconds', 0.0)
        agent_id = result.engineer_id

        if agent_id not in agent_aggregates:
            agent_aggregates[agent_id] = {
                "cost": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "iterations_total": 0,
                "max_iterations_per_task": result.max_iterations,
                "duration": 0,
                "tasks_completed": 0,
                "tasks_successful": 0,
                "tasks": [],
            }

        agg = agent_aggregates[agent_id]
        agg["cost"] += result.cost
        agg["prompt_tokens"] += result.prompt_tokens
        agg["completion_tokens"] += result.completion_tokens
        agg["total_tokens"] += result.total_tokens
        agg["iterations_total"] += result.actual_iterations
        agg["duration"] += duration
        agg["tasks_completed"] += 1
        if result.success:
            agg["tasks_successful"] += 1

        task_entry = {
            "round": result.round_num,
            "task_id": result.task_id,
            "actual_iterations": result.actual_iterations,
            "cost": result.cost,
            "duration": duration,
            "success": result.success,
        }
        if result.task_node_id:
            task_entry["task_node_id"] = result.task_node_id
        if result.file_path:
            task_entry["file_path"] = result.file_path
        agg["tasks"].append(task_entry)

        total_subagent_cost += result.cost
        total_subagent_prompt += result.prompt_tokens
        total_subagent_completion += result.completion_tokens
        total_subagent_tokens += result.total_tokens
        total_subagent_duration += duration

    total_work_duration = manager_duration + total_subagent_duration

    manager_section = {
        "cost": manager_cost,
        "prompt_tokens": manager_prompt,
        "completion_tokens": manager_completion,
        "total_tokens": manager_total,
        "duration": manager_duration,
    }

    if manager_cost_breakdown:
        operations = {
            "analysis": {
                "cost": manager_cost_breakdown.get("analysis_cost", 0),
                "tokens": manager_cost_breakdown.get("analysis_tokens", 0),
                "duration": manager_cost_breakdown.get("analysis_duration", 0),
            },
            "delegation": {
                "cost": manager_cost_breakdown.get("delegation_cost", 0),
                "tokens": manager_cost_breakdown.get("delegation_tokens", 0),
                "duration": manager_cost_breakdown.get("delegation_duration", 0),
            },
            "assign_task": {
                "cost": manager_cost_breakdown.get("assign_task_cost", 0),
                "tokens": manager_cost_breakdown.get("assign_task_tokens", 0),
                "duration": manager_cost_breakdown.get("assign_task_duration", 0),
            },
            "review": {
                "cost": manager_cost_breakdown.get("review_cost", 0),
                "tokens": manager_cost_breakdown.get("review_tokens", 0),
                "duration": manager_cost_breakdown.get("review_duration", 0),
            },
            "final_review": {
                "cost": manager_cost_breakdown.get("final_review_cost", 0),
                "tokens": manager_cost_breakdown.get("final_review_tokens", 0),
                "duration": manager_cost_breakdown.get("final_review_duration", 0),
            },
        }
        # Exploration (commit0 only passes this key)
        if "exploration_cost" in manager_cost_breakdown:
            operations["exploration"] = {
                "cost": manager_cost_breakdown.get("exploration_cost", 0),
                "tokens": manager_cost_breakdown.get("exploration_tokens", 0),
                "duration": manager_cost_breakdown.get("exploration_duration", 0),
            }
        # Test duration (paperbench only passes this key)
        if "test_duration" in manager_cost_breakdown:
            operations["test"] = {
                "duration": manager_cost_breakdown.get("test_duration", 0),
            }
        manager_section["operations"] = operations

    # Judge section (present when judge data is in breakdown)
    judge_cost = 0.0
    judge_prompt = 0
    judge_completion = 0
    judge_section = None
    if "judge_token_usage" in manager_cost_breakdown:
        judge_token_usage = manager_cost_breakdown["judge_token_usage"]
        judge_cost = manager_cost_breakdown.get("judge_cost", 0.0)
        judge_prompt = sum(u.get("in", 0) for u in judge_token_usage.values())
        judge_completion = sum(u.get("out", 0) for u in judge_token_usage.values())
        judge_section = {
            "model": manager_cost_breakdown.get("judge_model"),
            "cost": judge_cost,
            "prompt_tokens": judge_prompt,
            "completion_tokens": judge_completion,
            "total_tokens": judge_prompt + judge_completion,
            "duration": manager_cost_breakdown.get("test_duration", 0),
        }

    grand_total_cost = manager_cost + total_subagent_cost + judge_cost
    cost_data = {
        "model": model,
        "manager": manager_section,
        "subagents": agent_aggregates,
        "total": {
            "cost": grand_total_cost,
            "prompt_tokens": manager_prompt + total_subagent_prompt + judge_prompt,
            "completion_tokens": manager_completion + total_subagent_completion + judge_completion,
            "total_tokens": manager_total + total_subagent_tokens + judge_prompt + judge_completion,
            "duration": total_work_duration,
        },
    }
    if subagent_model:
        cost_data["subagent_model"] = subagent_model
    if paper_id:
        cost_data["paper_id"] = paper_id
    if judge_section:
        cost_data["judge"] = judge_section

    if wall_clock_duration is not None:
        cost_data["total"]["wall_clock_duration"] = wall_clock_duration

    with open(output_dir / "cost.json", "w") as f:
        json.dump(cost_data, f, indent=2)
    print(f"[Cost] Saved to: {output_dir / 'cost.json'}")
    if judge_section:
        print(f"[Cost] Manager: ${manager_cost:.4f}, Subagents: ${total_subagent_cost:.4f}, Judge: ${judge_cost:.4f}, Total: ${grand_total_cost:.4f}")
    else:
        print(f"[Cost] Manager: ${manager_cost:.4f}, Subagents: ${total_subagent_cost:.4f}, Total: ${grand_total_cost:.4f}")


def build_llm_kwargs(model_name):
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("Please set LLM_API_KEY environment variable")
    base_url = os.getenv("LLM_BASE_URL")
    if not base_url:
        raise ValueError("Please set LLM_BASE_URL environment variable")
    return {
        "model": model_name,
        "api_key": SecretStr(api_key),
        "base_url": base_url,
    }


def filter_kwargs(kwargs, key_map):
    """Pick keys from kwargs, remap names, drop empty strings."""
    return {v: kwargs[k] for k, v in key_map.items() if k in kwargs and kwargs[k] != ""}


def build_task_module(task, **kwargs):
    if task == "commit0":
        init = filter_kwargs(kwargs, {
            "repo": "repo_name",
            "base_branch": "base_branch",
            "docker_image_prefix": "docker_image_prefix",
            "dataset_path": "dataset_path",
        })
        return Commit0Task(Commit0Config(**init))
    elif task == "paperbench":
        init = filter_kwargs(kwargs, {
            "paper_id": "paper_id",
            "docker_image": "docker_image",
            "paperbench_dir": "paperbench_dir",
            "test_max_depth": "test_max_depth",
            "test_reproduce_timeout": "test_reproduce_timeout",
            "judge_type": "judge_type",
            "judge_model": "judge_model",
            "code_dev": "code_dev",
            "output_dir": "output_dir",
        })
        return PaperbenchTask(PaperbenchConfig(**init))
    else:
        raise ValueError(f"Unknown task: {task}. Available: commit0, paperbench")


def build_output_dir(task, model_name, workflow_config, multi_agent=True, **kwargs):
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name

    if workflow_config.subagent_model:
        sub_short = workflow_config.subagent_model.split("/")[-1] if "/" in workflow_config.subagent_model else workflow_config.subagent_model
        model_short = f"{model_short}+{sub_short}"

    params = (
        f"manageriters={workflow_config.manager_max_iterations}"
        f"_subagents={workflow_config.max_subagents}"
        f"_subiters={workflow_config.subagent_max_iterations}"
        f"_rchats={workflow_config.max_rounds_chat}"
    )

    mode = "multi-agent" if multi_agent else "single-agent"

    if task == "commit0":
        repo = kwargs.get("repo", "minitorch")
        return str(Path("outputs") / "commit0" / model_short / repo / mode / params)
    elif task == "paperbench":
        paper_id = kwargs.get("paper_id", "rice")
        code_dev_str = "true" if kwargs.get("code_dev", True) else "false"
        params += f"_codedev={code_dev_str}"
        return str(Path("outputs") / "paperbench" / model_short / paper_id / mode / params)
    else:
        return str(Path("outputs") / task / model_short / mode / params)


def detect_platform():
    machine = plat.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def get_manager_summary(analysis_result, delegation_plan, identifier, phase="all"):
    """Unified summary for both commit0 (repo_name) and paperbench (paper_info)."""
    lines = []

    if phase in ("analysis", "all") and analysis_result:
        if isinstance(identifier, str) or (identifier is None and not analysis_result.task_tree):
            # commit0 path (identifier is repo_name string)
            lines.extend([
                f"\nRepository Analysis Summary",
                f"=" * 40,
                f"Repository: {identifier}",
                f"Context: {analysis_result.repo_context}",
                f"Functions to implement: {analysis_result.total_funcs}",
                f"Files: {len(analysis_result.pass_files)}",
            ])
            for f in analysis_result.pass_files[:10]:
                funcs = analysis_result.functions_by_file.get(f, [])
                lines.append(f"  - {f} ({len(funcs)} funcs)")
            if len(analysis_result.pass_files) > 10:
                lines.append(f"  ... and {len(analysis_result.pass_files) - 10} more")
        else:
            # paperbench path
            lines.extend([
                f"\nPaper Analysis Summary",
                f"=" * 40,
                f"Paper: {identifier.title if identifier else 'Unknown'}",
                f"Context: {analysis_result.paper_context}",
                f"Total tasks: {analysis_result.total_tasks}",
                f"Leaf tasks: {len(analysis_result.leaf_tasks)}",
            ])
            for cat, count in analysis_result.task_categories.items():
                lines.append(f"  - {cat}: {count}")

    if phase in ("delegation", "all") and delegation_plan:
        lines.extend([
            f"\nTask Delegation Summary",
            f"=" * 40,
            f"Agents for first round: {delegation_plan.num_agents}",
            f"Reasoning: {delegation_plan.reasoning}",
            f"First round tasks: {len(delegation_plan.first_round_tasks)}",
            f"Remaining tasks: {len(delegation_plan.remaining_tasks)}",
        ])

    return "\n".join(lines) if lines else "No results yet."


def generate_patch(workspace, repo_dir, base_commit, subagent_results):
    patch_content = ""
    agent_contributions = []

    for result in subagent_results:
        if result.success and result.git_diff:
            agent_contributions.append({
                "engineer_id": result.engineer_id,
                "task_id": result.task_id,
                "file_path": result.file_path,
                "files_modified": result.files_modified,
                "commit_hash": result.commit_hash,
                "round_num": result.round_num,
            })

    header = "# Multi-Agent Patch Summary\n"
    header += f"# Base commit: {base_commit}\n"
    header += "#\n"
    header += "# Agent Contributions:\n"

    for contrib in agent_contributions:
        header += f"#   - {contrib['engineer_id']} (round {contrib['round_num']}): {contrib['task_id']}\n"
        header += f"#     File: {contrib['file_path']}\n"
        if contrib['files_modified']:
            header += f"#     Modified: {', '.join(contrib['files_modified'])}\n"
        if contrib['commit_hash']:
            header += f"#     Commit: {contrib['commit_hash']}\n"

    header += "#\n"
    header += "# " + "=" * 70 + "\n\n"

    result = workspace.execute_command(
        f"cd {repo_dir} && git diff {base_commit} HEAD --no-color",
        timeout=600
    )

    if result.exit_code == 0 and result.stdout.strip():
        patch_content = header + result.stdout.strip()
    else:
        patch_content = header + "# No changes detected\n"

    return patch_content, agent_contributions


def download_file_via_base64(workspace, remote_path, local_path, chunk_size=1024*1024):
    try:
        size_result = workspace.execute_command(f"stat -c%s {remote_path}", timeout=30)
        if size_result.exit_code != 0:
            print(f"[Download] Error: Cannot stat file: {size_result.stderr}")
            return False

        file_size = int(size_result.stdout.strip())
        print(f"[Download] File size: {file_size / (1024*1024):.2f} MB")

        with open(local_path, 'wb') as f:
            offset = 0
            while offset < file_size:
                cmd = f"dd if={remote_path} bs=1 skip={offset} count={chunk_size} 2>/dev/null | base64 -w 0"
                result = workspace.execute_command(cmd, timeout=120)
                if result.exit_code != 0:
                    print(f"[Download] Error reading chunk at offset {offset}: {result.stderr}")
                    return False

                chunk_data = base64.b64decode(result.stdout)
                f.write(chunk_data)
                offset += len(chunk_data)

                progress = (offset / file_size) * 100
                print(f"[Download] Progress: {progress:.1f}%", end='\r')

                if len(chunk_data) < chunk_size:
                    break

        print(f"\n[Download] Successfully saved to {local_path}")
        return True

    except Exception as e:
        print(f"[Download] Error: {e}")
        return False
