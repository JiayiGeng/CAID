import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import fire
import litellm
from openhands.sdk import LLM
from openhands.workspace import DockerDevWorkspace, DockerWorkspace

import core.patches  
from config import WorkflowConfig
from core.manager import Manager
from core.subagent import SubAgentRunner, run_subagents_parallel
from core.utils import (
    OutputLogger,
    TeeLogger,
    build_llm_kwargs,
    build_task_module,
    build_output_dir,
    cleanup_stale_containers,
    detect_platform,
    download_file_via_base64,
    extract_conversation_metrics,
    generate_patch,
    get_manager_summary,
    load_prompts,
    save_all_costs,
    serialize_event,
)
from tasks.commit0 import Commit0Task

litellm.set_verbose = False
litellm.drop_params = True


async def run_workflow_inner(task, workflow_config, task_module, multi_agent=True, **kwargs):
    start_time = datetime.now()

    print("=" * 70)
    if multi_agent:
        print(f"Multi-Agent Delegation Workflow ({task})")
    else:
        print(f"Single-Agent Baseline Mode ({task})")
    print("=" * 70)

    is_commit0 = isinstance(task_module, Commit0Task)

    llm_kwargs = build_llm_kwargs(workflow_config.model)
    llm = LLM(**llm_kwargs)

    # Setup subagent LLM (falls back to manager LLM if not specified)
    if workflow_config.subagent_model:
        subagent_llm_kwargs = build_llm_kwargs(workflow_config.subagent_model)
        subagent_llm = LLM(**subagent_llm_kwargs)
    else:
        subagent_llm = llm

    output_logger = OutputLogger(workflow_config.output_dir)

    print("\nConfiguration:")
    print(f"- Model: {workflow_config.model}")
    print(f"- Max Iterations: {workflow_config.manager_max_iterations}")
    print(f"- Output Directory: {workflow_config.output_dir}")
    print(f"- Multi-Agent: {multi_agent}")

    print(f"\n[LLM] model={workflow_config.model}")
    if workflow_config.subagent_model:
        print(f"[LLM] subagent_model={workflow_config.subagent_model}")
    print(f"\n[Setup] Output logger initialized: {output_logger.output_file}")

    # Clean up stale containers from previous runs
    cleanup_stale_containers(verbose=True)

    workspace_config = task_module.get_workspace_config()

    sdk_source_dir = os.getenv(
        "SDK_SOURCE_DIR",
        str(Path(__file__).resolve().parent.parent / "software-agent-sdk"),
    )

    print("\n[Setup] Creating Docker workspace...")

    original_cwd = os.getcwd()
    os.chdir(sdk_source_dir)

    if workspace_config.get("base_image"):
        workspace_ctx = DockerDevWorkspace(
            base_image=workspace_config["base_image"],
            server_image=None,
            target=workspace_config.get("target", "source-minimal"),
            host_port=None,
            platform="linux/amd64",
            detach_logs=False,
        )
    else:
        workspace_ctx = DockerWorkspace(
            server_image=workspace_config["server_image"],
            host_port=None,
            platform=detect_platform(),
            detach_logs=False,
        )

    os.chdir(original_cwd)

    prompts = load_prompts(task)

    subagents = []
    subagent_results = []
    runners = []

    with workspace_ctx as workspace:

        print("[Setup] Docker workspace ready")

        manager = Manager(
            llm=llm,
            workspace=workspace,
            task=task_module,
            config=workflow_config,
            output_logger=output_logger,
            prompts=prompts,
        )

        try:
            manager.setup_workspace()

            # Single-agent baseline mode
            if not multi_agent:
                print("\n" + "-" * 60)
                print("Step 3: Initialize Single Agent")
                print("-" * 60)
                manager.setup(mode="single_agent")

                print("\n" + "-" * 60)
                if is_commit0:
                    print("Step 4: Run Single Agent (Implement All Functions)")
                else:
                    print("Step 4: Run Single Agent (Reproduce Paper)")
                print("-" * 60)
                runtime_start = datetime.now()
                single_agent_result = manager.run_single_agent()

                runtime_end = datetime.now()
                runtime_seconds = (runtime_end - runtime_start).total_seconds()
                runtime_file = Path(workflow_config.output_dir) / "runtime.txt"
                with open(runtime_file, "w") as f:
                    f.write(f"{runtime_seconds:.1f}")
                print(f"[Runtime] {runtime_seconds:.1f}s saved to {runtime_file}")

                if is_commit0:
                    print("\n" + "-" * 60)
                    print("Step 5: Run Pytest")
                    print("-" * 60)

                    pytest_results = task_module.evaluate(workspace)

                    report_file = Path(workflow_config.output_dir) / "report.json"
                    exit_code_file = Path(workflow_config.output_dir) / f"{task_module.config.repo_name}_pytest_exit_code.txt"
                    test_output_file = Path(workflow_config.output_dir) / f"{task_module.config.repo_name}_test_output.txt"

                    with open(report_file, "w") as f:
                        f.write(pytest_results["report_json"])
                    with open(exit_code_file, "w") as f:
                        f.write(pytest_results["exit_code"])
                    with open(test_output_file, "w") as f:
                        f.write(pytest_results["test_output"])

                    print(f"\nPytest files saved:")
                    print(f"- {report_file}")
                    print(f"- {exit_code_file}")
                    print(f"- {test_output_file}")

                    # Save final repo state as tarball
                    print("\n[Tarball] Saving final repo state...")
                    repo_name = task_module.config.repo_name
                    tarball_name = f"{repo_name}_repo.tar.gz"
                    tarball_cmd = f"cd /workspace && tar -czf {tarball_name} {repo_name}_repo"
                    tar_result = workspace.execute_command(tarball_cmd, timeout=300)
                    if tar_result.exit_code == 0:
                        final_repo_dir = Path(workflow_config.output_dir) / "final_repo"
                        final_repo_dir.mkdir(parents=True, exist_ok=True)
                        tarball_local_path = final_repo_dir / f"{repo_name}.tar.gz"
                        success = download_file_via_base64(workspace, f"/workspace/{tarball_name}", str(tarball_local_path))
                        if not success:
                            print(f"[Tarball] Warning: Download failed")
                    else:
                        print(f"[Tarball] Warning: Failed to create tarball: {tar_result.stderr}")

                    # Save costs
                    total_time = (datetime.now() - start_time).total_seconds()
                    manager_metrics = extract_conversation_metrics(manager.conversation)
                    manager_metrics["duration"] = single_agent_result["duration"]
                    save_all_costs(workflow_config.output_dir, manager_metrics, [], wall_clock_duration=total_time, model=workflow_config.model)
                else:
                    # Paperbench single-agent: run test
                    print("\n" + "-" * 60)
                    print("Step 5: Run Test (PaperBench Judge)")
                    print("-" * 60)

                    test_start = datetime.now()
                    test_result = task_module.evaluate(workspace)
                    test_duration = (datetime.now() - test_start).total_seconds()
                    test_end = datetime.now()

                    print(f"\n[Test] Results:")
                    print(f"- reproduce.sh exists: {test_result['reproduce_script_exists']}")
                    print(f"- reproduce.sh success: {test_result['reproduce_success']}")
                    print(f"- reproduce duration: {test_result['reproduce_duration']:.1f}s")
                    if test_result.get('judge_score') is not None:
                        print(f"- Judge score: {test_result['judge_score']:.4f}")
                        print(f"- Judge nodes: {test_result['judge_num_nodes']}")

                    # Save grade.json
                    grade_path = Path(workflow_config.output_dir) / "grade.json"
                    grade_output = {
                        "paper_id": task_module.config.paper_id,
                        "agent_model": workflow_config.model,
                        "judge_output": {
                            "judge_type": test_result["judge_type"],
                            "score": test_result["judge_score"],
                            "num_leaf_nodes": test_result["judge_num_nodes"],
                            "num_invalid_leaf_nodes": test_result["judge_num_invalid_nodes"],
                            "judge_model": test_result["judge_model"],
                            "max_depth": test_result["max_depth"],
                            "graded_task_tree": test_result["graded_task_tree"],
                        },
                        "reproduction_metadata": {
                            "repro_script_exists": test_result["reproduce_script_exists"],
                            "repro_success": test_result["reproduce_success"],
                            "repro_duration": test_result["reproduce_duration"],
                            "repro_log": test_result["reproduce_log"],
                        },
                        "score": test_result["judge_score"],
                        "tested_at": test_end.isoformat(),
                        "total_duration": test_duration,
                    }
                    with open(grade_path, "w") as f:
                        json.dump(grade_output, f, indent=2)
                    print(f"[Grade] Saved to {grade_path}")

                    # Save costs
                    total_time = (datetime.now() - start_time).total_seconds()
                    manager_metrics = extract_conversation_metrics(manager.conversation)
                    manager_metrics["duration"] = single_agent_result["duration"]

                    test_result_data = test_result or {}
                    manager_cost_breakdown = {
                        "test_duration": test_duration,
                        "judge_token_usage": test_result_data.get("judge_token_usage", {}),
                        "judge_cost": test_result_data.get("judge_cost", 0.0),
                        "judge_model": test_result_data.get("judge_model"),
                    }
                    save_all_costs(workflow_config.output_dir, manager_metrics, [],
                                   wall_clock_duration=total_time,
                                   manager_cost_breakdown=manager_cost_breakdown,
                                   model=workflow_config.model,
                                   paper_id=getattr(task_module.config, 'paper_id', None))

                print("\n" + "=" * 70)
                print("Single-Agent Baseline Complete")
                print("=" * 70)
                print(f"Total time: {(datetime.now() - start_time).total_seconds():.1f}s")
                print(f"Iterations used: {single_agent_result['iterations']}")
                return

            # Multi-agent workflow continues below
            print("\n" + "-" * 60)
            print("Step 3: Initialize Manager Agent")
            print("-" * 60)
            manager.setup(mode="multi_agent")

            print("\n" + "-" * 60)
            if is_commit0:
                print("Step 4: Scan and Analyze Repository")
            else:
                print("Step 4: Scan and Analyze Paper")
            print("-" * 60)
            runtime_start = datetime.now()
            manager.scan_and_analyze()

            if is_commit0:
                print(get_manager_summary(manager.analysis_result, None, task_module.config.repo_name, "analysis"))
            else:
                print(get_manager_summary(manager.analysis_result, None, getattr(manager.task, 'paper_info', None), "analysis"))

            print("\n" + "-" * 60)
            print("Step 5: Delegate Tasks")
            print("-" * 60)
            manager.delegate_tasks()

            if is_commit0:
                print(get_manager_summary(manager.analysis_result, manager.delegation_plan, task_module.config.repo_name, "delegation"))
            else:
                print(get_manager_summary(manager.analysis_result, manager.delegation_plan, getattr(manager.task, 'paper_info', None), "delegation"))

            # commit0: Save manager events after scan + delegate (before subagents)
            if is_commit0 and manager.conversation:
                events = list(manager.conversation.state.events)
                engineer_id = "manager"
                print(f"[Manager] Saving {len(events)} events to {engineer_id}_events.jsonl...")
                for idx, event in enumerate(events):
                    serialized = serialize_event(event, idx)
                    serialized["engineer_id"] = engineer_id
                    serialized["phase"] = "scan_and_delegate"
                    serialized["start_time"] = serialized.get("timestamp")
                    if idx + 1 < len(events):
                        next_ts = getattr(events[idx + 1], 'timestamp', None)
                        serialized["end_time"] = next_ts
                    else:
                        serialized["end_time"] = manager.delegation_end_time.isoformat() if manager.delegation_end_time else None
                    output_logger.log_agent_event(engineer_id, serialized)

            print("\n" + "-" * 60)
            print("Step 6: Onboard Subagents")
            print("-" * 60)
            subagents = manager.onboard_subagents()

            print("\n[Verification] Listing git worktrees:")
            result = workspace.execute_command(
                f"cd {manager.repo_dir} && git worktree list",
                timeout=30
            )
            if result.exit_code == 0:
                print(result.stdout)

            print("\n" + "-" * 60)
            print("Step 7: Setup Subagents")
            print("-" * 60)

            ready_subagents = [s for s in subagents if s.status == "ready"]
            print(f"[Setup] Setting up {len(ready_subagents)} subagent runners...")

            runners = []
            for subagent in ready_subagents:
                runner = SubAgentRunner(
                    llm=subagent_llm,
                    workspace=workspace,
                    subagent=subagent,
                    prompts=prompts,
                    task_module=task_module,
                    max_iterations=workflow_config.subagent_max_iterations,
                    max_rounds_chat=workflow_config.max_rounds_chat,
                    output_dir=workflow_config.output_dir,
                    output_logger=output_logger,
                )
                runner.setup()
                runners.append(runner)

                if is_commit0:
                    output_logger.log_manager_instruction(
                        engineer_id=subagent.engineer_id,
                        task_id=subagent.task_id,
                        file_path=subagent.file_path,
                        functions_to_implement=subagent.functions_to_implement,
                        instruction=subagent.instruction,
                    )
                else:
                    output_logger.log_manager_instruction(
                        engineer_id=subagent.engineer_id,
                        task_id=subagent.task_id,
                        task_node_id=subagent.task_node_id,
                        requirements=subagent.requirements,
                        instruction=subagent.instruction,
                    )

            print(f"[Setup] {len(runners)} subagent runners ready")

            print("\n" + "-" * 60)
            if is_commit0:
                print("Step 8: Run Subagents in Parallel")
            else:
                print("Step 8: Run Subagents First Round")
            print("-" * 60)

            # commit0: enable background exploration; paperbench: no
            enable_bg_exploration = is_commit0

            subagent_results = await run_subagents_parallel(
                runners,
                manager=manager,
                task_module=task_module,
                output_logger=output_logger,
                enable_background_exploration=enable_bg_exploration,
                max_subagents=workflow_config.max_subagents,
            )

            if not is_commit0:
                # Paperbench: "All Rounds Complete" summary
                print("\n" + "-" * 60)
                print("All Rounds Complete")
                print("-" * 60)
                print(f"  Total results: {len(subagent_results)}")
                for r in subagent_results:
                    status = "SUCCESS" if r.success else "FAILED"
                    print(f"- {r.engineer_id} round {r.round_num}: {status} ({r.duration_seconds:.1f}s, ${r.cost:.4f})")

                for runner in runners:
                    runner.cleanup()

                # Paperbench: final review
                print("\n" + "-" * 60)
                print("Step 9: Manager Final Review")
                print("-" * 60)
                manager.final_review_all(subagent_results, max_iterations=30)

            else:
                # Commit0: final review
                print("\n" + "-" * 60)
                print("Step 8.5: Manager Final Review")
                print("-" * 60)
                manager.final_review_all(subagent_results, max_iterations=30)

            # Save runtime (before evaluation)
            runtime_end = datetime.now()
            runtime_seconds = (runtime_end - runtime_start).total_seconds()
            runtime_file = Path(workflow_config.output_dir) / "runtime.txt"
            with open(runtime_file, "w") as f:
                f.write(f"{runtime_seconds:.1f}")
            print(f"\n[Runtime] {runtime_seconds:.1f}s saved to {runtime_file}")

            # Commit0: cleanup runners after runtime save
            if is_commit0:
                for runner in runners:
                    runner.cleanup()

            # Save costs
            print("\n" + "-" * 60)
            print("Step 9: Collect and Save Results")
            print("-" * 60)

            manager_metrics = extract_conversation_metrics(manager.conversation)

            analysis_duration = 0
            if manager.analysis_end_time and manager.analysis_start_time:
                analysis_duration = (manager.analysis_end_time - manager.analysis_start_time).total_seconds()
            delegation_duration = 0
            if manager.delegation_end_time and manager.delegation_start_time:
                delegation_duration = (manager.delegation_end_time - manager.delegation_start_time).total_seconds()

            if is_commit0:
                manager_duration = (
                    analysis_duration + delegation_duration
                    + manager.assign_task_total_time + manager.review_total_time
                    + manager.final_review_total_time
                )
                manager_metrics["duration"] = manager_duration

                manager_cost_breakdown = {
                    "analysis_cost": manager.analysis_cost,
                    "analysis_tokens": manager.analysis_tokens,
                    "analysis_duration": analysis_duration,
                    "delegation_cost": manager.delegation_cost,
                    "delegation_tokens": manager.delegation_tokens,
                    "delegation_duration": delegation_duration,
                    "assign_task_cost": manager.assign_task_total_cost,
                    "assign_task_tokens": manager.assign_task_total_tokens,
                    "assign_task_duration": manager.assign_task_total_time,
                    "review_cost": manager.review_total_cost,
                    "review_tokens": manager.review_total_tokens,
                    "review_duration": manager.review_total_time,
                    "exploration_cost": manager.exploration_cost,
                    "exploration_tokens": manager.exploration_tokens,
                    "exploration_duration": manager.exploration_total_time,
                    "final_review_cost": manager.final_review_cost,
                    "final_review_tokens": manager.final_review_tokens,
                    "final_review_duration": manager.final_review_total_time,
                }

                save_all_costs(
                    workflow_config.output_dir,
                    manager_metrics,
                    subagent_results,
                    wall_clock_duration=runtime_seconds,
                    manager_cost_breakdown=manager_cost_breakdown,
                    model=workflow_config.model,
                )

                # Generate patch
                base_commit = None
                for s in subagents:
                    if s.base_commit:
                        base_commit = s.base_commit
                        break

                if base_commit:
                    patch_content, _ = generate_patch(
                        workspace, manager.repo_dir, base_commit, subagent_results
                    )
                    patch_file = Path(workflow_config.output_dir) / "patch.diff"
                    with open(patch_file, "w") as f:
                        f.write(patch_content)
                    print(f"[Patch] Saved to {patch_file}")

                # Run pytest
                print("\n" + "-" * 60)
                print("Step 10: Run Final Pytest")
                print("-" * 60)

                pytest_results = task_module.evaluate(workspace)

                report_file = Path(workflow_config.output_dir) / "report.json"
                exit_code_file = Path(workflow_config.output_dir) / f"{task_module.config.repo_name}_pytest_exit_code.txt"
                test_output_file = Path(workflow_config.output_dir) / f"{task_module.config.repo_name}_test_output.txt"

                with open(report_file, "w") as f:
                    f.write(pytest_results["report_json"])
                with open(exit_code_file, "w") as f:
                    f.write(pytest_results["exit_code"])
                with open(test_output_file, "w") as f:
                    f.write(pytest_results["test_output"])

                print(f"\nPytest files saved:")
                print(f"- {report_file}")
                print(f"- {exit_code_file}")
                print(f"- {test_output_file}")

                # Save final repo state as tarball
                print("\n[Tarball] Saving final repo state...")
                repo_name = task_module.config.repo_name
                tarball_name = f"{repo_name}_repo.tar.gz"
                tarball_cmd = f"cd /workspace && tar -czf {tarball_name} {repo_name}_repo"
                tar_result = workspace.execute_command(tarball_cmd, timeout=300)
                if tar_result.exit_code == 0:
                    final_repo_dir = Path(workflow_config.output_dir) / "final_repo"
                    final_repo_dir.mkdir(parents=True, exist_ok=True)
                    tarball_local_path = final_repo_dir / f"{repo_name}.tar.gz"
                    success = download_file_via_base64(workspace, f"/workspace/{tarball_name}", str(tarball_local_path))
                    if not success:
                        print(f"[Tarball] Warning: Download failed")
                else:
                    print(f"[Tarball] Warning: Failed to create tarball: {tar_result.stderr}")

            else:
                # Paperbench post-parallel flow
                manager_duration = (
                    analysis_duration + delegation_duration
                    + manager.assign_task_total_time + manager.review_total_time
                    + manager.final_review_total_time + manager.test_total_time
                )
                manager_metrics["duration"] = manager_duration

                # Run test (paperbench judge)
                print("\n" + "-" * 60)
                print("Step 10: Run Test (PaperBench Judge)")
                print("-" * 60)

                test_start = datetime.now()
                test_result = task_module.evaluate(workspace)
                test_duration = (datetime.now() - test_start).total_seconds()
                manager.test_total_time = test_duration
                manager.test_result = test_result

                test_end = datetime.now()

                print(f"\n[Test] Results:")
                print(f"- reproduce.sh exists: {test_result['reproduce_script_exists']}")
                print(f"- reproduce.sh success: {test_result['reproduce_success']}")
                print(f"- reproduce duration: {test_result['reproduce_duration']:.1f}s")
                if test_result.get('judge_score') is not None:
                    print(f"- Judge score: {test_result['judge_score']:.4f}")
                    print(f"- Judge nodes: {test_result['judge_num_nodes']}")

                # Save grade.json (same format as original paperbench)
                grade_path = Path(workflow_config.output_dir) / "grade.json"
                grade_output = {
                    "paper_id": task_module.config.paper_id,
                    "agent_model": workflow_config.model,
                    "judge_output": {
                        "judge_type": test_result["judge_type"],
                        "score": test_result["judge_score"],
                        "num_leaf_nodes": test_result["judge_num_nodes"],
                        "num_invalid_leaf_nodes": test_result["judge_num_invalid_nodes"],
                        "judge_model": test_result["judge_model"],
                        "max_depth": test_result["max_depth"],
                        "graded_task_tree": test_result["graded_task_tree"],
                    },
                    "reproduction_metadata": {
                        "repro_script_exists": test_result["reproduce_script_exists"],
                        "repro_success": test_result["reproduce_success"],
                        "repro_duration": test_result["reproduce_duration"],
                        "repro_log": test_result["reproduce_log"],
                    },
                    "score": test_result["judge_score"],
                    "tested_at": test_end.isoformat(),
                    "total_duration": test_duration,
                }
                with open(grade_path, "w") as f:
                    json.dump(grade_output, f, indent=2)
                print(f"[Grade] Saved to {grade_path}")

                # Recalculate manager_duration with test time
                manager_duration = (
                    analysis_duration + delegation_duration
                    + manager.assign_task_total_time + manager.review_total_time
                    + manager.final_review_total_time + manager.test_total_time
                )
                manager_metrics["duration"] = manager_duration

                test_result_data = manager.test_result or {}
                manager_cost_breakdown = {
                    "analysis_cost": manager.analysis_cost,
                    "analysis_tokens": manager.analysis_tokens,
                    "analysis_duration": analysis_duration,
                    "delegation_cost": manager.delegation_cost,
                    "delegation_tokens": manager.delegation_tokens,
                    "delegation_duration": delegation_duration,
                    "assign_task_cost": manager.assign_task_total_cost,
                    "assign_task_tokens": manager.assign_task_total_tokens,
                    "assign_task_duration": manager.assign_task_total_time,
                    "review_cost": manager.review_total_cost,
                    "review_tokens": manager.review_total_tokens,
                    "review_duration": manager.review_total_time,
                    "final_review_cost": manager.final_review_cost,
                    "final_review_tokens": manager.final_review_tokens,
                    "final_review_duration": manager.final_review_total_time,
                    "test_duration": manager.test_total_time,
                    "judge_token_usage": test_result_data.get("judge_token_usage", {}),
                    "judge_cost": test_result_data.get("judge_cost", 0.0),
                    "judge_model": test_result_data.get("judge_model"),
                }

                save_all_costs(
                    workflow_config.output_dir,
                    manager_metrics,
                    subagent_results,
                    wall_clock_duration=runtime_seconds,
                    manager_cost_breakdown=manager_cost_breakdown,
                    model=workflow_config.model,
                    subagent_model=workflow_config.subagent_model,
                    paper_id=getattr(task_module.config, 'paper_id', None),
                )

                # Paperbench: Save manager events after everything (test included)
                if manager.conversation:
                    events = list(manager.conversation.state.events)
                    engineer_id = "manager"
                    print(f"[Manager] Saving {len(events)} events to {engineer_id}_events.jsonl...")
                    for idx, event in enumerate(events):
                        serialized = serialize_event(event, idx)
                        serialized["engineer_id"] = engineer_id
                        serialized["phase"] = "scan_and_delegate"
                        serialized["start_time"] = serialized.get("timestamp")
                        if idx + 1 < len(events):
                            next_ts = getattr(events[idx + 1], 'timestamp', None)
                            serialized["end_time"] = next_ts
                        else:
                            serialized["end_time"] = manager.delegation_end_time.isoformat() if manager.delegation_end_time else None
                        output_logger.log_agent_event(engineer_id, serialized)

        finally:
            manager.cleanup()

    total_time = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("Workflow Complete")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")

    if is_commit0:
        if subagent_results:
            merged_count = len([r for r in subagent_results if r.merged])
            committed_count = len([r for r in subagent_results if r.success])
            recovered_count = len([r for r in subagent_results if r.merged and not r.success])
            failed_count = len([r for r in subagent_results if not r.merged])
            total_cost = sum(r.cost for r in subagent_results)
            print("\nSubagent Results:")
            print(f"- Total: {len(subagent_results)}")
            print(f"- Merged: {merged_count} (committed: {committed_count}, recovered: {recovered_count})")
            print(f"- Failed: {failed_count}")
            print(f"- Total cost: ${total_cost:.4f}")

            for result in subagent_results:
                if result.merged and result.success:
                    status = "SUCCESS"
                elif result.merged and not result.success:
                    status = "RECOVERED"
                else:
                    status = "FAILED"
                print(f"- {result.engineer_id}: {status}")
                if result.commit_hash:
                    print(f"  Commit: {result.commit_hash}")
                if result.merge_method:
                    print(f"  Merge method: {result.merge_method}")
                if result.error and not result.merged:
                    print(f"  Error: {result.error[:100]}")

        print("\nOutput files:")
        print(f"- {workflow_config.output_dir}/delegations.json")
        print(f"- {workflow_config.output_dir}/cost.json")
        print(f"- {workflow_config.output_dir}/outputs.jsonl")
        print(f"- {workflow_config.output_dir}/runtime.txt")
        if multi_agent:
            print(f"- {workflow_config.output_dir}/patch.diff")
        repo_name = task_module.config.repo_name
        print(f"- {workflow_config.output_dir}/report.json")
        print(f"- {workflow_config.output_dir}/{repo_name}_pytest_exit_code.txt")
        print(f"- {workflow_config.output_dir}/{repo_name}_test_output.txt")
        print(f"- {workflow_config.output_dir}/final_repo/{repo_name}.tar.gz (for retest.py)")
    else:
        print("\nOutput files saved:")
        print(f"- {workflow_config.output_dir}/outputs.jsonl")
        print(f"- {workflow_config.output_dir}/cost.json")
        print(f"- {workflow_config.output_dir}/runtime.txt")
        print(f"- {workflow_config.output_dir}/agent_events/manager_events.jsonl")
        print(f"- {workflow_config.output_dir}/delegations.json")
        print(f"Output directory: {workflow_config.output_dir}")


async def run_workflow(task, workflow_config, task_module, multi_agent=True, **kwargs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = str(Path(workflow_config.output_dir) / f"run_{timestamp}.log")

    with TeeLogger(log_path):
        return await run_workflow_inner(
            task, workflow_config, task_module,
            multi_agent=multi_agent, **kwargs,
        )


def main(task="commit0", model=None, multi_agent=True,
         max_iterations=50, max_subagents=4, sub_iterations=None,
         rounds_of_chat=2, subagent_model=None, output_dir=None, **kwargs):
    model_name = model or os.getenv("LLM_MODEL", "litellm_proxy/neulab/gpt-5-mini")
    subagent_model_name = subagent_model or os.getenv("LLM_SUBAGENT_MODEL")
    subagent_iters = sub_iterations if sub_iterations is not None else 50

    workflow_config = WorkflowConfig(
        model=model_name,
        subagent_model=subagent_model_name,
        manager_max_iterations=max_iterations,
        max_subagents=max_subagents,
        subagent_max_iterations=subagent_iters,
        max_rounds_chat=rounds_of_chat,
    )

    task_module = build_task_module(task, **kwargs)

    if output_dir:
        workflow_config.output_dir = output_dir
    else:
        workflow_config.output_dir = build_output_dir(
            task, model_name, workflow_config, multi_agent=multi_agent, **kwargs,
        )

    Path(workflow_config.output_dir).mkdir(parents=True, exist_ok=True)

    # Sync output_dir to task_module config (used by judge log_dir etc.)
    if hasattr(task_module.config, 'output_dir'):
        task_module.config.output_dir = workflow_config.output_dir

    print(f"[Config] {workflow_config}")

    asyncio.run(run_workflow(
        task, workflow_config, task_module,
        multi_agent=multi_agent, **kwargs,
    ))


if __name__ == "__main__":
    fire.Fire(main)
