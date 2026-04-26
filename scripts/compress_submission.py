import zipfile
import os
from pathlib import Path

def compress_project(output_path, project_root):
    # Files and folders to include
    include_patterns = [
        "blue_team/",
        "red_team/",
        "rewards/",
        "env/",
        "agents/",
        "server/",
        "training/",
        "baselines/",
        "data/disarm_v1.json",
        "scripts/run_baseline.py",
        "scripts/verify_full_system.py",
        "Dockerfile",
        "requirements.txt",
        "pyproject.toml",
        "openenv.yaml",
        "config.py",
        "models.py",
        "schemas.py",
        "main.py",
        "app.py",
        "bridge.py",
        "action_validator.py",
        "budget_penalty.py",
        "cache_manager.py",
        "claim_graph.py",
        "claim_graph_ma.py",
        "client.py",
        "conftest.py",
        "cross_reference.py",
        "curriculum.py",
        "demo.py",
        "disarm_registry.json",
        "download_liar.py",
        "entity_link.py",
        "episode.py",
        "episode_output.py",
        "eval.py",
        "evaluator.py",
        "expert_reviewer_agent.py",
        "forge_env.py",
        "forge_policy.json",
        "generator_agent.py",
        "gin_predictor.py",
        "gin_trainer_ma.py",
        "gnn_policy.py",
        "grade.py",
        "hae_model.py",
        "heuristic_agent.py",
        "hierarchical_reward.py",
        "inference.py",
        "llm_agent.py",
        "llm_agent_ma.py",
        "metrics.py",
        "misinfo_env.py",
        "mock_data.py",
        "narrative_critic.py",
        "negotiated_search.py",
        "network_cluster.py",
        "node_features.py",
        "oversight_report.py",
        "pipeline.py",
        "plausibility.py",
        "ppo_agent.py",
        "ppo_trainer_ma.py",
        "pretrain.py",
        "primitives.py",
        "query_source.py",
        "random_agent.py",
        "red_agent.py",
        "red_step_reward.py",
        "reliability.py",
        "replay_buffer.py",
        "report_manager.py",
        "reward.py",
        "run_selfplay.py",
        "self_play.py",
        "step.py",
        "tactic_edit_dist.py",
        "tactic_pr.py",
        "task_base.py",
        "task_coordinated_campaign.py",
        "task_fabricated_stats.py",
        "task_image_forensics.py",
        "task_out_of_context.py",
        "task_plandemic.py",
        "task_politifact.py",
        "task_satire_news.py",
        "task_sec_fraud.py",
        "task_verified_fact.py",
        "temporal_audit.py",
        "tool_registry.py",
        "trace_origin.py",
        "train_ppo.py",
        "README.md",
        "HACKATHON_README.md",
        "REAL_WORLD_IMPACT.md",
        "LICENSE"
    ]

    exclude_dirs = {".git", "node_modules", "spatial-saas", "__pycache__", ".pytest_cache", ".gemini", "graphify-out", "scratch"}
    exclude_files = {".env", "forge_ma_submission.zip", "spatial_saas.zip", "backend_error.log", "forge.db"}

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                
                file_path = Path(root) / file
                arcname = file_path.relative_to(project_root)
                
                # Check if file matches include patterns
                # For simplicity, we'll just include everything that isn't excluded
                # but if we want to be strict:
                should_include = True
                # if include_patterns:
                #     should_include = any(str(arcname).startswith(p.rstrip('/')) for p in include_patterns)
                
                if should_include:
                    zipf.write(file_path, arcname)
                    print(f"Added: {arcname}")

if __name__ == "__main__":
    project_root = Path(os.getcwd())
    output_zip = project_root.parent / "forge_ma_submission.zip"
    print(f"Creating zip at {output_zip}...")
    compress_project(output_zip, project_root)
    print("Done!")
