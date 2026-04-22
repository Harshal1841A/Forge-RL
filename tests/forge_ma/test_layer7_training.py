"""
Layer 7 — Training tests.
Covers: ForgeEnv Gymnasium API contract, PPOTrainer episode collection,
TrainingStats accumulation, EpisodeOutput returned from full run.
"""
import pytest
from env.forge_env import ForgeEnv, ForgeEnvConfig
from env.episode_output import EpisodeOutput
from training.ppo_trainer_ma import PPOTrainer, TrainingStats



@pytest.fixture
def env():
    cfg = ForgeEnvConfig(budget=5, seed=0)
    return ForgeEnv(cfg)


@pytest.fixture
def trainer():
    cfg = ForgeEnvConfig(budget=5, seed=7)
    return PPOTrainer(
        env_config=cfg,
        n_episodes_per_generation=2,
        max_generations=2,
        use_trl=False,
    )


# ─────────────────────────────────────────────
# ForgeEnv — Gymnasium contract
# ─────────────────────────────────────────────
class TestForgeEnv:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "claim_text" in obs
        assert "budget_remaining" in obs
        assert isinstance(info, dict)
        assert "true_chain" in info

    def test_obs_budget_at_full_after_reset(self, env):
        obs, _ = env.reset()
        assert obs["budget_remaining"] == env.budget

    def test_step_returns_5_tuple(self, env):
        env.reset()
        result = env.step()
        assert len(result) == 5

    def test_step_reward_is_float(self, env):
        env.reset()
        _, reward, terminated, truncated, info = env.step()
        # intermediate steps return 0.0
        assert isinstance(reward, float)

    def test_budget_decrements_per_step(self, env):
        obs, _ = env.reset()
        initial = obs["budget_remaining"]
        obs2, *_ = env.step()
        assert obs2["budget_remaining"] == initial - 1

    def test_terminates_at_budget(self, env):
        env.reset()
        terminated = False
        for _ in range(env.budget + 2):
            _, _, terminated, truncated, _ = env.step()
            if terminated or truncated:
                break
        assert terminated is True

    def test_episode_output_set_after_done(self, env):
        env.reset()
        for _ in range(env.budget):
            _, _, terminated, _, _ = env.step()
            if terminated:
                break
        assert env.episode_output is not None
        assert isinstance(env.episode_output, EpisodeOutput)

    def test_step_raises_after_done(self, env):
        env.reset()
        for _ in range(env.budget):
            _, _, terminated, _, _ = env.step()
            if terminated:
                break
        with pytest.raises(RuntimeError):
            env.step()

    def test_reset_clears_episode_output(self, env):
        env.reset()
        for _ in range(env.budget):
            _, _, t, _, _ = env.step()
            if t:
                break
        assert env.episode_output is not None
        env.reset()
        assert env.episode_output is None

    def test_red_chain_respects_k_max(self, env):
        from env.primitives import K_MAX
        env.reset()
        for _ in range(env.budget):
            obs, _, t, _, _ = env.step()
            assert len(obs["red_chain"]) <= K_MAX
            if t:
                break

    def test_reward_in_range_at_terminal(self, env):
        env.reset()
        final_reward = 0.0
        for _ in range(env.budget):
            _, r, t, _, _ = env.step()
            if t:
                final_reward = r
                break
        assert -1.0 <= final_reward <= 1.0


# ─────────────────────────────────────────────
# TrainingStats
# ─────────────────────────────────────────────
class TestTrainingStats:
    def test_initial_state(self):
        s = TrainingStats()
        assert s.episodes_run == 0
        assert s.total_reward == 0.0

    def test_update_increments_count(self, env):
        env.reset()
        for _ in range(env.budget):
            _, r, t, _, info = env.step()
            if t:
                ep = info["episode_output"]
                break
        s = TrainingStats()
        s.update(ep, r)
        assert s.episodes_run == 1
        assert s.total_reward == pytest.approx(r)
        assert s.mean_reward == pytest.approx(r)

    def test_summary_keys(self):
        s = TrainingStats()
        summary = s.summary()
        required = {"generation", "episodes_run", "mean_reward",
                    "max_reward", "min_reward", "over_budget_pct", "chain_accuracy"}
        assert required.issubset(summary.keys())


# ─────────────────────────────────────────────
# PPOTrainer
# ─────────────────────────────────────────────
class TestPPOTrainer:
    def test_run_single_episode_returns_output(self, trainer):
        ep = trainer.run_single_episode()
        assert isinstance(ep, EpisodeOutput)

    def test_train_returns_stats(self, trainer):
        stats = trainer.train()
        assert isinstance(stats, TrainingStats)
        assert stats.episodes_run > 0

    def test_train_episodes_logged(self, trainer):
        stats = trainer.train()
        # 2 gens × 2 episodes = 4
        assert stats.episodes_run == 4

    def test_mean_reward_in_range(self, trainer):
        stats = trainer.train()
        assert -1.0 <= stats.mean_reward <= 1.0

    def test_history_length_matches_episodes(self, trainer):
        stats = trainer.train()
        assert len(stats.episode_history) == stats.episodes_run
