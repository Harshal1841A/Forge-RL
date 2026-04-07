"""
Self-Play Training Loop — Population-Based Training (PBT) for generators.
Implements PAIRED-style regret-maximising curriculum.

Architecture:
  - Pool of N generator variants (default 8)
  - Each generator battles the investigator for M episodes
  - Bottom-25% generators replaced by mutations of top-25%
  - ELO ratings track relative strength
  - Regret = protagonist_reward - antagonist_reward (PAIRED signal)
"""

from __future__ import annotations
import logging
import random
from typing import List, Tuple  # noqa: F401 – Optional removed (unused)

from agents.adversarial.generator_agent import GeneratorAgent, ALL_TACTICS
from agents.ppo_agent import PPOAgent
from agents.heuristic_agent import HeuristicAgent
from env.misinfo_env import MisInfoForensicsEnv
import config

logger = logging.getLogger(__name__)


class SelfPlayTrainer:
    """
    Runs adversarial self-play between a population of generators
    and a single investigator (PPO or heuristic).
    """

    def __init__(
        self,
        obs_dim: int,
        population_size: int = config.GENERATOR_POPULATION_SIZE,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.population: List[GeneratorAgent] = self._init_population(population_size, seed)
        self.investigator = PPOAgent(obs_dim=obs_dim)
        self.antagonist = HeuristicAgent()   # fixed weaker reference for PAIRED

        self.generation = 0
        self.history: List[dict] = []

    def _init_population(self, n: int, seed: int) -> List[GeneratorAgent]:
        styles = ["tabloid", "academic", "official", "social", "neutral"]
        generators = []
        for i in range(n):
            tactic_bias = list(self.rng.sample(ALL_TACTICS, k=2))
            style = styles[i % len(styles)]
            generators.append(GeneratorAgent(
                agent_id=f"gen_{i}",
                tactic_bias=tactic_bias,
                register_style=style,
                seed=seed + i,
            ))
        return generators

    def run_generation(
        self,
        episodes_per_generator: int = 20,
        difficulty: int = 1,
    ) -> dict:
        """
        Run one PBT generation:
        1. Each generator produces episodes
        2. Investigator plays each episode
        3. ELOs updated
        4. Bottom 25% replaced
        5. Investigator updated once
        """
        self.generation += 1
        logger.info("=== Self-Play Generation %d ===", self.generation)

        gen_scores: List[Tuple[GeneratorAgent, float]] = []

        for gen in self.population:
            gen_reward_sum = 0.0
            inv_reward_sum = 0.0
            ant_reward_sum = 0.0
            episodes_done = 0

            for ep in range(episodes_per_generator):
                ep_seed = self.rng.randint(0, 2**20)
                graph = gen.generate(difficulty=difficulty)

                # Investigator episode
                inv_reward = self._run_episode(graph, self.investigator, ep_seed)
                # Antagonist episode (fixed reference)
                ant_reward = self._run_heuristic_episode(graph)

                # PAIRED regret signal
                regret = inv_reward - ant_reward
                gen_reward_sum += -regret   # generator rewarded when regret is HIGH
                inv_reward_sum += inv_reward
                ant_reward_sum += ant_reward
                episodes_done += 1

                gen.update_elo(investigator_won=(inv_reward > 0.5))

            mean_inv_reward = inv_reward_sum / max(episodes_done, 1)
            mean_regret = (inv_reward_sum - ant_reward_sum) / max(episodes_done, 1)
            gen_scores.append((gen, mean_regret))
            logger.info(
                "  Generator %s: inv_reward=%.3f regret=%.3f ELO=%d",
                gen.agent_id, mean_inv_reward, mean_regret, gen.elo,
            )

        # ── PBT Selection ─────────────────────────────────────────────────────
        gen_scores.sort(key=lambda x: x[1], reverse=True)   # high regret = good generator
        n = len(self.population)
        n_replace = max(1, n // 4)
        top_gens = [g for g, _ in gen_scores[:n - n_replace]]
        new_gens = []
        for i in range(n_replace):
            parent = top_gens[i % len(top_gens)]
            mutant = parent.mutate(seed=self.rng.randint(0, 2**20))
            new_gens.append(mutant)
            logger.info("  Replaced bottom generator with mutant of %s", parent.agent_id)
        self.population = top_gens + new_gens

        # ── Investigator PPO update ───────────────────────────────────────────
        best_gen = gen_scores[0][0]
        env = self._build_env_from_generator(best_gen, difficulty)
        rollout_stats = self.investigator.collect_rollout(env)
        update_stats = self.investigator.update()

        stats = {
            "generation": self.generation,
            "best_generator": best_gen.agent_id,
            "best_gen_elo": best_gen.elo,
            "inv_mean_reward": rollout_stats.get("mean_reward", 0.0),
            "ppo_pg_loss": update_stats.get("pg_loss", 0.0),
            "ppo_entropy": update_stats.get("entropy", 0.0),
            "population_elos": [g.elo for g in self.population],
        }
        self.history.append(stats)
        return stats

    def _run_episode(
        self,
        graph,
        agent: PPOAgent,
        seed: int,
    ) -> float:
        """Run a single episode against a fixed graph; return total reward."""
        env = MisInfoForensicsEnv(difficulty=graph.difficulty)
        obs, _ = env.reset(seed=seed)
        # Manually inject the generated graph
        env.graph = graph
        env._claim_embedding = env._embed(graph.root.text)

        total_reward = 0.0
        done = False
        while not done:
            action, _, _ = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        return total_reward

    def _run_heuristic_episode(self, graph) -> float:
        """Run heuristic antagonist on same graph."""
        env = MisInfoForensicsEnv(difficulty=graph.difficulty)
        obs, _ = env.reset()
        env.graph = graph
        env._claim_embedding = env._embed(graph.root.text)
        self.antagonist.reset()

        total_reward = 0.0
        done = False
        while not done:
            action = self.antagonist.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        return total_reward

    def _build_env_from_generator(self, gen: GeneratorAgent, difficulty: int) -> MisInfoForensicsEnv:
        """Build a fresh env configured to sample from the generator."""
        env = MisInfoForensicsEnv(difficulty=difficulty)
        # Monkey-patch task list to use generator's bias

        class _GenTask:
            task_id = gen.agent_id

            def generate(self, difficulty=1, seed=0):
                return gen.generate(difficulty=difficulty)

            def oracle_steps(self, g):
                return 5

            def has_manipulation(self, g):
                return True
        env.tasks = [_GenTask()]
        return env

    def save_population(self, path: str) -> None:
        import json
        import os
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/population.json", "w") as f:
            json.dump([g.to_dict() for g in self.population], f, indent=2)
        self.investigator.save(f"{path}/investigator.pt")
        logger.info("Saved self-play population → %s", path)
