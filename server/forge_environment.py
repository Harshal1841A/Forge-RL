# server/forge_environment.py
import uuid
import random
from openenv.core.env_server import Environment
from models import ForgeAction, ForgeObservation, ForgeState

ACTIONS = [
    "query_source",        # 0
    "cross_reference",     # 1
    "network_cluster",     # 2
    "temporal_audit",      # 3
    "entity_link",         # 4
    "context_retrieve",    # 5
    "flag_manipulation",   # 6
    "image_verify",        # 7
    "citation_check",      # 8
    "amplification_scan",  # 9
    "submit_verdict_misinfo",    # 10
    "submit_verdict_satire",     # 11
    "submit_verdict_verified",   # 12
]

VERDICT_ACTIONS = {10: "misinfo", 11: "satire", 12: "verified"}


class ForgeEnvironment(Environment):

    def __init__(self):
        self._episode_id = ""
        self._steps = 0
        self._max_steps = 10
        self._task = None
        self._true_label = "misinfo"
        self._claim_text = ""
        self._evidence_coverage = 0.0
        self._verdict_submitted = None
        self._manipulation_flagged = False
        self._total_reward = 0.0
        self._useful_tools = 0

    def reset(self) -> ForgeObservation:
        from env.tasks import TASK_REGISTRY
        self._episode_id = str(uuid.uuid4())
        self._steps = 0
        self._verdict_submitted = None
        self._total_reward = 0.0
        self._evidence_coverage = 0.0
        self._manipulation_flagged = False
        self._useful_tools = 0

        # Pick random task
        task_name = random.choice(list(TASK_REGISTRY.keys()))
        self._task = TASK_REGISTRY[task_name]()
        graph = self._task.generate(difficulty=random.randint(1, 3),
                                    seed=random.randint(0, 9999))
        self._true_label = graph.true_label
        self._claim_text = graph.root.text if graph and graph.root else ""

        return ForgeObservation(
            done=False,
            reward=0.0,
            episode_id=self._episode_id,
            claim_text=self._claim_text,
            evidence_coverage=0.0,
            source_diversity=0.0,
            contradiction_count=0,
            manipulation_flagged=False,
            budget_remaining=1.0,
            steps_used=0,
            actions_available=ACTIONS,
        )

    def step(self, action: ForgeAction) -> ForgeObservation:
        if self._steps >= self._max_steps:
            return self._terminal_obs(reward=0.001)

        act_idx = int(action.action)
        if not (0 <= act_idx <= 12):
            act_idx = 0

        self._steps += 1
        done = False
        step_reward = -0.01  # step cost

        if act_idx in VERDICT_ACTIONS:
            # Episode ends
            done = True
            predicted = VERDICT_ACTIONS[act_idx]
            correct = (predicted == self._true_label)
            base = 0.90 if correct else 0.001
            efficiency = min(5 / max(self._steps, 1), 1.0) * 0.2
            coverage_bonus = self._evidence_coverage * 0.1
            budget_ratio = self._steps / self._max_steps
            budget_bonus = 0.05 if (self._useful_tools >= 2 and budget_ratio < 0.60) else 0.0
            step_reward = float(max(0.001, min(0.999, base + efficiency + coverage_bonus + budget_bonus)))
            self._verdict_submitted = predicted
        else:
            # Investigation tool used
            self._useful_tools += 1
            self._evidence_coverage = min(1.0, self._evidence_coverage + 0.08)
            if act_idx == 6:  # flag_manipulation
                self._manipulation_flagged = True

        if self._steps >= self._max_steps and not done:
            done = True
            step_reward = max(0.001, step_reward - 0.50)  # over-budget penalty

        self._total_reward += step_reward

        return ForgeObservation(
            done=done,
            reward=step_reward,
            episode_id=self._episode_id,
            claim_text=self._claim_text,
            evidence_coverage=round(self._evidence_coverage, 4),
            source_diversity=round(min(self._useful_tools / 5.0, 1.0), 4),
            contradiction_count=max(0, self._useful_tools - 3),
            manipulation_flagged=self._manipulation_flagged,
            budget_remaining=round(1.0 - self._steps / self._max_steps, 4),
            steps_used=self._steps,
            actions_available=ACTIONS,
            metadata={"true_label_hidden": True, "total_reward": self._total_reward}
        )

    def state(self) -> ForgeState:
        return ForgeState(
            episode_id=self._episode_id,
            step_count=self._steps,
            task_id=getattr(self._task, "task_id", "unknown"),
            verdict_submitted=self._verdict_submitted,
            total_reward=round(self._total_reward, 4),
        )

    def _terminal_obs(self, reward: float) -> ForgeObservation:
        return ForgeObservation(
            done=True,
            reward=reward,
            episode_id=self._episode_id,
            claim_text=self._claim_text,
            evidence_coverage=self._evidence_coverage,
            source_diversity=0.0,
            contradiction_count=0,
            manipulation_flagged=self._manipulation_flagged,
            budget_remaining=0.0,
            steps_used=self._steps,
            actions_available=[],
        )
