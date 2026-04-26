# models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from openenv.core.env_server import Action, Observation, State


@dataclass
class ForgeAction(Action):
    """
    13 possible actions.
    0-9: investigation tools
    10: submit_verdict_misinfo
    11: submit_verdict_satire
    12: submit_verdict_verified
    """
    action: int = 0


@dataclass
class ForgeObservation(Observation):
    """
    Observation returned after every step.
    done, reward, metadata are inherited from Observation base class.
    Do NOT redefine them here.
    """
    claim_text: str = ""
    evidence_coverage: float = 0.0
    source_diversity: float = 0.0
    contradiction_count: int = 0
    manipulation_flagged: bool = False
    budget_remaining: float = 1.0
    steps_used: int = 0
    episode_id: str = ""
    actions_available: List[str] = field(default_factory=list)


@dataclass
class ForgeState(State):
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    verdict_submitted: Optional[str] = None
    total_reward: float = 0.0
