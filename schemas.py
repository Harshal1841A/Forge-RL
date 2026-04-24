"""
server/schemas.py — OpenEnv Compatible API schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional


class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    difficulty: int = 1
    seed: Optional[int] = None
    use_live_tools: bool = False
    agent_id: Optional[str] = None


class ResetResponse(BaseModel):
    episode_id: str
    observation: List[float]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    episode_id: Optional[str] = None  # Optional for single-tenant mode
    action: int = Field(..., ge=0, le=12)


class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateRequest(BaseModel):
    episode_id: Optional[str] = None


class StateResponse(BaseModel):
    episode_id: str
    observation: List[float]
    typed_observation: Optional['Observation'] = None
    done: bool
    total_reward: float
    info: Dict[str, Any]


class Observation(BaseModel):
    """OpenEnv typed observation."""
    episode_id: str
    vector: List[float] = Field(..., description="Flat observation vector, length 3859")
    claim_text: str = Field(default="")
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    source_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_count: int = Field(default=0, ge=0)
    manipulation_flagged: bool = Field(default=False)
    budget_remaining: float = Field(default=1.0, ge=0.0, le=1.0)
    steps_used: int = Field(default=0, ge=0)


class Action(BaseModel):
    """OpenEnv typed action."""
    action: int = Field(..., ge=0, le=12)
    episode_id: Optional[str] = None
    action_name: Optional[str] = None


class Reward(BaseModel):
    """OpenEnv typed reward."""
    value: float = Field(..., ge=0.001, le=0.999)
    done: bool
    shaped: bool = True
    components: Dict[str, float] = Field(default_factory=dict)


class GradeResponse(BaseModel):
    episode_id: str
    verdict: Optional[str] = None
    true_label: str
    correct: bool
    accuracy: float
    manipulation_detected: bool
    evidence_coverage: float
    steps_used: int
    efficiency_score: float
    total_reward: float
    grade_breakdown: Dict[str, float]

    @field_validator("accuracy", "total_reward", "efficiency_score", mode="before")
    @classmethod
    def clip_to_open_interval(cls, v):
        return round(float(max(0.001, min(0.999, float(v)))), 4)
