# client.py
from openenv_core import HTTPEnvClient, StepResult
from models import ForgeAction, ForgeObservation, ForgeState
from typing import Any


class ForgeEnv(HTTPEnvClient[ForgeAction, ForgeObservation]):
    """
    Client for FORGE-MA misinformation forensics environment.

    Usage:
        # Sync
        with ForgeEnv(base_url="https://YOUR-USERNAME-forge-ma.hf.space").sync() as env:
            obs = env.reset()
            result = env.step(ForgeAction(action=0))

        # From HF Hub (auto-pulls Docker)
        env = ForgeEnv.from_hub("YOUR_USERNAME/forge-ma")
    """

    def _step_payload(self, action: ForgeAction) -> dict:
        return {"action": int(action.action)}

    def _parse_result(self, payload: dict) -> StepResult[ForgeObservation]:
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=ForgeObservation(**{
                k: v for k, v in obs_data.items()
                if k in ForgeObservation.__dataclass_fields__
            }),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> Any:
        return ForgeState(**{
            k: v for k, v in payload.items()
            if k in ForgeState.__dataclass_fields__
        })
