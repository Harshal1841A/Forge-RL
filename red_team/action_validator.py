"""
Red Team Action Validator.
SPEC (Master Prompt §Layer4):
  - Hard-enforce K_MAX = 4 on tactic chain length
  - Reject any TA-prefix DISARM IDs (only T-prefix allowed)
  - Reject action if it would produce a duplicate consecutive primitive
  - All rejections are silent (return False, no exceptions)
"""
import re
from typing import List
from env.primitives import PrimitiveType, K_MAX

# Pattern: T followed by digits only (e.g. T1234.001). TA-prefix is forbidden.
_VALID_DISARM_RE = re.compile(r'^T\d{4}(\.\d{3})?$')

# HIGH BUG 3 — DISARM documentation drift:
# PRD v8.1 Section 1.1 maps NETWORK_AMPLIFY → TA15 (threat-actor object).
# red_agent._disarm_for() maps it to ["T0049"].
# This validator enforces T-prefix only (regex above rejects TA-prefix).
# DO NOT update the mapping to TA15 — it would silently block every
# NETWORK_AMPLIFY action, locking the primitive out of the Red Team.
# NOTE: PRD v8.1 specifies TA15 but ActionValidator enforces T-prefix only;
# using T0049 instead. Update the PRD comment if the DISARM spec changes.


def _is_valid_disarm_id(disarm_id: str) -> bool:
    """Return True iff disarm_id matches T-prefix pattern (not TA-)."""
    return bool(_VALID_DISARM_RE.match(disarm_id))


def validate_chain(candidate_chain: List[PrimitiveType]) -> bool:
    """
    Gate 1 — Chain length: must not exceed K_MAX.
    """
    if len(candidate_chain) > K_MAX:
        return False
    return True


def validate_no_consecutive_duplicate(candidate_chain: List[PrimitiveType]) -> bool:
    """
    Gate 2 — No identical adjacent primitives (e.g. [P1, P1] blocked).
    A repeat at non-adjacent positions is allowed.
    """
    for i in range(len(candidate_chain) - 1):
        if candidate_chain[i] == candidate_chain[i + 1]:
            return False
    return True


def validate_disarm_tags(disarm_ids: List[str]) -> bool:
    """
    Gate 3 — All DISARM IDs in the proposed action must pass T-prefix check.
    Returns False if any entry is TA-prefixed or malformed.
    """
    for did in disarm_ids:
        if not _is_valid_disarm_id(did):
            return False
    return True


def validate_action(candidate_chain: List[PrimitiveType],
                    disarm_ids: List[str]) -> bool:
    """
    Master gate: all three checks must pass.
    Designed for silent failure (returns bool, never raises).
    """
    try:
        return (
            validate_chain(candidate_chain)
            and validate_no_consecutive_duplicate(candidate_chain)
            and validate_disarm_tags(disarm_ids)
        )
    except Exception:
        return False
