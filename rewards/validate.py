"""
Reward validation utilities for FORGE-RL.
Run: python -m rewards.validate
Verifies all reward components stay within (0.001, 0.999).
"""
from __future__ import annotations
import random
from typing import List

from env.primitives import PrimitiveType


def _random_chain(max_k: int = 4) -> List[PrimitiveType]:
    k = random.randint(0, max_k)
    return random.sample(list(PrimitiveType), min(k, len(PrimitiveType)))


def validate_ted(n: int = 100) -> bool:
    from rewards.tactic_edit_dist import tactic_edit_distance
    failures = []
    for _ in range(n):
        pred = _random_chain()
        true = _random_chain()
        score = tactic_edit_distance(pred, true)
        if not (0.001 <= score <= 0.999):
            failures.append((pred, true, score))
    if failures:
        print(f"TED FAIL: {len(failures)}/{n} out of range")
        for p, t, s in failures[:3]:
            print(f"  pred={p} true={t} score={s}")
        return False
    print(f"TED OK: {n} samples all in (0.001, 0.999)")
    return True


def validate_f1(n: int = 100) -> bool:
    from rewards.tactic_pr import tactic_f1
    failures = []
    for _ in range(n):
        pred = _random_chain()
        true = _random_chain()
        score = tactic_f1(pred, true)
        if not (0.001 <= score <= 0.999):
            failures.append((pred, true, score))
    if failures:
        print(f"F1 FAIL: {len(failures)}/{n} out of range")
        return False
    print(f"F1 OK: {n} samples all in (0.001, 0.999)")
    return True


def validate_budget(n: int = 100) -> bool:
    from rewards.budget_penalty import compute_budget_penalty
    failures = []
    for _ in range(n):
        steps   = random.randint(1, 15)
        budget  = random.randint(5, 10)
        useful  = random.randint(0, steps)
        result  = compute_budget_penalty(steps, budget, useful)
        if result.total > 0.10 or result.total < -1.0:
            failures.append((steps, budget, useful, result.total))
    if failures:
        print(f"Budget FAIL: {len(failures)}/{n} out of range")
        return False
    print(f"Budget OK: {n} samples all in (-1.0, 0.10)")
    return True


def validate_hierarchical(n: int = 200) -> bool:
    from rewards.hierarchical_reward import compute_reward
    from rewards.budget_penalty import BudgetPenaltyResult
    failures = []
    consensus_levels = ["unanimous", "majority_3", "split_2_2", "all_different"]
    expert_decisions = ["APPROVE", "REJECT"]

    for _ in range(n):
        pred_chains = [_random_chain() for _ in range(4)]
        true_chain  = _random_chain()
        steps = random.randint(1, 10)
        result = compute_reward(
            predicted_chains=pred_chains,
            true_chain=true_chain,
            claim_text_before="WHO confirmed vaccines are safe for adults.",
            claim_text_after="WHO confirmed vaccines are safe for adults.",
            consensus_level=random.choice(consensus_levels),
            expert_decision=random.choice(expert_decisions),
            steps_taken=steps,
            budget_limit=10,
            useful_tools_called=random.randint(0, steps),
        )
        if not (0.001 <= result.total <= 0.999):
            failures.append(result)

    if failures:
        print(f"Hierarchical FAIL: {len(failures)}/{n} out of range")
        for r in failures[:3]:
            print(f"  {r}")
        return False
    print(f"Hierarchical OK: {n} samples all in (0.001, 0.999)")
    return True


def validate_all() -> bool:
    print("=== FORGE-RL Reward Validation ===\n")
    results = [
        validate_ted(),
        validate_f1(),
        validate_budget(),
        validate_hierarchical(),
    ]
    print()
    if all(results):
        print("ALL REWARD VALIDATIONS PASSED")
        print("Rewards are OpenEnv compliant (0.001, 0.999)")
        return True
    else:
        failed = sum(1 for r in results if not r)
        print(f"FAILED: {failed}/4 validations. Fix before training.")
        return False


if __name__ == "__main__":
    import sys
    ok = validate_all()
    sys.exit(0 if ok else 1)
