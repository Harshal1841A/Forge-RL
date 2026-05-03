from env.primitives import PrimitiveType
from typing import List
import numpy as np


def tactic_edit_distance(
    pred: List[PrimitiveType],
    true: List[PrimitiveType],
) -> float:
    """
    Normalised edit distance between two primitive chains.
    Returns values in [0.001, 0.999]:
      - 0.001 = identical (or both-empty → trivially correct)
      - 0.999 = maximally different

    Clipped to open interval so that downstream components multiplying
    by TED never produce exact 0 or 1, preserving gradient signal.
    """
    n, m = len(pred), len(true)
    if n == 0 and m == 0:
        # Both empty: treat as identical (distance 0), clip to floor
        return 0.001

    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == true[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    raw = float(dp[n][m]) / max(n, m, 1)
    # Clip to open interval (0.001, 0.999) so score is never exactly 0 or 1
    return float(np.clip(raw, 0.001, 0.999))
