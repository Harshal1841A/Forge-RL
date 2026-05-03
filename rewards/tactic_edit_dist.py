from env.primitives import PrimitiveType
from typing import List
import numpy as np

def tactic_edit_distance(
    pred: List[PrimitiveType],
    true: List[PrimitiveType],
) -> float:
    """Normalised edit distance between two primitive chains (0=identical, 1=max diff)."""
    n, m = len(pred), len(true)
    if n == 0 and m == 0:
        return 0.0
    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i-1] == true[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return float(dp[n][m]) / max(n, m, 1)
