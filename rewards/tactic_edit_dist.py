def tactic_edit_distance(predicted: list, true_chain: list) -> float:
    """
    TED = 1 - (edit_distance(predicted, true) / max(len(predicted), len(true), 1))
    Returns float in [0.0, 1.0]. 1.0 = perfect match.
    Handles k=0 (empty predicted) and k=1 to k=4.
    NEVER returns exactly 0.0 or 1.0 — clip to (0.001, 0.999).
    """
    n = len(predicted)
    m = len(true_chain)
    
    if n == 0 and m == 0:
        dist = 0
    elif n == 0:
        dist = m
    elif m == 0:
        dist = n
    else:
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
            
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if predicted[i - 1] == true_chain[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost # substitution
                )
        dist = dp[n][m]

    max_len = max(n, m, 1)
    ted = 1.0 - (dist / max_len)
    
    # Clip strictly inside (0.001, 0.999) per instructions
    return max(0.001, min(0.999, ted))
