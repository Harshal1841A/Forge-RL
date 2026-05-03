import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)

run_cmd("python -m rewards.validate")

code_2 = """
from rewards.hierarchical_reward import EXPERT_APPROVE_BONUS, EXPERT_REJECT_BONUS
from agents.expert_reviewer_agent import ExpertReviewerAgent
e = ExpertReviewerAgent(mode="ising")
assert e.bonus_reward('APPROVE') == EXPERT_APPROVE_BONUS == 0.15, \
    f'APPROVE mismatch: {e.bonus_reward("APPROVE")} vs {EXPERT_APPROVE_BONUS}'
assert e.bonus_reward('REJECT') == EXPERT_REJECT_BONUS == -0.10, \
    f'REJECT mismatch: {e.bonus_reward("REJECT")} vs {EXPERT_REJECT_BONUS}'
print('Expert bonus consistent: APPROVE=+0.15, REJECT=-0.10 OK')
"""
run_cmd(f'python -c "{code_2}"')

code_3 = """
from rewards.hierarchical_reward import _compute_plausibility_delta
from env.claim_graph_ma import ClaimGraph, ClaimNode, EvidenceEdge

def _make_graph(n_adv_nodes=0):
    root = ClaimNode(id='root', text='WHO study confirms 87% risk reduction.',
                     domain='root', trust_score=0.8)
    nodes = [root]
    edges = []
    for i in range(n_adv_nodes):
        adv = ClaimNode(id=f'adv{i}', text='adversarial', domain='adversarial',
                        trust_score=0.1, injected=True)
        nodes.append(adv)
        edges.append(EvidenceEdge('root', f'adv{i}', 'adversarial', 0.4, True))
    return ClaimGraph(nodes=nodes, edges=edges, root_id='root')

before = _make_graph(0)
after  = _make_graph(3)
delta  = _compute_plausibility_delta('same text', 'same text', before, after)
assert delta != 0.0, f'Plausibility delta is 0.0 — graph path not working!'
print(f'Plausibility delta OK: {delta:.4f} (non-zero)')
"""
run_cmd(f'python -c "{code_3}"')

code_4 = """
from env.reward import verdict_reward
r = verdict_reward('verified', 'real', 0.9, 5, 10, False, False)
assert r > 0.5, f'verified vs real should be correct, got {r}'
r2 = verdict_reward('real', 'real', 0.9, 5, 10, False, False)
assert abs(r - r2) < 0.01, f'verified and real should give same reward'
print(f'Verdict normalisation OK: verified={r:.3f}, real={r2:.3f}')
"""
run_cmd(f'python -c "{code_4}"')

code_5 = """
from rewards.budget_penalty import compute_budget_penalty
# Use only 3 of 10 steps with 2 useful tools
result = compute_budget_penalty(steps_taken=3, budget_limit=10, useful_tools_called=2)
assert result.efficiency_bonus > 0, 'Efficiency bonus should be positive'
assert result.total > 0, f'Budget total should be positive for efficient agent, got {result.total}'
print(f'Budget efficiency OK: bonus={result.efficiency_bonus}, total={result.total}')
"""
run_cmd(f'python -c "{code_5}"')

code_6 = """
from rewards.tactic_pr import compute_tactic_pr
result = compute_tactic_pr([], [])
assert result['f1'] == 0.5, f'Both-empty F1 should be 0.5, got {result["f1"]}'
print(f'F1 both-empty OK: {result}')
"""
run_cmd(f'python -c "{code_6}"')

code_7 = """
from rewards.hierarchical_reward import _chain_entropy_bonus
from env.primitives import PrimitiveType as P
chains = [
    [P.SOURCE_LAUNDER, P.TEMPORAL_SHIFT],
    [P.QUOTE_FABRICATE, P.NETWORK_AMPLIFY],
    [P.SOURCE_LAUNDER, P.CITATION_FORGE],
    [P.ENTITY_SUBSTITUTE],
]
bonus = _chain_entropy_bonus(chains, mean_ted=0.30)
assert bonus > 0.0, f'Entropy bonus should fire at TED=0.30, got {bonus}'
print(f'Entropy bonus at TED=0.30 OK: {bonus:.4f}')

bonus_low = _chain_entropy_bonus(chains, mean_ted=0.20)
assert bonus_low == 0.0, f'Entropy bonus should NOT fire at TED=0.20'
print(f'Entropy bonus at TED=0.20 correctly 0.0')
"""
run_cmd(f'python -c "{code_7}"')

code_8 = """
import random
from rewards.hierarchical_reward import compute_reward
from env.primitives import PrimitiveType
prims = list(PrimitiveType)
fails = 0
for _ in range(500):
    pred = [random.sample(prims, random.randint(0,3)) for _ in range(4)]
    true = random.sample(prims, random.randint(1,4))
    r = compute_reward(
        predicted_chains=pred, true_chain=true,
        claim_text_before='Same text', claim_text_after='Same text',
        consensus_level=random.choice(['unanimous','majority_3','split_2_2','all_different']),
        expert_decision=random.choice(['APPROVE','REJECT']),
        steps_taken=random.randint(1,10), budget_limit=10,
        useful_tools_called=random.randint(0,5),
    )
    if not (-1.0 <= r.total <= 1.0):
        fails += 1
        print(f'OOB: {r}')
print(f'Range check: {500-fails}/500 passed' + (' — ALL OK' if fails==0 else f' — {fails} FAILED'))
assert fails == 0
"""
run_cmd(f'python -c "{code_8}"')
