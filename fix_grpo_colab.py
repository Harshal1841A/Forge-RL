"""Apply all 6 fixes to forge_grpo_colab.ipynb"""
import json

NB_PATH = "training/forge_grpo_colab.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ─── CELL 3 (index 2): FIX 1 + FIX 2 + FIX 3 ────────────────────────────────
c3 = "".join(cells[2]["source"])

# FIX 2: TOOL_ACTIONS
c3 = c3.replace(
    "TOOL_ACTIONS = [0, 5, 2]  # query_source, temporal_audit, cross_reference",
    "# 0=query_source, 3=temporal_audit, 1=cross_reference (5=context_retrieve, NOT temporal)\nTOOL_ACTIONS = [0, 3, 1]"
)

# FIX 1: _parse_verdict
old_parse = """def _parse_verdict(text):
    t = text.lower()
    if any(w in t for w in ['fabricat']): return 12
    if any(w in t for w in ['out of context', 'recontextual', 'cropped']): return 11
    if any(w in t for w in ['satire', 'parody', 'joke']): return 10
    if any(w in t for w in ['misinfo', 'false', 'manipulat', 'mislead']): return 9
    if any(w in t for w in ['verified', 'true', 'legitimate', 'accurate']): return 8
    return 9"""

new_parse = """# Action space: 10=misinfo, 11=satire, 12=verified
# fabricated and out_of_context → closest is misinfo (10)
def _parse_verdict(text: str) -> int:
    t = text.lower()
    # Check satire first (distinct signal words)
    if any(w in t for w in ['satire', 'parody', 'joke', 'humor', 'comedic']):
        return 11
    # Check verified (must come before 'false' check to avoid substring collision)
    if any(w in t for w in ['verified', 'legitimate', 'accurate', 'credible', 'true claim']):
        return 12
    # Misinfo umbrella (fabricated, out_of_context, mislead all → misinfo)
    if any(w in t for w in ['misinfo', 'false', 'manipulat', 'mislead',
                             'fabricat', 'out of context', 'deceptive', 'disinformation']):
        return 10
    return 10  # safe default: misinfo (a terminal verdict, not a tool)"""

c3 = c3.replace(old_parse, new_parse)

# FIX 2 continued: tool loop + verdict with done guard
old_tool_loop = """            done = False
            for act in TOOL_ACTIONS:
                if done: break
                _, done = _safe_step(env, act)
            verdict_action = _parse_verdict(comp_text)
            reward, _ = _safe_step(env, verdict_action)"""

new_tool_loop = """            done = False
            for act in TOOL_ACTIONS:
                if done:
                    break
                try:
                    _, done = _safe_step(env, act)
                except Exception:
                    break  # budget exhausted or env error — stop tools
            
            # Only submit verdict if episode not already done
            if not done:
                verdict_action = _parse_verdict(comp_text)
                try:
                    reward, _ = _safe_step(env, verdict_action)
                except Exception:
                    reward = 0.001
            else:
                # Env terminated before verdict — penalise lightly
                reward = 0.001"""

c3 = c3.replace(old_tool_loop, new_tool_loop)

# FIX 3: reward_fn signature + kwargs fallback
old_reward_sig = """def reward_fn(prompts, completions, claim_texts=None, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):"""

new_reward_sig = """def reward_fn(prompts, completions, claim_texts=None, **kwargs):
    # GRPOTrainer passes dataset cols via kwargs — also check kwargs
    if claim_texts is None:
        claim_texts = kwargs.get('claim_texts', None)
    rewards = []
    for i, completion in enumerate(completions):"""

c3 = c3.replace(old_reward_sig, new_reward_sig)

cells[2]["source"] = c3.splitlines(keepends=True)

# ─── CELL 5 (index 4): FIX 5 ─────────────────────────────────────────────────
c5 = "".join(cells[4]["source"])

# Fix default_verdict
c5 = c5.replace(
    "def evaluate_heuristic(n_episodes=20, default_verdict=9):",
    "def evaluate_heuristic(n_episodes=20, default_verdict=10):  # 10=misinfo, was 9=tool"
)

# Fix tool loop + done guard in evaluate_heuristic
old_heur_loop = """            done = False
            for act in TOOL_ACTIONS:
                if done: break
                _, done = _safe_step(env, act)
            reward, _ = _safe_step(env, default_verdict)"""

new_heur_loop = """            done = False
            for act in TOOL_ACTIONS:
                if done:
                    break
                try:
                    _, done = _safe_step(env, act)
                except Exception:
                    break
            if not done:
                try:
                    reward, _ = _safe_step(env, default_verdict)
                except Exception:
                    reward = 0.001
            else:
                reward = 0.001"""

c5 = c5.replace(old_heur_loop, new_heur_loop)

cells[4]["source"] = c5.splitlines(keepends=True)

# ─── CELL 7 (index 6): FIX 4 + FIX 6 ────────────────────────────────────────
c7 = "".join(cells[6]["source"])

# FIX 6: done guard in evaluate_trained
old_trained_loop = """            done = False
            for act in TOOL_ACTIONS:
                if done: break
                _, done = _safe_step(env, act)
            reward, _ = _safe_step(env, _parse_verdict(response))"""

new_trained_loop = """            done = False
            for act in TOOL_ACTIONS:
                if done:
                    break
                try:
                    _, done = _safe_step(env, act)
                except Exception:
                    break
            if not done:
                try:
                    reward, _ = _safe_step(env, _parse_verdict(response))
                except Exception:
                    reward = 0.001
            else:
                reward = 0.001"""

c7 = c7.replace(old_trained_loop, new_trained_loop)

# FIX 4: log history reward keys
old_log = """for l in trainer.state.log_history:
    for key in ('rewards/mean', 'reward/mean', 'reward'):
        if key in l:
            steps.append(l['step']); rew.append(l[key]); break"""

new_log = """REWARD_LOG_KEYS = (
    'rewards/mean', 'reward/mean', 'reward',
    'train/reward', 'rewards/reward_mean',   # TRL version variants
    'mean_reward',
)
for l in trainer.state.log_history:
    if 'step' not in l:
        continue
    for key in REWARD_LOG_KEYS:
        if key in l:
            steps.append(l['step'])
            rew.append(l[key])
            break"""

c7 = c7.replace(old_log, new_log)

cells[6]["source"] = c7.splitlines(keepends=True)

# ─── Save ─────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

# ─── Verify ───────────────────────────────────────────────────────────────────
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

full = "\n".join("".join(c["source"]) for c in nb2["cells"])

checks = [
    ("_parse_verdict returns 10/11/12 only", "return 9" not in full and "return 8" not in full),
    ("verified before false", full.index("'verified'") < full.index("'false'")),
    ("default return is 10", "return 10  # safe default" in full),
    ("TOOL_ACTIONS = [0, 3, 1]", "TOOL_ACTIONS = [0, 3, 1]" in full),
    ("try/except around _safe_step", full.count("try:\n                    _, done = _safe_step") >= 3),
    ("verdict skipped if done", full.count("if not done:") >= 3),
    ("default_verdict=10", "default_verdict=10" in full),
    ("REWARD_LOG_KEYS", "REWARD_LOG_KEYS" in full),
    ("claim_texts kwarg fallback", "claim_texts = kwargs.get('claim_texts', None)" in full),
]

print("\n=== VERIFICATION ===")
all_pass = True
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok: all_pass = False

if all_pass:
    print("\nAll 6 fixes applied and verified.")
else:
    print("\nSome checks failed -- review manually.")
