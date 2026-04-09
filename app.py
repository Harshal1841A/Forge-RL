import gradio as gr
import sys
import time
import random
import re
from pathlib import Path
import logging

# ─── Environment & Agent Imports ──────────────────────────────────────────
import config
from env.misinfo_env import MisInfoForensicsEnv, ACTIONS
from agents.llm_agent import LLMAgent
from env.tasks import TASK_REGISTRY

# ─── Debug logging: write to file for traceability, keep stdout clean ─────────
_debug_handler = logging.FileHandler("forge_debug.log", mode="a", encoding="utf-8")
_debug_handler.setLevel(logging.DEBUG)
_debug_handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
logging.getLogger("env").addHandler(_debug_handler)
logging.getLogger("env").setLevel(logging.DEBUG)
logging.getLogger("env").propagate = False  # don't leak to stdout
logging.getLogger("agents").addHandler(_debug_handler)
logging.getLogger("agents").setLevel(logging.DEBUG)
logging.getLogger("agents").propagate = False
_app_logger = logging.getLogger("forge.app")
_app_logger.addHandler(_debug_handler)
_app_logger.setLevel(logging.DEBUG)
_app_logger.propagate = False

# ─── Task metadata ────────────────────────────────────────────────────────────
TASK_META = {
    "fabricated_stats":     {"icon": "📊", "code": "FAB_STAT"},
    "out_of_context":       {"icon": "🔀", "code": "OOC_STRIP"},
    "coordinated_campaign": {"icon": "🤖", "code": "BOT_CAMP"},
    "satire_news":          {"icon": "🎭", "code": "SAT_PARSE"},
    "verified_fact":        {"icon": "✅", "code": "VER_FACT"},
    "politifact_liar":      {"icon": "🏛️", "code": "POL_LIAR"},
    "image_forensics":      {"icon": "🖼️", "code": "IMG_FRNSC"},
    "sec_fraud":            {"icon": "💰", "code": "SEC_FRAUD"},
}

ACTION_ICONS = {
    "query_source": "🔍", "trace_origin": "🔍", "cross_reference": "🔍",
    "request_context": "📄", "entity_link": "🔗", "temporal_audit": "⏱️",
    "network_cluster": "🕸️", "flag_manipulation": "🚩",
    "submit_verdict_real": "✅", "submit_verdict_misinfo": "❌",
    "submit_verdict_satire": "🎭", "submit_verdict_out_of_context": "✂️",
    "submit_verdict_fabricated": "⚠️",
}

# ─── CSS — Dynamic Animated Nebula Aesthetic ──────────────────────────────────
FORGE_CSS = """

:root {
    --bg-void: #020617;
    --card-bg: rgba(15, 23, 42, 0.45);
    --card-bg-hover: rgba(15, 23, 42, 0.65);
    --border-glass: rgba(255, 255, 255, 0.08);
    --border-glass-bright: rgba(255, 255, 255, 0.15);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --shadow-soft: 0 10px 40px rgba(0, 0, 0, 0.5);
    --font-ui: 'Inter', system-ui, -apple-system, sans-serif;
    --radius-lg: 20px;
}

/* 1. VIBRANT NEBULA BACKGROUND ANIMATION */
@keyframes nebulaFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

body {
    background: var(--bg-void) !important;
}

.gradio-container {
    background: linear-gradient(-45deg, #020617, #1e1b4b, #2e1065, #083344, #0f172a, #4a044e, #020617) !important;
    background-size: 400% 400% !important;
    animation: nebulaFlow 20s ease infinite !important;
    font-family: var(--font-ui) !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
}

/* 2. GLOWING GLASS CARDS ANIMATION */
@keyframes cardPulse {
    0% { box-shadow: 0 0 15px rgba(139, 92, 246, 0.1); }
    50% { box-shadow: 0 0 35px rgba(6, 182, 212, 0.2); }
    100% { box-shadow: 0 0 15px rgba(139, 92, 246, 0.1); }
}

.glass-panel {
    background: var(--card-bg) !important;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-soft);
    animation: cardPulse 8s infinite alternate ease-in-out;
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), background 0.4s ease;
}

.glass-panel:hover {
    background: var(--card-bg-hover) !important;
    border-color: rgba(236, 72, 153, 0.3) !important;
    transform: translateY(-4px);
}

.claim-text {
    font-size: clamp(22px, 2vw, 26px) !important;
    font-weight: 500;
    color: var(--text-primary);
    text-align: center;
    line-height: 1.5;
    padding: 30px 20px;
}

/* Elegant Text Gradient Highlight */
@keyframes textGlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.highlight-entity {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f472b6, #38bdf8);
    background-size: 200% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    font-weight: 700;
    animation: textGlow 3s linear infinite;
}

/* 3. SHINING LAUNCH BUTTON ANIMATION */
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

button.primary {
    background: linear-gradient(110deg, #6366f1 0%, #ec4899 40%, #c084fc 50%, #ec4899 60%, #6366f1 100%) !important;
    background-size: 200% auto !important;
    border: none !important;
    color: white !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4) !important;
    animation: shimmer 5s linear infinite !important;
    transition: transform 0.2s !important;
    padding: 14px 24px !important;
}

button.primary:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 10px 30px rgba(6, 182, 212, 0.6) !important;
}

/* 4. SLIDE UP FADE ACTION LOGS */
@keyframes slideUpFade {
    0% { opacity: 0; transform: translateY(20px) scale(0.95); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}

.log-entry {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #818cf8;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 10px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.log-time { 
    font-size: 11px;
    color: #38bdf8;
    font-weight: 600;
}

.log-text {
    font-size: 13px;
    color: var(--text-primary);
}

/* 5. FLOATING 3D CARDS ANIMATION */
.meta-grid {
    display: grid; 
    grid-template-columns: repeat(3, 1fr); 
    gap: 14px; 
    padding: 0 30px 30px 30px;
}
.meta-card {
    background: rgba(255,255,255,0.03);
    padding: 16px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid var(--border-glass);
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.meta-card:hover { 
    transform: translateY(-8px) scale(1.02); 
    background: rgba(255,255,255,0.08);
    border-color: #38bdf8;
    box-shadow: 0 10px 20px rgba(6, 182, 212, 0.2);
}
.meta-label { font-size: 10px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
.meta-value { font-size: 16px; color: #f8fafc; font-weight: 600; margin-top: 6px; }

/* 6. STATUS DOT ANIMATIONS */
@keyframes radar-pulse {
    0% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(6, 182, 212, 0); }
    100% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border-glass);
}
.status-online { border-color: rgba(6, 182, 212, 0.3); color: #22d3ee; }
.dot {
    width: 8px; height: 8px; border-radius: 50%; background: #06b6d4;
    animation: radar-pulse 2s infinite;
}

/* 7. SCANNER LINE ANIMATION for Center Panel */
@keyframes scanline {
    0% { top: 0%; opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

.scanner-container {
    position: relative;
    overflow: hidden;
    border-radius: var(--radius-lg);
}

.scanner-container::after {
    content: '';
    position: absolute;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.8), transparent);
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
    animation: scanline 4s linear infinite;
    pointer-events: none;
    z-index: 10;
}

/* Gradio 6.x Component Overrides — force dark theme on all form elements */
.gradio-container input,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .secondary-wrap,
.gradio-container ul.options li {
    background: rgba(15, 23, 42, 0.6) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
}

/* Force dark backgrounds on all Gradio block/panel wrappers */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .compact,
.gradio-container .contain,
.gradio-container .gap,
.gradio-container div[class*="row"],
.gradio-container div[class*="column"] {
    background: transparent !important;
    border-color: rgba(255, 255, 255, 0.06) !important;
}

/* Dropdown and slider wrapper panels */
.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-compact,
.gradio-container .gr-padded,
.gradio-container .gr-group,
.gradio-container [class*="block_"] {
    background: transparent !important;
}

/* Labels */
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container span[data-testid="block-label"],
.gradio-container .gradio-slider label,
.gradio-container .gradio-dropdown label {
    color: var(--text-secondary) !important;
    font-family: var(--font-ui) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Dropdown specific */
.gradio-container .gradio-dropdown,
.gradio-container [data-testid="dropdown"],
.gradio-container .multiselect-dropdown {
    background: rgba(15, 23, 42, 0.6) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Slider specific */
.gradio-container .gradio-slider input[type="range"] {
    accent-color: #818cf8 !important;
}
.gradio-container .gradio-slider input[type="number"] {
    background: rgba(15, 23, 42, 0.8) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Row variant=compact — Gradio 6 wraps this with a visible border+bg */
.gradio-container .row,
.gradio-container [class*="row_"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Status bar bottom section */
.statusbar-container {
    margin-top: 12px;
}

"""

# ─── HTML Builders ────────────────────────────────────────────────────────────

def _topnav_html():
    return f"""
    <div style="display:flex; align-items:center; justify-content:space-between; padding: 20px 30px; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.05);">
        <div style="display:flex; align-items:center; gap:16px;">
            <div style="width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg, #06b6d4, #ec4899); display:flex; align-items:center; justify-content:center; color:white; font-size:20px; box-shadow: 0 0 20px rgba(236, 72, 153, 0.4);">🛡️</div>
            <div>
                <div style="font-weight:700; font-size:20px; letter-spacing:-0.02em; color:white;">MisInfo Forensics</div>
                <div style="font-size:12px; color:#38bdf8; font-weight:500; letter-spacing:0.04em;">Vibrant Analysis Core</div>
            </div>
        </div>
        <div class="status-badge status-online">
            <div class="dot"></div> System Ready
        </div>
    </div>
    """

def _statusbar_html(node_status="OPTIMAL", packets="0.0K/S"):
    return f"""
    <div style="font-size:11px; color:#94a3b8; display:flex; justify-content:space-between; margin-top:12px; padding:8px; background:rgba(0,0,0,0.3); border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
        <span>NET_LINK: <span style="color:#06b6d4; font-weight:600;">{node_status}</span></span>
        <span>BANDWIDTH: <span style="color:#06b6d4;">{packets}</span></span>
    </div>
    """

def _left_panel_idle():
    return f"""
    <div style="padding:24px;">
        <div style="font-size:11px; color:var(--text-secondary); font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:20px;">Live Feed</div>
        <div style="text-align:center; padding:50px 20px; color:var(--text-secondary);">
            <div style="font-size:24px; margin-bottom:12px; opacity:0.5;">📡</div>
            <div style="font-size:13px; font-weight:500;">Awaiting Task Initialization</div>
        </div>
    </div>
    """

def _left_panel_active(log_entries):
    entries_html = ""
    for t, msg in log_entries:
        entries_html += f'<div class="log-entry"><span class="log-time">{t}</span><span class="log-text">{msg}</span></div>'
    return f"""
    <div style="padding:24px;">
        <div style="font-size:11px; color:#c084fc; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;">
            <span>Live Feed</span>
            <span style="font-size:20px; animation: textGlow 1.5s infinite alternate;">⚡</span>
        </div>
        {entries_html}
    </div>
    """

def _center_idle():
    return """
    <div class="scanner-container" style="height:350px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:var(--text-secondary);">
        <div style="font-size:50px; margin-bottom:24px; opacity:0.8; filter: drop-shadow(0 0 10px rgba(6, 182, 212, 0.4));">🔍</div>
        <div style="font-size:15px; font-weight:500; letter-spacing:0.02em;">Select forensic protocol below</div>
    </div>
    """

def _center_active(claim_text, task_id, virality, source_dom, status_str):
    highlighted = re.sub(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', r'<span class="highlight-entity">\1</span>', claim_text)
    return f"""
    <div class="scanner-container" style="padding: 24px; min-height:350px;">
        <div style="display:flex; justify-content:center; margin-bottom:20px;">
            <div class="status-badge" style="background:rgba(236,72,153,0.1); border-color:rgba(236,72,153,0.3); color:#fbcfe8;">
                Target • {task_id.replace('_', ' ').title()}
            </div>
        </div>
        <div class="claim-text">"{highlighted}"</div>
        <div class="meta-grid">
            <div class="meta-card"><div class="meta-label">Node Source</div><div class="meta-value">{source_dom.title()}</div></div>
            <div class="meta-card"><div class="meta-label">Viral Index</div><div class="meta-value">{virality*100:.1f}%</div></div>
            <div class="meta-card"><div class="meta-label">Status</div><div class="meta-value" style="color:#38bdf8;">{status_str}</div></div>
        </div>
    </div>
    """

def _right_panel_idle():
    return """
    <div style="padding:24px; height:100%; display:flex; align-items:center; justify-content:center; color:var(--text-secondary);">
        <div style="text-align:center;">
            <div style="font-size:35px; opacity:0.4; margin-bottom:12px; filter: drop-shadow(0 0 5px rgba(255,255,255,0.2));">🧠</div>
            <div style="font-size:13px; font-weight:500;">Brain Offline</div>
        </div>
    </div>
    """

def _right_panel_active(think, predict, fsm_state, step_num, coverage, contras):
    return f"""
    <div style="padding:24px;">
        <div style="font-size:11px; color:#f472b6; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:20px;">Agent Thought Stream</div>
        
        <div style="background:rgba(0,0,0,0.3); border:1px solid rgba(236, 72, 153, 0.2); border-radius:14px; padding:18px; margin-bottom:20px; box-shadow: inset 0 0 20px rgba(0,0,0,0.5);">
            <div style="font-size:13px; line-height:1.6; color:#e2e8f0; font-weight:400; font-style:italic;">
                "{think[:350]}..."
            </div>
        </div>

        <div style="display:flex; gap:10px; font-size:10px; font-weight:700; color:var(--text-secondary);">
            <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:10px; border-radius:10px; flex:1; text-align:center;">STEP<br><span style="font-size:16px; color:white; font-weight:600;">{step_num}</span></div>
            <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:10px; border-radius:10px; flex:1; text-align:center;">COV<br><span style="font-size:16px; color:#38bdf8; font-weight:600;">{coverage:.0%}</span></div>
            <div style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.3); padding:10px; border-radius:10px; flex:1; text-align:center; color:#c4b5fd;">STATE<br><span style="font-size:11px;">{fsm_state[:8]}</span></div>
        </div>
    </div>
    """

def _right_panel_done(verdict, true_label, correct, steps, reward, confidence):
    badge_bg = "rgba(16, 185, 129, 0.1)" if correct else "rgba(239, 68, 68, 0.1)"
    badge_border = "rgba(16, 185, 129, 0.4)" if correct else "rgba(239, 68, 68, 0.4)"
    badge_color = "#34d399" if correct else "#fca5a5"
    badge_text = "VERIFIED REAL" if correct else "MISINFO FLAGGED"
    icon = "✅" if correct else "🚨"
    return f"""
    <div style="padding:24px;">
        <div style="font-size:11px; color:#f8fafc; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:20px;">Resolution</div>
        
        <div style="text-align:center; padding:30px 20px; border-radius:18px; background:{badge_bg}; border: 2px solid {badge_border}; margin-bottom:20px; animation: slideUpFade 0.7s cubic-bezier(0.16, 1, 0.3, 1);">
            <div style="font-size:36px; margin-bottom:12px; filter:drop-shadow(0 0 10px {badge_color});">{icon}</div>
            <div style="font-size:22px; font-weight:700; color:{badge_color}; letter-spacing:0.02em;">{badge_text}</div>
            <div style="font-size:12px; font-weight:600; color:white; opacity:0.9; margin-top:10px; background:rgba(0,0,0,0.5); padding:4px 10px; border-radius:10px; display:inline-block;">Confidence {confidence:.1%}</div>
        </div>
        
        <div style="background:rgba(0,0,0,0.3); border-radius:14px; border:1px solid rgba(255,255,255,0.1); padding:16px;">
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:10px; margin-bottom:10px;">
                <span style="font-size:12px; color:var(--text-secondary); font-weight:500;">Ground Truth</span>
                <span style="font-size:13px; color:white; font-weight:600;">{true_label.title()}</span>
            </div>
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:10px; margin-bottom:10px;">
                <span style="font-size:12px; color:var(--text-secondary); font-weight:500;">Steps</span>
                <span style="font-size:13px; color:white; font-weight:600;">{steps}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:12px; color:var(--text-secondary); font-weight:500;">Reward</span>
                <span style="font-size:13px; color:#34d399; font-weight:700;">{reward:.2f}</span>
            </div>
        </div>
    </div>
    """

# ─── Investigation Logic ──────────────────────────────────────────────────────

def investigate(task_name, difficulty):
    try:
        yield from _investigate_inner(task_name, int(difficulty))
    except Exception as exc:
        import traceback
        # Log the full traceback server-side for debugging
        _app_logger.error("Investigation failed: %s\n%s", exc, traceback.format_exc())
        # Sanitized error panel — NO raw tracebacks in the frontend
        error_type = type(exc).__name__
        err_panel = f"""
        <div id="center-panel">
            <div style="
                background:rgba(255,45,107,0.06); border:1px solid rgba(255,45,107,0.2);
                border-radius:6px; padding:20px; font-family:'Share Tech Mono',monospace;
                font-size:11px; color:#ff2d6b; max-width:480px;
            ">
                <div style="letter-spacing:0.15em; margin-bottom:8px;">SYSTEM_ERROR</div>
                <div style="color:#94a3b8; font-size:12px; margin-top:8px;">
                    Error type: <b>{error_type}</b><br>
                    The investigation encountered an unexpected error.<br>
                    Details have been logged to <code>forge_debug.log</code>.
                </div>
            </div>
        </div>"""
        yield (_left_panel_idle(), err_panel, _right_panel_idle(),
               _statusbar_html("ERROR", "0.0K/S"), gr.update(interactive=True))

def _investigate_inner(task_name, difficulty):
    agent = LLMAgent()
    if task_name == "All Tasks (Random)":
        env = MisInfoForensicsEnv(difficulty=difficulty)
    else:
        env = MisInfoForensicsEnv(task_names=[task_name], difficulty=difficulty)

    ep_seed = int(time.time()) % 100000
    obs, info = env.reset(seed=ep_seed)
    if hasattr(agent, "reset"):
        agent.reset()

    claim_text = env.graph.root.text if env.graph else ""
    task_id = info.get("task_id", "unknown")
    source_dom = env.graph.root.domain if env.graph else "unknown"
    virality = env.graph.root.virality_score if env.graph else 0.5

    log_entries = []
    start_time = time.time()

    def _ts():
        elapsed = time.time() - start_time
        return f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

    log_entries.append((_ts(), f'Task init: <b style="color:white;">{task_id.replace("_", " ").title()}</b>'))
    
    yield (
        _left_panel_active(log_entries),
        _center_active(claim_text, task_id, virality, source_dom, "Waking API..."),
        _right_panel_idle(),
        _statusbar_html("ACTIVE", f"{random.uniform(0.8,2.0):.1f}K/S"),
        gr.update(interactive=False),
    )

    done = False
    step_info = {}
    final_reward = 0.0
    INVESTIGATION_TIMEOUT = 180  # 3 minute hard ceiling for the entire investigation

    while not done:
        # ── Timeout guard: prevent infinite hangs from stuck API calls ─────
        if time.time() - start_time > INVESTIGATION_TIMEOUT:
            _app_logger.warning("Investigation timed out after %ds", INVESTIGATION_TIMEOUT)
            log_entries.append((_ts(), '⏱️ <b style="color:#fca5a5;">Investigation Timeout</b>'))
            break

        context = {
            "steps": env.steps,
            "max_steps": env.max_steps,
            "coverage": env.graph.evidence_coverage if env.graph else 0.0,
            "contradictions": env.graph.contradiction_surface_area if env.graph else 0,
            "last_tool_result": step_info.get("tool_result"),
            "claim_text": claim_text,
            "task_name": task_id,
            "true_label_hint": None,
        }
        action = agent.act(obs, context=context)
        action_name = ACTIONS[action]
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        final_reward = reward

        coverage = env.graph.evidence_coverage if env.graph else 0.0
        contras = env.graph.contradiction_surface_area if env.graph else 0
        last = agent.reasoning_log[-1] if agent.reasoning_log else {}
        think = last.get("think", "Reviewing entity correlations...")
        predict = last.get("predict", "")

        icon = ACTION_ICONS.get(action_name, "⚡")
        formatted_action = action_name.replace("_", " ").title()
        log_entries.append((_ts(), f'{icon} <b>{formatted_action}</b> <span style="color:#34d399; float:right; font-size:12px; font-weight:700;">{reward:+.2f}</span>'))
        visible_log = log_entries[-10:]
        fsm_state = getattr(agent, "_fsm_state", "Compute")

        yield (
            _left_panel_active(visible_log),
            _center_active(claim_text, task_id, virality, source_dom, "Analyzing Elements..."),
            _right_panel_active(think, predict, fsm_state, env.steps, coverage, contras),
            _statusbar_html("ACTIVE", f"{random.uniform(1.0,3.5):.1f}K/S"),
            gr.update(interactive=False),
        )

    true_label = env.graph.true_label if env.graph else "unknown"
    verdict = step_info.get("verdict")
    correct = (verdict == true_label)
    confidence = env._estimate_confidence() if hasattr(env, "_estimate_confidence") else 0.85

    yield (
        _left_panel_active(log_entries[-12:]),
        _center_active(claim_text, task_id, virality, source_dom, "VERIFIED" if correct else "FLAGGED"),
        _right_panel_done(verdict, true_label, correct, env.steps, final_reward, confidence),
        _statusbar_html("OPTIMAL" if correct else "ALERT", f"{random.uniform(0.5,1.5):.1f}K/S"),
        gr.update(interactive=True),
    )

# ─── Gradio App ───────────────────────────────────────────────────────────────

NEBULA_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="fuchsia",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
    block_border_width="0px",
    input_background_fill="rgba(0,0,0,0.3)",
    input_border_color="rgba(255,255,255,0.15)",
    input_border_width="1px",
    input_radius="10px",
    background_fill_primary="transparent",
)

with gr.Blocks(title="Forensics AI // Vibrant Nebula") as demo:
    gr.HTML(_topnav_html())

    with gr.Row(elem_classes=["main-container"]):
        # Left panel: Logs
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            left_panel = gr.HTML(value=_left_panel_idle())

        # Center panel: Investigation
        with gr.Column(scale=6):
            with gr.Column(elem_classes=["glass-panel"]):
                center_panel = gr.HTML(value=_center_idle())
                
                with gr.Row(variant="compact"):
                    with gr.Column(scale=3):
                        task_dd = gr.Dropdown(
                            choices=["All Tasks (Random)"] + list(TASK_REGISTRY.keys()),
                            value="All Tasks (Random)",
                            label="Investigation Protocol",
                        )
                    with gr.Column(scale=1):
                        diff_sl = gr.Slider(
                            minimum=1, maximum=4, step=1, value=1,
                            label="Depth Level",
                        )
                
                with gr.Row():
                    start_btn = gr.Button("Launch Deep Analysis", variant="primary")
                
                statusbar = gr.HTML(value=_statusbar_html("IDLE", "0.0K/S"))

        # Right panel: Reasoning
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            right_panel = gr.HTML(value=_right_panel_idle())

    # Event Wiring
    start_btn.click(
        fn=investigate,
        inputs=[task_dd, diff_sl],
        outputs=[left_panel, center_panel, right_panel, statusbar, start_btn],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, theme=NEBULA_THEME, css=FORGE_CSS)