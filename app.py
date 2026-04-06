"""
FORGE — MisInfo Forensics Investigation AI
Crystalline Gradio UI with premium glassmorphism design and micro-animations.
"""

import gradio as gr
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from env.misinfo_env import MisInfoForensicsEnv, ACTIONS, VERDICT_ACTIONS
from agents.llm_agent import LLMAgent
from env.tasks import TASK_REGISTRY
import logging

logging.getLogger("env").setLevel(logging.CRITICAL)
logging.getLogger("agents").setLevel(logging.CRITICAL)

# ─── Task metadata for the UI ────────────────────────────────────────────────
TASK_META = {
    "fabricated_stats":       {"icon": "📊", "color": "#ef4444", "desc": "Fabricated statistics attributed to institutions"},
    "out_of_context":         {"icon": "🔀", "color": "#f97316", "desc": "Media stripped of original context"},
    "coordinated_campaign":   {"icon": "🤖", "color": "#a855f7", "desc": "Bot-amplified coordinated campaigns"},
    "satire_news":            {"icon": "🎭", "color": "#06b6d4", "desc": "Satire misinterpreted as real news"},
    "verified_fact":          {"icon": "✅", "color": "#22c55e", "desc": "Legitimate verified factual claims"},
    "politifact_liar":        {"icon": "🏛️", "color": "#3b82f6", "desc": "Real Politifact claims from LIAR dataset"},
    "image_forensics":        {"icon": "🖼️", "color": "#ec4899", "desc": "AI-generated or manipulated images"},
    "sec_fraud":              {"icon": "💰", "color": "#eab308", "desc": "Securities fraud & market manipulation"},
}

# ─── Action icons ─────────────────────────────────────────────────────────────
ACTION_ICONS = {
    "query_source": "🔍", "trace_origin": "🕵️", "cross_reference": "🔗",
    "request_context": "📄", "entity_link": "🏷️", "temporal_audit": "⏰",
    "network_cluster": "🕸️", "flag_manipulation": "🚩",
    "submit_verdict_real": "✅", "submit_verdict_misinfo": "⚠️",
    "submit_verdict_satire": "🎭", "submit_verdict_out_of_context": "🔀",
    "submit_verdict_fabricated": "❌",
}

VERDICT_EMOJI = {
    "real": "✅", "misinfo": "⚠️", "satire": "🎭",
    "out_of_context": "🔀", "fabricated": "❌",
}

# ─── Crystalline CSS ─────────────────────────────────────────────────────────
CRYSTALLINE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & Base ── */
:root {
    --crystal-bg: #06070d;
    --crystal-surface: rgba(15, 18, 35, 0.85);
    --crystal-glass: rgba(30, 35, 65, 0.45);
    --crystal-border: rgba(100, 120, 255, 0.12);
    --crystal-glow: rgba(99, 102, 241, 0.15);
    --crystal-accent: #818cf8;
    --crystal-accent-bright: #a5b4fc;
    --crystal-text: #e2e8f0;
    --crystal-text-dim: #94a3b8;
    --crystal-success: #34d399;
    --crystal-danger: #f87171;
    --crystal-warn: #fbbf24;
    --crystal-info: #60a5fa;
    --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-glass: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.04) 100%);
    --gradient-aurora: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--crystal-bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--crystal-text) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Animated aurora background ── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(99,102,241,0.06), transparent),
        radial-gradient(ellipse 60% 60% at 80% 20%, rgba(168,85,247,0.05), transparent),
        radial-gradient(ellipse 50% 70% at 50% 80%, rgba(56,189,248,0.04), transparent);
    pointer-events: none;
    z-index: 0;
    animation: auroraShift 20s ease-in-out infinite alternate;
}

@keyframes auroraShift {
    0%   { opacity: 0.6; transform: scale(1.0); }
    50%  { opacity: 1.0; transform: scale(1.05); }
    100% { opacity: 0.7; transform: scale(0.98); }
}

/* ── Floating crystal particles ── */
.gradio-container::after {
    content: '';
    position: fixed;
    width: 300px; height: 300px;
    top: 10%; right: 5%;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    border-radius: 50%;
    animation: floatParticle 15s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes floatParticle {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%  { transform: translate(-40px, 30px) scale(1.1); }
    66%  { transform: translate(20px, -20px) scale(0.95); }
}

/* ── Header hero section ── */
#forge-hero {
    position: relative;
    text-align: center;
    padding: 48px 24px 36px;
    margin-bottom: 8px;
    overflow: hidden;
    z-index: 1;
}

#forge-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg,
        rgba(99,102,241,0.06) 0%,
        transparent 60%,
        rgba(168,85,247,0.04) 100%);
    border-bottom: 1px solid var(--crystal-border);
    z-index: -1;
}

/* ── Glass panels ── */
.panel, .glass-panel {
    background: var(--crystal-glass) !important;
    backdrop-filter: blur(20px) saturate(1.3) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.3) !important;
    border: 1px solid var(--crystal-border) !important;
    border-radius: 16px !important;
    box-shadow:
        0 4px 24px rgba(0,0,0,0.2),
        inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.panel:hover, .glass-panel:hover {
    border-color: rgba(100, 120, 255, 0.25) !important;
    box-shadow:
        0 8px 32px rgba(99,102,241,0.08),
        0 4px 24px rgba(0,0,0,0.2),
        inset 0 1px 0 rgba(255,255,255,0.06) !important;
}

/* ── Gradio block overrides ── */
.block, .gr-group, .gr-box, .gr-form, .gr-panel, fieldset, .form {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
}

/* ── Buttons ── */
button.primary, button.lg {
    background: var(--gradient-primary) !important;
    border: 1px solid rgba(129, 140, 248, 0.3) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 32px !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.25) !important;
    letter-spacing: 0.02em !important;
    z-index: 1 !important;
}

button.primary:hover, button.lg:hover {
    transform: translateY(-2px) !important;
    box-shadow:
        0 8px 25px rgba(99,102,241,0.35),
        0 0 40px rgba(99,102,241,0.1) !important;
    border-color: rgba(165, 180, 252, 0.5) !important;
}

button.primary:active, button.lg:active {
    transform: translateY(0) !important;
}

/* Button shimmer effect */
button.primary::after, button.lg::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: linear-gradient(
        45deg,
        transparent 30%,
        rgba(255,255,255,0.08) 50%,
        transparent 70%
    );
    animation: shimmer 3s ease-in-out infinite;
    z-index: -1;
}

@keyframes shimmer {
    0%   { transform: translateX(-100%) rotate(45deg); }
    100% { transform: translateX(100%) rotate(45deg); }
}

/* ── Dropdowns & Sliders ── */
.gr-dropdown, select, .gr-input, input, textarea, .single-select, input[type="number"] {
    background: rgba(20, 24, 45, 0.8) !important;
    background-color: rgba(20, 24, 45, 0.8) !important;
    border: 1px solid rgba(130, 140, 240, 0.3) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.3s ease !important;
}

/* Ensure inner items in dropdowns are styled correctly */
.gr-dropdown *, select *, input::placeholder, textarea::placeholder {
    color: #f1f5f9 !important;
    background-color: transparent !important;
}

/* Dropdown option menu background fix */
ul.options, .gr-dropdown .options {
    background: rgba(20, 24, 45, 0.95) !important;
}
ul.options *, .gr-dropdown .options * {
    color: #f1f5f9 !important;
}

.gr-dropdown:focus, select:focus, .gr-input:focus, input:focus, textarea:focus {
    border-color: var(--crystal-accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    outline: none !important;
}

/* Labels and Helper Text */
label, .gr-label, label span, .gr-form-label, legend {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

span.text-gray-500, p.text-gray-500, [data-testid="block-info"], .gr-block-info {
    color: #94a3b8 !important;
    font-weight: 400 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}

/* Slider track */
input[type="range"] {
    accent-color: var(--crystal-accent) !important;
}

/* ── Investigation logs (textbox) ── */
textarea {
    background: rgba(8, 10, 22, 0.8) !important;
    border: 1px solid var(--crystal-border) !important;
    border-radius: 12px !important;
    color: var(--crystal-text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    transition: border-color 0.3s ease !important;
}

textarea:focus {
    border-color: rgba(99,102,241,0.3) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.05) !important;
}

/* ── Stats cards ── */
.stat-card {
    background: var(--crystal-glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--crystal-border);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-primary);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.stat-card:hover {
    border-color: rgba(99,102,241,0.25);
    transform: translateY(-2px);
}

.stat-card:hover::before { opacity: 1; }

.stat-number {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #c4b5fd, #e9d5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    filter: brightness(1.3);
}

.stat-label {
    font-size: 12px;
    color: var(--crystal-text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Task selector cards ── */
.task-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 12px;
    padding: 8px 0;
}

.task-card {
    background: var(--crystal-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--crystal-border);
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.task-card:hover {
    border-color: rgba(99,102,241,0.35);
    transform: translateY(-3px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.15);
}

.task-card .task-icon {
    font-size: 28px;
    margin-bottom: 6px;
    display: block;
}

.task-card .task-name {
    font-size: 12px;
    font-weight: 600;
    color: var(--crystal-text);
    letter-spacing: 0.02em;
}

/* ── Result verdict badges ── */
.verdict-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    animation: badgePop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@keyframes badgePop {
    0%   { transform: scale(0.5); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

.verdict-correct {
    background: rgba(52, 211, 153, 0.15);
    color: var(--crystal-success);
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.verdict-wrong {
    background: rgba(248, 113, 113, 0.15);
    color: var(--crystal-danger);
    border: 1px solid rgba(248, 113, 113, 0.3);
}

/* ── Step timeline ── */
.step-entry {
    padding: 8px 0;
    border-left: 2px solid var(--crystal-border);
    padding-left: 16px;
    margin-left: 8px;
    position: relative;
    animation: stepFadeIn 0.3s ease-out;
}

.step-entry::before {
    content: '';
    position: absolute;
    left: -5px; top: 14px;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--crystal-accent);
    box-shadow: 0 0 8px rgba(99,102,241,0.4);
}

@keyframes stepFadeIn {
    0%   { opacity: 0; transform: translateX(-10px); }
    100% { opacity: 1; transform: translateX(0); }
}

/* ── Progress bar ── */
.progress-bar-container {
    background: rgba(15, 18, 35, 0.6);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    border: 1px solid var(--crystal-border);
}

.progress-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: var(--gradient-primary);
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.progress-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    animation: progressShimmer 2s infinite;
}

@keyframes progressShimmer {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99,102,241,0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.35); }

/* ── Markdown overrides ── */
.prose h1, .prose h2, .prose h3, h1, h2, h3 {
    color: var(--crystal-text) !important;
}

.prose p, p { color: var(--crystal-text-dim) !important; }
.prose a, a { color: var(--crystal-accent-bright) !important; }

/* ── Accordion overrides ── */
.gr-accordion {
    background: var(--crystal-glass) !important;
    border: 1px solid var(--crystal-border) !important;
    border-radius: 12px !important;
}

/* ── HTML component backgrounds ── */
.gr-html, .prose {
    background: transparent !important;
}

/* ── Pulse animation for live status ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.pulse { animation: pulse 2s ease-in-out infinite; }

/* ── Fade-in for sections ── */
@keyframes fadeInUp {
    0%   { opacity: 0; transform: translateY(16px); }
    100% { opacity: 1; transform: translateY(0); }
}

.fade-in { animation: fadeInUp 0.5s ease-out; }

/* ── Glow ring animation for active investigation ── */
@keyframes glowRing {
    0%   { box-shadow: 0 0 5px rgba(99,102,241,0.2); }
    50%  { box-shadow: 0 0 20px rgba(99,102,241,0.3), 0 0 40px rgba(99,102,241,0.1); }
    100% { box-shadow: 0 0 5px rgba(99,102,241,0.2); }
}

.glow-active {
    animation: glowRing 2s ease-in-out infinite;
}
"""


# ── HTML Builders ─────────────────────────────────────────────────────────────

def build_hero_html():
    return """
    <div id="forge-hero">
        <div style="font-size: 48px; margin-bottom: 8px; filter: drop-shadow(0 0 20px rgba(99,102,241,0.3));">🔬</div>
        <h1 style="
            font-size: 36px; font-weight: 900; margin: 0 0 6px;
            background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; letter-spacing: -0.02em;
        ">FORGE</h1>
        <p style="
            font-size: 15px; color: #94a3b8; margin: 0 0 4px;
            font-weight: 400; letter-spacing: 0.06em; text-transform: uppercase;
        ">Forensic RL Graph Environment</p>
        <p style="
            font-size: 13px; color: #64748b; margin: 0;
            max-width: 520px; margin: 4px auto 0;
            line-height: 1.6;
        ">AI-powered misinformation investigation using reinforcement learning,
        graph reasoning, and LLM chain-of-thought analysis</p>

        <div style="display: flex; justify-content: center; gap: 24px; margin-top: 20px;">
            <div class="stat-card" style="min-width: 110px;">
                <div class="stat-number">{n_tasks}</div>
                <div class="stat-label">Task Types</div>
            </div>
            <div class="stat-card" style="min-width: 110px;">
                <div class="stat-number">13</div>
                <div class="stat-label">Actions</div>
            </div>
            <div class="stat-card" style="min-width: 110px;">
                <div class="stat-number">5</div>
                <div class="stat-label">Verdicts</div>
            </div>
            <div class="stat-card" style="min-width: 110px;">
                <div class="stat-number">∞</div>
                <div class="stat-label">Scenarios</div>
            </div>
        </div>
    </div>
    """.format(n_tasks=len(TASK_REGISTRY))


def build_task_cards_html():
    cards = ""
    for tid, meta in TASK_META.items():
        if tid in TASK_REGISTRY:
            nice_name = tid.replace("_", " ").title()
            cards += f"""
            <div class="task-card" title="{meta['desc']}">
                <span class="task-icon">{meta['icon']}</span>
                <span class="task-name">{nice_name}</span>
            </div>"""
    return f'<div class="task-grid">{cards}</div>'


def build_idle_status():
    return """
    <div style="
        text-align: center; padding: 60px 24px;
        color: #64748b; font-size: 14px;
    ">
        <div style="font-size: 56px; margin-bottom: 16px; opacity: 0.4;">🔍</div>
        <p style="margin: 0; font-weight: 500; color: #94a3b8;">Ready to investigate</p>
        <p style="margin: 4px 0 0; font-size: 13px;">Select a task and press <strong style="color: #818cf8;">Start Investigation</strong></p>
    </div>
    """


def build_step_html(step_num, action_name, max_steps, coverage, contradictions):
    icon = ACTION_ICONS.get(action_name, "🔧")
    pct = min(100, int((step_num / max(max_steps, 1)) * 100))
    is_verdict = action_name.startswith("submit_verdict")
    nice_action = action_name.replace("_", " ").title()

    step_class = "step-entry"
    action_style = f"color: {'#34d399' if is_verdict else '#818cf8'}; font-weight: 600;"

    return f"""
    <div class="{step_class}">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="font-size: 16px;">{icon}</span>
            <span style="{action_style}; font-size: 14px;">{nice_action}</span>
            <span style="margin-left: auto; font-size: 11px; color: #64748b;">Step {step_num}/{max_steps}</span>
        </div>
        <div class="progress-bar-container" style="margin-top: 4px;">
            <div class="progress-bar-fill" style="width: {pct}%;"></div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 6px; font-size: 11px; color: #94a3b8;">
            <span>Coverage: {coverage:.0%}</span>
            <span>Contradictions: {contradictions}</span>
        </div>
    </div>
    """


def build_result_html(verdict, true_label, correct, steps_used, max_steps, reward):
    verdict_icon = VERDICT_EMOJI.get(verdict, "❓")
    true_icon = VERDICT_EMOJI.get(true_label, "❓")
    badge_class = "verdict-correct" if correct else "verdict-wrong"
    badge_text = "CORRECT" if correct else "INCORRECT"
    result_emoji = "🎉" if correct else "💔"

    return f"""
    <div style="
        padding: 28px; text-align: center;
        animation: fadeInUp 0.5s ease-out;
    ">
        <div style="font-size: 48px; margin-bottom: 12px;">{result_emoji}</div>

        <span class="verdict-badge {badge_class}">{badge_text}</span>

        <div style="
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 16px; margin-top: 24px; max-width: 400px; margin-left: auto; margin-right: auto;
        ">
            <div class="stat-card">
                <div style="font-size: 24px; margin-bottom: 4px;">{verdict_icon}</div>
                <div style="font-size: 14px; font-weight: 700; color: #e2e8f0;">{(verdict or 'N/A').replace('_',' ').title()}</div>
                <div class="stat-label">Predicted</div>
            </div>
            <div class="stat-card">
                <div style="font-size: 24px; margin-bottom: 4px;">{true_icon}</div>
                <div style="font-size: 14px; font-weight: 700; color: #e2e8f0;">{true_label.replace('_',' ').title()}</div>
                <div class="stat-label">True Label</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="font-size: 22px;">{steps_used}/{max_steps}</div>
                <div class="stat-label">Steps Used</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="font-size: 22px;">{reward:.3f}</div>
                <div class="stat-label">Reward</div>
            </div>
        </div>
    </div>
    """


def build_claim_panel(claim_text, task_id, difficulty):
    meta = TASK_META.get(task_id, {"icon": "🔬", "color": "#818cf8", "desc": ""})
    nice_task = task_id.replace("_", " ").title()
    diff_dots = "●" * difficulty + "○" * (4 - difficulty)
    diff_color = ["#34d399", "#fbbf24", "#f97316", "#ef4444"][difficulty - 1]

    return f"""
    <div style="
        background: rgba(15, 18, 35, 0.6);
        border: 1px solid rgba(100, 120, 255, 0.12);
        border-radius: 14px; padding: 20px;
        margin-bottom: 12px;
        animation: fadeInUp 0.4s ease-out;
    ">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 22px;">{meta['icon']}</span>
            <span style="font-weight: 700; color: #e2e8f0; font-size: 15px;">{nice_task}</span>
            <span style="
                margin-left: auto; font-size: 13px; letter-spacing: 0.15em;
                color: {diff_color}; font-weight: 600;
            ">{diff_dots}</span>
        </div>
        <div style="
            font-style: italic; color: #cbd5e1;
            font-size: 14px; line-height: 1.6;
            padding: 12px 16px;
            background: rgba(8, 10, 22, 0.5);
            border-radius: 10px;
            border-left: 3px solid {meta['color']};
        ">"{claim_text}"</div>
    </div>
    """


def build_thinking_html(think_text, predict_text):
    if not think_text:
        return ""
    return f"""
    <div style="
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 10px; padding: 12px 16px;
        margin-top: 8px; font-size: 12px; line-height: 1.6;
    ">
        <div style="color: #a5b4fc; font-weight: 600; margin-bottom: 4px;">💭 Agent Reasoning</div>
        <div style="color: #94a3b8;">{think_text[:300]}</div>
        {f'<div style="color: #64748b; margin-top: 4px; font-style: italic;">Prediction: {predict_text[:150]}</div>' if predict_text else ''}
    </div>
    """


# ── Investigation Logic ──────────────────────────────────────────────────────

def investigate(task_name, difficulty):
    agent = LLMAgent()

    if task_name == "All Tasks (Random)":
        env = MisInfoForensicsEnv(difficulty=int(difficulty))
    else:
        env = MisInfoForensicsEnv(task_names=[task_name], difficulty=int(difficulty))
    ep_seed = int(time.time()) % 100000
    obs, info = env.reset(seed=ep_seed)

    if hasattr(agent, "reset"):
        agent.reset()

    claim_text = env.graph.root.text if env.graph else ""
    task_id = info.get("task_id", "unknown")

    # Build initial state
    claim_html = build_claim_panel(claim_text, task_id, int(difficulty))
    steps_html = ""
    thinking_html = ""
    result_html = """
        <div style="text-align: center; padding: 40px; color: #64748b;">
            <div style="font-size: 32px; margin-bottom: 8px;" class="pulse">🔬</div>
            <p style="font-size: 13px; margin: 0;">Investigation in progress…</p>
        </div>
    """

    # Status indicator
    status_html = f"""
        <div style="
            display: flex; align-items: center; gap: 8px;
            padding: 8px 16px; font-size: 13px;
            background: rgba(99,102,241,0.08);
            border: 1px solid rgba(99,102,241,0.15);
            border-radius: 8px;
        ">
            <div style="
                width: 8px; height: 8px; border-radius: 50%;
                background: #818cf8;
            " class="pulse"></div>
            <span style="color: #a5b4fc; font-weight: 500;">Investigating {task_id.replace('_',' ').title()}…</span>
        </div>
    """

    yield claim_html, steps_html + thinking_html, result_html, status_html

    done = False
    verdict = None
    step_info = {}
    final_reward = 0.0

    while not done:
        context = {
            "steps": env.steps,
            "max_steps": env.max_steps,
            "coverage": env.graph.evidence_coverage if env.graph else 0.0,
            "contradictions": env.graph.contradiction_surface_area if env.graph else 0,
            "last_tool_result": step_info.get("tool_result"),
            "claim_text": env.graph.root.text if env.graph else "",
        }
        action = agent.act(obs, context=context)
        action_name = ACTIONS[action]
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        final_reward = reward

        if step_info.get("verdict"):
            verdict = step_info["verdict"]

        # Build step visualization
        coverage = env.graph.evidence_coverage if env.graph else 0.0
        contras = env.graph.contradiction_surface_area if env.graph else 0
        steps_html += build_step_html(env.steps, action_name, env.max_steps, coverage, contras)

        # Agent reasoning
        last_thought = agent.reasoning_log[-1] if agent.reasoning_log else {}
        thinking_html = build_thinking_html(
            last_thought.get("think", ""),
            last_thought.get("predict", ""),
        )

        yield claim_html, steps_html + thinking_html, result_html, status_html

    # Final result
    true_label = env.graph.true_label if env.graph else "unknown"
    correct = (verdict == true_label)

    result_html = build_result_html(
        verdict, true_label, correct,
        env.steps, env.max_steps, final_reward,
    )

    status_color = "#34d399" if correct else "#f87171"
    status_text = "Verdict Correct! ✅" if correct else "Verdict Incorrect ❌"
    status_html = f"""
        <div style="
            display: flex; align-items: center; gap: 8px;
            padding: 8px 16px; font-size: 13px;
            background: rgba({'52,211,153' if correct else '248,113,113'},0.08);
            border: 1px solid rgba({'52,211,153' if correct else '248,113,113'},0.2);
            border-radius: 8px;
        ">
            <div style="
                width: 8px; height: 8px; border-radius: 50%;
                background: {status_color};
            "></div>
            <span style="color: {status_color}; font-weight: 600;">{status_text}</span>
            <span style="margin-left: auto; color: #94a3b8; font-size: 12px;">{env.steps} steps · {final_reward:.3f} reward</span>
        </div>
    """

    yield claim_html, steps_html + thinking_html, result_html, status_html


# ── Gradio App ────────────────────────────────────────────────────────────────

CRYSTALLINE_THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="*neutral_950",
    block_background_fill="transparent",
    block_border_width="0px",
    panel_background_fill="transparent",
    input_background_fill="*neutral_900",
)

with gr.Blocks(
    title="FORGE — MisInfo Forensics Investigation AI",
    theme=CRYSTALLINE_THEME,
    css=CRYSTALLINE_CSS,
) as demo:

    # ── Hero header ──
    gr.HTML(build_hero_html())

    # ── Task gallery ──
    with gr.Accordion("📋 Available Task Scenarios", open=False):
        gr.HTML(build_task_cards_html())

    # ── Control panel ──
    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            task_choices = ["All Tasks (Random)"] + list(TASK_REGISTRY.keys())
            task_dropdown = gr.Dropdown(
                choices=task_choices,
                value="All Tasks (Random)",
                label="🔬 Investigation Scenario",
                info="Choose a specific misinformation type or random selection",
            )
        with gr.Column(scale=1):
            difficulty_slider = gr.Slider(
                minimum=1, maximum=4, step=1, value=1,
                label="⚡ Difficulty Level",
                info="Higher = more tactics, noisy evidence",
            )
        with gr.Column(scale=1, min_width=180):
            start_btn = gr.Button(
                "🚀 Start Investigation",
                variant="primary",
                size="lg",
            )

    # ── Status bar ──
    status_display = gr.HTML(
        value="""<div style="
            display: flex; align-items: center; gap: 8px;
            padding: 8px 16px; font-size: 13px;
            background: rgba(30, 35, 65, 0.45);
            border: 1px solid rgba(100, 120, 255, 0.12);
            border-radius: 8px;
        ">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #64748b;"></div>
            <span style="color: #94a3b8;">Idle — Ready for investigation</span>
        </div>"""
    )

    # ── Main content ──
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            # Claim under investigation
            claim_display = gr.HTML(value=build_idle_status())

            # Investigation timeline
            with gr.Accordion("🕵️ Investigation Timeline", open=True):
                timeline_display = gr.HTML(value="")

        with gr.Column(scale=2):
            # Result panel
            gr.HTML("""
                <div style="
                    font-size: 12px; text-transform: uppercase;
                    letter-spacing: 0.08em; color: #64748b;
                    font-weight: 600; margin-bottom: 8px; padding-left: 4px;
                ">📊 Investigation Result</div>
            """)
            result_display = gr.HTML(
                value="""
                <div style="
                    text-align: center; padding: 48px 24px;
                    color: #475569; font-size: 13px;
                ">
                    <div style="font-size: 40px; margin-bottom: 12px; opacity: 0.3;">📊</div>
                    <p style="margin: 0;">Results will appear here</p>
                </div>
                """
            )

    # ── Footer ──
    gr.HTML("""
        <div style="
            text-align: center; padding: 24px 0 16px;
            border-top: 1px solid rgba(100, 120, 255, 0.08);
            margin-top: 24px;
        ">
            <p style="font-size: 12px; color: #475569; margin: 0;">
                FORGE v1.0 · 100% Free-Tier · PyTorch + Gymnasium + FastAPI
            </p>
            <p style="font-size: 11px; color: #334155; margin: 4px 0 0;">
                Powered by Groq (LLaMA 3), Sentence-Transformers, and Open-Source APIs
            </p>
        </div>
    """)

    # ── Wire events ──
    start_btn.click(
        fn=investigate,
        inputs=[task_dropdown, difficulty_slider],
        outputs=[claim_display, timeline_display, result_display, status_display],
    )


if __name__ == "__main__":
    demo.launch()
