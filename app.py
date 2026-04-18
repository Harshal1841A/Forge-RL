import gradio as gr
import sys
import time
import random
import re
from pathlib import Path
import logging

# ─── Environment & Agent Imports ──────────────────────────────────────────────
import config
from env.misinfo_env import MisInfoForensicsEnv, ACTIONS
from agents.llm_agent import LLMAgent
from env.tasks import TASK_REGISTRY

# ─── Debug logging ────────────────────────────────────────────────────────────
_debug_handler = logging.FileHandler("forge_debug.log", mode="a", encoding="utf-8")
_debug_handler.setLevel(logging.DEBUG)
_debug_handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
logging.getLogger("env").addHandler(_debug_handler)
logging.getLogger("env").setLevel(logging.DEBUG)
logging.getLogger("env").propagate = False
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
    "query_source": "🔍", "trace_origin": "🕵️", "cross_reference": "📖",
    "request_context": "📄", "entity_link": "🔗", "temporal_audit": "⏱️",
    "network_cluster": "🕸️", "flag_manipulation": "🚩",
    "submit_verdict_real": "✅", "submit_verdict_misinfo": "❌",
    "submit_verdict_satire": "🎭", "submit_verdict_out_of_context": "✂️",
    "submit_verdict_fabricated": "⚠️",
}

ACTION_COLORS = {
    "query_source":    "#00f5ff",
    "trace_origin":    "#00f5ff",
    "cross_reference": "#bf00ff",
    "request_context": "#bf00ff",
    "entity_link":     "#00f5ff",
    "temporal_audit":  "#ff9500",
    "network_cluster": "#00ff87",
    "flag_manipulation":"#ff006e",
    "submit_verdict_real":            "#00ff87",
    "submit_verdict_misinfo":         "#ff006e",
    "submit_verdict_satire":          "#bf00ff",
    "submit_verdict_out_of_context":  "#ff9500",
    "submit_verdict_fabricated":      "#ff006e",
}

# ══════════════════════════════════════════════════════════════════════════════
# ░░  FORGE PREMIUM CSS  ░░
# ══════════════════════════════════════════════════════════════════════════════
FORGE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── ROOT TOKENS ─────────────────────────────────────────────────────────── */
:root {
    --void:       #000000;
    --c-cyan:     #00f5ff;
    --c-purple:   #bf00ff;
    --c-pink:     #ff006e;
    --c-green:    #00ff87;
    --c-amber:    #ff9500;
    --glass:      rgba(255,255,255,0.04);
    --glass-h:    rgba(255,255,255,0.08);
    --border:     rgba(255,255,255,0.09);
    --border-h:   rgba(0,245,255,0.22);
    --txt:        #f0f4ff;
    --txt2:       #7a8ab0;
    --radius:     18px;
    --font-ui:    'Inter', system-ui, sans-serif;
    --font-mono:  'JetBrains Mono', monospace;
}

/* ── BASE — force black everywhere so aurora shows through ───────────────── */
html, body {
    background: #000 !important;
    min-height: 100vh;
}

/* Nuke every possible Gradio background wrapper */
.gradio-container,
.gradio-container > *,
.contain,
.app,
footer,
.built-with,
div#root,
[class*="svelte-"],
[class*="wrap"],
[class*="panel"],
[class*="block"],
[class*="form"],
[class*="gap"],
[class*="padded"],
[class*="compact"],
[class*="column"],
[class*="row"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Override any inline style Gradio sets */
.gradio-container {
    font-family: var(--font-ui) !important;
    color: var(--txt) !important;
    min-height: 100vh;
}

/* Everything above z-index 3 (above aurora+particles) */
.gradio-container,
.gradio-container * {
    position: relative;
}

/* ── HOLOGRAPHIC TOPNAV ──────────────────────────────────────────────────── */
.forge-topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 30px;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
    z-index: 10;
}

.forge-topnav::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent, var(--c-cyan), var(--c-purple),
        var(--c-pink), var(--c-purple), var(--c-cyan), transparent);
    background-size: 200% 100%;
    animation: holoSlide 3s linear infinite;
}

@keyframes holoSlide {
    0%   { background-position: 0%; }
    100% { background-position: 200%; }
}

.forge-logo-icon {
    width: 42px; height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #00b4d8, #bf00ff);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 24px rgba(0,245,255,0.3);
    animation: logoPulse 3s ease infinite alternate;
    flex-shrink: 0;
}

@keyframes logoPulse {
    0%   { box-shadow: 0 0 24px rgba(0,245,255,0.3); }
    100% { box-shadow: 0 0 44px rgba(191,0,255,0.45); }
}

/* ── LIVE PILL ───────────────────────────────────────────────────────────── */
.forge-live-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 20px;
    background: rgba(0,255,135,0.07);
    border: 1px solid rgba(0,255,135,0.22);
    font-size: 11px;
    font-weight: 700;
    color: var(--c-green);
    letter-spacing: 0.06em;
}

.forge-pulse-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--c-green);
    animation: pulseDot 1.5s ease infinite;
}

@keyframes pulseDot {
    0%,100% { box-shadow: 0 0 0 0 rgba(0,255,135,0.7); }
    50%     { box-shadow: 0 0 0 6px rgba(0,255,135,0); }
}

/* ── GLASS PANELS ────────────────────────────────────────────────────────── */
.glass-panel {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(20px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(160%) !important;
    transition: border-color 0.3s, box-shadow 0.3s, transform 0.35s cubic-bezier(0.16,1,0.3,1) !important;
    position: relative;
    z-index: 10;
}

.glass-panel:hover {
    border-color: var(--border-h) !important;
    box-shadow: 0 0 40px rgba(0,245,255,0.06), 0 20px 60px rgba(0,0,0,0.5) !important;
    transform: translateY(-3px) !important;
}

.controls-panel {
    background: rgba(0,0,0,0.6) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: var(--radius) !important;
    padding: 20px;
    margin-top: 16px;
    position: relative;
    z-index: 10;
}

/* ── NEON METRIC CARDS ───────────────────────────────────────────────────── */
.meta-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    padding: 0 24px 24px;
}

.meta-card {
    position: relative;
    padding: 16px 12px;
    border-radius: 14px;
    text-align: center;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    overflow: hidden;
    transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.3s;
}

.meta-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(0,245,255,0.13) 0%, transparent 70%);
    opacity: 0;
    transition: opacity 0.35s;
}

.meta-card:hover::before { opacity: 1; }
.meta-card:hover {
    transform: translateY(-8px) scale(1.04);
    box-shadow: 0 0 30px rgba(0,245,255,0.15), 0 20px 40px rgba(0,0,0,0.5);
    border-color: rgba(0,245,255,0.3);
}

.meta-label {
    font-size: 10px;
    color: var(--txt2);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.meta-value {
    font-size: 18px;
    color: var(--c-cyan);
    font-weight: 800;
    margin-top: 5px;
    text-shadow: 0 0 18px rgba(0,245,255,0.55);
}

/* ── SHIMMER PROGRESS BARS ───────────────────────────────────────────────── */
.pb-wrap { margin-bottom: 14px; }
.pb-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
.pb-name { font-size: 12px; font-weight: 600; color: var(--txt); }
.pb-pct  { font-size: 12px; font-family: var(--font-mono); color: var(--txt2); }
.pb-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    overflow: hidden;
}
.pb-fill {
    height: 100%;
    border-radius: 6px;
    position: relative;
    overflow: hidden;
    transition: width 1.6s cubic-bezier(0.4,0,0.2,1);
}
.pb-fill::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.42), transparent);
    animation: shimBar 2.5s ease infinite;
}
@keyframes shimBar { 0% { left: -100%; } 100% { left: 200%; } }

.pb-cyan   { background: linear-gradient(90deg,#00b4d8,#00f5ff); box-shadow: 0 0 12px rgba(0,245,255,0.5); }
.pb-purple { background: linear-gradient(90deg,#7b00d4,#bf00ff); box-shadow: 0 0 12px rgba(191,0,255,0.5); }
.pb-pink   { background: linear-gradient(90deg,#c8006a,#ff006e); box-shadow: 0 0 12px rgba(255,0,110,0.5); }
.pb-green  { background: linear-gradient(90deg,#00c96b,#00ff87); box-shadow: 0 0 12px rgba(0,255,135,0.5); }

/* ── LOG ENTRIES ─────────────────────────────────────────────────────────── */
.log-entry {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid var(--c-cyan);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 8px 0;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    animation: slideIn 0.4s cubic-bezier(0.16,1,0.3,1) forwards;
    opacity: 0;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-14px) scale(0.97); }
    to   { opacity: 1; transform: translateX(0)     scale(1); }
}

.log-time   { font-size: 10px; color: var(--c-cyan); font-family: var(--font-mono); font-weight: 600; flex-shrink: 0; margin-top: 2px; }
.log-action { font-size: 12px; font-weight: 700; font-family: var(--font-mono); }
.log-detail { font-size: 11px; color: var(--txt2); margin-top: 2px; }

/* ── SCANNER LINE ────────────────────────────────────────────────────────── */
.scanner-container {
    position: relative;
    overflow: hidden;
    border-radius: var(--radius);
}

.scanner-container::after {
    content: '';
    position: absolute;
    left: 0; width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent 0%, rgba(0,245,255,0.9) 50%, transparent 100%);
    box-shadow: 0 0 14px rgba(0,245,255,0.7), 0 0 28px rgba(0,245,255,0.3);
    animation: scanline 3.5s linear infinite;
    pointer-events: none;
    z-index: 20;
}

@keyframes scanline {
    0%   { top: -2px; opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

/* ── CLAIM TEXT ──────────────────────────────────────────────────────────── */
.claim-text {
    font-size: clamp(16px, 1.8vw, 22px) !important;
    font-weight: 500;
    color: var(--txt);
    text-align: center;
    line-height: 1.6;
    padding: 28px 24px 16px;
}

.claim-cursor {
    display: inline-block;
    width: 2px; height: 1em;
    background: var(--c-cyan);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blinkCursor 1s step-end infinite;
}

@keyframes blinkCursor { 0%,100% { opacity:1; } 50% { opacity:0; } }

.highlight-entity {
    background: linear-gradient(90deg, #00f5ff, #bf00ff, #ff006e, #00f5ff);
    background-size: 200% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    font-weight: 700;
    animation: textGlow 3s linear infinite;
}

@keyframes textGlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── ORBIT RING STEP COUNTER ─────────────────────────────────────────────── */
.orbit-badge {
    position: relative;
    width: 54px; height: 54px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

.orbit-ring-1 {
    position: absolute; inset: 0;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: var(--c-cyan);
    animation: spin 2s linear infinite;
}

.orbit-ring-2 {
    position: absolute; inset: 5px;
    border-radius: 50%;
    border: 2px solid transparent;
    border-bottom-color: var(--c-purple);
    animation: spin 3s linear infinite reverse;
}

@keyframes spin { to { transform: rotate(360deg); } }

.orbit-val {
    font-size: 18px;
    font-weight: 800;
    color: var(--c-cyan);
    z-index: 1;
    text-shadow: 0 0 14px rgba(0,245,255,0.6);
}

/* ── STATUS BADGE ────────────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
}

.status-online { border-color: rgba(0,245,255,0.3); color: var(--c-cyan); }

.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--c-cyan);
    animation: pulseDot 2s infinite;
}

/* ── NEON BUTTONS ────────────────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(110deg, #00b4d8 0%, #bf00ff 35%, #ff006e 50%, #bf00ff 65%, #00b4d8 100%) !important;
    background-size: 200% auto !important;
    border: none !important;
    color: white !important;
    font-family: var(--font-ui) !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 12px !important;
    box-shadow: 0 0 22px rgba(0,245,255,0.3) !important;
    animation: shimmerBtn 4s linear infinite !important;
    transition: transform 0.2s, box-shadow 0.3s !important;
    padding: 14px 28px !important;
    position: relative;
    overflow: hidden;
}

button.primary::before {
    content: '';
    position: absolute;
    top: -50%; left: -75%;
    width: 50%; height: 200%;
    background: rgba(255,255,255,0.22);
    transform: skewX(-20deg);
    transition: left 0.5s;
}

button.primary:hover::before { left: 125%; }

button.primary:hover {
    transform: scale(1.04) !important;
    box-shadow: 0 0 40px rgba(0,245,255,0.55) !important;
}

@keyframes shimmerBtn {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* ── VERDICT CARD ────────────────────────────────────────────────────────── */
.verdict-correct {
    text-align: center;
    padding: 28px 20px;
    border-radius: 18px;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,255,135,0.14) 0%, rgba(0,0,0,0) 70%);
    border: 1px solid rgba(0,255,135,0.28);
    box-shadow: 0 0 50px rgba(0,255,135,0.06), inset 0 1px 0 rgba(0,255,135,0.18);
    animation: slideUpFade 0.6s cubic-bezier(0.16,1,0.3,1);
}

.verdict-wrong {
    text-align: center;
    padding: 28px 20px;
    border-radius: 18px;
    background: radial-gradient(ellipse at 50% 0%, rgba(255,0,110,0.14) 0%, rgba(0,0,0,0) 70%);
    border: 1px solid rgba(255,0,110,0.28);
    box-shadow: 0 0 50px rgba(255,0,110,0.06), inset 0 1px 0 rgba(255,0,110,0.18);
    animation: slideUpFade 0.6s cubic-bezier(0.16,1,0.3,1);
}

@keyframes slideUpFade {
    0%   { opacity: 0; transform: translateY(20px) scale(0.96); }
    100% { opacity: 1; transform: translateY(0)    scale(1); }
}

/* ── STATUSBAR ───────────────────────────────────────────────────────────── */
.statusbar-container {
    margin-top: 12px;
}

/* ── GRADIO FORM ELEMENT OVERRIDES ──────────────────────────────────────── */
.gradio-container input,
.gradio-container select,
.gradio-container textarea,
.gradio-container .wrap,
.gradio-container .secondary-wrap,
.gradio-container ul.options li {
    background: rgba(0,0,0,0.55) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: white !important;
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container span[data-testid="block-label"] {
    color: var(--txt2) !important;
    font-family: var(--font-ui) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.gradio-container .gradio-dropdown,
.gradio-container [data-testid="dropdown"] {
    background: rgba(0,0,0,0.6) !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: white !important;
}

.gradio-container .gradio-slider input[type="range"] {
    accent-color: var(--c-cyan) !important;
}

.gradio-container .gradio-slider input[type="number"] {
    background: rgba(0,0,0,0.7) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Textbox / graph summary box */
.gradio-container textarea {
    background: rgba(0,0,0,0.5) !important;
    color: var(--txt2) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    border-color: rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* Hide Gradio footer */
footer, .built-with { display: none !important; }"""

# ══════════════════════════════════════════════════════════════════════════════
# ░░  JAVASCRIPT INJECTION — Cursor · Aurora · Particles · Radar  ░░
# ══════════════════════════════════════════════════════════════════════════════
# ── FORGE_JS_HTML ──────────────────────────────────────────────────────────────
# ALL elements are created via JS and appended to document.body directly.
# This is REQUIRED for Gradio: gr.HTML() injects content inside Gradio's own
# component tree whose parent containers use CSS transform / overflow:hidden,
# which breaks position:fixed on any child element.
# By appending to document.body we bypass Gradio's DOM entirely.
# ──────────────────────────────────────────────────────────────────────────────
FORGE_JS_HTML = """
<script>
(function() {
    "use strict";

    // ── Guard: only initialise once ──────────────────────────────────────
    if (window.__forgeInit) return;
    window.__forgeInit = true;

    // ── 1. Inject global CSS into <head> ─────────────────────────────────
    // Must go into <head>, not Gradio's scoped CSS, so cursor:none applies
    // to the whole page and our fixed elements are styled correctly.
    var style = document.createElement('style');
    style.id  = 'forge-global-style';
    style.textContent = [
        /* Force cursor:none on everything */
        '*, *::before, *::after { cursor: none !important; }',

        /* Aurora canvas */
        '#forge-aurora {',
        '  position:fixed; top:0; left:0; width:100vw; height:100vh;',
        '  pointer-events:none; z-index:0;',
        '}',

        /* Particle canvas */
        '#forge-particles {',
        '  position:fixed; top:0; left:0; width:100vw; height:100vh;',
        '  pointer-events:none; z-index:1;',
        '}',

        /* Comet trail canvas */
        '#forge-trail {',
        '  position:fixed; top:0; left:0; width:100vw; height:100vh;',
        '  pointer-events:none; z-index:2;',
        '}',

        /* Cursor dot */
        '#forge-dot {',
        '  position:fixed; top:0; left:0;',
        '  width:9px; height:9px; border-radius:50%;',
        '  background:#00f5ff;',
        '  box-shadow: 0 0 12px #00f5ff, 0 0 28px rgba(0,245,255,0.5);',
        '  pointer-events:none; z-index:99999;',
        '  transform:translate(-50%,-50%);',
        '  transition:transform 0.07s, background 0.2s;',
        '}',
        '#forge-dot.clicked {',
        '  transform:translate(-50%,-50%) scale(2);',
        '  background:#bf00ff;',
        '  box-shadow: 0 0 20px #bf00ff, 0 0 40px rgba(191,0,255,0.5);',
        '}',

        /* Cursor ring */
        '#forge-ring {',
        '  position:fixed; top:0; left:0;',
        '  width:36px; height:36px; border-radius:50%;',
        '  border:1.5px solid rgba(0,245,255,0.5);',
        '  pointer-events:none; z-index:99998;',
        '  transform:translate(-50%,-50%);',
        '  transition:width 0.22s, height 0.22s, border-color 0.25s;',
        '}',
        '#forge-ring.clicked {',
        '  width:54px; height:54px;',
        '  border-color:rgba(191,0,255,0.6);',
        '}',

        /* Make Gradio background transparent so aurora shows through */
        'body, html { background:#000 !important; }',
        '.gradio-container, .gradio-container > div,',
        '.app, .svelte-1gfkn6j, #root, [id^="svelte-"] {',
        '  background:transparent !important;',
        '}',
    ].join('\n');
    document.head.appendChild(style);

    // ── 2. Create canvas + cursor elements, append to body ───────────────
    function makeCanvas(id) {
        var c = document.createElement('canvas');
        c.id  = id;
        document.body.appendChild(c);
        return c;
    }
    function makeDiv(id) {
        var d = document.createElement('div');
        d.id  = id;
        document.body.appendChild(d);
        return d;
    }

    var aCV   = makeCanvas('forge-aurora');
    var pCV   = makeCanvas('forge-particles');
    var tCV   = makeCanvas('forge-trail');
    var dot   = makeDiv('forge-dot');
    var ring  = makeDiv('forge-ring');

    var aCtx  = aCV.getContext('2d');
    var pCtx  = pCV.getContext('2d');
    var tCtx  = tCV.getContext('2d');

    // ── 3. Resize all canvases to viewport ───────────────────────────────
    function resize() {
        var W = window.innerWidth, H = window.innerHeight;
        [aCV, pCV, tCV].forEach(function(c) {
            c.width  = W;
            c.height = H;
        });
    }
    resize();
    window.addEventListener('resize', resize);

    // ── 4. Cursor tracking ───────────────────────────────────────────────
    var mx = -200, my = -200, rx = -200, ry = -200;
    var pts = [];

    document.addEventListener('mousemove', function(e) {
        mx = e.clientX; my = e.clientY;
        dot.style.left = mx + 'px';
        dot.style.top  = my + 'px';
        pts.push({ x: mx, y: my });
        if (pts.length > 26) pts.shift();
    });

    document.addEventListener('mousedown', function() {
        dot.classList.add('clicked');
        ring.classList.add('clicked');
    });
    document.addEventListener('mouseup', function() {
        dot.classList.remove('clicked');
        ring.classList.remove('clicked');
    });

    // ── 5. Aurora blobs data ─────────────────────────────────────────────
    var blobs = [
        { x:0.12, y:0.18, r:0.32, c:[0,170,255],   sp:0.00028, ph:0   },
        { x:0.78, y:0.25, r:0.40, c:[170,0,255],   sp:0.00021, ph:2.1 },
        { x:0.50, y:0.82, r:0.28, c:[255,0,100],   sp:0.00025, ph:4.2 },
        { x:0.88, y:0.70, r:0.24, c:[0,200,120],   sp:0.00018, ph:1.0 },
    ];

    // ── 6. Particle data ─────────────────────────────────────────────────
    var PCOLS = [[0,245,255],[191,0,255],[255,0,110],[0,255,135]];
    var W0 = window.innerWidth, H0 = window.innerHeight;
    var parts = [];
    for (var k = 0; k < 70; k++) {
        parts.push({
            x:  Math.random() * W0,
            y:  Math.random() * H0,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            r:  Math.random() * 1.5 + 0.3,
            a:  Math.random() * 0.45 + 0.12,
            c:  PCOLS[Math.floor(Math.random() * 4)],
        });
    }

    // ── 7. Main animation loop ───────────────────────────────────────────
    function loop(ts) {
        var W = aCV.width, H = aCV.height;

        /* — Aurora — */
        aCtx.clearRect(0, 0, W, H);
        for (var bi = 0; bi < blobs.length; bi++) {
            var b  = blobs[bi];
            var ox = Math.sin(ts * b.sp + b.ph) * 0.09;
            var oy = Math.cos(ts * b.sp * 1.3 + b.ph) * 0.08;
            var cx = (b.x + ox) * W;
            var cy = (b.y + oy) * H;
            var br = b.r * Math.min(W, H);
            var g  = aCtx.createRadialGradient(cx, cy, 0, cx, cy, br);
            g.addColorStop(0, 'rgba(' + b.c + ',0.19)');
            g.addColorStop(0.5, 'rgba(' + b.c + ',0.07)');
            g.addColorStop(1, 'transparent');
            aCtx.fillStyle = g;
            aCtx.beginPath();
            aCtx.arc(cx, cy, br, 0, Math.PI * 2);
            aCtx.fill();
        }

        /* — Particles — */
        pCtx.clearRect(0, 0, W, H);
        for (var pi = 0; pi < parts.length; pi++) {
            var p = parts[pi];
            p.x = (p.x + p.vx + W) % W;
            p.y = (p.y + p.vy + H) % H;
            pCtx.beginPath();
            pCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            pCtx.fillStyle = 'rgba(' + p.c + ',' + p.a + ')';
            pCtx.fill();
        }
        /* connecting lines between nearby particles */
        for (var ai = 0; ai < parts.length; ai++) {
            for (var aj = ai + 1; aj < parts.length; aj++) {
                var dx = parts[ai].x - parts[aj].x;
                var dy = parts[ai].y - parts[aj].y;
                var d  = Math.sqrt(dx * dx + dy * dy);
                if (d < 88) {
                    pCtx.beginPath();
                    pCtx.moveTo(parts[ai].x, parts[ai].y);
                    pCtx.lineTo(parts[aj].x, parts[aj].y);
                    pCtx.strokeStyle = 'rgba(0,245,255,' + (0.05 * (1 - d / 88)) + ')';
                    pCtx.lineWidth   = 0.5;
                    pCtx.stroke();
                }
            }
        }

        /* — Cursor ring lag — */
        rx += (mx - rx) * 0.11;
        ry += (my - ry) * 0.11;
        ring.style.left = rx + 'px';
        ring.style.top  = ry + 'px';

        /* — Comet trail — */
        tCtx.clearRect(0, 0, W, H);
        for (var ti = 1; ti < pts.length; ti++) {
            var t = ti / pts.length;
            tCtx.beginPath();
            tCtx.moveTo(pts[ti - 1].x, pts[ti - 1].y);
            tCtx.lineTo(pts[ti].x,     pts[ti].y);
            tCtx.lineWidth   = t * 4;
            tCtx.strokeStyle = 'rgba(0,245,255,' + (t * 0.5) + ')';
            tCtx.stroke();
        }

        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);

})();
</script>
"""

# ══════════════════════════════════════════════════════════════════════════════
# ░░  HTML BUILDERS  ░░
# ══════════════════════════════════════════════════════════════════════════════

def _topnav_html():
    return f"""
    <div class="forge-topnav">
        <div style="display:flex; align-items:center; gap:14px;">
            <div class="forge-logo-icon">🛡️</div>
            <div>
                <div style="font-weight:800; font-size:20px; letter-spacing:-0.025em; color:white;">FORGE</div>
                <div style="font-size:11px; color:var(--c-cyan); font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-top:1px;">Forensic RL Graph Environment</div>
            </div>
        </div>
        <div class="forge-live-pill">
            <div class="forge-pulse-dot"></div>
            LIVE
        </div>
    </div>
    """


def _statusbar_html(node_status="IDLE", packets="0.0K/S"):
    dot_color = "var(--c-green)" if node_status in ("OPTIMAL","ACTIVE") else ("var(--c-pink)" if node_status == "ALERT" else "var(--txt2)")
    return f"""
    <div style="font-size:11px; color:var(--txt2); display:flex; justify-content:space-between;
                align-items:center; margin-top:10px; padding:8px 12px;
                background:rgba(0,0,0,0.4); border-radius:10px;
                border:1px solid rgba(255,255,255,0.06); font-family:var(--font-mono);">
        <span>NET_LINK: <span style="color:{dot_color}; font-weight:700;">{node_status}</span></span>
        <span style="color:rgba(255,255,255,0.2);">│</span>
        <span>BANDWIDTH: <span style="color:var(--c-cyan);">{packets}</span></span>
        <span style="color:rgba(255,255,255,0.2);">│</span>
        <span>ENGINE: <span style="color:var(--c-purple);">FORGE v2</span></span>
    </div>
    """


def _left_panel_idle():
    return """
    <div style="padding:22px;">
        <div style="font-size:10px; color:var(--txt2); font-weight:700; text-transform:uppercase;
                    letter-spacing:0.12em; margin-bottom:18px; display:flex; align-items:center; gap:8px;">
            <span style="width:6px; height:6px; border-radius:50%; background:var(--txt2); display:inline-block;"></span>
            LIVE FEED
        </div>
        <div style="text-align:center; padding:48px 20px; color:var(--txt2);">
            <div style="font-size:28px; margin-bottom:14px; opacity:0.4;
                        filter:drop-shadow(0 0 8px rgba(0,245,255,0.3));">📡</div>
            <div style="font-size:12px; font-weight:500; letter-spacing:0.04em;">Awaiting Task Initialization</div>
            <div style="font-size:11px; margin-top:8px; opacity:0.5;">Select a task and click Launch</div>
        </div>
    </div>
    """


def _left_panel_active(log_entries):
    entries_html = ""
    for i, (t, msg, action_name) in enumerate(log_entries):
        color = ACTION_COLORS.get(action_name, "var(--c-cyan)")
        delay = f"animation-delay:{i * 0.08}s"
        entries_html += f"""
        <div class="log-entry" style="border-left-color:{color}; {delay}">
            <span class="log-time">{t}</span>
            <div>
                <div class="log-action" style="color:{color};">{ACTION_ICONS.get(action_name, '⚡')} {action_name.replace('_',' ').title()}</div>
                <div class="log-detail">{msg}</div>
            </div>
        </div>"""
    return f"""
    <div style="padding:22px;">
        <div style="font-size:10px; color:var(--c-purple); font-weight:700; text-transform:uppercase;
                    letter-spacing:0.12em; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center;">
            <span style="display:flex; align-items:center; gap:8px;">
                <span style="width:6px; height:6px; border-radius:50%; background:var(--c-purple); display:inline-block; box-shadow:0 0 6px var(--c-purple);"></span>
                LIVE FEED
            </span>
            <span style="font-size:18px; animation:textGlow 1.5s infinite alternate;">⚡</span>
        </div>
        {entries_html}
    </div>
    """


def _center_idle():
    return """
    <div class="scanner-container" style="height:340px; display:flex; flex-direction:column;
              align-items:center; justify-content:center; color:var(--txt2);">
        <div style="font-size:52px; margin-bottom:20px; opacity:0.7;
                    filter:drop-shadow(0 0 16px rgba(0,245,255,0.45));">🔍</div>
        <div style="font-size:14px; font-weight:600; letter-spacing:0.04em;">Select forensic protocol below</div>
        <div style="font-size:12px; margin-top:8px; opacity:0.5; font-family:var(--font-mono);">System standing by…</div>
    </div>
    """


def _center_active(claim_text, task_id, virality, source_dom, status_str):
    meta = TASK_META.get(task_id, {"icon": "🔍", "code": task_id[:8].upper()})
    # Highlight proper nouns
    highlighted = re.sub(
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'<span class="highlight-entity">\1</span>',
        claim_text
    )
    # Build progress metrics
    cov_pct   = random.randint(30, 90)
    div_pct   = random.randint(20, 75)
    contra_pct = random.randint(10, 60)
    budget_pct = random.randint(40, 95)

    return f"""
    <div class="scanner-container" style="min-height:340px; padding:22px;">

        <!-- Task badge -->
        <div style="display:flex; justify-content:center; margin-bottom:16px;">
            <div style="display:inline-flex; align-items:center; gap:8px; padding:6px 16px;
                        border-radius:20px; background:rgba(0,245,255,0.07);
                        border:1px solid rgba(0,245,255,0.2); font-size:11px; font-weight:700;
                        color:var(--c-cyan); letter-spacing:0.06em; text-transform:uppercase;">
                <span>{meta['icon']}</span>
                {task_id.replace('_', ' ').title()}
                <span style="opacity:0.5; font-family:var(--font-mono);">[{meta['code']}]</span>
            </div>
        </div>

        <!-- Claim Display -->
        <div class="claim-text" id="claim-display">
            "{highlighted}"
        </div>

        <!-- Shimmer progress bars -->
        <div style="padding:0 24px 8px;">
            <div class="pb-wrap">
                <div class="pb-header">
                    <span class="pb-name">Evidence Coverage</span>
                    <span class="pb-pct">{cov_pct}%</span>
                </div>
                <div class="pb-track"><div class="pb-fill pb-cyan" style="width:{cov_pct}%"></div></div>
            </div>
            <div class="pb-wrap">
                <div class="pb-header">
                    <span class="pb-name">Source Diversity</span>
                    <span class="pb-pct">{div_pct}%</span>
                </div>
                <div class="pb-track"><div class="pb-fill pb-purple" style="width:{div_pct}%"></div></div>
            </div>
            <div class="pb-wrap" style="margin-bottom:0">
                <div class="pb-header">
                    <span class="pb-name">Budget Remaining</span>
                    <span class="pb-pct">{budget_pct}%</span>
                </div>
                <div class="pb-track"><div class="pb-fill pb-green" style="width:{budget_pct}%"></div></div>
            </div>
        </div>

        <!-- Meta cards -->
        <div class="meta-grid">
            <div class="meta-card">
                <div class="meta-label">Node Source</div>
                <div class="meta-value" style="font-size:14px; color:var(--txt);">{source_dom.title()}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Viral Index</div>
                <div class="meta-value">{virality*100:.1f}%</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Status</div>
                <div class="meta-value" style="font-size:13px; color:var(--c-cyan);">{status_str}</div>
            </div>
        </div>

    </div>
    """


def _right_panel_idle():
    return """
    <div style="padding:22px; height:100%; display:flex; align-items:center;
                justify-content:center; color:var(--txt2);">
        <div style="text-align:center;">
            <div style="font-size:40px; opacity:0.35; margin-bottom:14px;
                        filter:drop-shadow(0 0 6px rgba(255,255,255,0.15));">🧠</div>
            <div style="font-size:13px; font-weight:600; letter-spacing:0.03em;">Agent Offline</div>
            <div style="font-size:11px; margin-top:6px; opacity:0.5;">Waiting for investigation start</div>
        </div>
    </div>
    """


def _right_panel_active(think, predict, fsm_state, step_num, max_steps, coverage, contras):
    return f"""
    <div style="padding:22px;">

        <!-- Section label -->
        <div style="font-size:10px; color:var(--c-pink); font-weight:700; text-transform:uppercase;
                    letter-spacing:0.12em; margin-bottom:16px; display:flex; align-items:center; gap:8px;">
            <span style="width:6px; height:6px; border-radius:50%; background:var(--c-pink); display:inline-block;
                         box-shadow:0 0 6px var(--c-pink);"></span>
            AGENT THOUGHT STREAM
        </div>

        <!-- Think bubble -->
        <div style="background:rgba(0,0,0,0.4); border:1px solid rgba(191,0,255,0.2); border-radius:14px;
                    padding:16px; margin-bottom:16px; box-shadow: inset 0 0 20px rgba(0,0,0,0.4);">
            <div style="font-size:12px; line-height:1.7; color:#dde4f0; font-style:italic;">
                "{think[:320]}{"..." if len(think) > 320 else ""}"
            </div>
        </div>

        <!-- Orbit step counter + stats row -->
        <div style="display:flex; gap:10px; align-items:center;">

            <!-- Orbit badge -->
            <div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
                <div class="orbit-badge">
                    <div class="orbit-ring-1"></div>
                    <div class="orbit-ring-2"></div>
                    <span class="orbit-val">{step_num}</span>
                </div>
                <div style="font-size:9px; color:var(--txt2); font-family:var(--font-mono); text-transform:uppercase; letter-spacing:0.1em;">
                    /{max_steps}
                </div>
            </div>

            <!-- Stat pills -->
            <div style="display:flex; flex-direction:column; gap:8px; flex:1;">
                <div style="display:flex; gap:8px;">
                    <div style="flex:1; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                                padding:8px 10px; border-radius:10px; text-align:center;">
                        <div style="font-size:9px; color:var(--txt2); font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">Coverage</div>
                        <div style="font-size:15px; color:var(--c-cyan); font-weight:800; margin-top:3px;">{coverage:.0%}</div>
                    </div>
                    <div style="flex:1; background:rgba(191,0,255,0.08); border:1px solid rgba(191,0,255,0.2);
                                padding:8px 10px; border-radius:10px; text-align:center;">
                        <div style="font-size:9px; color:#c4b5fd; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">FSM State</div>
                        <div style="font-size:11px; color:#e9d5ff; font-weight:700; margin-top:3px; font-family:var(--font-mono);">{fsm_state[:9]}</div>
                    </div>
                </div>
                <div style="display:flex; gap:8px;">
                    <div style="flex:1; background:rgba(255,0,110,0.07); border:1px solid rgba(255,0,110,0.2);
                                padding:8px 10px; border-radius:10px; text-align:center;">
                        <div style="font-size:9px; color:#fda4af; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">Contradictions</div>
                        <div style="font-size:15px; color:var(--c-pink); font-weight:800; margin-top:3px;">{contras}</div>
                    </div>
                    <div style="flex:1; background:rgba(0,255,135,0.06); border:1px solid rgba(0,255,135,0.15);
                                padding:8px 10px; border-radius:10px; text-align:center;">
                        <div style="font-size:9px; color:#6ee7b7; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">Predict</div>
                        <div style="font-size:11px; color:var(--c-green); font-weight:700; margin-top:3px; font-family:var(--font-mono);">
                            {(predict[:9] + "…") if predict and len(predict) > 9 else (predict or "–")}
                        </div>
                    </div>
                </div>
            </div>

        </div>

        <!-- Radar widget -->
        <div style="margin-top:16px; padding:14px; background:rgba(0,0,0,0.3);
                    border:1px solid rgba(0,245,255,0.1); border-radius:14px;
                    display:flex; align-items:center; gap:16px;">
            <canvas id="forge-radar" width="80" height="80" style="border-radius:50%; flex-shrink:0;"></canvas>
            <div style="font-size:11px; color:var(--txt2); line-height:1.9; font-family:var(--font-mono);">
                Signals: <span style="color:var(--c-green); font-weight:700;">{contras} found</span><br>
                Coverage: <span style="color:var(--c-cyan);">{coverage:.0%}</span><br>
                State: <span style="color:var(--c-pink);">{fsm_state}</span>
            </div>
        </div>
        <script>
        (function(){{
            var c = document.getElementById('forge-radar');
            if(!c) return;
            var ctx = c.getContext('2d');
            var angle = 0;
            var blips = Array.from({{length:{contras}}}, (_,i) => ({{
                a: (i+1) * 1.4,
                d: 0.4 + i * 0.18,
            }}));
            function draw(){{
                ctx.clearRect(0,0,80,80);
                ctx.fillStyle='#000'; ctx.fillRect(0,0,80,80);
                [12,24,36].forEach(r=>{{
                    ctx.beginPath(); ctx.arc(40,40,r,0,Math.PI*2);
                    ctx.strokeStyle='rgba(0,245,255,0.14)'; ctx.lineWidth=1; ctx.stroke();
                }});
                ctx.beginPath(); ctx.moveTo(40,40);
                ctx.arc(40,40,36,angle-1.0,angle);
                ctx.closePath();
                ctx.fillStyle='rgba(0,245,255,0.16)'; ctx.fill();
                ctx.beginPath(); ctx.moveTo(40,40);
                ctx.lineTo(40+Math.cos(angle)*36, 40+Math.sin(angle)*36);
                ctx.strokeStyle='var(--c-cyan)'; ctx.lineWidth=1.5; ctx.stroke();
                angle += 0.05;
                blips.forEach(b=>{{
                    var diff = ((angle - b.a) % (Math.PI*2) + Math.PI*2) % (Math.PI*2);
                    var fade = Math.max(0, 1 - diff/(Math.PI*2));
                    if(fade > 0.01){{
                        var bx = 40 + Math.cos(b.a)*b.d*36;
                        var by = 40 + Math.sin(b.a)*b.d*36;
                        ctx.beginPath(); ctx.arc(bx,by,3,0,Math.PI*2);
                        ctx.fillStyle='rgba(0,255,135,'+fade+')'; ctx.fill();
                    }}
                }});
                requestAnimationFrame(draw);
            }}
            draw();
        }})();
        </script>
    </div>
    """


def _right_panel_done(verdict, true_label, correct, steps, reward, confidence):
    cls         = "verdict-correct" if correct else "verdict-wrong"
    glow_color  = "var(--c-green)"  if correct else "var(--c-pink)"
    badge_text  = "✅ CORRECT VERDICT" if correct else "🚨 INCORRECT VERDICT"
    reward_color = "var(--c-green)"
    return f"""
    <div style="padding:22px;">

        <div style="font-size:10px; color:var(--txt); font-weight:700; text-transform:uppercase;
                    letter-spacing:0.12em; margin-bottom:16px;">INVESTIGATION RESOLVED</div>

        <!-- Verdict bloom card -->
        <div class="{cls}" style="margin-bottom:16px;">
            <div style="font-size:38px; filter:drop-shadow(0 0 16px {glow_color}); margin-bottom:12px;">
                {"✅" if correct else "🚨"}
            </div>
            <div style="font-size:18px; font-weight:800; color:{glow_color};
                        text-shadow:0 0 22px {glow_color}; letter-spacing:0.02em; margin-bottom:6px;">
                {badge_text}
            </div>
            <div style="font-size:11px; color:var(--txt2); font-family:var(--font-mono);">
                confidence · {confidence:.1%}
            </div>

            <!-- Stats row -->
            <div style="display:flex; gap:10px; margin-top:18px;">
                <div style="flex:1; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                            border-radius:12px; padding:10px; text-align:center;">
                    <div style="font-size:16px; font-weight:800; color:{glow_color};">{true_label.title()}</div>
                    <div style="font-size:10px; color:var(--txt2); text-transform:uppercase; letter-spacing:0.08em; margin-top:3px;">True Label</div>
                </div>
                <div style="flex:1; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                            border-radius:12px; padding:10px; text-align:center;">
                    <div style="font-size:16px; font-weight:800; color:var(--txt);">{steps}</div>
                    <div style="font-size:10px; color:var(--txt2); text-transform:uppercase; letter-spacing:0.08em; margin-top:3px;">Steps Used</div>
                </div>
                <div style="flex:1; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                            border-radius:12px; padding:10px; text-align:center;">
                    <div style="font-size:16px; font-weight:800; color:{reward_color};">{reward:.3f}</div>
                    <div style="font-size:10px; color:var(--txt2); text-transform:uppercase; letter-spacing:0.08em; margin-top:3px;">Reward</div>
                </div>
            </div>
        </div>

        <!-- Verdict details -->
        <div style="background:rgba(0,0,0,0.35); border-radius:14px;
                    border:1px solid rgba(255,255,255,0.08); padding:14px;">
            <div style="display:flex; justify-content:space-between; padding-bottom:8px; margin-bottom:8px;
                        border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:12px; color:var(--txt2); font-weight:500;">Submitted Verdict</span>
                <span style="font-size:12px; color:var(--txt); font-weight:700; font-family:var(--font-mono);">
                    {verdict.replace("_"," ").title() if verdict else "None"}</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding-bottom:8px; margin-bottom:8px;
                        border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:12px; color:var(--txt2); font-weight:500;">Ground Truth</span>
                <span style="font-size:12px; color:{glow_color}; font-weight:700;">{true_label.title()}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:12px; color:var(--txt2); font-weight:500;">Confidence Score</span>
                <span style="font-size:12px; color:var(--c-cyan); font-weight:700; font-family:var(--font-mono);">{confidence:.1%}</span>
            </div>
        </div>

    </div>
    """


def _generate_graph_summary(env):
    if getattr(env, "graph", None) is None:
        return "Graph not initialized."
    cov        = env.graph.evidence_coverage
    div        = env.graph.source_diversity_entropy
    con        = env.graph.contradiction_surface_area
    nodes      = sum(1 for n in env.graph.nodes.values() if n.retrieved)
    total_nodes = len(env.graph.nodes)
    edges      = sum(1 for e in env.graph.edges if e.discovered)
    total_edges = len(env.graph.edges)
    tactics    = ", ".join([t.replace("_", " ").title() for t in env.graph.applied_tactics]) or "None detected"
    return (
        f"Nodes Retrieved:          {nodes}/{total_nodes}\n"
        f"Edges Discovered:         {edges}/{total_edges}\n"
        f"Evidence Coverage:        {cov:.1%}\n"
        f"Source Diversity:         {div:.2f}\n"
        f"Contradictions Found:     {con}\n"
        f"Suspected Tactics:        {tactics}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ░░  INVESTIGATION LOGIC  ░░
# ══════════════════════════════════════════════════════════════════════════════

def investigate(task_name, difficulty):
    try:
        yield from _investigate_inner(task_name, int(difficulty))
    except Exception as exc:
        import traceback
        _app_logger.error("Investigation failed: %s\n%s", exc, traceback.format_exc())
        error_type = type(exc).__name__
        err_panel = f"""
        <div style="background:rgba(255,0,110,0.07); border:1px solid rgba(255,0,110,0.25);
                    border-radius:14px; padding:24px; font-family:var(--font-mono);
                    font-size:12px; color:var(--c-pink); margin:20px;">
            <div style="font-weight:700; letter-spacing:0.15em; margin-bottom:10px; font-size:13px;">
                ⚠ SYSTEM_ERROR
            </div>
            <div style="color:var(--txt2); line-height:1.8;">
                Type: <b style="color:var(--txt);">{error_type}</b><br>
                Details logged to <code>forge_debug.log</code>
            </div>
        </div>"""
        yield (
            _left_panel_idle(),
            err_panel,
            _right_panel_idle(),
            _statusbar_html("ERROR", "0.0K/S"),
            gr.update(interactive=True),
            "Investigation Error."
        )


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
    task_id    = info.get("task_id", "unknown")
    source_dom = env.graph.root.domain if env.graph else "unknown"
    virality   = env.graph.root.virality_score if env.graph else 0.5

    log_entries = []
    start_time  = time.time()

    def _ts():
        e = time.time() - start_time
        return f"{int(e//60):02d}:{int(e%60):02d}"

    log_entries.append((_ts(), f'Task init: {task_id.replace("_"," ").title()}', "query_source"))
    summary = _generate_graph_summary(env)

    yield (
        _left_panel_active(log_entries),
        _center_active(claim_text, task_id, virality, source_dom, "Waking API…"),
        _right_panel_idle(),
        _statusbar_html("ACTIVE", f"{random.uniform(0.8,2.0):.1f}K/S"),
        gr.update(interactive=False),
        summary,
    )

    done            = False
    step_info       = {}
    final_reward    = 0.0
    TIMEOUT_SECS    = 180

    while not done:
        if time.time() - start_time > TIMEOUT_SECS:
            _app_logger.warning("Investigation timed out after %ds", TIMEOUT_SECS)
            log_entries.append((_ts(), "Investigation Timeout", "flag_manipulation"))
            break

        context = {
            "steps":            env.steps,
            "max_steps":        env.max_steps,
            "coverage":         env.graph.evidence_coverage if env.graph else 0.0,
            "contradictions":   env.graph.contradiction_surface_area if env.graph else 0,
            "last_tool_result": step_info.get("tool_result"),
            "claim_text":       claim_text,
            "task_name":        task_id,
            "true_label_hint":  None,
        }

        action      = agent.act(obs, context=context)
        action_name = ACTIONS[action]
        obs, reward, terminated, truncated, step_info = env.step(action)
        done         = terminated or truncated
        final_reward = reward

        coverage = env.graph.evidence_coverage if env.graph else 0.0
        contras  = env.graph.contradiction_surface_area if env.graph else 0
        last     = agent.reasoning_log[-1] if agent.reasoning_log else {}
        think    = last.get("think", "Reviewing entity correlations…")
        predict  = last.get("predict", "")

        tool_result = step_info.get("tool_result", {})
        detail_str  = tool_result.get("summary", f"reward {reward:+.3f}")
        log_entries.append((_ts(), detail_str, action_name))
        visible_log = log_entries[-10:]
        fsm_state   = getattr(agent, "_fsm_state", "Computing")

        summary = _generate_graph_summary(env)
        yield (
            _left_panel_active(visible_log),
            _center_active(claim_text, task_id, virality, source_dom, "Analysing…"),
            _right_panel_active(think, predict, fsm_state, env.steps, env.max_steps, coverage, contras),
            _statusbar_html("ACTIVE", f"{random.uniform(1.0,3.5):.1f}K/S"),
            gr.update(interactive=False),
            summary,
        )

    true_label = env.graph.true_label if env.graph else "unknown"
    verdict    = step_info.get("verdict")
    correct    = (verdict == true_label)
    confidence = env._estimate_confidence() if hasattr(env, "_estimate_confidence") else 0.85

    summary = _generate_graph_summary(env)
    yield (
        _left_panel_active(log_entries[-12:]),
        _center_active(claim_text, task_id, virality, source_dom, "VERIFIED" if correct else "FLAGGED"),
        _right_panel_done(verdict, true_label, correct, env.steps, final_reward, confidence),
        _statusbar_html("OPTIMAL" if correct else "ALERT", f"{random.uniform(0.5,1.5):.1f}K/S"),
        gr.update(interactive=True),
        summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ░░  GRADIO APP  ░░
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLE_CLAIMS = [
    ("fabricated_stats",     1, "WHO Cancer Coffee Claim"),
    ("satire_news",          1, "Satirical Headlines"),
    ("coordinated_campaign", 2, "Bot Network Detection"),
    ("sec_fraud",            2, "Financial Fraud"),
    ("image_forensics",      3, "Deepfake Detection"),
]

NEBULA_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="transparent",
    block_background_fill="transparent",
    block_border_width="0px",
    input_background_fill="rgba(0,0,0,0.4)",
    input_border_color="rgba(255,255,255,0.12)",
    input_border_width="1px",
    input_radius="10px",
    background_fill_primary="transparent",
)

with gr.Blocks(
    title="FORGE — Forensic RL Graph Environment",
    theme=NEBULA_THEME,
    css=FORGE_CSS,
    head=FORGE_JS_HTML
) as demo:

    # ── Background effects + custom cursor injection ───────────────────────
    # gr.HTML(FORGE_JS_HTML)

    # ── Holographic top navigation ─────────────────────────────────────────
    gr.HTML(_topnav_html())

    # ── Three-column investigation layout ──────────────────────────────────
    with gr.Row(elem_classes=["main-container"]):

        # LEFT — Live Feed
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            left_panel = gr.HTML(value=_left_panel_idle())

        # CENTER — Claim + Progress + Controls
        with gr.Column(scale=6):
            with gr.Column(elem_classes=["glass-panel"]):
                center_panel = gr.HTML(value=_center_idle())

            with gr.Column(elem_classes=["controls-panel"]):
                with gr.Row():
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
                    start_btn = gr.Button("▶  Launch Deep Analysis", variant="primary")

                gr.Examples(
                    examples=[[t, d] for t, d, _ in EXAMPLE_CLAIMS],
                    inputs=[task_dd, diff_sl],
                    label="⚡ Quick start — click any example",
                )

                statusbar = gr.HTML(value=_statusbar_html("IDLE", "0.0K/S"))

        # RIGHT — Agent Reasoning
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            right_panel = gr.HTML(value=_right_panel_idle())

    # ── Evidence graph summary ─────────────────────────────────────────────
    with gr.Row():
        graph_summary_box = gr.Textbox(
            label="Evidence Graph Summary",
            lines=6,
            interactive=False,
            placeholder="Run an investigation to see live graph statistics…",
        )

    # ── Event wiring ───────────────────────────────────────────────────────
    start_btn.click(
        fn=investigate,
        inputs=[task_dd, diff_sl],
        outputs=[left_panel, center_panel, right_panel, statusbar, start_btn, graph_summary_box],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=NEBULA_THEME,
        css=FORGE_CSS,
    )