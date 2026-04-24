import gradio as gr
import os

# FORGE-MA v9.0 Glassmorphic Dark UI Theme
# Injected via gr.Blocks(css=FORGE_CSS)
FORGE_CSS = """
.gradio-container {
    background: radial-gradient(circle at top left, #0d1117, #010409) !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Glassmorphic Cards */
.forge-card {
    background: rgba(22, 27, 34, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(48, 54, 61, 0.8) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    transition: transform 0.2s ease, border-color 0.2s ease !important;
}
.forge-card:hover {
    border-color: #58a6ff !important;
    transform: translateY(-2px);
}

/* Animated Background Dots */
.forge-dots {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: radial-gradient(#30363d 1px, transparent 1px);
    background-size: 32px 32px;
    opacity: 0.15;
    pointer-events: none;
    z-index: -1;
}

/* Accents & Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #1f6feb, #58a6ff) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
.gr-button-primary:hover {
    filter: brightness(1.1);
    box-shadow: 0 0 15px rgba(88, 166, 255, 0.4);
}

/* Metric Badges */
.forge-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 700;
    text-transform: uppercase;
}
.badge-teal { background: rgba(56, 178, 172, 0.2); color: #38b2ac; border: 1px solid #38b2ac; }
.badge-purple { background: rgba(159, 122, 234, 0.2); color: #9f7aea; border: 1px solid #9f7aea; }
"""

def dummy_investigate(claim, task):
    import time
    time.sleep(1)
    verdict = {"Fabricated": 0.85, "Real": 0.15} if task == "fabricated_stats" else {"Satire": 0.9, "Real": 0.1}
    chain = "1. Information Gathering\n2. Cross-referencing\n3. Source Verification\n4. Network Analysis"
    return verdict, 0.85, chain

def launch_demo():
    with gr.Blocks(title="FORGE-MA | Forensic RL Multi-Agent") as demo:
        gr.HTML('<div class="forge-dots"></div>')
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# 🏆 FORGE-MA")
                gr.Markdown("### Forensic RL Graph Environment — Multi-Agent")
            with gr.Column(scale=0):
                gr.HTML('<span class="forge-badge badge-teal">V9.0 Hardened</span>')
                gr.HTML('<span class="forge-badge badge-purple">OpenEnv Ready</span>')

        with gr.Tab("Investigation Dashboard"):
            with gr.Row():
                with gr.Column(elem_classes=["forge-card"]):
                    gr.Markdown("## 📋 Claim Analysis")
                    claim_input = gr.Textbox(label="Claim Text", placeholder="Enter claim to investigate...")
                    task_select = gr.Dropdown(choices=["fabricated_stats", "out_of_context", "plandemic"], label="Task Scenario", value="fabricated_stats")
                    investigate_btn = gr.Button("Begin Investigation", variant="primary")
                
                with gr.Column(elem_classes=["forge-card"]):
                    gr.Markdown("## 🔬 Society of Thought")
                    gr.Markdown("*Multi-Provider Consensus Engine*")
                    verdict_output = gr.Label(label="Final Verdict")
                    confidence_bar = gr.Slider(0, 1, label="Consensus Confidence", interactive=False)

        with gr.Tab("Forensic Evidence"):
            with gr.Row():
                with gr.Column(elem_classes=["forge-card"]):
                    gr.Markdown("### 🔗 Tactic Chain (Reconstructed)")
                    chain_output = gr.Code(label="DISARM Tactic Sequence", language="markdown")
                with gr.Column(elem_classes=["forge-card"]):
                    gr.Markdown("### 📊 Metrics")
                    gr.Code(label="Reward Breakdown", language="markdown")

        with gr.Tab("STIX 2.1 Export"):
            stix_output = gr.JSON(label="STIX 2.1 Bundle")
            download_btn = gr.Button("Download Bundle")

        gr.Markdown("---")
        gr.Markdown("© 2026 FORGE Research Team | Meta × HuggingFace OpenEnv Hackathon")

        # Wire up the dummy investigation
        investigate_btn.click(
            fn=dummy_investigate,
            inputs=[claim_input, task_select],
            outputs=[verdict_output, confidence_bar, chain_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7861, css=FORGE_CSS)

if __name__ == "__main__":
    launch_demo()

