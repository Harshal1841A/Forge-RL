"""
Task: ImageForensicsTask — AI-Generated Deepfake Image Detection (v2.0)

Simulates a forensic investigation into a viral image claim. The agent must
determine whether an image is real or AI-generated (deepfake / diffusion model
output) using simulated ELA (Error Level Analysis) and diffusion artifact signatures
surfaced through the existing tool set.

Tools overloaded for image forensics:
- temporal_audit: Reads EXIF metadata and ELA score for the image
- cross_reference: Reverse image search — looks for source image discrepancies
- entity_link: Verifies whether depicted persons/organisations actually exist
- trace_origin: Traces the posting origin timestamp
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
from env.tasks.task_base import BaseTask


_IMAGE_SCENARIOS = [
    {
        "claim": "Leaked satellite photo shows secret military base being constructed in northern Finland.",
        "image_desc": "Satellite imagery shared via Telegram group. Resolution too high for public-tier satellites.",
        "ela_score": 0.84,          # >0.7 → strong AI-generation signal
        "diffusion_signature": True,
        "source_domain": "tgchannel-leaked.ru",
    },
    {
        "claim": "Photo shows world leader shaking hands with known terrorist at undisclosed location.",
        "image_desc": "JPEG posted to anonymous imageboard. Face-swap artifacts detected around neck boundary.",
        "ela_score": 0.91,
        "diffusion_signature": True,
        "source_domain": "deepstatealert.net",
    },
    {
        "claim": "Newly surfaced photo shows police brutality incident that mainstream media is hiding.",
        "image_desc": "Image circulating on social media. Reverse-image search finds original unedited photo from 2016.",
        "ela_score": 0.72,
        "diffusion_signature": False,   # not AI — just re-contextualized old photo
        "source_domain": "citizenjustice.info",
    },
    {
        "claim": "Authentic video still from private Epstein island shows prominent politician.",
        "image_desc": "Low-quality JPEG, face region shows diffusion noise pattern. No EXIF metadata present.",
        "ela_score": 0.88,
        "diffusion_signature": True,
        "source_domain": "truthunleashed.co",
    },
    {
        "claim": "Real photo of UFO taken by US Air Force pilot over Pacific Ocean last week.",
        "image_desc": "High-resolution PNG with missing EXIF. Noise pattern inconsistent with camera sensor.",
        "ela_score": 0.79,
        "diffusion_signature": True,
        "source_domain": "ufodisclosure.net",
    },
]

_TRUE_IMAGE_SCENARIOS = [
    {
        "claim": "Live photo from the international space station captured today.",
        "image_desc": "High-resolution JPG uploaded directly from official agency channels.",
        "ela_score": 0.05,
        "diffusion_signature": False,
        "source_domain": "nasa.gov",
    },
    {
        "claim": "Official portrait released by the newly elected prime minister.",
        "image_desc": "Standard press image containing verified EXIF data from professional camera.",
        "ela_score": 0.02,
        "diffusion_signature": False,
        "source_domain": "gov.uk",
    },
]

_DEBUNK_DOMAINS = ["bellingcat.com", "snopes.com", "bbc.com/verify", "reuters.com"]


class ImageForensicsTask(BaseTask):
    """
    Agent investigates a viral image/video claim to determine if it is
    AI-generated (fabricated) or a real-but-miscontextualized photograph.

    Tools are used in image-forensics mode:
    - temporal_audit: surfaces ELA_score and EXIF anomalies
    - cross_reference: reveals reverse-image-search matches or lack thereof
    """

    task_id = "image_forensics"
    description = (
        "Agent investigates a viral image claim to determine if it is AI-generated "
        "(deepfake / Stable Diffusion output) or a real photograph used out of context."
    )

    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        rng = random.Random(seed)
        is_true = rng.random() > 0.5

        if is_true:
            scenario = rng.choice(_TRUE_IMAGE_SCENARIOS)
            ela_score = scenario["ela_score"]
            has_diffusion = scenario["diffusion_signature"]
            true_label = "real"
            trust_score = 0.95
        else:
            scenario = rng.choice(_IMAGE_SCENARIOS)
            ela_score = scenario["ela_score"]
            has_diffusion = scenario["diffusion_signature"]
            true_label = "fabricated" if has_diffusion else "out_of_context"
            trust_score = 0.1

        graph_id = str(uuid.uuid4())
        root_id = "node_root"

        # ── Root node: the viral image claim ─────────────────────────────────
        root = ClaimNode(
            node_id=root_id,
            text=scenario["claim"],
            source_url=f"https://{scenario['source_domain']}/post/{rng.randint(1000, 9999)}",
            domain=scenario["source_domain"],
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 14)),
            virality_score=rng.uniform(0.7, 0.99),
            trust_score=trust_score,
            metadata={
                # These are surfaced by temporal_audit (ELA / EXIF forensics)
                "image_description": scenario["image_desc"],
                "ela_score": ela_score,
                "diffusion_signature": has_diffusion,
                "exif_missing": has_diffusion,       # AI images usually lack EXIF
                "forensics_domain": "image",
            },
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label=true_label,
            difficulty=difficulty,
            applied_tactics=["splice_image_caption"] if not has_diffusion else [
                "fabricate_statistic", "splice_image_caption"
            ],
        )
        graph.add_node(root)

        # ── ELA Analysis node (surfaced by temporal_audit or cross_reference) ──
        ela_id = "node_ela_analysis"
        if is_true:
            ela_verdict = "AUTHENTIC"
            ela_desc = "LOW — consistent with original camera compression"
            ela_rel = "supports"
        else:
            ela_verdict = "AI-GENERATED" if has_diffusion else "AUTHENTIC_BUT_RECIRCULATED"
            ela_desc = "HIGH — diffusion artifact pattern detected" if has_diffusion else "moderate — possible compression artifact"
            ela_rel = "contradicts" if has_diffusion else "cites"

        ela = ClaimNode(
            node_id=ela_id,
            text=(
                f"ELA Analysis: ela_score={ela_score:.2f} "
                f"({ela_desc}). "
                f"Verdict: {ela_verdict}."
            ),
            source_url="https://fotoforensics.com/analysis",
            domain="fotoforensics.com",
            timestamp=datetime.utcnow() - timedelta(hours=rng.randint(1, 48)),
            virality_score=0.05,
            trust_score=0.85,
            metadata={"forensics_type": "ela", "ela_score": ela_score},
        )
        graph.add_node(ela)
        graph.add_edge(EvidenceEdge(
            edge_id="e_ela", src_id=ela_id, tgt_id=root_id,
            relation=ela_rel,
            weight=1.0 - ela_score if is_true else ela_score,
        ))

        # ── OSINT node: reverse-image-search / Fact Check ─────────────
        deb_domain = rng.choice(_DEBUNK_DOMAINS)
        deb_id = "node_debunk"

        if is_true:
            deb_text = "OSINT finding: Image traces directly back to official source without alterations."
            deb_rel = "supports"
        elif has_diffusion:
            deb_text = (
                "OSINT: No authentic source found for image. "
                "Diffusion noise detected in pixel frequency analysis. "
                "Image likely generated by Stable Diffusion or Midjourney."
            )
            deb_rel = "debunks"
        else:
            deb_text = (
                f"Reverse image search: original photo dated "
                f"{(datetime.utcnow() - timedelta(days=rng.randint(365, 2000))).strftime('%Y-%m-%d')} "
                f"— not from the claimed time/location."
            )
            deb_rel = "debunks"

        deb = ClaimNode(
            node_id=deb_id,
            text=deb_text,
            source_url=f"https://{deb_domain}/fact-check/{rng.randint(10000, 99999)}",
            domain=deb_domain,
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 7)),
            virality_score=0.1,
            trust_score=0.93,
        )
        graph.add_node(deb)
        graph.add_edge(EvidenceEdge(
            edge_id="e_deb", src_id=deb_id, tgt_id=root_id,
            relation=deb_rel, weight=0.95,
        ))

        # ── Difficulty: chained bot amplification nodes ─────────────────────────
        # DEPTH SCALING: bot accounts form a chain (bot_0→bot_1→bot_2) instead of
        # all connecting directly to root.  At difficulty >= 3, a hidden
        # whistleblower node with forensic metadata is only reachable by
        # traversing the full chain.
        prev_chain_id = root_id
        for i in range(difficulty - 1):
            bot_id = f"node_bot_{i}"
            bot = ClaimNode(
                node_id=bot_id,
                text=f"Automated account amplifying the image claim (bot account {i+1}).",
                source_url=f"https://twitter.com/bot_account_{rng.randint(1000, 9999)}",
                domain="twitter.com",
                timestamp=datetime.utcnow() - timedelta(hours=rng.randint(1, 72)),
                virality_score=rng.uniform(0.5, 0.85),
                trust_score=0.05,
                metadata={"is_bot": True},
            )
            graph.add_node(bot)
            graph.add_edge(EvidenceEdge(
                edge_id=f"e_bot_{i}", src_id=prev_chain_id, tgt_id=bot_id,
                relation="amplifies", weight=rng.uniform(0.6, 0.9),
            ))
            prev_chain_id = bot_id

        # ── Hidden forensic whistleblower at chain depth (difficulty >= 3) ────
        if difficulty >= 3 and not is_true:
            whistleblower_id = "node_whistleblower"
            whistleblower = ClaimNode(
                node_id=whistleblower_id,
                text=(
                    f"Whistleblower metadata leak: image EXIF reconstruction reveals "
                    f"Stable Diffusion v2.1 model hash in PNG metadata chunk. "
                    f"Original prompt recovered from C2PA watermark."
                ),
                source_url="https://forensicarchive.org/exif-reconstruction/report",
                domain="forensicarchive.org",
                timestamp=datetime.utcnow() - timedelta(hours=rng.randint(2, 24)),
                virality_score=0.01,
                trust_score=0.92,
                metadata={"forensics_type": "metadata_reconstruction", "definitive": True},
            )
            graph.add_node(whistleblower)
            graph.add_edge(EvidenceEdge(
                edge_id="e_whistleblower", src_id=prev_chain_id, tgt_id=whistleblower_id,
                relation="debunks", weight=0.99,
            ))

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        # Minimum: temporal_audit (ELA) + cross_reference (reverse image) + submit
        return 2 + (graph.difficulty - 1)

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return graph.true_label == "fabricated"

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        """
        Hard multimodal task grader.
        Partial credit:
          +0.3  used temporal_audit (checks EXIF/metadata)
          +0.2  used trace_origin (finds original image source)
          +0.1  used entity_link (verifies depicted entities)
          +0.4  submitted correct final verdict

        Exploit resistance:
        - Requires >= 2 unique investigation tools
        - Requires a submitted verdict for score > 0.3
        """
        import numpy as np
        score = 0.001
        actions = [s.get("action", "") for s in episode_trace if "action" in s]

        # ── Exploit guard 1: tool diversity requirement ─────────────────────
        investigation_tools = [
            a for a in actions
            if not a.startswith("submit_verdict") and a != "flag_manipulation"
        ]
        unique_tools = len(set(investigation_tools))
        if unique_tools < 2:
            final_verdict = next(
                (a.replace("submit_verdict_", "") for a in reversed(actions)
                 if a.startswith("submit_verdict_")), None
            )
            if final_verdict == graph.true_label:
                return float(np.clip(0.4, 0.001, 0.999))
            return 0.001

        # ── Exploit guard 2: verdict required ───────────────────────────────
        final_verdict = next(
            (a.replace("submit_verdict_", "") for a in reversed(actions)
             if a.startswith("submit_verdict_")), None
        )
        if final_verdict is None:
            if "temporal_audit" in actions:
                score += 0.3
            if "trace_origin" in actions:
                score += 0.2
            if "entity_link" in actions:
                score += 0.1
            return float(np.clip(score * 0.3, 0.001, 0.999))

        # ── Standard grading ────────────────────────────────────────────────
        if "temporal_audit" in actions:
            score += 0.3
        if "trace_origin" in actions:
            score += 0.2
        if "entity_link" in actions:
            score += 0.1

        if final_verdict == graph.true_label:
            score += 0.4
        elif final_verdict is not None:
            misinfo = {"misinfo", "satire", "out_of_context", "fabricated"}
            if final_verdict in misinfo and graph.true_label in misinfo:
                score += 0.2

        return float(np.clip(score, 0.001, 0.999))
