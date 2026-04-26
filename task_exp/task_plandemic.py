"""
Task: Plandemic Case Study
Tactics: coordinated_campaign, cherry_pick_study, fabricated_stats
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta
from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
from env.tasks.task_base import BaseTask

_PLANDEMIC_CLAIMS = [
    {
        "text": "The COVID-19 vaccine contains tracking microchips developed by Bill Gates.",
        "correct_stat": "The Gates Foundation funds vaccine research, but no microchips or tracking devices are in any vaccine. Vials only contain mRNA, lipids, salts, and sugars.",
        "real_domain": "reuters.com",
        "fake_domain": "truth-about-plandemic.org",
        "tactic": "fabricated_stats",
    },
    {
        "text": "Wearing masks activates the coronavirus and makes you sick with COVID-19.",
        "correct_stat": "Masks prevent viral transmission. They do not 'activate' any virus. Doctors have worn masks safely for decades.",
        "real_domain": "who.int",
        "fake_domain": "real-health-freedom.net",
        "tactic": "coordinated_campaign",
    },
    {
        "text": "Hospitals are deliberately inflating COVID death numbers to get more funding.",
        "correct_stat": "Hospitals are audited. Excess mortality data confirms the death toll without relying purely on hospital coding.",
        "real_domain": "cdc.gov",
        "fake_domain": "awake-patriot-news.com",
        "tactic": "fabricated_stats",
    },
]

_DEBUNKING_DOMAINS = [
    "snopes.com", "factcheck.org", "politifact.com",
    "reuters.com/fact-check", "apnews.com/fact-check",
]

class PlandemicTask(BaseTask):
    task_id = "plandemic"
    description = (
        "The agent investigates viral claims from the 'Plandemic' documentary series."
    )

    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        rng = random.Random(seed)
        
        # Plandemic scenarios are always misinfo/fabricated for the sake of this case study,
        # but we add a true claim to keep it balanced if needed. We'll force it to fabricated.
        template = rng.choice(_PLANDEMIC_CLAIMS)
        domain = template["fake_domain"]
        trust_score = 0.1
        true_label = "fabricated"
        edge_rel = "contradicts"
        applied_tactics = [template["tactic"], "coordinated_campaign"]

        # Try to use LLM to dynamically generate a claim variant
        try:
            from agents.llm_agent_ma import LLMAgent
            import config
            agent = LLMAgent(
                system_prompt="You are an expert at generating conspiracy claims. Provide a single claim text.", 
                provider=config.AGENT_NEGOTIATED_PROVIDER
            )
            prompt = f"Generate a novel false claim about the Plandemic matching this theme: {template['text']}. Return only the claim text, no quotes or prefixes."
            dynamic_text = agent.query(prompt)
            if dynamic_text and "MOCK:" not in dynamic_text:
                template["text"] = dynamic_text.strip().strip('"')
        except Exception as e:
            pass # Fallback to static template
        
        graph_id = str(uuid.uuid4())
        root_id = "node_root"

        root = ClaimNode(
            node_id=root_id,
            text=template["text"],
            source_url=f"https://{domain}/video-{rng.randint(1000,9999)}",
            domain=domain,
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 30)),
            virality_score=rng.uniform(0.8, 0.99), # High virality
            trust_score=trust_score,
        )

        graph = ClaimGraph(
            graph_id=graph_id,
            root_claim_id=root_id,
            true_label=true_label,
            difficulty=difficulty,
            applied_tactics=applied_tactics,
        )
        graph.add_node(root)

        # Authoritative debunking node
        auth_id = "node_authority"
        auth_text = f"Official fact check by {template['real_domain']} — {template['correct_stat']}"

        auth = ClaimNode(
            node_id=auth_id,
            text=auth_text,
            source_url=f"https://{template['real_domain']}/fact-check/",
            domain=template["real_domain"],
            timestamp=datetime.utcnow() - timedelta(days=rng.randint(60, 365)),
            virality_score=0.4,
            trust_score=0.95,
        )
        graph.add_node(auth)
        graph.add_edge(EvidenceEdge(
            edge_id="e1", src_id=root_id, tgt_id=auth_id,
            relation=edge_rel, weight=0.9,
        ))

        # Amplifier nodes (coordinated campaign)
        prev_chain_id = root_id
        for i in range(difficulty):
            amp_domain = f"alt-news-{rng.randint(1,99)}.org"
            amp_id = f"node_amp_{i}"
            amp = ClaimNode(
                node_id=amp_id,
                text=f"Must watch: Plandemic exposes the truth. {template['text']}",
                source_url=f"https://{amp_domain}/watch",
                domain=amp_domain,
                timestamp=datetime.utcnow() - timedelta(days=rng.randint(1, 5)),
                virality_score=rng.uniform(0.6, 0.8),
                trust_score=0.2,
            )
            graph.add_node(amp)
            graph.add_edge(EvidenceEdge(
                edge_id=f"e_amp_{i}", src_id=prev_chain_id, tgt_id=amp_id,
                relation="supports", weight=rng.uniform(0.7, 0.9),
            ))
            prev_chain_id = amp_id

        return graph

    def oracle_steps(self, graph: ClaimGraph) -> int:
        return 3 + (graph.difficulty - 1)

    def has_manipulation(self, graph: ClaimGraph) -> bool:
        return graph.true_label == "fabricated"

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        import numpy as np
        actions = [s.get("action", "") for s in episode_trace if "action" in s]

        investigation_tools = [
            a for a in actions
            if not a.startswith("submit_verdict") and a != "flag_manipulation"
        ]
        unique_tools = len(set(investigation_tools))
        
        final_verdict = next(
            (a.replace("submit_verdict_", "") for a in reversed(actions)
             if a.startswith("submit_verdict_")), None
        )

        score = 0.001
        if final_verdict == graph.true_label:
            score += 0.5
        
        if unique_tools >= 2:
            score += 0.3
            
        used_key_tools = any(a in ["cross_reference", "network_cluster"] for a in actions)
        if used_key_tools:
            score += 0.2

        return float(np.clip(score, 0.001, 0.999))
