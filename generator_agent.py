"""
GeneratorAgent — adversarial misinformation generator.
Creates synthetic claims using compositional tactics to challenge the investigator.
Trained via self-play signal: rewarded when investigator FAILS.

100% free — uses local template composition + optional Groq (free tier).
"""

from __future__ import annotations
import logging
import random
from typing import Any, Dict, List, Optional

from env.claim_graph import ClaimGraph, TacticType
from env.tasks import TASK_REGISTRY
import config

logger = logging.getLogger(__name__)

ALL_TACTICS: List[TacticType] = [
    "fabricate_statistic", "strip_context", "backdate_article",
    "misattribute_quote", "amplify_via_bot_network",
    "splice_image_caption", "cherry_pick_study", "translate_without_context",
    "parody_taken_literally",
]

# Linguistic register styles for diversity
REGISTER_TEMPLATES = {
    "tabloid": "SHOCKING: {core}",
    "academic": "A recent peer-reviewed study suggests that {core}",
    "official": "According to official government data, {core}",
    "social": "🚨 SHARE BEFORE DELETED 🚨 {core} #truth #exposed",
    "neutral": "{core}",
}


class GeneratorAgent:
    """
    Population-member generator that creates adversarial misinformation claims.
    Each generator has a tactic_bias (preferred tactic mix) and register_style.
    """

    def __init__(
        self,
        agent_id: str = "gen_0",
        tactic_bias: Optional[List[TacticType]] = None,
        register_style: str = "neutral",
        elo: int = config.ELO_INITIAL,
        seed: int = 42,
        use_llm: bool = False,         # optional Groq enhancement (free)
    ):
        self.agent_id = agent_id
        self.tactic_bias = tactic_bias or random.sample(ALL_TACTICS, k=2)
        self.register_style = register_style
        self.elo = elo
        self.rng = random.Random(seed)
        self.use_llm = use_llm and bool(config.HF_TOKEN)
        self._openai = None
        if self.use_llm:
            self._init_openai()

        # All task generators
        self.task_generators = {k: v() for k, v in TASK_REGISTRY.items()}
        self.wins = 0
        self.losses = 0

    def _init_openai(self):
        try:
            from openai import OpenAI
            self._openai = OpenAI(
                base_url=config.API_BASE_URL,
                api_key=config.HF_TOKEN or "dummy-key"
            )
        except Exception as e:
            logger.debug("OpenAI init failed in generator: %s", e)

    def generate(self, difficulty: int = 1) -> ClaimGraph:
        """
        Compose k tactics (Poisson-sampled) and produce a ClaimGraph.
        If Groq is available, use it to diversify the claim text.
        """
        seed = self.rng.randint(0, 2**20)

        # Sample k tactics (biased towards this generator's preferences)
        all_pool = self.tactic_bias * 3 + ALL_TACTICS   # bias: 3:1 ratio
        k = max(1, min(difficulty, 4))
        tactics = list(dict.fromkeys(self.rng.choices(all_pool, k=k)))[:k]

        # Pick task type based on dominant tactic
        task_name = self._tactic_to_task(tactics[0])
        task = self.task_generators[task_name]
        graph = task.generate(difficulty=difficulty, seed=seed)

        # Override tactics to match what generator chose
        graph.applied_tactics = tactics

        # Apply register style to root claim text
        template = REGISTER_TEMPLATES.get(self.register_style, "{core}")
        graph.nodes[graph.root_claim_id].text = template.format(
            core=graph.nodes[graph.root_claim_id].text
        )

        # Optionally use LLM to make claim more convincing
        if self.use_llm and self._openai:
            try:
                graph = self._llm_enhance(graph)
            except Exception as e:
                logger.debug("LLM enhancement failed: %s", e)

        return graph

    def _tactic_to_task(self, tactic: TacticType) -> str:
        tactic_task_map = {
            "fabricate_statistic": "fabricated_stats",
            "cherry_pick_study": "fabricated_stats",
            "misattribute_quote": "fabricated_stats",
            "strip_context": "out_of_context",
            "backdate_article": "out_of_context",
            "translate_without_context": "out_of_context",
            "amplify_via_bot_network": "coordinated_campaign",
            "splice_image_caption": "out_of_context",
            "parody_taken_literally": "satire_news",
        }
        return tactic_task_map.get(tactic, "fabricated_stats")

    def _llm_enhance(self, graph: ClaimGraph) -> ClaimGraph:
        """Use LLM to make the claim text more realistic/convincing."""
        original = graph.root.text
        prompt = (
            f"Rewrite this misinformation claim to sound more credible and realistic, "
            f"while keeping it false. Apply tactic: {graph.applied_tactics[0]}. "
            f"Return ONLY the rewritten claim text, nothing else.\n\nOriginal: {original}"
        )
        resp = self._openai.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a misinformation research tool for safety testing."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        enhanced = resp.choices[0].message.content.strip()
        if enhanced and len(enhanced) > 20:
            graph.nodes[graph.root_claim_id].text = enhanced
        return graph

    def update_elo(self, investigator_won: bool) -> None:
        """Update ELO after one bout."""
        expected_gen = self.elo / (self.elo + config.ELO_INITIAL)
        result = 0.0 if investigator_won else 1.0
        # LOW BUG 1 FIX: int() truncates small deltas (e.g. 0.8 → 0) so
        # close ELO matches never register. round() correctly rounds 0.8 → 1.
        self.elo += round(config.ELO_K_FACTOR * (result - expected_gen))
        if investigator_won:
            self.losses += 1
        else:
            self.wins += 1

    def mutate(self, seed: int) -> "GeneratorAgent":
        """Produce a mutated copy (for PBT)."""
        rng = random.Random(seed)
        new_tactic = rng.choice(ALL_TACTICS)
        new_tactics = list(set(self.tactic_bias + [new_tactic]))[:3]
        styles = list(REGISTER_TEMPLATES.keys())
        new_style = rng.choice(styles)
        return GeneratorAgent(
            agent_id=f"{self.agent_id}_mut{seed}",
            tactic_bias=new_tactics,
            register_style=new_style,
            elo=self.elo,
            seed=seed,
            use_llm=self.use_llm,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "tactic_bias": self.tactic_bias,
            "register_style": self.register_style,
            "elo": self.elo,
            "wins": self.wins,
            "losses": self.losses,
        }
