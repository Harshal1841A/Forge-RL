import numpy as np
from agents.llm_agent_ma import LLMAgent
from env.primitives import PrimitiveType

# ── 13 Investigation Tools ─────────────────────────────────────────────────────
TOOLS = [
    "trace_origin", "query_source", "temporal_audit", "quote_search",
    "cross_reference", "entity_link", "network_cluster", "context_expand",
    "image_reverse_search", "domain_whois", "sentiment_analysis",
    "semantic_similarity", "authorship_verification"
]

# ── Semantic Primitive → Tools mapping (High Bug H4 fix) ──────────────────────
# Replaces the broken modulo mapping (gin_prior_13[i % 13] += gin_probs[i])
# which had zero semantic relationship between primitive index and tool index.
#
# Each primitive maps to 1-3 tools that are actually useful for detecting it:
PRIMITIVE_TO_TOOLS: dict[PrimitiveType, list[str]] = {
    PrimitiveType.SOURCE_LAUNDER:    ["trace_origin", "query_source", "domain_whois"],
    PrimitiveType.TEMPORAL_SHIFT:    ["temporal_audit", "cross_reference"],
    PrimitiveType.ENTITY_SUBSTITUTE: ["entity_link", "image_reverse_search", "authorship_verification"],
    PrimitiveType.QUOTE_FABRICATE:   ["quote_search", "authorship_verification", "semantic_similarity"],
    PrimitiveType.CONTEXT_STRIP:     ["context_expand", "cross_reference", "sentiment_analysis"],
    PrimitiveType.CITATION_FORGE:    ["query_source", "cross_reference", "domain_whois"],
    PrimitiveType.NETWORK_AMPLIFY:   ["network_cluster", "trace_origin", "domain_whois"],
    PrimitiveType.SATIRE_REFRAME:    ["sentiment_analysis", "context_expand", "semantic_similarity"],
}

# Ordered list of all primitives (must match GINPredictor.prims order)
_PRIMS_ORDERED = [
    PrimitiveType.SOURCE_LAUNDER,
    PrimitiveType.TEMPORAL_SHIFT,
    PrimitiveType.ENTITY_SUBSTITUTE,
    PrimitiveType.QUOTE_FABRICATE,
    PrimitiveType.CONTEXT_STRIP,
    PrimitiveType.CITATION_FORGE,
    PrimitiveType.NETWORK_AMPLIFY,
    PrimitiveType.SATIRE_REFRAME,
]


class NegotiatedSearch:
    """
    Pre-analysis: generates ToolPreferenceVectors before investigation.
    Uses multi-provider agents to eliminate shared-model bias:
      Historian → Cerebras / Llama-3.1 70B
      Critic    → Mistral  / mistral-small-latest
    Produces V_ensemble to guide Auditor’s tool selection.
    """
    def __init__(self):
        import config
        self.historian = LLMAgent(
            system_prompt="You are Context Historian. Analyse temporal and provenance cues in the claim. Respond in JSON with keys: tool_preferences (dict mapping tool names to 0–1 scores).",
            provider=config.AGENT_HISTORIAN_PROVIDER,    # cerebras
            api_key=config.CEREBRAS_API_KEY,
        )
        self.critic = LLMAgent(
            system_prompt="You are Narrative Critic. Analyse narrative style, framing and emotional language. Respond in JSON with keys: tool_preferences (dict mapping tool names to 0–1 scores).",
            provider=config.AGENT_CRITIC_PROVIDER,       # mistral
            api_key=config.MISTRAL_API_KEY,
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _gin_probs_to_tool_vector(self, gin_probs: np.ndarray) -> np.ndarray:
        """
        Convert 8-dim GIN presence probabilities to 13-dim tool preference vector.

        High Bug H4 fix: replaces broken modulo mapping with a semantic
        PRIMITIVE_TO_TOOLS dictionary lookup.

        For each primitive i, gin_probs[i] contributes equally to all tools
        semantically associated with that primitive. The contribution is
        weighted by the detection probability — if GIN thinks SOURCE_LAUNDER
        is likely (high prob), preference for trace_origin/query_source rises.
        """
        tool_index = {t: i for i, t in enumerate(TOOLS)}
        gin_prior_13 = np.zeros(13)

        for prim_idx, prim in enumerate(_PRIMS_ORDERED):
            prob = gin_probs[prim_idx]
            associated_tools = PRIMITIVE_TO_TOOLS.get(prim, [])
            if associated_tools:
                per_tool_weight = prob / len(associated_tools)
                for tool_name in associated_tools:
                    if tool_name in tool_index:
                        gin_prior_13[tool_index[tool_name]] += per_tool_weight

        # Normalise to sum=1
        total = gin_prior_13.sum()
        if total > 0:
            gin_prior_13 = gin_prior_13 / total

        return gin_prior_13

    def generate_vectors(self, claim: str, gin_model) -> np.ndarray:
        """
        Step 1: Historian Groq call — temporal-focused prompt, output 13-dim vector
        Step 2: Critic Groq call — style-focused prompt, output 13-dim vector
        Step 3: GIN prior — zero-shot on root-only graph → semantic tool mapping
        Step 4: V_ensemble = softmax(0.50*v_hist + 0.30*v_crit + 0.20*normalize(gin_prior))
        """
        # Step 1: Historian
        hist_resp = self.historian.query(self._hist_prompt(claim))
        hist_dict = self.historian.parse_json(hist_resp).get("tool_preferences", {})
        v_hist = np.array([hist_dict.get(t, 0.0) for t in TOOLS], dtype=float)

        # Step 2: Critic
        crit_resp = self.critic.query(self._critic_prompt(claim))
        crit_dict = self.critic.parse_json(crit_resp).get("tool_preferences", {})
        v_crit = np.array([crit_dict.get(t, 0.0) for t in TOOLS], dtype=float)

        # Step 3: GIN prior — root-only graph, semantic mapping
        class DummyGraph:
            def __init__(self):
                import torch
                self.x = torch.zeros((1, 10))
                self.edge_index = torch.zeros((2, 0), dtype=torch.long)
                self.batch = torch.zeros((1,), dtype=torch.long)

        gin_res = gin_model.predict_chain(DummyGraph())
        gin_probs = gin_res["presence_probs"]

        # Semantic primitive → tool mapping (H4 fix)
        gin_prior_13 = self._gin_probs_to_tool_vector(gin_probs)

        # Normalise v_hist and v_crit
        sum_hist = v_hist.sum()
        if sum_hist > 0:
            v_hist = v_hist / sum_hist

        sum_crit = v_crit.sum()
        if sum_crit > 0:
            v_crit = v_crit / sum_crit

        # Step 4: V_ensemble
        v_ensemble_raw = 0.50 * v_hist + 0.30 * v_crit + 0.20 * gin_prior_13
        V_ensemble = self._softmax(v_ensemble_raw)

        return V_ensemble

    def _hist_prompt(self, claim: str) -> str:
        """Temporal-focused prompt. Asks for JSON with tool_preferences dict."""
        return (f"Analyze claim temporally: '{claim}'.\n"
                "Return JSON with 'tool_preferences' mapping each of the 13 tools to a float 0.0-1.0 "
                "representing relevance for temporal investigation.")

    def _critic_prompt(self, claim: str) -> str:
        """Style-focused prompt. Asks for JSON with tool_preferences dict."""
        return (f"Analyze claim stylistically: '{claim}'.\n"
                "Return JSON with 'tool_preferences' mapping each of the 13 tools to a float 0.0-1.0 "
                "representing relevance for narrative critique.")
