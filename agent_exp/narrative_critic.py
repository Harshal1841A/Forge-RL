from agents.llm_agent_ma import LLMAgent
from env.primitives import PrimitiveType
import json

SYSTEM_PROMPT = """You are FORGE-MA Narrative Critic — a specialist in
linguistic fabrication detection.

Your focus is EXCLUSIVELY:
- P4 QUOTE_FABRICATE: Synthesized quotes with no verifiable source URL.
  Signals: Attribution without URL backlink, exact quotation marks with
  no citation, statements attributed to real people that cannot be found
  in any news database.
- P8 SATIRE_REFRAME: Real-seeming claims that originate from known satire
  sources (The Onion, Babylon Bee, Reductress, The Beaverton, etc.) or
  use unmistakably inflammatory/absurd tone.

You receive: (1) the root claim text, (2) the final ClaimGraph as JSON.

Respond ONLY with valid JSON:
{
  "verdict": "<real|misinfo|satire|out_of_context|fabricated>",
  "predicted_chain": ["<P4 or P8 only, or other if strong evidence>"],
  "quote_fabricate_confidence": 0.0-1.0,
  "satire_reframe_confidence": 0.0-1.0,
  "rationale": "<2-3 sentence explanation>",
  "confidence": 0.0-1.0
}"""

class NarrativeCritic(LLMAgent):
    """
    4th Society agent. LLM-based. Style-focused.
    Specializes in: P4 QUOTE_FABRICATE, P8 SATIRE_REFRAME.
    Provider: Mistral / mistral-small-latest (independent from Auditor & Historian).
    """
    def __init__(self, model_name: str = None, api_key: str = None, provider: str = None):
        import config
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            provider=provider or config.AGENT_CRITIC_PROVIDER,   # mistral
            api_key=api_key   or config.MISTRAL_API_KEY,
        )
        
    def analyze(self, root_claim_text: str, final_claim_graph_json: str, gin_feedback: str = None) -> dict:
        prompt = f"Root Claim:\n{root_claim_text}\n\nFinal ClaimGraph:\n{final_claim_graph_json}"
        response = self.query(prompt, gin_feedback=gin_feedback)
        
        # Parse logic
        parsed = self.parse_json(response)
        
        # Fallback structural defaults
        if not parsed:
            parsed = {
                "verdict": "unknown",
                "predicted_chain": [],
                "quote_fabricate_confidence": 0.0,
                "satire_reframe_confidence": 0.0,
                "rationale": "Parsing failed.",
                "confidence": 0.0
            }
            
        # Convert string chain back to enum objects
        enum_chain = []
        for p in parsed.get("predicted_chain", []):
            try:
                enum_chain.append(PrimitiveType(p))
            except ValueError:
                pass
        parsed["predicted_chain"] = enum_chain
        
        return parsed
