import json
import logging
from typing import Optional
import config

logger = logging.getLogger(__name__)

# ── Provider registry — all use OpenAI-compatible API ──────────────────────────
_PROVIDER_CONFIG = {
    "groq": {
        "base_url": config.API_BASE_URL,
        "api_key":  config.OPENAI_API_KEY,
        "model":    config.MODEL_NAME,
    },
    "cerebras": {
        "base_url": config.CEREBRAS_BASE_URL,
        "api_key":  config.CEREBRAS_API_KEY,
        "model":    config.CEREBRAS_MODEL,
    },
    "mistral": {
        "base_url": config.MISTRAL_BASE_URL,
        "api_key":  config.MISTRAL_API_KEY,
        "model":    config.MISTRAL_MODEL,
    },
    "openrouter": {
        "base_url": config.OPENROUTER_BASE_URL,
        "api_key":  config.OPENROUTER_API_KEY,
        "model":    config.OPENROUTER_MODEL,
    },
}


class LLMAgent:
    """
    Multi-provider LLM Agent.
    Each forensic role is backed by a different AI to prevent shared-model bias.
      Forensic Auditor   → Groq   (Llama 3 70B)
      Context Historian  → Cerebras (Llama 3.1 70B)
      Narrative Critic   → Mistral  (mistral-small-latest)
      NegotiatedSearch   → OpenRouter (Llama 3 8B free)
    Falls back to deterministic mock logic when API call fails.
    """
    def __init__(
        self,
        system_prompt: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        self.history = []

        # Resolve provider config
        prov = provider or "groq"
        prov_cfg = _PROVIDER_CONFIG.get(prov, _PROVIDER_CONFIG["groq"])

        self.provider    = prov
        self.model_name  = model_name or prov_cfg["model"]
        self.api_key     = api_key    or prov_cfg["api_key"] or config.OPENAI_API_KEY
        self.base_url    = base_url   or prov_cfg["base_url"]

        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            
            # Validate API key presence
            if not self.api_key or self.api_key == "dummy-key":
                logger.warning(
                    f"[LLMAgent/{self.provider}] WARNING: api_key is empty or dummy! "
                    f"Will fall back to mock responses."
                )
            else:
                logger.info(
                    f"[LLMAgent/{self.provider}] api_key validated (len={len(self.api_key)})"
                )
            
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "dummy-key",
            )
            logger.info(
                f"[LLMAgent] provider={self.provider} model={self.model_name} "
                f"base={self.base_url} (api_key_set={bool(self.api_key and self.api_key != 'dummy-key')})"
            )
        except ImportError:
            logger.warning("openai package not installed. Run: pip install openai")

    def query(self, user_prompt: str, gin_feedback: str = None) -> str:
        """
        Queries the assigned AI provider.  Falls back to deterministic mock on failure.
        """
        prompt = user_prompt
        if gin_feedback:
            prompt += f"\n\n[GIN Topology Hint]: {gin_feedback}"

        self.history.append({"role": "user", "content": prompt})

        # ── Live API call ─────────────────────────────────────────────────────
        if self._client:
            messages = [{"role": "system", "content": self.system_prompt}] + self.history
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                )
                response = resp.choices[0].message.content
                self.history.append({"role": "assistant", "content": response})
                logger.debug(f"[{self.provider}] response: {response[:120]}")
                return response
            except Exception as e:
                logger.warning(
                    f"[LLMAgent] {self.provider} call failed ({e}). Using mock fallback."
                )

        # ── Deterministic mock fallback ───────────────────────────────────────
        prompt_lower = prompt.lower()
        verdict    = "unknown"
        confidence = 0.5
        rationale  = "The evidence is inconclusive."
        chain      = []

        if any(kw in prompt_lower for kw in ["2015", "2016", "2017", "2018", "2019",
                                               "misdate", "temporal", "mislabelled",
                                               "mislabeled", "old video", "repost"]):
            verdict, confidence = "misinfo", 0.85
            rationale = "Identified temporal manipulation regarding dates."
            # MEDIUM BUG 2 FIX: use PrimitiveType enum members, NOT lowercase strings.
            # PrimitiveType("temporal_shift") raises ValueError; the silent except
            # was leaving the mock chain empty, defeating the fallback's purpose.
            from env.primitives import PrimitiveType as _PT
            if _PT.TEMPORAL_SHIFT not in chain:
                chain.append(_PT.TEMPORAL_SHIFT)
            if "historian" in (self.model_name or "").lower():
                confidence = 0.95

        if any(kw in prompt_lower for kw in ["fabricat", "fake", "retract", "leaked",
                                               "hoax", "forged"]):
            verdict, confidence = "fabricated", 0.88
            rationale = "Evidence points to fabricated quotes or retracted sources."
            from env.primitives import PrimitiveType as _PT
            if _PT.QUOTE_FABRICATE not in chain:
                chain.append(_PT.QUOTE_FABRICATE)

        if any(kw in prompt_lower for kw in ["satire", "parody", "onion", "babylon bee"]):
            verdict, confidence = "satire", 0.92
            rationale = "Context identifies the origin as a satirical publication."
            from env.primitives import PrimitiveType as _PT
            if _PT.SATIRE_REFRAME not in chain:
                chain.append(_PT.SATIRE_REFRAME)

        response = json.dumps({
            "verdict": verdict,
            "predicted_chain": chain,
            "rationale": rationale,
            "confidence": confidence,
        })
        self.history.append({"role": "assistant", "content": response})
        return response

    def parse_json(self, response: str) -> dict:
        try:
            start = response.find('{')
            end   = response.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(response[start:end])
            return json.loads(response)
        except Exception:
            return {}
