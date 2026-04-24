"""
FORGE — Forensic RL Graph Environment
Global Configuration (100% free-tier / open-source)
"""

# dataclasses import removed — was only used incorrectly for POLICY_HIDDEN_DIMS
from typing import List
import os

from dotenv import load_dotenv
# Load .env file automatically if it exists (for local development)
load_dotenv()

# ─── LLM — Multi-Provider (no shared model bias) ──────────────────────────────
# Groq  → Forensic Auditor  (llama3-70b, free at console.groq.com)
HF_TOKEN: str          = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY: str    = os.getenv("OPENAI_API_KEY", "")          # Groq key
OPENAI_API_KEY_AUDITOR: str = os.getenv("OPENAI_API_KEY_AUDITOR", OPENAI_API_KEY)
API_BASE_URL: str      = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str        = os.getenv("MODEL_NAME", "llama3-70b-8192")

# Cerebras → Context Historian (llama3.1-70b, free at cloud.cerebras.ai)
CEREBRAS_API_KEY: str  = os.getenv("CEREBRAS_API_KEY", OPENAI_API_KEY)
CEREBRAS_BASE_URL: str = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL: str    = os.getenv("CEREBRAS_MODEL", "llama3.1-70b")

# Mistral → Narrative Critic (mistral-small-latest, free at console.mistral.ai)
MISTRAL_API_KEY: str   = os.getenv("MISTRAL_API_KEY", OPENAI_API_KEY)
MISTRAL_BASE_URL: str  = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
MISTRAL_MODEL: str     = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# OpenRouter → NegotiatedSearch agents (free models, free at openrouter.ai)
OPENROUTER_API_KEY: str  = os.getenv("OPENROUTER_API_KEY", OPENAI_API_KEY)
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL: str    = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free")

# Per-agent provider routing
AGENT_AUDITOR_PROVIDER: str    = os.getenv("AGENT_AUDITOR_PROVIDER", "groq")
AGENT_HISTORIAN_PROVIDER: str  = os.getenv("AGENT_HISTORIAN_PROVIDER", "cerebras")
AGENT_CRITIC_PROVIDER: str     = os.getenv("AGENT_CRITIC_PROVIDER", "mistral")
AGENT_NEGOTIATED_PROVIDER: str = os.getenv("AGENT_NEGOTIATED_PROVIDER", "openrouter")

# Legacy aliases — kept for backward compat
OPENAI_API_KEY_HISTORIAN: str = os.getenv("OPENAI_API_KEY_HISTORIAN", CEREBRAS_API_KEY)
OPENAI_API_KEY_CRITIC: str    = os.getenv("OPENAI_API_KEY_CRITIC", MISTRAL_API_KEY)

HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # local, free


# ─── Free External APIs ───────────────────────────────────────────────────────
WIKIPEDIA_API_URL: str = "https://en.wikipedia.org/api/rest_v1"
WAYBACK_API_URL: str = "http://archive.org/wayback/available"
WIKIDATA_SPARQL_URL: str = "https://query.wikidata.org/sparql"
MEDIASTACK_API_KEY: str = os.getenv("MEDIASTACK_API_KEY", "")  # free 500/month
GNEWS_API_KEY: str = os.getenv("GNEWS_API_KEY", "")            # free 100/day

# ─── Environment ──────────────────────────────────────────────────────────────
MAX_EPISODE_STEPS: int = 20
BASE_EPISODE_STEPS: int = 12
STEP_COMPLEXITY_BONUS: int = 3       # extra steps per extra tactic
TOOL_CALL_TIMEOUT_SEC: float = 8.0
TOOL_CACHE_TTL_SEC: int = 21600      # 6 hours

# ─── Reward Shaping ───────────────────────────────────────────────────────────
# Clip bounds (open interval for TED scorer compliance)
REWARD_CLIP_MIN: float = -1.0
REWARD_CLIP_MAX: float = 1.0

# Terminal verdict rewards
REWARD_CORRECT_VERDICT: float    = 0.75
REWARD_WRONG_VERDICT: float      = -0.40
REWARD_FALSE_POSITIVE: float     = -0.20   # extra penalty: misinfo flagged on real content

# Step-level signals
REWARD_STEP_PENALTY: float          = -0.02   # small cost per action taken
REWARD_DUPLICATE_TOOL_PENALTY: float = -0.10  # penalty for repeated tool call

# Manipulation detection signals
REWARD_MANIPULATION_FLAG: float    =  0.15   # correct flag of manipulation
REWARD_MANIPULATION_PENALTY: float = -0.10   # missed or false manipulation flag

# ─── Potential-Based Shaping Weights Φ(s) ─────────────────────────────────────
# Φ(s) = w1·coverage + w2·diversity + w3·contradiction_area + w4·network_diameter
POTENTIAL_W1: float = 0.40   # evidence coverage weight
POTENTIAL_W2: float = 0.25   # source diversity weight
POTENTIAL_W3: float = 0.20   # contradiction surface area weight
POTENTIAL_W4: float = 0.15   # network diameter weight


# ─── RL Training ──────────────────────────────────────────────────────────────
PPO_LR: float = 3e-4
PPO_GAMMA: float = 0.99
PPO_GAE_LAMBDA: float = 0.95
PPO_CLIP_EPS: float = 0.2
PPO_ENTROPY_COEF: float = 0.01
PPO_VF_COEF: float = 0.5
PPO_TRAIN_BATCH: int = 2048
PPO_MINI_BATCH: int = 256
PPO_EPOCHS: int = 10
PPO_NUM_WORKERS: int = 4

# ─── GNN Policy ───────────────────────────────────────────────────────────────
GNN_HIDDEN_DIM: int = 128
GNN_NUM_LAYERS: int = 3
GNN_HEADS: int = 4               # for GAT
CLAIM_EMBED_DIM: int = 384       # all-MiniLM output dim
MAX_OBSERVATION_NODES: int = 10  # max graph nodes embedded in v2.0 multimodal obs
POLICY_HIDDEN_DIMS: List[int] = [256, 128]

# ─── Curriculum ───────────────────────────────────────────────────────────────
CURRICULUM_STAGES: List[dict] = [
    {"name": "stage0", "max_tactics": 1, "noisy_tools": False, "budget_mult": 1.5},
    {"name": "stage1", "max_tactics": 2, "noisy_tools": False, "budget_mult": 1.2},
    {"name": "stage2", "max_tactics": 3, "noisy_tools": True, "budget_mult": 1.0},
    {"name": "stage3", "max_tactics": 4, "noisy_tools": True, "budget_mult": 0.8},
]
CURRICULUM_GATE_REWARD: float = 0.70   # must achieve this mean reward to advance

# ─── Self-Play ────────────────────────────────────────────────────────────────
GENERATOR_POPULATION_SIZE: int = 8     # reduced for free compute
ELO_INITIAL: int = 1200
ELO_K_FACTOR: int = 32

# ─── Server ───────────────────────────────────────────────────────────────────
SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 7860
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./forge.db")  # SQLite = free

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"


# ─── API Key Validation & Logging (CRITICAL for debugging multi-provider issues) ──
import logging
_logger = logging.getLogger(__name__)

def _validate_api_keys():
    """
    Validate all provider API keys on config load.
    Logs warnings if keys are missing or defaulting unexpectedly.
    """
    _logger.info("[config] Initializing multi-provider key validation...")
    
    # Check primary keys
    if not OPENAI_API_KEY or OPENAI_API_KEY == "":
        _logger.warning("[config] OPENAI_API_KEY (Groq) not set - Auditor will fail")
    else:
        _logger.info("[config] Groq Auditor key configured")
    
    if not CEREBRAS_API_KEY or CEREBRAS_API_KEY == OPENAI_API_KEY:
        _logger.warning("[config] CEREBRAS_API_KEY not set - using fallback (Groq key)")
    else:
        _logger.info("[config] Cerebras Historian key configured")
    
    if not MISTRAL_API_KEY or MISTRAL_API_KEY == OPENAI_API_KEY:
        _logger.warning("[config] MISTRAL_API_KEY not set - using fallback (Groq key)")
    else:
        _logger.info("[config] Mistral Critic key configured")
    
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == OPENAI_API_KEY:
        _logger.warning("[config] OPENROUTER_API_KEY not set - using fallback (Groq key)")
    else:
        _logger.info("[config] OpenRouter NegotiatedSearch key configured")
    
    # Log provider routing
    _logger.info(f"[config] Provider routing: Auditor={AGENT_AUDITOR_PROVIDER}, "
                f"Historian={AGENT_HISTORIAN_PROVIDER}, Critic={AGENT_CRITIC_PROVIDER}, "
                f"NegotiatedSearch={AGENT_NEGOTIATED_PROVIDER}")

# Run validation on module load
_validate_api_keys()

