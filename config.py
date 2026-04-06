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

# ─── LLM (OpenEnv / Groq free tier API) ────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")                   # free at huggingface.co
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")       # free at console.groq.com
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3-8b-8192")
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
REWARD_CORRECT_VERDICT: float = 1.0
REWARD_WRONG_VERDICT: float = 0.0
REWARD_FALSE_POSITIVE: float = 0.0
REWARD_MANIPULATION_FLAG: float = 0.15
REWARD_STEP_PENALTY: float = -0.02
REWARD_DUPLICATE_TOOL_PENALTY: float = -0.05
POTENTIAL_W1: float = 0.4   # evidence coverage weight
POTENTIAL_W2: float = 0.3   # source diversity weight
POTENTIAL_W3: float = 0.2   # contradiction surface area
POTENTIAL_W4: float = 0.1   # reserved / network diameter (not currently used in potential fn)

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
    {"name": "stage2", "max_tactics": 3, "noisy_tools": True,  "budget_mult": 1.0},
    {"name": "stage3", "max_tactics": 4, "noisy_tools": True,  "budget_mult": 0.8},
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
