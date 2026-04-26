"""
Publishes the blog post from docs/blog_post.md to HuggingFace.
Run: python scripts/publish_blog.py

Requires: pip install huggingface_hub
Set: export HF_TOKEN=your_hf_token
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Run: pip install huggingface_hub")
    sys.exit(1)

HF_TOKEN = os.getenv("HF_TOKEN", "")
USERNAME = os.getenv("HF_USERNAME", "your-username")
BLOG_PATH = Path(__file__).parent.parent / "docs" / "blog_post.md"

if not HF_TOKEN:
    print("Set HF_TOKEN environment variable first")
    sys.exit(1)

if not BLOG_PATH.exists():
    print(f"Blog post not found at {BLOG_PATH}")
    sys.exit(1)

# Load real baseline numbers into the blog post
try:
    import json
    results = json.loads((Path(__file__).parent.parent / "baselines" / "results.json").read_text())
    baselines = results.get("forge_ma_baselines", {})
    v0_ted = baselines.get("v0_heuristic", {}).get("mean_ted", "0.XX")
    v1_ted = baselines.get("v1_llm", {}).get("mean_ted", "0.XX")
    print(f"Injecting real numbers: v0={v0_ted}, v1={v1_ted}")
except Exception:
    v0_ted, v1_ted = "0.XX", "0.XX"
    print("Baseline numbers not available yet — using placeholders")

blog_content = BLOG_PATH.read_text()
blog_content = blog_content.replace("V0_TED_PLACEHOLDER", str(v0_ted))
blog_content = blog_content.replace("V1_TED_PLACEHOLDER", str(v1_ted))

# Upload to HuggingFace blog
api = HfApi(token=HF_TOKEN)
repo_id = f"{USERNAME}/FORGE-RL"

try:
    api.upload_file(
        path_or_fileobj=blog_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    print(f"\nBlog post published!")
    print(f"URL: https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f"Upload failed: {e}")
    print("Manual option: paste blog_post.md into HuggingFace Space README")
