import uvicorn
import sys
import os
from pathlib import Path

# Ensure the root directory is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.main import app

def main():
    port = int(os.environ.get("SERVER_PORT", 7860))
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    uvicorn.run("server.app:app", host=host, port=port)

if __name__ == "__main__":
    main()
