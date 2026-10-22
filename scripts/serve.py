"""Run the Worldline server locally with label editing on.

Usage: python scripts/serve.py [--port 8765]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from worldline.server import main  # noqa: E402

if __name__ == "__main__":
    sys.argv += [] if "--labels" in sys.argv else ["--labels"]
    main()
