"""
AlphaTrade AI — Backend ASGI Entrypoint for Cloud Deployment (Render, Railway, Fly.io)
Enables cloud start commands:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import os
import sys

# Ensure repository root is on sys.path so 'backend.*' imports resolve cleanly
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from backend.app import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
