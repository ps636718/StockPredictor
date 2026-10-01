"""
AlphaTrade AI — Root ASGI Application Entrypoint
Enables starting the backend from repository root:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from backend.app import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
