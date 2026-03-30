"""FastAPI application for chess gameplay with trained neural network models."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import ChessEngine, list_wandb_artifacts

app = FastAPI(title="Searchless Chess UI")
engine = ChessEngine()

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    fen: str


class AIMoveRequest(BaseModel):
    fen: str
    top_k: int = 5
    temperature: float = 1.0


class LoadModelRequest(BaseModel):
    full_name: str  # W&B artifact qualified name


class LoadLocalRequest(BaseModel):
    path: str  # local .pt checkpoint path


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI page."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/status")
async def status():
    """Return current engine status and loaded model info."""
    return {
        "model_loaded": engine.is_loaded,
        "model_info": engine.model_info if engine.is_loaded else None,
        "device": str(engine.device),
    }


@app.get("/api/models")
async def list_models(
    project: str = Query(default="bdh-searchless-chess"),
    entity: str = Query(default=None),
):
    """List available model artifacts from W&B."""
    try:
        artifacts = list_wandb_artifacts(project=project, entity=entity)
        return {"artifacts": artifacts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load-wandb")
async def load_wandb_model(req: LoadModelRequest):
    """Download and load a model from W&B artifact.

    Uses SSE-style streaming to report progress.
    """
    async def generate():
        yield _sse("progress", {"stage": "downloading", "message": "Downloading artifact from W&B..."})
        await asyncio.sleep(0)  # yield control

        try:
            # Download in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None, lambda: engine.load_from_wandb(req.full_name)
            )
            yield _sse("progress", {"stage": "loaded", "message": "Model loaded successfully!"})
            yield _sse("done", {"model_info": info})
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/load-local")
async def load_local_model(req: LoadLocalRequest):
    """Load a model from a local .pt checkpoint path."""
    async def generate():
        yield _sse("progress", {"stage": "loading", "message": "Loading checkpoint..."})
        await asyncio.sleep(0)

        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None, lambda: engine.load_from_checkpoint(req.path)
            )
            yield _sse("progress", {"stage": "loaded", "message": "Model loaded successfully!"})
            yield _sse("done", {"model_info": info})
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/evaluate")
async def evaluate_position(req: EvalRequest):
    """Evaluate a board position."""
    if not engine.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: engine.evaluate_position(req.fen))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai-move")
async def ai_move(req: AIMoveRequest):
    """Get AI move for the given position."""
    if not engine.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: engine.get_ai_move(req.fen, req.top_k, req.temperature)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
