"""Chess engine wrapper — model loading from W&B, inference, AI move selection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.encoding import fen_to_tensor
from src.models.registry import create_model, count_parameters


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _load_env():
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
            env_file = p / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                return
    except ImportError:
        pass


def _get_wandb_api():
    """Return an authenticated wandb.Api instance."""
    _load_env()
    import wandb
    api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WEIGHTS_AND_BIASES_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    return wandb.Api()


def list_wandb_artifacts(
    project: str = "bdh-searchless-chess",
    entity: str | None = None,
) -> list[dict]:
    """List model artifacts from W&B.

    Returns list of dicts with keys:
        name, version, full_name, architecture, epoch, val_loss, created_at, size_bytes
    """
    api = _get_wandb_api()
    full_project = f"{entity}/{project}" if entity else project
    results = []

    try:
        artifact_type = api.artifact_type("model", full_project)
        for collection in artifact_type.collections():
            for version in collection.versions():
                meta = version.metadata or {}
                # Try to extract architecture from the artifact's logged run config
                arch = meta.get("architecture", None)
                config = None
                if not arch:
                    try:
                        run = version.logged_by()
                        if run and run.config:
                            config = run.config
                            arch = run.config.get("model", {}).get("architecture", "unknown")
                    except Exception:
                        arch = "unknown"
                results.append({
                    "name": version.name,
                    "version": version.version,
                    "full_name": version.qualified_name,
                    "architecture": arch,
                    "epoch": meta.get("epoch"),
                    "val_loss": meta.get("val_loss"),
                    "created_at": str(version.created_at) if hasattr(version, "created_at") else None,
                    "size_bytes": version.size,
                    "run_config": config,
                })
    except Exception as e:
        raise RuntimeError(f"Failed to list W&B artifacts: {e}") from e

    return results


def download_artifact(full_name: str, download_dir: str = "./model_cache") -> Path:
    """Download a W&B artifact and return the path to the checkpoint file."""
    api = _get_wandb_api()
    artifact = api.artifact(full_name)
    artifact_dir = Path(artifact.download(root=download_dir))
    # Find the .pt file inside the artifact directory
    pt_files = list(artifact_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in artifact {full_name}")
    return pt_files[0]


# ---------------------------------------------------------------------------
# Chess Engine
# ---------------------------------------------------------------------------

class ChessEngine:
    """Wraps a trained chess model for inference and move selection."""

    def __init__(self):
        self.model: nn.Module | None = None
        self.model_info: dict = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> dict:
        """Load model from a local .pt checkpoint file.

        Returns dict with model info (architecture, params, epoch, val_loss).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        model_config = config.get("model", {})

        if not model_config.get("architecture"):
            raise ValueError("Checkpoint does not contain model architecture in config")

        model = create_model(model_config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.model_info = {
            "architecture": model_config["architecture"],
            "params": count_parameters(model),
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "best_val_loss": ckpt.get("best_val_loss"),
            "checkpoint_path": str(checkpoint_path),
        }
        return self.model_info

    def load_from_wandb(self, full_name: str, cache_dir: str = "./model_cache") -> dict:
        """Download artifact from W&B and load model. Returns model info."""
        pt_path = download_artifact(full_name, download_dir=cache_dir)
        return self.load_from_checkpoint(pt_path)

    @torch.no_grad()
    def evaluate_position(self, fen: str) -> dict:
        """Evaluate a single position.

        Returns dict with:
            score: float in [-1, 1] from side-to-move perspective
            score_white: float in [-1, 1] from white's perspective
            centipawns: approximate centipawns from white's perspective
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        board = chess.Board(fen)
        tensor = fen_to_tensor(fen, always_white_perspective=True)
        batch = np.array([tensor], dtype=np.float32)
        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)

        score = float(self.model(batch_tensor).cpu().item())

        # Score is from side-to-move perspective. Convert to white perspective.
        score_white = score if board.turn == chess.WHITE else -score

        # Approximate centipawns via inverse tanh (clamp to avoid inf)
        clamped = max(-0.9999, min(0.9999, score_white))
        centipawns = int(np.arctanh(clamped) * 1000)

        return {
            "score": score,
            "score_white": score_white,
            "centipawns": centipawns,
        }

    @torch.no_grad()
    def get_ai_move(
        self,
        fen: str,
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> dict:
        """Select an AI move by evaluating all legal moves in batch.

        For each legal move:
          1. Push move, encode resulting board, pop move
          2. Batch-evaluate all resulting positions
          3. Negate scores (opponent perspective → our perspective)
          4. Select top-k, apply softmax with temperature, sample one

        Returns dict with:
            move_uci: selected move in UCI format
            move_san: selected move in SAN format
            all_evals: list of {move_uci, move_san, score, score_white} for all legal moves
            position_eval: eval of position after the selected move
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return {"move_uci": None, "move_san": None, "all_evals": [], "position_eval": None}

        # Encode all post-move positions
        tensors = []
        san_moves = []
        for move in legal_moves:
            san_moves.append(board.san(move))
            board.push(move)
            t = fen_to_tensor(board.fen(), always_white_perspective=True)
            tensors.append(t)
            board.pop()

        batch = np.array(tensors, dtype=np.float32)
        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        raw_scores = self.model(batch_tensor).cpu().numpy().flatten()

        # raw_scores are from opponent's (next side-to-move) perspective
        # Negate to get our perspective (we want moves that are BAD for opponent)
        our_scores = -raw_scores

        # Build all_evals list
        side_is_white = board.turn == chess.WHITE
        all_evals = []
        for i, move in enumerate(legal_moves):
            # score_white: from white's perspective for the position AFTER move
            # raw_scores[i] is from side-to-move perspective AFTER move
            # After our move, opponent is to move, so raw_scores[i] is opponent perspective
            # opponent perspective score for white: if we are white, opponent is black,
            # so score_white = -raw_scores[i] = our_scores[i]
            # if we are black, opponent is white, so score_white = raw_scores[i]
            score_white = our_scores[i] if side_is_white else raw_scores[i]
            clamped = float(max(-0.9999, min(0.9999, score_white)))
            cp = int(np.arctanh(clamped) * 1000)
            all_evals.append({
                "move_uci": move.uci(),
                "move_san": san_moves[i],
                "score": float(our_scores[i]),
                "score_white": float(score_white),
                "centipawns": cp,
            })

        # Sort by our_scores descending (best moves first from our perspective)
        indices = np.argsort(-our_scores)
        all_evals_sorted = [all_evals[i] for i in indices]

        # Top-k selection with temperature sampling
        k = min(top_k, len(legal_moves))
        top_indices = indices[:k]
        top_scores = our_scores[top_indices]

        # Softmax with temperature
        if temperature <= 0:
            # Greedy
            selected_idx = top_indices[0]
        else:
            logits = top_scores / temperature
            logits -= logits.max()  # numerical stability
            probs = np.exp(logits) / np.exp(logits).sum()
            selected_idx = int(np.random.choice(top_indices, p=probs))

        selected_move = legal_moves[selected_idx]

        # Position eval after selected move
        position_eval = all_evals[selected_idx]

        return {
            "move_uci": selected_move.uci(),
            "move_san": san_moves[selected_idx],
            "all_evals": all_evals_sorted,
            "position_eval": position_eval,
        }
