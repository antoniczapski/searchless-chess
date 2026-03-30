# Searchless Chess — Gameplay UI

Interactive chess UI for playing against your trained searchless chess models.  
Models are loaded from **Weights & Biases** artifacts or local `.pt` checkpoints.

## Quick Start

```bash
cd project/ui

# Install dependencies (if not already in your environment)
pip install -r requirements.txt

# Start the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

## Setup

### W&B Authentication

The app reads your W&B API key from environment variables. Set one of:

```bash
export WANDB_API_KEY=your_key_here
# or
export WEIGHTS_AND_BIASES_KEY=your_key_here
```

Or place it in a `.env` file anywhere in the project tree.

### Loading Models

1. **From W&B**: Enter your project name (default: `bdh-searchless-chess`), click **Fetch Models**, select an artifact, then **Load Weights**. The checkpoint will be downloaded and cached in `./model_cache/`.

2. **From local file**: Enter the absolute path to a `.pt` checkpoint and click **Load Local**.

## Features

- **Legal move enforcement** — only valid moves are allowed (chess.js client-side + python-chess server-side)
- **Position evaluation bar** — chess.com-style vertical eval bar showing the model's assessment
- **AI move selection** — evaluates all legal moves in a single batch, selects from top-K with temperature sampling
- **AI move analysis** — shows the top-10 candidate moves with centipawn evaluations
- **Full game controls** — New Game, Undo Move, Flip Board, Play as White/Black
- **Tunable AI** — adjust Top-K and Temperature sliders to control AI play style
- **Game-over detection** — checkmate, stalemate, draw by repetition, 50-move rule, insufficient material

## Architecture

```
ui/
├── app.py              # FastAPI server (routes + SSE streaming)
├── engine.py           # Chess engine wrapper (W&B download, model inference, move selection)
├── requirements.txt
├── README.md
└── static/
    ├── index.html      # Main page
    ├── css/style.css   # Dark theme styling
    └── js/app.js       # Frontend game logic (chessboard.js + chess.js)
```

**Backend** reuses `src.data.encoding.fen_to_tensor()` and `src.models.registry.create_model()` from the training codebase.  
**Frontend** uses [chessboard.js](https://chessboardjs.com/) for the board and [chess.js](https://github.com/jhlywa/chess.js) for move validation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Engine status + loaded model info |
| `GET` | `/api/models?project=...&entity=...` | List W&B model artifacts |
| `POST` | `/api/load-wandb` | Download & load W&B artifact (SSE stream) |
| `POST` | `/api/load-local` | Load local `.pt` checkpoint (SSE stream) |
| `POST` | `/api/evaluate` | Evaluate a FEN position |
| `POST` | `/api/ai-move` | Get AI move with full analysis |
