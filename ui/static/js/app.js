/* =============================================
   Searchless Chess UI — Frontend Game Logic
   ============================================= */

// ---- State ----
let board = null;           // chessboard.js instance
let game = new Chess();     // chess.js instance (legal move engine)
let playerColor = 'white';  // 'white' or 'black'
let modelLoaded = false;
let aiThinking = false;
let moveHistory = [];       // [{moveNum, white, black, fenAfterWhite, fenAfterBlack}]
let boardFlipped = false;

// ---- DOM refs ----
const $board = $('#board');
const $moveHistory = $('#move-history');
const $aiAnalysis = $('#ai-analysis');
const $evalBarWhite = $('#eval-bar-white');
const $evalText = $('#eval-text');
const $evalCp = $('#eval-cp');
const $progressContainer = $('#progress-container');
const $progressBar = $('#progress-bar');
const $progressText = $('#progress-text');
const $modelStatus = $('#model-status');
const $noModelOverlay = $('#no-model-overlay');
const $gameOverlay = $('#game-overlay');
const $overlayMessage = $('#overlay-message');
const $thinkingIndicator = $('#thinking-indicator');
const $selectArtifact = $('#select-artifact');

// ---- Init ----
$(document).ready(function () {
    initBoard();
    bindEvents();
    checkStatus();
});

// ===================== BOARD INIT =====================

function initBoard() {
    const cfg = {
        draggable: true,
        position: 'start',
        pieceTheme: 'https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/img/chesspieces/wikipedia/{piece}.png',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
    };
    board = Chessboard('board', cfg);
}

// ===================== EVENT BINDINGS =====================

function bindEvents() {
    $('#btn-fetch-models').on('click', fetchModels);
    $('#btn-load-model').on('click', loadWandbModel);
    $('#btn-load-local').on('click', loadLocalModel);
    $('#btn-new-game').on('click', newGame);
    $('#btn-overlay-new').on('click', () => { $gameOverlay.hide(); newGame(); });
    $('#btn-undo').on('click', undoMove);
    $('#btn-flip').on('click', flipBoard);
    $('#btn-play-white').on('click', () => setPlayerColor('white'));
    $('#btn-play-black').on('click', () => setPlayerColor('black'));

    $('#slider-topk').on('input', function () {
        $('#topk-value').text(this.value);
    });
    $('#slider-temp').on('input', function () {
        $('#temp-value').text(parseFloat(this.value).toFixed(1));
    });
}

// ===================== MODEL MANAGEMENT =====================

async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.model_loaded) {
            modelLoaded = true;
            setModelStatus('loaded', formatModelInfo(data.model_info));
            $noModelOverlay.hide();
        }
    } catch (e) {
        console.error('Status check failed:', e);
    }
}

async function fetchModels() {
    const project = $('#wandb-project').val().trim();
    const entity = $('#wandb-entity').val().trim();
    if (!project) { showToast('Enter a W&B project name', 'error'); return; }

    $('#btn-fetch-models').prop('disabled', true).text('Fetching...');

    try {
        let url = `/api/models?project=${encodeURIComponent(project)}`;
        if (entity) url += `&entity=${encodeURIComponent(entity)}`;
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to fetch models');
        }
        const data = await res.json();
        populateArtifactDropdown(data.artifacts);
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        $('#btn-fetch-models').prop('disabled', false).text('Fetch Models');
    }
}

function populateArtifactDropdown(artifacts) {
    $selectArtifact.empty();
    if (!artifacts.length) {
        $selectArtifact.append('<option value="">No artifacts found</option>');
        $selectArtifact.prop('disabled', true);
        $('#btn-load-model').prop('disabled', true);
        showToast('No model artifacts found in project', 'info');
        return;
    }

    $selectArtifact.append('<option value="">— select a model —</option>');
    artifacts.forEach(a => {
        const arch = a.architecture || '?';
        const epoch = a.epoch != null ? `ep${a.epoch}` : '';
        const vl = a.val_loss != null ? `loss=${a.val_loss.toFixed(4)}` : '';
        const label = `[${arch}] ${a.name} ${epoch} ${vl}`.trim();
        $selectArtifact.append(`<option value="${a.full_name}">${label}</option>`);
    });
    $selectArtifact.prop('disabled', false);
    $('#btn-load-model').prop('disabled', false);
    showToast(`Found ${artifacts.length} model artifact(s)`, 'success');
}

async function loadWandbModel() {
    const fullName = $selectArtifact.val();
    if (!fullName) { showToast('Select a model artifact first', 'error'); return; }
    await loadModelSSE('/api/load-wandb', { full_name: fullName });
}

async function loadLocalModel() {
    const path = $('#local-path').val().trim();
    if (!path) { showToast('Enter a checkpoint path', 'error'); return; }
    await loadModelSSE('/api/load-local', { path: path });
}

async function loadModelSSE(url, body) {
    setModelStatus('loading', 'Loading...');
    showProgress('Initializing...');

    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line

            let eventType = null;
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ') && eventType) {
                    const data = JSON.parse(line.slice(6));
                    handleSSEEvent(eventType, data);
                    eventType = null;
                }
            }
        }
    } catch (e) {
        setModelStatus('error', 'Load failed');
        hideProgress();
        showToast(`Failed to load model: ${e.message}`, 'error');
    }
}

function handleSSEEvent(event, data) {
    if (event === 'progress') {
        const pct = data.stage === 'downloading' ? 30 :
                    data.stage === 'loading' ? 50 :
                    data.stage === 'loaded' ? 100 : 0;
        updateProgress(pct, data.message);
    } else if (event === 'done') {
        modelLoaded = true;
        setModelStatus('loaded', formatModelInfo(data.model_info));
        hideProgress();
        $noModelOverlay.hide();
        showToast('Model loaded successfully!', 'success');
        // If playing as black, trigger AI first move
        if (playerColor === 'black' && game.turn() === 'w') {
            triggerAIMove();
        }
    } else if (event === 'error') {
        modelLoaded = false;
        setModelStatus('error', 'Load failed');
        hideProgress();
        showToast(data.message, 'error');
    }
}

function formatModelInfo(info) {
    if (!info) return 'Unknown';
    const arch = info.architecture || '?';
    const params = info.params ? `${(info.params / 1e6).toFixed(1)}M` : '';
    const ep = info.epoch != null ? `ep${info.epoch}` : '';
    return `${arch} ${params} ${ep}`.trim();
}

// ===================== CHESS GAME LOGIC =====================

function onDragStart(source, piece, position, orientation) {
    // Don't allow moves if game is over, AI is thinking, or no model
    if (game.game_over()) return false;
    if (aiThinking) return false;
    if (!modelLoaded) return false;

    // Only allow the player to move their own pieces
    if (playerColor === 'white' && piece.search(/^b/) !== -1) return false;
    if (playerColor === 'black' && piece.search(/^w/) !== -1) return false;

    // Only move when it's the player's turn
    if (playerColor === 'white' && game.turn() !== 'w') return false;
    if (playerColor === 'black' && game.turn() !== 'b') return false;

    return true;
}

function onDrop(source, target, piece, newPos, oldPos, orientation) {
    // Try the move — chess.js validates legality
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q',  // always promote to queen (simplification)
    });

    if (move === null) return 'snapback'; // illegal move

    updateMoveHistory();
    evaluateAndUpdateBar();

    // Check if game is over after player move
    if (checkGameOver()) return;

    // Trigger AI move
    triggerAIMove();
}

function onSnapEnd() {
    board.position(game.fen());
}

async function triggerAIMove() {
    if (!modelLoaded || game.game_over()) return;

    aiThinking = true;
    $thinkingIndicator.show();

    const topK = parseInt($('#slider-topk').val());
    const temperature = parseFloat($('#slider-temp').val());

    try {
        const res = await fetch('/api/ai-move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                fen: game.fen(),
                top_k: topK,
                temperature: temperature,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'AI move failed');
        }

        const data = await res.json();

        if (data.move_uci) {
            // Apply AI move
            const from = data.move_uci.substring(0, 2);
            const to = data.move_uci.substring(2, 4);
            const promotion = data.move_uci.length > 4 ? data.move_uci[4] : undefined;

            const move = game.move({ from, to, promotion });
            if (move) {
                board.position(game.fen());
                updateMoveHistory();
                updateEvalBar(data.position_eval);
                displayAIAnalysis(data.all_evals);
                checkGameOver();
            }
        }
    } catch (e) {
        showToast(`AI error: ${e.message}`, 'error');
    } finally {
        aiThinking = false;
        $thinkingIndicator.hide();
    }
}

// ===================== EVAL BAR =====================

async function evaluateAndUpdateBar() {
    if (!modelLoaded) return;
    try {
        const res = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fen: game.fen() }),
        });
        if (res.ok) {
            const data = await res.json();
            updateEvalBar({
                score_white: data.score_white,
                centipawns: data.centipawns,
            });
        }
    } catch (e) {
        console.error('Eval failed:', e);
    }
}

function updateEvalBar(evalData) {
    if (!evalData) return;

    const scoreWhite = evalData.score_white;
    const cp = evalData.centipawns;

    // Map score_white [-1, 1] to bar fill percentage [0%, 100%]
    const fillPct = Math.max(0, Math.min(100, (scoreWhite + 1) / 2 * 100));
    $evalBarWhite.css('height', fillPct + '%');

    // Display text
    const sign = scoreWhite >= 0 ? '+' : '';
    $evalText.text(sign + scoreWhite.toFixed(2));
    $evalCp.text((cp >= 0 ? '+' : '') + cp + ' cp');

    // Color the text
    $evalText.css('color', scoreWhite >= 0 ? 'var(--success)' : 'var(--danger)');
}

// ===================== AI ANALYSIS PANEL =====================

function displayAIAnalysis(allEvals) {
    if (!allEvals || !allEvals.length) {
        $aiAnalysis.html('<p class="muted">No analysis available</p>');
        return;
    }

    // Show top 10 moves
    const top = allEvals.slice(0, 10);
    let html = '';
    top.forEach((ev, i) => {
        const cpSign = ev.centipawns >= 0 ? '+' : '';
        const cpClass = ev.centipawns >= 0 ? 'eval-positive' : 'eval-negative';
        html += `<div class="analysis-move">
            <span class="move-name">${i + 1}. ${ev.move_san}</span>
            <span class="move-eval ${cpClass}">${cpSign}${ev.centipawns} cp</span>
        </div>`;
    });
    $aiAnalysis.html(html);
}

// ===================== MOVE HISTORY =====================

function updateMoveHistory() {
    const history = game.history();
    let html = '';
    for (let i = 0; i < history.length; i += 2) {
        const moveNum = Math.floor(i / 2) + 1;
        const whiteMove = history[i] || '';
        const blackMove = history[i + 1] || '';
        const isLast = (i + 2 >= history.length);
        html += `<div class="move-row ${isLast ? 'move-highlight' : ''}">
            <span class="move-number">${moveNum}.</span>
            <span class="move-white">${whiteMove}</span>
            <span class="move-black">${blackMove}</span>
        </div>`;
    }
    $moveHistory.html(html);
    // Auto-scroll to bottom
    $moveHistory.scrollTop($moveHistory[0].scrollHeight);
}

// ===================== GAME CONTROLS =====================

function newGame() {
    game.reset();
    board.start();
    moveHistory = [];
    $moveHistory.html('');
    $aiAnalysis.html('<p class="muted">Make a move to see AI analysis</p>');
    $gameOverlay.hide();
    resetEvalBar();

    // If playing as black and model loaded, AI makes first move
    if (modelLoaded && playerColor === 'black') {
        triggerAIMove();
    }
}

function undoMove() {
    if (aiThinking) return;
    if (game.history().length === 0) return;

    // Undo two half-moves (player + AI), or one if only one move was made
    game.undo(); // undo AI move (or player's if AI hasn't moved)
    if (game.history().length > 0) {
        // Check if it's now the player's turn; if not, undo one more
        const isPlayerTurn = (playerColor === 'white' && game.turn() === 'w') ||
                             (playerColor === 'black' && game.turn() === 'b');
        if (!isPlayerTurn) {
            game.undo();
        }
    }

    board.position(game.fen());
    updateMoveHistory();
    $gameOverlay.hide();
    evaluateAndUpdateBar();
}

function flipBoard() {
    board.flip();
    boardFlipped = !boardFlipped;
}

function setPlayerColor(color) {
    if (aiThinking) return;

    playerColor = color;
    $('#btn-play-white').toggleClass('active', color === 'white');
    $('#btn-play-black').toggleClass('active', color === 'black');

    // Set board orientation
    board.orientation(color);

    // Start new game with new color
    newGame();
}

function resetEvalBar() {
    $evalBarWhite.css('height', '50%');
    $evalText.text('0.00').css('color', 'var(--text-primary)');
    $evalCp.text('0 cp');
}

// ===================== GAME OVER =====================

function checkGameOver() {
    if (!game.game_over()) return false;

    let message = '';
    if (game.in_checkmate()) {
        const winner = game.turn() === 'w' ? 'Black' : 'White';
        message = `Checkmate! ${winner} wins!`;
    } else if (game.in_stalemate()) {
        message = 'Stalemate! Draw.';
    } else if (game.in_threefold_repetition()) {
        message = 'Draw by repetition.';
    } else if (game.in_draw()) {
        message = 'Draw!';
    }

    if (message) {
        $overlayMessage.text(message);
        $gameOverlay.show();
    }
    return true;
}

// ===================== UI HELPERS =====================

function setModelStatus(status, text) {
    $modelStatus
        .removeClass('status-none status-loading status-loaded status-error')
        .addClass('status-' + status)
        .text(text);
}

function showProgress(text) {
    $progressContainer.show();
    $progressBar.css('width', '10%');
    $progressText.text(text);
}

function updateProgress(pct, text) {
    $progressBar.css('width', pct + '%');
    $progressText.text(text);
}

function hideProgress() {
    setTimeout(() => {
        $progressContainer.hide();
        $progressBar.css('width', '0%');
    }, 500);
}

// Toast notifications
let toastContainer = null;

function showToast(message, type = 'info') {
    if (!toastContainer) {
        toastContainer = $('<div id="toast-container"></div>').appendTo('body');
    }
    const toast = $(`<div class="toast toast-${type}">${escapeHtml(message)}</div>`);
    toastContainer.append(toast);
    setTimeout(() => toast.fadeOut(300, () => toast.remove()), 4000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
