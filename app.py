from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import numpy as np
import math
import time
import logging
from collections import OrderedDict
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# --- Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_COUNT = 4
AI_DEPTH = 20
BASE_TIMEOUT = 8
TRANS_TABLE_SIZE = 2**24

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

# --- LimitedDict for Transposition Table ---
class LimitedDict(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)

# --- Global Variables ---
transposition_table: LimitedDict = LimitedDict(maxsize=TRANS_TABLE_SIZE)
killer_moves: Dict[int, List[int]] = {d: [] for d in range(AI_DEPTH + 1)}
history_scores: Dict[Tuple[int, int], int] = {}

# --- Helper Functions ---
def board_to_key(board: np.ndarray) -> str:
    return board.tobytes().hex()

def is_valid_location(board: np.ndarray, col: int) -> bool:
    return board[ROW_COUNT - 1, col] == 0

def get_next_open_row(board: np.ndarray, col: int) -> Optional[int]:
    for r in range(ROW_COUNT):
        if board[r, col] == 0:
            return r
    return None

def get_valid_moves(board: np.ndarray) -> List[int]:
    valid_moves = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    center_col = COLUMN_COUNT // 2
    if center_col in valid_moves:
        valid_moves.remove(center_col)
        valid_moves.insert(0, center_col)
    return valid_moves

def winning_move(board: np.ndarray, piece: int) -> bool:
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if all(board[r, c + i] == piece for i in range(WIN_COUNT)):
                return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all(board[r + i, c] == piece for i in range(WIN_COUNT)):
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all(board[r + i, c + i] == piece for i in range(WIN_COUNT)):
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all(board[r - i, c + i] == piece for i in range(WIN_COUNT)):
                return True
    return False

def has_three_piece_threat(board: np.ndarray, piece: int) -> bool:
    temp_board = board.copy()
    for c in range(COLUMN_COUNT):
        row = get_next_open_row(temp_board, c)
        if row is not None:
            temp_board[row, c] = piece
            if winning_move(temp_board, piece):
                return True
            temp_board[row, c] = 0
    return False

def terminal_node(board: np.ndarray) -> bool:
    return winning_move(board, 1) or winning_move(board, 2) or not get_valid_moves(board)

# --- Evaluation Function ---
def evaluate_window(window: List[int], piece: int) -> int:
    score = 0
    opp_piece = 3 - piece
    piece_count = window.count(piece)
    opp_count = window.count(opp_piece)
    empty_count = window.count(0)

    if piece_count == 4:
        score += 100000
    elif piece_count == 3 and empty_count == 1:
        score += 500
    elif piece_count == 2 and empty_count == 2:
        score += 50

    if opp_count == 4:
        score -= 100000
    elif opp_count == 3 and empty_count == 1:
        score -= 5000
    elif opp_count == 2 and empty_count == 2:
        score -= 10

    return score

def evaluate_board(board: np.ndarray, piece: int) -> int:
    score = 0
    center_array = [int(board[r, COLUMN_COUNT // 2]) for r in range(ROW_COUNT)]
    center_count = center_array.count(piece)
    score += center_count * 6

    for r in range(ROW_COUNT):
        row_array = [int(board[r, c]) for c in range(COLUMN_COUNT)]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WIN_COUNT]
            score += evaluate_window(window, piece)

    for c in range(COLUMN_COUNT):
        col_array = [int(board[r, c]) for r in range(ROW_COUNT)]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WIN_COUNT]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i, c + i] for i in range(WIN_COUNT)]
            score += evaluate_window(window, piece)

    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i, c + i] for i in range(WIN_COUNT)]
            score += evaluate_window(window, piece)

    logging.debug(f"Evaluation for player {piece}: score={score}")
    return score

# --- Move Sorting ---
def sort_moves(board: np.ndarray, moves: List[int], piece: int, depth: int) -> List[int]:
    scored_moves = []
    opp_piece = 3 - piece
    center_col = COLUMN_COUNT // 2
    current_killers = killer_moves.get(depth, [])

    for col in moves:
        score = 0
        temp_board = board.copy()
        row = get_next_open_row(temp_board, col)
        if row is None:
            continue

        temp_board[row, col] = piece
        if winning_move(temp_board, piece):
            score += 1000000
            logging.debug(f"Player win detected: col={col}, score={score}")
        else:
            temp_board[row, col] = opp_piece
            if winning_move(temp_board, opp_piece):
                score += 1000000
                logging.debug(f"Opponent win detected: col={col}, score={score}")
            elif has_three_piece_threat(temp_board, opp_piece):
                score += 750000
                logging.debug(f"Opponent threat detected: col={col}, score={score}")

            if score < 750000:
                temp_board[row, col] = piece
                score += evaluate_board(temp_board, piece) * 0.1
                if col == center_col:
                    score += 10
                elif abs(col - center_col) == 1:
                    score += 5
                if col in current_killers:
                    score += 100
                score += history_scores.get((depth, col), 0) * 0.01

        temp_board[row, col] = 0
        scored_moves.append((score, col))
        logging.debug(f"Move scored: col={col}, score={score}")

    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [col for score, col in scored_moves]

# --- Minimax ---
def minimax(board: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool,
            piece: int, start_time: float) -> Tuple[Optional[int], float]:
    if time.time() - start_time > BASE_TIMEOUT:
        return None, evaluate_board(board, piece if maximizing_player else 3 - piece)

    board_key = board_to_key(board)
    if board_key in transposition_table:
        stored_depth, stored_score, stored_best_move = transposition_table[board_key]
        if stored_depth >= depth:
            return stored_best_move, stored_score

    valid_moves = get_valid_moves(board)
    if depth == 0 or terminal_node(board):
        if terminal_node(board):
            if winning_move(board, piece):
                return None, 1000000 + depth
            elif winning_move(board, 3 - piece):
                return None, -1000000 - depth
            return None, 0
        return None, evaluate_board(board, piece)

    moves_order = sort_moves(board, valid_moves, piece if maximizing_player else 3 - piece, depth)
    best_move = moves_order[0] if moves_order else None

    if maximizing_player:
        value = -math.inf
        for col in moves_order:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            board[row, col] = piece
            _, score = minimax(board, depth - 1, alpha, beta, False, piece, start_time)
            board[row, col] = 0
            if score > value:
                value = score
                best_move = col
            alpha = max(alpha, value)
            if alpha >= beta:
                killer_moves.setdefault(depth, []).append(col)
                if len(killer_moves[depth]) > 2:
                    killer_moves[depth].pop(0)
                history_scores[(depth, col)] = history_scores.get((depth, col), 0) + (2 ** depth)
                break
        transposition_table[board_key] = (depth, value, best_move)
        return best_move, value

    else:
        value = math.inf
        for col in moves_order:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            board[row, col] = 3 - piece
            _, score = minimax(board, depth - 1, alpha, beta, True, piece, start_time)
            board[row, col] = 0
            if score < value:
                value = score
                best_move = col
            beta = min(beta, value)
            if alpha >= beta:
                killer_moves.setdefault(depth, []).append(col)
                if len(killer_moves[depth]) > 2:
                    killer_moves[depth].pop(0)
                history_scores[(depth, col)] = history_scores.get((depth, col), 0) + (2 ** depth)
                break
        transposition_table[board_key] = (depth, value, best_move)
        return best_move, value

# --- Main Search Function ---
def find_best_move(board: np.ndarray, piece: int, valid_moves: List[int]) -> int:
    start_time = time.time()
    if not valid_moves:
        logging.error("No valid moves provided in request!")
        fallback_moves = get_valid_moves(board)
        return fallback_moves[0] if fallback_moves else 0

    if np.all(board == 0):
        center_col = COLUMN_COUNT // 2
        if center_col in valid_moves:
            logging.info(f"First move (player {piece}): Center col {center_col}")
            return center_col
        return valid_moves[0]

    best_move_overall = valid_moves[0]
    best_score_overall = -math.inf
    last_completed_depth = 0
    initial_moves_order = sort_moves(board, valid_moves, piece, AI_DEPTH)

    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = piece
            if winning_move(board, piece):
                board[row, col] = 0
                logging.info(f"Immediate win found for player {piece} at col {col}")
                return col
            board[row, col] = 0

    opp_piece = 3 - piece
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = opp_piece
            if winning_move(board, opp_piece):
                board[row, col] = 0
                logging.info(f"Immediate block found for player {piece} at col {col}")
                return col
            board[row, col] = 0

    for depth in range(1, AI_DEPTH + 1):
        current_best_move_depth = None
        current_best_score_depth = -math.inf
        alpha = -math.inf
        beta = math.inf
        moves_to_search = initial_moves_order

        for col in moves_to_search:
            if time.time() - start_time > BASE_TIMEOUT:
                logging.warning(f"Timeout during iterative deepening at depth {depth}. Using best move from depth {last_completed_depth}.")
                return best_move_overall

            row = get_next_open_row(board, col)
            if row is None:
                continue

            board[row, col] = piece
            _, score = minimax(board, depth - 1, alpha, beta, False, piece, start_time)
            board[row, col] = 0

            if score > current_best_score_depth:
                current_best_score_depth = score
                current_best_move_depth = col

            alpha = max(alpha, score)

        if time.time() - start_time <= BASE_TIMEOUT and current_best_move_depth is not None:
            best_move_overall = current_best_move_depth
            best_score_overall = current_best_score_depth
            last_completed_depth = depth
            if best_move_overall in initial_moves_order:
                initial_moves_order.remove(best_move_overall)
                initial_moves_order.insert(0, best_move_overall)

            logging.info(f"Depth {depth} completed. Best move: {best_move_overall}, Score: {best_score_overall:.0f}, Time: {time.time() - start_time:.3f}s")

            if best_score_overall >= 1000000:
                logging.info(f"Winning move found at depth {depth}. Move: {best_move_overall}")
                break

    if best_move_overall not in valid_moves:
        logging.error(f"Selected best move {best_move_overall} is invalid! Valid moves: {valid_moves}")
        return valid_moves[0]

    return best_move_overall

# --- API Endpoint ---
@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")

        board = np.array(game_state.board, dtype=np.int32)
        board.flags.writeable = True
        player = game_state.current_player
        valid_moves = game_state.valid_moves

        logging.info(f"Received board (player {player}):\n{board}\nValid moves: {valid_moves}")

        selected_move = find_best_move(board, player, valid_moves)

        if selected_move not in valid_moves:
            logging.warning(f"Minimax did not return a valid move ({selected_move}). Falling back to first valid move.")
            selected_move = valid_moves[0]

        logging.info(f"AI (player {player}) selected move: {selected_move}")
        return AIResponse(move=selected_move)

    except Exception as e:
        logging.error(f"Error in make_move: {str(e)}", exc_info=True)
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý nước đi: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)