import numpy as np
import math
import time
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict

# --- Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_COUNT = 4
AI_DEPTH = 12  # Increased from 9 for deeper search
BASE_TIMEOUT = 9  # Increased from 3.8 for deeper searches
TRANS_TABLE_SIZE = 2**24

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

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
        score += 500  # Increased from 100 for stronger offense
    elif piece_count == 2 and empty_count == 2:
        score += 50   # Increased from 5 for stronger setups

    if opp_count == 4:
        score -= 100000
    elif opp_count == 3 and empty_count == 1:
        score -= 5000  # Increased from -80 for stronger defense
    elif opp_count == 2 and empty_count == 2:
        score -= 10    # Increased from -3 for better threat awareness

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
                score += 1000000  # Increased from 500000 for critical blocks
                logging.debug(f"Opponent win detected: col={col}, score={score}")
            elif has_three_piece_threat(temp_board, opp_piece):
                score += 750000  # New: Prioritize blocking opponent threats
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

# --- Main Search Function (Iterative Deepening) ---
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

# --- Process Request Function ---
def process_request(request):
    """
    Process a request from the game and return the best move.
    
    Args:
        request (dict): A dictionary containing:
            - board: 6x7 list representing the game board (0=empty, 1=player1, 2=player2)
            - current_player: Integer (1 or 2) indicating the current player
            - valid_moves: List of integers representing valid columns
    
    Returns:
        int: The optimal column for the current player to move
    """
    try:
        # Validate request
        if not isinstance(request, dict):
            logging.error("Request must be a dictionary")
            raise ValueError("Request must be a dictionary")
        
        required_keys = ["board", "current_player", "valid_moves"]
        for key in required_keys:
            if key not in request:
                logging.error(f"Missing required key in request: {key}")
                raise ValueError(f"Missing required key in request: {key}")
        
        board_list = request["board"]
        if not isinstance(board_list, list) or len(board_list) != ROW_COUNT:
            logging.error("Board must be a 6x7 list")
            raise ValueError("Board must be a 6x7 list")
        for row in board_list:
            if not isinstance(row, list) or len(row) != COLUMN_COUNT:
                logging.error("Each row in board must be a list of length 7")
                raise ValueError("Each row in board must be a list of length 7")
            for val in row:
                if val not in [0, 1, 2]:
                    logging.error(f"Invalid value in board: {val}. Must be 0, 1, or 2")
                    raise ValueError(f"Invalid value in board: {val}. Must be 0, 1, or 2")
        
        current_player = request["current_player"]
        if not isinstance(current_player, int) or current_player not in [1, 2]:
            logging.error(f"Invalid current_player: {current_player}. Must be 1 or 2")
            raise ValueError(f"Invalid current_player: {current_player}. Must be 1 or 2")
        
        valid_moves = request["valid_moves"]
        if not isinstance(valid_moves, list):
            logging.error("valid_moves must be a list")
            raise ValueError("valid_moves must be a list")
        for col in valid_moves:
            if not isinstance(col, int) or col < 0 or col >= COLUMN_COUNT:
                logging.error(f"Invalid column in valid_moves: {col}. Must be between 0 and 6")
                raise ValueError(f"Invalid column in valid_moves: {col}. Must be between 0 and 6")
        
        # Convert board to NumPy array
        board = np.array(board_list, dtype=int)
        
        # Find the best move using the provided valid_moves
        best_move = find_best_move(board, current_player, valid_moves)
        
        # Ensure the move is in valid_moves
        if best_move not in valid_moves:
            logging.warning(f"Best move {best_move} not in valid_moves {valid_moves}. Choosing first valid move.")
            best_move = valid_moves[0] if valid_moves else 0
        
        return best_move
    
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return valid_moves[0] if valid_moves else 0

# --- Test the AI ---
if __name__ == "__main__":
    request = {
        "board": [
            [1, 2, 1, 1, 2, 1, 1],
            [0, 0, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ],
        "current_player": 1,
        "valid_moves": [0, 1, 2, 3, 4, 5, 6]
    }
    best_move = process_request(request)
    print(f"Best move: {best_move}")