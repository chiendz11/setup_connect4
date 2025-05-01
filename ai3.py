import numpy as np
import math
import time
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict
import random

# --- Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_COUNT = 4
AI_DEPTH = 12
BASE_TIMEOUT = 5
TRANS_TABLE_SIZE = 2**24

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# --- Zobrist Hashing Initialization ---
ZOBRIST_KEYS = np.zeros((ROW_COUNT, COLUMN_COUNT, 3), dtype=np.uint64)
for r in range(ROW_COUNT):
    for c in range(COLUMN_COUNT):
        for val in range(3):  # 0, 1, 2
            ZOBRIST_KEYS[r, c, val] = np.uint64(random.getrandbits(64))

# --- LimitedDict for Transposition Table ---
class LimitedDict(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize
        self.depths = {}  # Store depth for each key

    def __setitem__(self, key, value):
        depth = value[0]  # Stored depth
        super().__setitem__(key, value)
        self.depths[key] = depth
        if len(self) > self.maxsize:
            # Remove entry with lowest depth
            min_depth_key = min(self.depths, key=lambda k: self.depths[k])
            self.pop(min_depth_key)
            del self.depths[min_depth_key]

# --- Global Variables ---
transposition_table: LimitedDict = LimitedDict(maxsize=TRANS_TABLE_SIZE)
killer_moves: Dict[int, List[int]] = {d: [] for d in range(AI_DEPTH + 1)}
history_scores: Dict[Tuple[int, int], int] = {}

# --- Helper Functions ---
def zobrist_hash(board: np.ndarray) -> int:
    hash_value = np.uint64(0)
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            val = int(board[r, c])
            hash_value ^= ZOBRIST_KEYS[r, c, val]
    return int(hash_value)

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

def terminal_node(board: np.ndarray) -> bool:
    return winning_move(board, 1) or winning_move(board, 2) or not get_valid_moves(board)

# --- Evaluation Function ---
def evaluate_window(window: List[int], piece: int) -> int:
    opp_piece = 3 - piece
    piece_count = window.count(piece)
    opp_count = window.count(opp_piece)
    empty_count = window.count(0)

    # Winning moves
    if piece_count == 4:
        return 1000000
    if opp_count == 4:
        return -1000000

    # Immediate threats
    if piece_count == 3 and empty_count == 1:
        return 50000
    if opp_count == 3 and empty_count == 1:
        # Vertical threats are more dangerous
        if window == [opp_piece, opp_piece, opp_piece, 0] or window == [0, opp_piece, opp_piece, opp_piece]:
            return -500000
        return -100000

    # Development positions
    score = 0
    if empty_count > 0:
        if piece_count == 2 and empty_count == 2:
            score += 500
        if opp_count == 2 and empty_count == 2:
            score -= 800

    return score

def evaluate_board(board: np.ndarray, piece: int) -> int:
    score = 0
    opp_piece = 3 - piece
    
    # Check vertical threats first (most critical)
    for c in range(COLUMN_COUNT):
        col_array = [int(board[r, c]) for r in range(ROW_COUNT)]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WIN_COUNT]
            if window.count(opp_piece) == 3 and window.count(0) == 1:
                empty_pos = r + window.index(0)
                if empty_pos == 0 or board[empty_pos-1, c] != 0:
                    score -= 1000000  # Immediate loss if not blocked
            window_score = evaluate_window(window, piece)
            if abs(window_score) > 1000:
                score += window_score

    # Check horizontal threats
    for r in range(ROW_COUNT):
        row_array = [int(board[r, c]) for c in range(COLUMN_COUNT)]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WIN_COUNT]
            window_score = evaluate_window(window, piece)
            if abs(window_score) > 1000:
                score += window_score

    # Check diagonal threats
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            # Diagonal \
            window = [board[r + i, c + i] for i in range(WIN_COUNT)]
            window_score = evaluate_window(window, piece)
            if abs(window_score) > 1000:
                score += window_score
            
            # Diagonal /
            window = [board[r + 3 - i, c + i] for i in range(WIN_COUNT)]
            window_score = evaluate_window(window, piece)
            if abs(window_score) > 1000:
                score += window_score

    # Center control
    center_array = [int(board[r, COLUMN_COUNT // 2]) for r in range(ROW_COUNT)]
    center_count = center_array.count(piece)
    score += center_count * 100

    return score

# --- Move Sorting ---
def sort_moves(board: np.ndarray, moves: List[int], piece: int, depth: int) -> List[int]:
    scored_moves = []
    opp_piece = 3 - piece
    center_col = COLUMN_COUNT // 2

    for col in moves:
        score = 0
        row = get_next_open_row(board, col)
        if row is None:
            continue

        # Check for immediate win
        temp_board = board.copy()
        temp_board[row, col] = piece
        if winning_move(temp_board, piece):
            return [col]  # Return immediately if winning move found

        # Check if opponent can win in next move
        temp_board[row, col] = opp_piece
        if winning_move(temp_board, opp_piece):
            score += 900000  # Very high priority for blocking moves

        # Check for vertical threats
        if row >= 2:
            below_count = 0
            for i in range(1, 4):
                if row - i >= 0 and board[row - i, col] == opp_piece:
                    below_count += 1
                else:
                    break
            if below_count == 2:
                score += 800000  # High priority for blocking vertical threats

        # Strategic position scoring
        temp_board[row, col] = piece
        score += evaluate_board(temp_board, piece) * 0.1

        # Prefer center and near-center columns
        if col == center_col:
            score += 2000
        elif abs(col - center_col) == 1:
            score += 1000
        elif abs(col - center_col) == 2:
            score += 500

        # History and killer move bonuses
        if col in killer_moves.get(depth, []):
            score += 3000
        score += history_scores.get((depth, col), 0) * 0.5

        scored_moves.append((score, col))

    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [col for score, col in scored_moves]

# --- Minimax ---
def minimax(board: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool,
            piece: int, start_time: float) -> Tuple[Optional[int], float]:
    # Quick timeout check
    if time.time() - start_time > BASE_TIMEOUT * 0.95:
        return None, evaluate_board(board, piece if maximizing_player else 3 - piece)

    # Check transposition table
    board_key = zobrist_hash(board)
    if board_key in transposition_table:
        stored_depth, stored_score, stored_move, stored_flag = transposition_table[board_key]
        if stored_depth >= depth:
            if stored_flag == 'exact':
                return stored_move, stored_score
            elif stored_flag == 'lower' and stored_score >= beta:
                return stored_move, stored_score
            elif stored_flag == 'upper' and stored_score <= alpha:
                return stored_move, stored_score

    # Terminal node check
    if winning_move(board, piece):
        return None, 1000000 + depth
    if winning_move(board, 3 - piece):
        return None, -1000000 - depth
    
    valid_moves = get_valid_moves(board)
    if not valid_moves or depth == 0:
        return None, evaluate_board(board, piece if maximizing_player else 3 - piece)

    # Principal Variation Search
    moves = sort_moves(board, valid_moves, piece if maximizing_player else 3 - piece, depth)
    best_move = moves[0]
    
    if maximizing_player:
        best_score = -math.inf
        first_move = True
        
        for col in moves:
            row = get_next_open_row(board, col)
            if row is None:
                continue

            board[row, col] = piece
            
            # Full window search for first move
            if first_move:
                _, score = minimax(board, depth - 1, alpha, beta, False, piece, start_time)
                first_move = False
            else:
                # Null window search for remaining moves
                _, score = minimax(board, depth - 1, alpha, alpha + 1, False, piece, start_time)
                if alpha < score < beta:  # If fail-high, do a full re-search
                    _, score = minimax(board, depth - 1, alpha, beta, False, piece, start_time)
            
            board[row, col] = 0

            if score > best_score:
                best_score = score
                best_move = col

            alpha = max(alpha, best_score)
            if alpha >= beta:
                # Update killer moves and history scores
                killer_moves.setdefault(depth, []).append(col)
                if len(killer_moves[depth]) > 2:
                    killer_moves[depth].pop(0)
                history_scores[(depth, col)] = history_scores.get((depth, col), 0) + (2 ** depth)
                break
    else:
        best_score = math.inf
        first_move = True
        
        for col in moves:
            row = get_next_open_row(board, col)
            if row is None:
                continue

            board[row, col] = 3 - piece
            
            # Full window search for first move
            if first_move:
                _, score = minimax(board, depth - 1, alpha, beta, True, piece, start_time)
                first_move = False
            else:
                # Null window search for remaining moves
                _, score = minimax(board, depth - 1, beta - 1, beta, True, piece, start_time)
                if alpha < score < beta:  # If fail-low, do a full re-search
                    _, score = minimax(board, depth - 1, alpha, beta, True, piece, start_time)
            
            board[row, col] = 0

            if score < best_score:
                best_score = score
                best_move = col

            beta = min(beta, best_score)
            if alpha >= beta:
                # Update killer moves and history scores
                killer_moves.setdefault(depth, []).append(col)
                if len(killer_moves[depth]) > 2:
                    killer_moves[depth].pop(0)
                history_scores[(depth, col)] = history_scores.get((depth, col), 0) + (2 ** depth)
                break

    # Store position in transposition table
    flag = 'exact'
    if best_score <= alpha:
        flag = 'upper'
    elif best_score >= beta:
        flag = 'lower'
    transposition_table[board_key] = (depth, best_score, best_move, flag)

    return best_move, best_score

# --- Main Search Function (Iterative Deepening) ---
def find_best_move(board: np.ndarray, piece: int, valid_moves: List[int]) -> int:
    start_time = time.time()
    if not valid_moves:
        logging.error("No valid moves provided in request!")
        fallback_moves = get_valid_moves(board)
        return fallback_moves[0] if fallback_moves else 0

    # Check for immediate win
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = piece
            if winning_move(board, piece):
                board[row, col] = 0
                logging.info(f"Immediate win found for player {piece} at col {col}")
                return col
            board[row, col] = 0

    # Check for opponent's immediate win (especially vertical threats)
    opp_piece = 3 - piece
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = opp_piece
            if winning_move(board, opp_piece):
                board[row, col] = 0
                # Check if it's a vertical threat
                if row >= 3 and all(board[r, col] == opp_piece for r in range(row-3, row)):
                    logging.info(f"Critical vertical block found at col {col}")
                    return col
                logging.info(f"Immediate block found at col {col}")
                return col
            board[row, col] = 0

    # Opening move
    if np.all(board == 0):
        center_col = COLUMN_COUNT // 2
        if center_col in valid_moves:
            logging.info(f"First move (player {piece}): Center col {center_col}")
            return center_col
        return valid_moves[0]

    # Regular search with iterative deepening
    best_move = valid_moves[0]
    best_score = -math.inf
    moves_order = sort_moves(board, valid_moves, piece, AI_DEPTH)

    # Adaptive depth based on game phase
    piece_count = np.count_nonzero(board)
    if piece_count <= 12:  # Endgame
        max_depth = AI_DEPTH  # Use maximum depth in endgame
    elif piece_count <= 24:  # Midgame
        max_depth = AI_DEPTH - 2  # Slightly reduced depth in midgame
    else:  # Opening
        max_depth = AI_DEPTH - 4  # Lower depth in opening to save time

    for depth in range(1, max_depth + 1):
        if time.time() - start_time > BASE_TIMEOUT * 0.8:
            break

        current_best_move = None
        current_best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        for col in moves_order:
            row = get_next_open_row(board, col)
            if row is None:
                continue

            board[row, col] = piece
            _, score = minimax(board, depth - 1, alpha, beta, False, piece, start_time)
            board[row, col] = 0

            if score > current_best_score:
                current_best_score = score
                current_best_move = col

            if score >= 1000000 - depth:  # Found winning move
                logging.info(f"Winning move found at depth {depth}. Move: {col}")
                return col

            alpha = max(alpha, score)

        if current_best_move is not None:
            best_move = current_best_move
            best_score = current_best_score
            
            # Update move ordering
            if best_move in moves_order:
                moves_order.remove(best_move)
                moves_order.insert(0, best_move)

            logging.info(f"Depth {depth} completed. Best move: {best_move}, Score: {best_score:.0f}, Time: {time.time() - start_time:.3f}s")

    if best_move not in valid_moves:
        logging.error(f"Selected best move {best_move} is invalid! Valid moves: {valid_moves}")
        return valid_moves[0]

    return best_move

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
            [2, 1, 1, 1, 2, 1, 1],
            [0, 2, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ],
        "current_player": 2,
        "valid_moves": [0, 1, 2, 3, 4, 5, 6]
    }
    best_move = process_request(request)
    print(f"Best move: {best_move}")