import numpy as np
import math
import time
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict
import concurrent.futures
import random

# --- Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_COUNT = 4
AI_DEPTH = 8  # Giảm độ sâu để tăng tốc
BASE_TIMEOUT = 5  # Giảm timeout để tránh treo
TRANS_TABLE_SIZE = 2**20  # Giảm kích thước bảng transposition

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# --- Zobrist Hashing ---
zobrist_table = None

def init_zobrist():
    global zobrist_table
    zobrist_table = np.random.randint(1, 2**64, size=(ROW_COUNT, COLUMN_COUNT, 3), dtype=np.uint64)

def board_to_key(board: np.ndarray) -> int:
    global zobrist_table
    if zobrist_table is None:
        init_zobrist()
    hash_value = 0
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            if board[r, c] != 0:
                hash_value ^= zobrist_table[r, c, board[r, c]]
    return hash_value

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

def winning_move(board: np.ndarray, piece: int, row: int, col: int) -> bool:
    # Kiểm tra chỉ các hướng liên quan đến (row, col)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Hàng ngang, dọc, chéo
    for dr, dc in directions:
        count = 1
        # Kiểm tra theo hướng dương
        for i in range(1, WIN_COUNT):
            r, c = row + dr * i, col + dc * i
            if 0 <= r < ROW_COUNT and 0 <= c < COLUMN_COUNT and board[r, c] == piece:
                count += 1
            else:
                break
        # Kiểm tra theo hướng âm
        for i in range(1, WIN_COUNT):
            r, c = row - dr * i, col - dc * i
            if 0 <= r < ROW_COUNT and 0 <= c < COLUMN_COUNT and board[r, c] == piece:
                count += 1
            else:
                break
        if count >= WIN_COUNT:
            return True
    return False

def terminal_node(board: np.ndarray) -> bool:
    return winning_move(board, 1, 0, 0) or winning_move(board, 2, 0, 0) or not get_valid_moves(board)

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
        score += 100
    elif piece_count == 2 and empty_count == 2:
        score += 5

    if opp_count == 4:
        score -= 100000
    elif opp_count == 3 and empty_count == 1:
        score -= 80
    elif opp_count == 2 and empty_count == 2:
        score -= 3

    return score

def evaluate_board(board: np.ndarray, piece: int, row: int, col: int) -> int:
    score = 0
    center_col = COLUMN_COUNT // 2
    if col == center_col:
        score += 6
    elif abs(col - center_col) <= 1:
        score += 3

    # Chỉ đánh giá các cửa sổ liên quan đến (row, col)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        for offset in range(-3, 1):
            window = []
            for i in range(WIN_COUNT):
                r, c = row + (offset + i) * dr, col + (offset + i) * dc
                if 0 <= r < ROW_COUNT and 0 <= c < COLUMN_COUNT:
                    window.append(int(board[r, c]))
                else:
                    window = []
                    break
            if window:
                score += evaluate_window(window, piece)
    return score

# --- Move Sorting ---
def sort_moves(board: np.ndarray, moves: List[int], piece: int, depth: int) -> List[int]:
    scored_moves = []
    center_col = COLUMN_COUNT // 2
    current_killers = killer_moves.get(depth, [])

    for col in moves:
        score = 0
        row = get_next_open_row(board, col)
        if row is None:
            continue

        board[row, col] = piece
        if winning_move(board, piece, row, col):
            score += 1000000
        else:
            score += evaluate_board(board, piece, row, col) * 0.1
            if col == center_col:
                score += 10
            elif abs(col - center_col) == 1:
                score += 5
            if col in current_killers:
                score += 100
            score += history_scores.get((depth, col), 0) * 0.01

        board[row, col] = 0
        scored_moves.append((score, col))

    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [col for _, col in scored_moves]

# --- Minimax with Parallelization ---
def minimax_parallel(board: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool,
                     piece: int, start_time: float, executor: concurrent.futures.Executor) -> Tuple[Optional[int], float]:
    if time.time() - start_time > BASE_TIMEOUT:
        return None, evaluate_board(board, piece if maximizing_player else 3 - piece, 0, 0)

    board_key = board_to_key(board)
    if board_key in transposition_table:
        stored_depth, stored_score, stored_best_move = transposition_table[board_key]
        if stored_depth >= depth:
            return stored_best_move, stored_score

    valid_moves = get_valid_moves(board)
    if depth == 0 or terminal_node(board):
        if terminal_node(board):
            if winning_move(board, piece, 0, 0):
                return None, 1000000 + depth
            elif winning_move(board, 3 - piece, 0, 0):
                return None, -1000000 - depth
            return None, 0
        return None, evaluate_board(board, piece, 0, 0)

    moves_order = sort_moves(board, valid_moves, piece if maximizing_player else 3 - piece, depth)
    best_move = moves_order[0] if moves_order else None

    if maximizing_player:
        value = -math.inf
        futures = []
        for col in moves_order:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            temp_board = board.copy()
            temp_board[row, col] = piece
            future = executor.submit(minimax_parallel, temp_board, depth - 1, alpha, beta, False, piece, start_time, executor)
            futures.append((col, future))

        for col, future in futures:
            _, score = future.result()
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
        futures = []
        for col in moves_order:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            temp_board = board.copy()
            temp_board[row, col] = 3 - piece
            future = executor.submit(minimax_parallel, temp_board, depth - 1, alpha, beta, True, piece, start_time, executor)
            futures.append((col, future))

        for col, future in futures:
            _, score = future.result()
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
        logging.error("No valid moves provided!")
        return valid_moves[0] if valid_moves else 0

    if np.all(board == 0):
        center_col = COLUMN_COUNT // 2
        if center_col in valid_moves:
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
            if winning_move(board, piece, row, col):
                board[row, col] = 0
                return col
            board[row, col] = 0

    opp_piece = 3 - piece
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = opp_piece
            if winning_move(board, opp_piece, row, col):
                board[row, col] = 0
                return col
            board[row, col] = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for depth in range(1, AI_DEPTH + 1):
            current_best_move_depth = None
            current_best_score_depth = -math.inf
            alpha = -math.inf
            beta = math.inf
            moves_to_search = initial_moves_order

            futures = []
            for col in moves_to_search:
                if time.time() - start_time > BASE_TIMEOUT:
                    logging.warning(f"Timeout at depth {depth}. Using move from depth {last_completed_depth}.")
                    return best_move_overall

                row = get_next_open_row(board, col)
                if row is None:
                    continue

                temp_board = board.copy()
                temp_board[row, col] = piece
                future = executor.submit(minimax_parallel, temp_board, depth - 1, alpha, beta, False, piece, start_time, executor)
                futures.append((col, future))

            for col, future in futures:
                _, score = future.result()
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
        logging.error(f"Invalid best move {best_move_overall}! Valid moves: {valid_moves}")
        return valid_moves[0]

    return best_move_overall

# --- Process Request Function ---
def process_request(request):
    try:
        if not isinstance(request, dict):
            logging.error("Request must be a dictionary")
            raise ValueError("Request must be a dictionary")

        required_keys = ["board", "current_player", "valid_moves"]
        for key in required_keys:
            if key not in request:
                logging.error(f"Missing key: {key}")
                raise ValueError(f"Missing key: {key}")

        board_list = request["board"]
        if not isinstance(board_list, list) or len(board_list) != ROW_COUNT:
            logging.error("Board must be a 6x7 list")
            raise ValueError("Board must be a 6x7 list")
        for row in board_list:
            if not isinstance(row, list) or len(row) != COLUMN_COUNT:
                logging.error("Each row must be a list of length 7")
                raise ValueError("Each row must be a list of length 7")
            for val in row:
                if val not in [0, 1, 2]:
                    logging.error(f"Invalid value in board: {val}")
                    raise ValueError(f"Invalid value in board: {val}")

        current_player = request["current_player"]
        if not isinstance(current_player, int) or current_player not in [1, 2]:
            logging.error(f"Invalid current_player: {current_player}")
            raise ValueError(f"Invalid current_player: {current_player}")

        valid_moves = request["valid_moves"]
        if not isinstance(valid_moves, list):
            logging.error("valid_moves must be a list")
            raise ValueError("valid_moves must be a list")
        for col in valid_moves:
            if not isinstance(col, int) or col < 0 or col >= COLUMN_COUNT:
                logging.error(f"Invalid column in valid_moves: {col}")
                raise ValueError(f"Invalid column in valid_moves: {col}")

        board = np.array(board_list, dtype=int)
        best_move = find_best_move(board, current_player, valid_moves)

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
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
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