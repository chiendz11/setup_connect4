import numpy as np
import math
import time
import random
from typing import List, Optional, Tuple, Dict
from collections import OrderedDict

# --- Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_COUNT = 4
AI_DEPTH = 10  # Độ sâu tìm kiếm tối đa
BASE_TIMEOUT = 5.0  # Thời gian giới hạn mỗi nước đi
TRANS_TABLE_SIZE = 1 << 20  # Kích thước bảng chuyển vị (~1 triệu mục)

# --- Zobrist Hashing Initialization ---
np.random.seed(42)
ZOBRIST_KEYS = np.zeros((ROW_COUNT, COLUMN_COUNT, 3), dtype=np.uint64)
for r in range(ROW_COUNT):
    for c in range(COLUMN_COUNT):
        for val in range(3):  # 0, 1, 2
            ZOBRIST_KEYS[r, c, val] = np.uint64(random.getrandbits(64))

# --- Transposition Table ---
class LimitedDict(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize
        self.depths = {}  # Lưu độ sâu cho mỗi mục

    def __setitem__(self, key, value):
        depth = value[0]  # Lưu độ sâu
        super().__setitem__(key, value)
        self.depths[key] = depth
        if len(self) > self.maxsize:
            min_depth_key = min(self.depths, key=lambda k: self.depths[k])
            self.pop(min_depth_key)
            del self.depths[min_depth_key]

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
    score = 0
    opp_piece = 3 - piece
    piece_count = window.count(piece)
    opp_count = window.count(opp_piece)
    empty_count = window.count(0)

    if piece_count == 4:
        score += 100000
    elif piece_count == 3 and empty_count == 1:
        score += 150
    elif piece_count == 2 and empty_count == 2:
        score += 10
    elif piece_count == 1 and empty_count == 3:
        score += 2

    if opp_count == 4:
        score -= 100000
    elif opp_count == 3 and empty_count == 1:
        score -= 120
    elif opp_count == 2 and empty_count == 2:
        score -= 8
    elif opp_count == 1 and empty_count == 3:
        score -= 1

    return score

def evaluate_board(board: np.ndarray, piece: int) -> int:
    score = 0
    opp_piece = 3 - piece

    # Center column
    center_array = [int(board[r, COLUMN_COUNT // 2]) for r in range(ROW_COUNT)]
    center_count = center_array.count(piece)
    score += center_count * 8

    # Standard window evaluation
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
            window = [board[r + 3 - i, c + i] for i in range(WIN_COUNT)]
            score += evaluate_window(window, piece)

    # Double threat detection
    threat_count = 0
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r, c + i] for i in range(WIN_COUNT)]
            if window.count(piece) == 3 and window.count(0) == 1:
                threat_count += 1
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            window = [board[r + i, c] for i in range(WIN_COUNT)]
            if window.count(piece) == 3 and window.count(0) == 1:
                threat_count += 1
    if threat_count >= 2:
        score += 500

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
            return [col]  # Trả về ngay nước đi thắng
        temp_board[row, col] = opp_piece
        if winning_move(temp_board, opp_piece):
            score += 1000000
        temp_board[row, col] = piece
        score += evaluate_board(temp_board, piece) * 0.1
        if col == center_col:
            score += 10
        elif abs(col - center_col) == 1:
            score += 5
        if col in current_killers:
            score += 100
        score += history_scores.get((depth, col), 0) * 0.01
        scored_moves.append((score, col))

    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [col for _, col in scored_moves]

# --- Minimax with Alpha-Beta Pruning ---
def minimax(board: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool,
            piece: int, start_time: float) -> Tuple[Optional[int], float]:
    if time.time() - start_time > BASE_TIMEOUT:
        return None, evaluate_board(board, piece if maximizing_player else 3 - piece)

    board_key = zobrist_hash(board)
    if board_key in transposition_table:
        stored_depth, stored_score, stored_best_move, stored_flag = transposition_table[board_key]
        if stored_depth >= depth:
            if stored_flag == 'exact':
                return stored_best_move, stored_score
            elif stored_flag == 'lower' and stored_score >= beta:
                return stored_best_move, stored_score
            elif stored_flag == 'upper' and stored_score <= alpha:
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

    # Null Move Pruning
    if depth > 2 and not maximizing_player and valid_moves:
        null_score = minimax(board, depth - 3, alpha, beta, True, piece, start_time)[1]
        if null_score <= alpha:
            return None, null_score

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
        flag = 'exact' if value > alpha and value < beta else 'lower' if value >= beta else 'upper'
        transposition_table[board_key] = (depth, value, best_move, flag)
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
        flag = 'exact' if value > alpha and value < beta else 'lower' if value >= beta else 'upper'
        transposition_table[board_key] = (depth, value, best_move, flag)
        return best_move, value

# --- Main Search Function (Iterative Deepening) ---
def find_best_move(board: np.ndarray, piece: int, valid_moves: List[int]) -> int:
    start_time = time.time()
    if not valid_moves:
        print("No valid moves provided in request!")
        fallback_moves = get_valid_moves(board)
        return fallback_moves[0] if fallback_moves else 0

    if np.all(board == 0):
        center_col = COLUMN_COUNT // 2
        if center_col in valid_moves:
            print(f"First move (player {piece}): Center col {center_col}")
            return center_col
        return valid_moves[0]

    best_move_overall = valid_moves[0]
    best_score_overall = -math.inf
    last_completed_depth = 0
    initial_moves_order = sort_moves(board, valid_moves, piece, AI_DEPTH)

    num_valid_moves = len(valid_moves)
    max_depth = min(AI_DEPTH, 8 + (7 - num_valid_moves))  # Điều chỉnh độ sâu dựa trên số nước đi
    timeout = BASE_TIMEOUT * (0.7 + 0.3 * (7 - num_valid_moves) / 7)  # Điều chỉnh thời gian

    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = piece
            if winning_move(board, piece):
                board[row, col] = 0
                print(f"Immediate win found for player {piece} at col {col}")
                return col
            board[row, col] = 0

    opp_piece = 3 - piece
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = opp_piece
            if winning_move(board, opp_piece):
                board[row, col] = 0
                print(f"Immediate block found for player {piece} at col {col}")
                return col
            board[row, col] = 0

    for depth in range(1, max_depth + 1):
        current_best_move_depth = None
        current_best_score_depth = -math.inf
        alpha = -math.inf
        beta = math.inf
        moves_to_search = initial_moves_order

        for col in moves_to_search:
            if time.time() - start_time > timeout:
                print(f"Timeout at depth {depth}. Using best move from depth {last_completed_depth}: {best_move_overall}")
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

        if time.time() - start_time <= timeout and current_best_move_depth is not None:
            best_move_overall = current_best_move_depth
            best_score_overall = current_best_score_depth
            last_completed_depth = depth
            if best_move_overall in initial_moves_order:
                initial_moves_order.remove(best_move_overall)
                initial_moves_order.insert(0, best_move_overall)
            print(f"Depth {depth} completed. Best move: {best_move_overall}, Score: {best_score_overall:.0f}, Time: {time.time() - start_time:.3f}s")

            if best_score_overall >= 1000000 - depth:
                print(f"Winning move found at depth {depth}. Move: {best_move_overall}")
                break

    if best_move_overall not in valid_moves:
        print(f"Selected best move {best_move_overall} is invalid! Valid moves: {valid_moves}")
        return valid_moves[0]

    return best_move_overall

# --- Process Request Function ---
def process_request(request):
    try:
        raw_board = request.get("board", [])
        current_player = request.get("current_player", 1)
        valid_moves = request.get("valid_moves", [0, 1, 2, 3, 4, 5, 6])

        board = np.array(raw_board, dtype=np.int32)
        if board.shape != (ROW_COUNT, COLUMN_COUNT):
            print("Invalid board shape, initializing empty board")
            board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int32)

        print(f"Received board:\n{board}")
        print(f"Current player: {current_player}")
        print(f"Valid moves: {valid_moves}")

        best_move = find_best_move(board, current_player, valid_moves)
        if best_move not in valid_moves:
            print(f"Best move {best_move} not in valid_moves {valid_moves}. Choosing first valid move.")
            best_move = valid_moves[0] if valid_moves else 0

        return best_move

    except Exception as e:
        print(f"Error in process_request: {e}")
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