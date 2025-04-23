import numpy as np
import time
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

# --- Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_COUNT = 4
AI_DEPTH = 12  # Độ sâu cao để bất bại
BASE_TIMEOUT = 9.0  # Giới hạn 9 giây
TRANS_TABLE_SIZE = 1000000

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
killer_moves: dict = {d: [] for d in range(AI_DEPTH + 1)}
history_scores: dict = {}
zobrist_table = np.random.randint(1, 2**63, (ROW_COUNT, COLUMN_COUNT, 3), dtype=np.int64)

# --- Helper Functions ---
def board_to_key(board: np.ndarray) -> int:
    """Tạo khóa Zobrist cho bàn cờ."""
    key = 0
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            piece = int(board[r, c])
            key ^= zobrist_table[r][c][piece]
    return key

def is_valid_location(board: np.ndarray, col: int) -> bool:
    """Kiểm tra cột hợp lệ."""
    return board[ROW_COUNT - 1, col] == 0

def get_next_open_row(board: np.ndarray, col: int) -> Optional[int]:
    """Tìm hàng trống thấp nhất."""
    for r in range(ROW_COUNT):
        if board[r, col] == 0:
            return r
    return None

def get_valid_moves(board: np.ndarray) -> List[int]:
    """Lấy danh sách cột hợp lệ, ưu tiên trung tâm."""
    valid_moves = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    center_col = COLUMN_COUNT // 2
    if center_col in valid_moves:
        valid_moves.remove(center_col)
        valid_moves.insert(0, center_col)
    return valid_moves

def winning_move(board: np.ndarray, piece: int) -> bool:
    """Kiểm tra chuỗi 4 quân thắng."""
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

def can_win_next(board: np.ndarray, piece: int) -> bool:
    """Kiểm tra thắng ngay trong nước tiếp theo."""
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            board[row, col] = piece
            if winning_move(board, piece):
                board[row, col] = 0
                return True
            board[row, col] = 0
    return False

def detect_double_threat(board: np.ndarray, piece: int, col: int) -> bool:
    """Phát hiện nước đi tạo hai cơ hội thắng."""
    row = get_next_open_row(board, col)
    if row is None:
        return False
    temp_board = board.copy()
    temp_board[row, col] = piece
    threat_count = 0
    for next_col in get_valid_moves(temp_board):
        next_row = get_next_open_row(temp_board, next_col)
        if next_row is None:
            continue
        temp_board[next_row, next_col] = piece
        if winning_move(temp_board, piece):
            threat_count += 1
        temp_board[next_row, next_col] = 0
    return threat_count >= 2

# --- Evaluation Function ---
def evaluate_window(window: List[int], piece: int) -> int:
    """Đánh giá cửa sổ 4 ô."""
    score = 0
    opp_piece = 3 - piece
    piece_count = window.count(piece)
    opp_count = window.count(opp_piece)
    empty_count = window.count(0)

    if piece_count == 4:
        score += 1000000000
    elif piece_count == 3 and empty_count == 1:
        score += 1000000
    elif piece_count == 2 and empty_count == 2:
        score += 10000

    if opp_count == 4:
        score -= 1000000000
    elif opp_count == 3 and empty_count == 1:
        score -= 8000000
    elif opp_count == 2 and empty_count == 2:
        score -= 8000

    return score

def evaluate_board(board: np.ndarray, piece: int) -> int:
    """Đánh giá bàn cờ."""
    board_key = board_to_key(board)
    if board_key in transposition_table and transposition_table[board_key][0] == -1:
        return transposition_table[board_key][1]

    score = 0
    center_array = [int(board[r, COLUMN_COUNT // 2]) for r in range(ROW_COUNT)]
    center_count = center_array.count(piece)
    score += center_count * 1000

    for r in range(ROW_COUNT):
        row_array = [int(board[r, c]) for c in range(COLUMN_COUNT)]
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window(row_array[c:c + WIN_COUNT], piece)

    for c in range(COLUMN_COUNT):
        col_array = [int(board[r, c]) for r in range(ROW_COUNT)]
        for r in range(ROW_COUNT - 3):
            score += evaluate_window(col_array[r:r + WIN_COUNT], piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i, c + i] for i in range(WIN_COUNT)]
            score += evaluate_window(window, piece)

    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i, c + i] for i in range(WIN_COUNT)]
            score += evaluate_window(window, piece)

    transposition_table[board_key] = (-1, score, 'exact', None)
    return score

# --- Move Sorting ---
def sort_moves(board: np.ndarray, moves: List[int], piece: int, depth: int) -> List[int]:
    """Sắp xếp nước đi để tối ưu hóa cắt tỉa."""
    scored_moves = []
    opp_piece = 3 - piece
    center_col = COLUMN_COUNT // 2
    current_killers = killer_moves.get(depth, [])

    for col in moves:
        score = 0
        row = get_next_open_row(board, col)
        if row is None:
            continue
        temp_board = board.copy()
        temp_board[row, col] = piece

        if winning_move(temp_board, piece):
            score += 1000000000
        elif detect_double_threat(board, piece, col):
            score += 50000000
        else:
            temp_board[row, col] = opp_piece
            if winning_move(temp_board, opp_piece):
                score += 40000000
            temp_board[row, col] = piece
            score += evaluate_board(temp_board, piece)
            if col == center_col:
                score += 2000
            elif abs(col - center_col) == 1:
                score += 1000
            if col in current_killers:
                score += 15000
            score += history_scores.get((depth, col), 0) * 0.2

        scored_moves.append((score, col))

    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [col for _, col in scored_moves]

# --- Negamax with Alpha-Beta Pruning ---
def negamax(board: np.ndarray, depth: int, alpha: int, beta: int, piece: int, start_time: float) -> Tuple[Optional[int], int]:
    """Negamax với alpha-beta pruning."""
    if time.time() - start_time > BASE_TIMEOUT:
        return None, evaluate_board(board, piece)

    board_key = board_to_key(board)
    if board_key in transposition_table:
        stored_depth, stored_score, stored_flag, stored_best_move = transposition_table[board_key]
        if stored_depth >= depth and stored_depth != -1:
            if stored_flag == 'exact':
                return stored_best_move, stored_score
            elif stored_flag == 'lower' and stored_score >= beta:
                return stored_best_move, stored_score
            elif stored_flag == 'upper' and stored_score <= alpha:
                return stored_best_move, stored_score

    if depth == 0 or winning_move(board, piece) or winning_move(board, 3 - piece):
        if winning_move(board, piece):
            return None, 1000000000 + depth
        if winning_move(board, 3 - piece):
            return None, -1000000000 - depth
        if not get_valid_moves(board):
            return None, 0
        return None, evaluate_board(board, piece)

    opp_piece = 3 - piece
    if can_win_next(board, piece):
        for col in range(COLUMN_COUNT):
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                board[row, col] = piece
                if winning_move(board, piece):
                    board[row, col] = 0
                    return col, 1000000000 + depth
                board[row, col] = 0
        moves = get_valid_moves(board)
    else:
        moves = get_valid_moves(board)

    if not moves:
        return None, 0

    max_score = (COLUMN_COUNT * ROW_COUNT + 1 - np.count_nonzero(board)) // 2
    if beta > max_score:
        beta = max_score
        if alpha >= beta:
            return None, beta

    min_score = -(COLUMN_COUNT * ROW_COUNT - np.count_nonzero(board)) // 2
    if alpha < min_score:
        alpha = min_score
        if alpha >= beta:
            return None, alpha

    moves_order = sort_moves(board, moves, piece, depth)
    best_move = moves_order[0] if moves_order else None
    value = -1000000000

    for col in moves_order:
        row = get_next_open_row(board, col)
        if row is None:
            continue
        board[row, col] = piece
        _, score = negamax(board, depth - 1, -beta, -alpha, opp_piece, start_time)
        score = -score
        board[row, col] = 0

        if score > value:
            value = score
            best_move = col
        alpha = max(alpha, value)
        if alpha >= beta:
            killer_moves[depth].append(col)
            if len(killer_moves[depth]) > 2:
                killer_moves[depth].pop(0)
            history_scores[(depth, col)] = history_scores.get((depth, col), 0) + (2 ** depth)
            break

    flag = 'exact'
    if value <= alpha:
        flag = 'upper'
    elif value >= beta:
        flag = 'lower'
    transposition_table[board_key] = (depth, value, flag, best_move)
    return best_move, value

# --- Main Search Function ---
def find_best_move(board: np.ndarray, piece: int, valid_moves: List[int]) -> int:
    """Tìm nước đi tốt nhất."""
    start_time = time.time()
    if not valid_moves:
        logging.error("No valid moves provided!")
        return get_valid_moves(board)[0] if get_valid_moves(board) else 0

    if np.all(board == 0):
        center_col = COLUMN_COUNT // 2
        if center_col in valid_moves:
            logging.info(f"First move (player {piece}): Center col {center_col}")
            return center_col
        return valid_moves[0]

    best_move = valid_moves[0]
    best_score = -1000000000
    last_completed_depth = 0

    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is None:
            continue
        board[row, col] = piece
        if winning_move(board, piece):
            board[row, col] = 0
            logging.info(f"Immediate win found at col {col}")
            return col
        board[row, col] = 0

    opp_piece = 3 - piece
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is None:
            continue
        board[row, col] = opp_piece
        if winning_move(board, opp_piece):
            board[row, col] = 0
            logging.info(f"Immediate block found at col {col}")
            return col
        board[row, col] = 0

    for col in valid_moves:
        if detect_double_threat(board, piece, col):
            logging.info(f"Double threat found at col {col}")
            return col

    moves_order = sort_moves(board, valid_moves, piece, AI_DEPTH)
    for depth in range(1, AI_DEPTH + 1):
        current_best_move = None
        current_best_score = -1000000000
        alpha = -1000000000
        beta = 1000000000

        for col in moves_order:
            if time.time() - start_time > BASE_TIMEOUT:
                logging.warning(f"Timeout at depth {depth}. Using best move from depth {last_completed_depth}.")
                return best_move

            row = get_next_open_row(board, col)
            if row is None:
                continue
            board[row, col] = piece
            _, score = negamax(board, depth - 1, alpha, beta, piece, start_time)
            score = -score
            board[row, col] = 0

            if score > current_best_score:
                current_best_score = score
                current_best_move = col
            alpha = max(alpha, score)

        if time.time() - start_time <= BASE_TIMEOUT and current_best_move is not None:
            best_move = current_best_move
            best_score = current_best_score
            last_completed_depth = depth
            if best_move in moves_order:
                moves_order.remove(best_move)
                moves_order.insert(0, best_move)

            logging.info(f"Depth {depth} completed. Best move: {best_move}, Score: {best_score}, Time: {time.time() - start_time:.3f}s")

            if best_score >= 1000000000:
                logging.info(f"Winning move found at depth {depth}. Move: {best_move}")
                break

    if best_move not in valid_moves:
        logging.error(f"Invalid best move {best_move}. Choosing first valid move.")
        return valid_moves[0]
    return best_move

# --- Process Request Function ---
def process_request(request):
    """Xử lý yêu cầu từ game.py."""
    try:
        if not isinstance(request, dict):
            logging.error("Request must be a dictionary")
            raise ValueError("Request must be a dictionary")

        required_keys = ["board", "current_player", "valid_moves"]
        for key in required_keys:
            if key not in request:
                logging.error(f"Missing required key: {key}")
                raise ValueError(f"Missing required key: {key}")

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
        if current_player not in [1, 2]:
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

        board = np.array(board_list, dtype=np.int32)
        best_move = find_best_move(board, current_player, valid_moves)

        if best_move not in valid_moves:
            logging.warning(f"Best move {best_move} not in valid_moves. Choosing first valid move.")
            best_move = valid_moves[0] if valid_moves else 0

        return best_move

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return valid_moves[0] if valid_moves else 0

# --- Test the AI ---
if __name__ == "__main__":
    request = {
        "board": [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
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