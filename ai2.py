import numpy as np
import time
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

# --- Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_COUNT = 4
AI_DEPTH = 8  # Giảm độ sâu để cải thiện hiệu suất
BASE_TIMEOUT = 5.0  # Tăng timeout để tìm kiếm sâu hơn
TRANS_TABLE_SIZE = 2000000

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
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
    key = 0
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            piece = int(board[r, c])
            key ^= zobrist_table[r][c][piece]
    return key

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
    """Kiểm tra chuỗi 4 quân thắng với log chi tiết."""
    # Horizontal
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r, c + i] == piece for i in range(WIN_COUNT)):
                logging.debug(f"Win detected: Horizontal at row {r}, cols {c} to {c+3}")
                return True

    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all(board[r + i, c] == piece for i in range(WIN_COUNT)):
                logging.debug(f"Win detected: Vertical at col {c}, rows {r} to {r+3}")
                return True

    # Positive diagonal (/)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r + i, c + i] == piece for i in range(WIN_COUNT)):
                logging.debug(f"Win detected: Positive diagonal from ({r},{c}) to ({r+3},{c+3})")
                return True

    # Negative diagonal (\)
    for r in range(WIN_COUNT - 1, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r - i, c + i] == piece for i in range(WIN_COUNT)):
                logging.debug(f"Win detected: Negative diagonal from ({r},{c}) to ({r-3},{c+3})")
                return True

    return False

def can_win_next(board: np.ndarray, piece: int) -> List[int]:
    """Thu thập tất cả cột dẫn đến thắng ngay với log chi tiết."""
    win_cols = []
    valid_moves = get_valid_moves(board)
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = piece
            if winning_move(board, piece):
                logging.debug(f"Win possible for piece {piece} at col {col}, row {row}")
                win_cols.append(col)
            board[row, col] = 0
        else:
            logging.debug(f"Invalid move for col {col}: No open row")
    logging.debug(f"can_win_next(piece={piece}) returns: {win_cols}")
    return win_cols

def is_safe_move(board: np.ndarray, col: int, piece: int) -> bool:
    """Kiểm tra xem nước đi có tạo mối đe dọa thắng ngay cho đối thủ hay không."""
    row = get_next_open_row(board, col)
    if row is None:
        return False
    temp_board = board.copy()
    temp_board[row, col] = piece
    opp_piece = 3 - piece
    threats = can_win_next(temp_board, opp_piece)
    return len(threats) == 0

def detect_double_threat(board: np.ndarray, piece: int, col: int) -> bool:
    """Kiểm tra xem nước đi có tạo ra ít nhất hai cơ hội thắng hay không."""
    row = get_next_open_row(board, col)
    if row is None:
        return False
    temp_board = board.copy()
    temp_board[row, col] = piece
    threat_count = 0
    for next_col in get_valid_moves(temp_board):
        next_row = get_next_open_row(temp_board, next_col)
        if next_row is not None:
            temp_board[next_row, next_col] = piece
            if winning_move(temp_board, piece):
                threat_count += 1
            temp_board[next_row, next_col] = 0
    return threat_count >= 2

def count_open_threes(board: np.ndarray, piece: int) -> int:
    """Đếm số chuỗi 3 quân mở (có thể mở rộng thành 4)."""
    count = 0
    # Horizontal
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r, c + i] for i in range(WIN_COUNT)]
            if window.count(piece) == 3 and window.count(0) == 1:
                count += 1
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            window = [board[r + i, c] for i in range(WIN_COUNT)]
            if window.count(piece) == 3 and window.count(0) == 1:
                count += 1
    # Positive diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i, c + i] for i in range(WIN_COUNT)]
            if window.count(piece) == 3 and window.count(0) == 1:
                count += 1
    # Negative diagonal
    for r in range(WIN_COUNT - 1, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i, c + i] for i in range(WIN_COUNT)]
            if window.count(piece) == 3 and window.count(0) == 1:
                count += 1
    return count

# --- Evaluation Function ---
def evaluate_window(window: List[int], piece: int, is_horizontal: bool = False) -> int:
    score = 0
    opp_piece = 3 - piece
    piece_count = window.count(piece)
    opp_count = window.count(opp_piece)
    empty_count = window.count(0)

    if piece_count == 4:
        score += 1000000000
    elif piece_count == 3 and empty_count == 1:
        score += 10000000
    elif piece_count == 2 and empty_count == 2:
        score += 100000

    if opp_count == 4:
        score -= 1000000000
    elif opp_count == 3 and empty_count == 1:
        score -= 500000000 if is_horizontal else 400000000  # Tăng phạt cho chuỗi ngang
    elif opp_count == 2 and empty_count == 2:
        score -= 200000 if is_horizontal else 150000

    return score

def evaluate_board(board: np.ndarray, piece: int) -> int:
    board_key = board_to_key(board)
    if board_key in transposition_table and transposition_table[board_key][0] == -1:
        return transposition_table[board_key][1]

    score = 0
    # Ưu tiên cột giữa
    center_array = [int(board[r, COLUMN_COUNT // 2]) for r in range(ROW_COUNT)]
    center_count = center_array.count(piece)
    score += center_count * 10000

    # Đếm chuỗi 3 quân mở
    open_threes = count_open_threes(board, piece)
    opp_open_threes = count_open_threes(board, 3 - piece)
    score += open_threes * 5000000
    score -= opp_open_threes * 6000000  # Phạt nặng hơn cho đối thủ

    # Đánh giá các hướng
    for r in range(ROW_COUNT):
        row_array = [int(board[r, c]) for c in range(COLUMN_COUNT)]
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window(row_array[c:c + WIN_COUNT], piece, is_horizontal=True)

    for c in range(COLUMN_COUNT):
        col_array = [int(board[r, c]) for r in range(ROW_COUNT)]
        for r in range(ROW_COUNT - 3):
            score += evaluate_window(col_array[r:r + WIN_COUNT], piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [int(board[r + i, c + i]) for i in range(WIN_COUNT)]
            score += evaluate_window(window, piece)

    for r in range(WIN_COUNT - 1, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [int(board[r - i, c + i]) for i in range(WIN_COUNT)]
            score += evaluate_window(window, piece)

    transposition_table[board_key] = (-1, score, 'exact', None)
    return score

# --- Move Sorting ---
def sort_moves(board: np.ndarray, moves: List[int], piece: int, depth: int) -> List[int]:
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
            score += 75000000
        temp_board[row, col] = opp_piece
        if winning_move(temp_board, opp_piece):
            score += 90000000
        temp_board[row, col] = piece
        score += evaluate_board(temp_board, piece)
        if col == center_col:
            score += 20000
        elif abs(col - center_col) == 1:
            score += 10000
        if col in current_killers:
            score += 50000
        score += history_scores.get((depth, col), 0) * 0.5

        scored_moves.append((score, col))

    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [col for _, col in scored_moves]

# --- Negamax with Alpha-Beta Pruning ---
def negamax(board: np.ndarray, depth: int, alpha: int, beta: int, piece: int, start_time: float) -> Tuple[Optional[int], int]:
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
    win_cols = can_win_next(board, piece)
    if win_cols:
        return win_cols[0], 1000000000 + depth

    block_cols = can_win_next(board, opp_piece)
    if block_cols:
        return block_cols[0], 900000000 + depth

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

    win_cols = can_win_next(board, piece)
    if win_cols and win_cols[0] in valid_moves:
        logging.info(f"Immediate win found at col {win_cols[0]}")
        return win_cols[0]

    opp_piece = 3 - piece
    block_cols = can_win_next(board, opp_piece)
    if len(block_cols) == 1 and block_cols[0] in valid_moves:
        logging.info(f"Immediate block found at col {block_cols[0]}")
        return block_cols[0]
    elif len(block_cols) > 1:
        # Ưu tiên nước đi an toàn
        safe_cols = [col for col in block_cols if col in valid_moves and is_safe_move(board, col, piece)]
        if safe_cols:
            logging.info(f"Safe block found at col {safe_cols[0]}")
            return safe_cols[0]
        # Ưu tiên chặn chuỗi ngang
        horizontal_threat_cols = []
        for col in block_cols:
            if col not in valid_moves:
                continue
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            temp_board[row, col] = opp_piece
            if winning_move(temp_board, opp_piece):
                for r in range(ROW_COUNT):
                    for c in range(COLUMN_COUNT - 3):
                        if all(temp_board[r, c + i] == opp_piece for i in range(WIN_COUNT)):
                            horizontal_threat_cols.append(col)
                            break
            temp_board[row, col] = 0
        if horizontal_threat_cols:
            best_block_col = None
            best_future_threats = float('inf')
            for col in horizontal_threat_cols:
                row = get_next_open_row(board, col)
                temp_board = board.copy()
                temp_board[row, col] = piece
                future_threats = len(can_win_next(temp_board, opp_piece))
                if future_threats < best_future_threats:
                    best_future_threats = future_threats
                    best_block_col = col
                elif future_threats == best_future_threats:
                    score = evaluate_board(temp_board, piece)
                    if best_block_col is None or score > evaluate_board(board, piece):
                        best_block_col = col
                temp_board[row, col] = 0
            if best_block_col is not None:
                logging.info(f"Blocking horizontal threat at col {best_block_col} with {best_future_threats} future threats")
                return best_block_col
        # Chọn cột với ít mối đe dọa nhất
        best_block_col = None
        best_future_threats = float('inf')
        for col in block_cols:
            if col not in valid_moves:
                continue
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            temp_board[row, col] = piece
            future_threats = len(can_win_next(temp_board, opp_piece))
            if future_threats < best_future_threats:
                best_future_threats = future_threats
                best_block_col = col
            elif future_threats == best_future_threats:
                score = evaluate_board(temp_board, piece)
                if best_block_col is None or score > evaluate_board(board, piece):
                    best_block_col = col
            temp_board[row, col] = 0
        if best_block_col is not None:
            logging.info(f"Multiple threats detected, blocking col {best_block_col} with {best_future_threats} future threats")
            return best_block_col

    for col in valid_moves:
        if detect_double_threat(board, piece, col):
            logging.info(f"Double threat found at col {col}")
            return col

    best_move = valid_moves[0]
    best_score = -1000000000
    last_completed_depth = 0

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
            _, score = negamax(board, depth - 1, -beta, -alpha, opp_piece, start_time)
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

# --- Test the AI with Critical Board State ---
if __name__ == "__main__":
    request = {
        "board": [
            [1, 1, 2, 1, 0, 2, 0],
            [1, 2, 2, 2, 0, 0, 0],
            [0, 1, 2, 1, 0, 0, 0],
            [0, 2, 1, 2, 0, 0, 0],
            [0, 0, 1, 2, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0]
        ],
        "current_player": 1,
        "valid_moves": [0, 1, 3, 4, 5, 6]
    }
    best_move = process_request(request)
    print(f"Best move: {best_move}")  # Mong đợi: 1