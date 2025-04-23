from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
import numpy as np
import math
import time
import logging
from collections import OrderedDict
from multiprocessing import Pool, cpu_count # Nếu bạn muốn dùng đa luồng
from fastapi.middleware.cors import CORSMiddleware

# --- Các hằng số (Copy từ AI Đối Thủ hoặc điều chỉnh) ---
ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_COUNT = 4
# Điều chỉnh độ sâu và thời gian nếu cần
# AI_DEPTH = 18 # Có thể bắt đầu với độ sâu thấp hơn để test, ví dụ 8-10
AI_DEPTH = 9 # Độ sâu ban đầu để test
BASE_TIMEOUT = 3.8 # Thời gian của bạn, nên thấp hơn đối thủ một chút để chắc chắn phản hồi kịp
TRANS_TABLE_SIZE = 1000000 # Điều chỉnh kích thước bảng nếu cần
# CPU_CORES = min(max(1, cpu_count() - 1), 8) # Bật nếu dùng Pool

# --- Cấu hình Logging (Tùy chọn) ---
logging.basicConfig(filename='my_ai.log', level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Các Model Pydantic (Giữ nguyên) ---
class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

# --- Lớp LimitedDict (Copy từ AI Đối Thủ) ---
class LimitedDict(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            # Chiến lược xóa đơn giản: xóa entry cũ nhất
            # Hoặc có thể xóa entry có độ sâu thấp nhất như AI đối thủ
            self.popitem(last=False) # Xóa entry cũ nhất (FIFO)
            # Hoặc:
            # min_depth_key = min(self, key=lambda k: self[k][0])
            # self.pop(min_depth_key)

# --- Các biến toàn cục (Copy và khởi tạo) ---
transposition_table: LimitedDict = LimitedDict(maxsize=TRANS_TABLE_SIZE)
# Bạn có thể dùng hoặc không dùng killer_moves/history_scores tùy vào độ phức tạp muốn xây dựng
killer_moves: Dict[int, List[int]] = {d: [] for d in range(AI_DEPTH + 1)}
history_scores: Dict[Tuple[int, int], int] = {}
# threat_scores: Dict[Tuple[int, int], int] = {} # Có thể thêm nếu muốn


# --- Các hàm Helper (Copy từ AI Đối Thủ) ---
# board_to_key, is_valid_location, get_next_open_row, get_valid_moves,
# winning_move, terminal_node

def board_to_key(board: np.ndarray) -> str:
    # Sử dụng tobytes().hex() như đối thủ hoặc thử Zobrist Hashing
    return board.tobytes().hex()

def is_valid_location(board: np.ndarray, col: int) -> bool:
    # Kiểm tra xem cột có còn chỗ trống ở hàng trên cùng không
    return board[0, col] == 0

def get_next_open_row(board: np.ndarray, col: int) -> Optional[int]:
    # Tìm hàng trống thấp nhất trong cột đã chọn
    for r in range(ROW_COUNT - 1, -1, -1):
        if board[r, col] == 0:
            return r
    return None # Should not happen if is_valid_location is checked first

def get_valid_moves(board: np.ndarray) -> List[int]:
    # Lấy danh sách các cột hợp lệ (còn ô trống)
    valid_moves = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_moves.append(col)
    # Ưu tiên cột giữa
    center_col = COLUMN_COUNT // 2
    if center_col in valid_moves:
        valid_moves.remove(center_col)
        valid_moves.insert(0, center_col)
    return valid_moves

def winning_move(board: np.ndarray, piece: int) -> bool:
    # Kiểm tra xem người chơi 'piece' đã thắng chưa
    # Check horizontal locations
    for c in range(COLUMN_COUNT - (WIN_COUNT - 1)):
        for r in range(ROW_COUNT):
            if all(board[r, c + i] == piece for i in range(WIN_COUNT)):
                return True
    # Check vertical locations
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - (WIN_COUNT - 1)):
            if all(board[r + i, c] == piece for i in range(WIN_COUNT)):
                return True
    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT - (WIN_COUNT - 1)):
        for r in range(ROW_COUNT - (WIN_COUNT - 1)):
            if all(board[r + i, c + i] == piece for i in range(WIN_COUNT)):
                return True
    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT - (WIN_COUNT - 1)):
        for r in range(WIN_COUNT - 1, ROW_COUNT):
            if all(board[r - i, c + i] == piece for i in range(WIN_COUNT)):
                return True
    return False

def terminal_node(board: np.ndarray) -> bool:
    # Kiểm tra trạng thái kết thúc game (thắng hoặc hòa)
    return winning_move(board, 1) or winning_move(board, 2) or len(get_valid_moves(board)) == 0


# --- Hàm Lượng Giá (Copy từ AI Đối Thủ và **CHỈNH SỬA Ở ĐÂY**) ---
def evaluate_window(window: List[int], piece: int) -> int:
    # Đánh giá một cửa sổ 4 ô
    score = 0
    opp_piece = 3 - piece
    piece_count = window.count(piece)
    opp_count = window.count(opp_piece)
    empty_count = window.count(0)

    if piece_count == 4:
        score += 100000  # Điểm thắng cao
    elif piece_count == 3 and empty_count == 1:
        score += 100  # Điểm cho 3 quân liên tiếp
    elif piece_count == 2 and empty_count == 2:
        score += 5   # Điểm cho 2 quân liên tiếp
    # *** CẢI TIẾN: Thêm điểm thưởng nếu 2 quân nằm giữa cửa sổ? ***
    # if window == [0, piece, piece, 0]: score += 3

    if opp_count == 4:
        score -= 100000 # Phạt nặng nếu đối thủ sắp thắng (dùng trong minimax, không phải immediate block)
    elif opp_count == 3 and empty_count == 1:
        score -= 80  # Phạt khi đối thủ có 3 quân (ít hơn điểm thưởng của mình)
        # *** CẢI TIẾN: Phạt nặng hơn nếu là lượt đối thủ và ô trống đó hợp lệ? ***
    elif opp_count == 2 and empty_count == 2:
        score -= 3

    return score

def evaluate_board(board: np.ndarray, piece: int) -> int:
    # Đánh giá toàn bộ bàn cờ
    score = 0
    opp_piece = 3 - piece

    # *** CẢI TIẾN: Thay đổi trọng số cột giữa? ***
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 6 # Điểm thưởng cột giữa

    # Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - (WIN_COUNT - 1)):
            window = row_array[c:c + WIN_COUNT]
            score += evaluate_window(window, piece)

    # Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - (WIN_COUNT - 1)):
            window = col_array[r:r + WIN_COUNT]
            score += evaluate_window(window, piece)

    # Score positive sloped diagonal
    for r in range(ROW_COUNT - (WIN_COUNT - 1)):
        for c in range(COLUMN_COUNT - (WIN_COUNT - 1)):
            window = [board[r + i, c + i] for i in range(WIN_COUNT)]
            score += evaluate_window(window, piece)

    # Score negative sloped diagonal
    for r in range(WIN_COUNT - 1, ROW_COUNT):
        for c in range(COLUMN_COUNT - (WIN_COUNT - 1)):
            window = [board[r - i, c + i] for i in range(WIN_COUNT)]
            score += evaluate_window(window, piece)

    # *** CẢI TIẾN: Thêm logic nhận dạng mối đe dọa phức tạp hơn? ***
    # Ví dụ: phát hiện dual threat (như AI đối thủ nhưng có thể với trọng số khác)

    return score

# --- Hàm Sắp xếp Nước đi (Copy và tùy chỉnh) ---
def sort_moves(board: np.ndarray, moves: List[int], piece: int, depth: int) -> List[int]:
    # Sắp xếp các nước đi để tối ưu Alpha-Beta
    # Ưu tiên: Thắng > Chặn > Tạo 3 quân > Tạo 2 quân > Cột giữa > Các nước khác
    # Có thể thêm Killer Moves / History Heuristics vào đây
    scored_moves = []
    opp_piece = 3 - piece
    center_col = COLUMN_COUNT // 2

    # Lấy killer moves cho độ sâu hiện tại (nếu dùng)
    current_killers = killer_moves.get(depth, [])

    for col in moves:
        score = 0
        temp_board = board.copy()
        row = get_next_open_row(temp_board, col)
        if row is None: continue # Bỏ qua nếu cột đầy (dù get_valid_moves đã lọc)

        # 1. Kiểm tra thắng ngay
        temp_board[row, col] = piece
        if winning_move(temp_board, piece):
            score += 1000000
        else:
            # 2. Kiểm tra chặn ngay
            temp_board[row, col] = opp_piece # Giả sử đối thủ đi vào ô này
            # *Nhưng* đối thủ sẽ đi vào ô *của họ* nếu họ có thể thắng
            can_opp_win = False
            temp_board_opp_check = board.copy()
            for opp_col in get_valid_moves(temp_board_opp_check):
                opp_row = get_next_open_row(temp_board_opp_check, opp_col)
                if opp_row is not None:
                    temp_board_opp_check[opp_row, opp_col] = opp_piece
                    if winning_move(temp_board_opp_check, opp_piece):
                        can_opp_win = True
                        # Nếu nước đi hiện tại (col) chặn được nước thắng đó của đối thủ
                        if opp_col == col:
                            score += 500000 # Điểm chặn thắng rất cao
                        temp_board_opp_check[opp_row, opp_col] = 0 # Reset
                        break # Chỉ cần tìm 1 nước thắng của đối thủ
                    temp_board_opp_check[opp_row, opp_col] = 0 # Reset

            # 3. Điểm lượng giá heuristic (nếu không phải thắng/chặn)
            if score < 500000: # Nếu chưa phải là nước thắng hoặc chặn
                temp_board[row, col] = piece # Đặt lại quân của mình để lượng giá
                score += evaluate_board(temp_board, piece) * 0.1 # Dùng điểm lượng giá với trọng số thấp hơn

                # 4. Điểm cột giữa
                if col == center_col:
                    score += 10
                elif abs(col - center_col) == 1:
                    score += 5

                # 5. Điểm Killer Moves (nếu dùng)
                if col in current_killers:
                    score += 100 # Ưu tiên killer move

                # 6. Điểm History Heuristics (nếu dùng)
                score += history_scores.get((depth, col), 0) * 0.01

        temp_board[row, col] = 0 # Reset lại bảng
        scored_moves.append((score, col))

    scored_moves.sort(key=lambda x: x[0], reverse=True)
    # logging.info(f"Sorted moves (depth {depth}): {scored_moves}")
    return [col for score, col in scored_moves]


# --- Hàm Minimax (Copy và tùy chỉnh) ---
def minimax(board: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool,
            piece: int, start_time: float) -> Tuple[Optional[int], float]:
    # Thuật toán Minimax với Alpha-Beta, Transposition Table, Time Limit

    # Kiểm tra thời gian
    if time.time() - start_time > BASE_TIMEOUT:
        # logging.warning(f"Timeout reached at depth {depth}")
        return None, evaluate_board(board, piece if maximizing_player else 3 - piece) # Trả về heuristic nếu hết giờ

    board_key = board_to_key(board)

    # Kiểm tra Transposition Table
    if board_key in transposition_table:
        stored_depth, stored_score, stored_best_move = transposition_table[board_key]
        if stored_depth >= depth:
            # logging.info(f"TT Hit: key={board_key[:5]}..., depth={depth}, stored_depth={stored_depth}")
            # Cần trả về cả best_move nếu có thể, nhưng hàm này chỉ trả score ở nút lá
            return stored_best_move, stored_score # Trả về score đã lưu

    valid_moves = get_valid_moves(board)
    is_terminal = terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, piece):
                return None, 1000000 + depth # Thắng càng sớm càng tốt
            elif winning_move(board, 3 - piece):
                return None, -1000000 - depth # Thua càng muộn càng tốt
            else: # Hòa
                return None, 0
        else: # Depth is zero
            return None, evaluate_board(board, piece) # Trả về điểm lượng giá

    # Sắp xếp nước đi
    moves_order = sort_moves(board, valid_moves, piece if maximizing_player else 3 - piece, depth)
    best_move = moves_order[0] if moves_order else None

    if maximizing_player:
        value = -math.inf
        for col in moves_order:
            row = get_next_open_row(board, col)
            if row is None: continue
            board[row, col] = piece # Make the move
            _, new_score = minimax(board, depth - 1, alpha, beta, False, piece, start_time)
            board[row, col] = 0 # Undo the move

            if new_score > value:
                value = new_score
                best_move = col
            alpha = max(alpha, value)
            if alpha >= beta:
                # Beta cutoff - lưu killer move và history score (nếu dùng)
                if col not in killer_moves.get(depth, []):
                    killer_moves.setdefault(depth, []).append(col)
                    if len(killer_moves[depth]) > 2: # Giữ 2 killer moves
                        killer_moves[depth].pop(0)
                history_scores[(depth, col)] = history_scores.get((depth, col), 0) + (2 ** depth)
                break # Prune
        # Lưu vào Transposition Table
        transposition_table[board_key] = (depth, value, best_move)
        return best_move, value

    else: # Minimizing player
        value = math.inf
        for col in moves_order:
            row = get_next_open_row(board, col)
            if row is None: continue
            board[row, col] = 3 - piece # Make the move (opponent's piece)
            _, new_score = minimax(board, depth - 1, alpha, beta, True, piece, start_time)
            board[row, col] = 0 # Undo the move

            if new_score < value:
                value = new_score
                best_move = col
            beta = min(beta, value)
            if alpha >= beta:
                # Alpha cutoff - lưu killer move và history score (nếu dùng)
                if col not in killer_moves.get(depth, []):
                    killer_moves.setdefault(depth, []).append(col)
                    if len(killer_moves[depth]) > 2:
                        killer_moves[depth].pop(0)
                history_scores[(depth, col)] = history_scores.get((depth, col), 0) + (2 ** depth)
                break # Prune
        # Lưu vào Transposition Table
        transposition_table[board_key] = (depth, value, best_move)
        return best_move, value

# --- Hàm Tìm kiếm Chính (Iterative Deepening) ---
def find_best_move(board: np.ndarray, piece: int, max_depth: int) -> int:
    start_time = time.time()
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return -1 # Không còn nước đi

    # 0. Nước đi đầu tiên -> cột giữa
    if np.all(board == 0):
        center_col = COLUMN_COUNT // 2
        if center_col in valid_moves:
            logging.info(f"First move (player {piece}): Center col {center_col}")
            return center_col
        else: # Nếu cột giữa không hợp lệ (không nên xảy ra)
            return valid_moves[0]

    best_move_overall = valid_moves[0] # Nước đi mặc định

    # 1. Kiểm tra thắng ngay
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = piece
            if winning_move(board, piece):
                board[row, col] = 0 # Undo
                logging.info(f"Immediate win found for player {piece} at col {col}")
                return col
            board[row, col] = 0 # Undo

    # 2. Kiểm tra chặn thắng ngay
    opp_piece = 3 - piece
    for col in valid_moves:
        row = get_next_open_row(board, col)
        if row is not None:
            board[row, col] = opp_piece
            if winning_move(board, opp_piece):
                board[row, col] = 0 # Undo
                logging.info(f"Immediate block found for player {piece} at col {col}")
                return col # Phải chặn nước này
            board[row, col] = 0 # Undo

    # 3. Iterative Deepening Search
    best_score_overall = -math.inf
    last_completed_depth = 0

    # Sắp xếp nước đi ban đầu một lần
    initial_moves_order = sort_moves(board, valid_moves, piece, max_depth) # Sắp xếp ở độ sâu cao nhất

    for depth in range(1, max_depth + 1):
        current_best_move_depth = -1
        current_best_score_depth = -math.inf
        alpha = -math.inf
        beta = math.inf

        # Dùng thứ tự đã sắp xếp
        moves_to_search = initial_moves_order

        for col in moves_to_search:
            if time.time() - start_time > BASE_TIMEOUT:
                logging.warning(f"Timeout during iterative deepening at depth {depth}. Using best move from depth {last_completed_depth}.")
                # Trả về best_move_overall từ độ sâu hoàn thành cuối cùng
                if best_move_overall not in valid_moves: # Kiểm tra lại nếu move cũ không còn hợp lệ
                    return valid_moves[0] if valid_moves else -1
                return best_move_overall

            row = get_next_open_row(board, col)
            if row is None: continue

            board[row, col] = piece
            _, score = minimax(board, depth - 1, alpha, beta, False, piece, start_time)
            board[row, col] = 0 # Undo move

            if score > current_best_score_depth:
                current_best_score_depth = score
                current_best_move_depth = col

            alpha = max(alpha, score) # Cập nhật alpha cho các nước đi cùng cấp

        # Kết thúc duyệt các nước đi ở độ sâu 'depth'
        if time.time() - start_time <= BASE_TIMEOUT: # Chỉ cập nhật nếu hoàn thành trong thời gian
            best_move_overall = current_best_move_depth
            best_score_overall = current_best_score_depth
            last_completed_depth = depth
            # Cập nhật lại thứ tự cho lần lặp sâu hơn (Principal Variation)
            if best_move_overall in initial_moves_order:
                initial_moves_order.remove(best_move_overall)
                initial_moves_order.insert(0, best_move_overall)

            logging.info(f"Depth {depth} completed. Best move: {best_move_overall}, Score: {best_score_overall:.0f}, Time: {time.time() - start_time:.3f}s")

            # Nếu tìm thấy nước thắng chắc chắn ở độ sâu này
            if best_score_overall >= 1000000:
                logging.info(f"Winning move found at depth {depth}. Move: {best_move_overall}")
                break # Dừng tìm kiếm

        else:
            logging.warning(f"Timeout after completing depth {last_completed_depth}. Using its result: {best_move_overall}")
            break # Dừng nếu hết giờ sau khi hoàn thành một độ sâu

    # Đảm bảo nước đi cuối cùng hợp lệ
    if best_move_overall not in valid_moves:
        logging.error(f"Selected best move {best_move_overall} is invalid! Fallback.")
        return valid_moves[0] if valid_moves else -1 # Fallback

    return best_move_overall


# --- API Endpoint ---
@app.post("/api/test")
async def health_check():
    return {"status":"ok","message":"Server is running"}

async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")

        # Chuyển đổi board sang numpy array
        board = np.array(game_state.board, dtype=np.int32)
        # Đảm bảo board có thể ghi được
        board.flags.writeable = True

        player = game_state.current_player
        valid_moves = game_state.valid_moves

        logging.info(f"Received board (player {player}):\n{board}\nValid moves: {valid_moves}")

        # Gọi hàm tìm kiếm chính
        # selected_move = find_best_move_simple(board, player) # Phiên bản đơn giản ban đầu
        selected_move = find_best_move(board, player, AI_DEPTH) # Phiên bản Iterative Deepening

        # Fallback nếu không tìm được nước đi (không nên xảy ra nếu valid_moves không rỗng)
        if selected_move == -1 or selected_move not in valid_moves:
            logging.warning(f"Minimax did not return a valid move ({selected_move}). Falling back to first valid move.")
            selected_move = valid_moves[0]

        logging.info(f"AI (player {player}) selected move: {selected_move}")
        return AIResponse(move=selected_move)

    except Exception as e:
        logging.error(f"Error in make_move: {str(e)}", exc_info=True)
        # Fallback an toàn: trả về nước đi hợp lệ đầu tiên nếu có lỗi
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        # Nếu không còn nước đi hợp lệ và có lỗi, raise HTTP exception
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý nước đi: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) # Chạy server