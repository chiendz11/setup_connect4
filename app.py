from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import math

# Khởi tạo FastAPI và middleware CORS
app = FastAPI()

# Định nghĩa mô hình dữ liệu
class GameState(BaseModel):
    board: List[List[int]]  # Bảng trò chơi 6x7
    current_player: int     # Người chơi hiện tại (1 hoặc 2)
    valid_moves: List[int]  # Danh sách các cột còn trống

class AIResponse(BaseModel):
    move: int               # Nước đi AI chọn

# Các hằng số
ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4
AI_DEPTH = 8  # Độ sâu tìm kiếm của minimax

# Hàm kiểm tra nước đi hợp lệ
def is_valid_location(board, col):
    return board[0][col] == 0

# Hàm tìm hàng trống tiếp theo trong cột
def get_next_open_row(board, col):
    for r in range(ROW_COUNT - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return None

# Hàm lấy danh sách các nước đi hợp lệ
def get_valid_moves(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]

# Hàm kiểm tra thắng (tối ưu hóa: chỉ kiểm tra quanh vị trí vừa đặt)
def winning_move(board, piece, row, col):
    # Kiểm tra hàng ngang
    for c in range(max(0, col - 3), min(COLUMN_COUNT - 3, col + 1)):
        if all(board[row][c + i] == piece for i in range(4)):
            return True

    # Kiểm tra cột dọc
    if row <= ROW_COUNT - 4:
        if all(board[row + i][col] == piece for i in range(4)):
            return True

    # Kiểm tra đường chéo tăng (/)
    for offset in range(-3, 1):
        r, c = row + offset, col - offset
        if 0 <= r <= ROW_COUNT - 4 and 0 <= c <= COLUMN_COUNT - 4:
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True

    # Kiểm tra đường chéo giảm (\)
    for offset in range(-3, 1):
        r, c = row - offset, col - offset
        if 3 <= r < ROW_COUNT and 0 <= c <= COLUMN_COUNT - 4:
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True

    return False

# Hàm đánh giá một window 4 ô
def evaluate_window(window, piece, opp_piece):
    score = 0
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 20  # Thưởng cho đường 3 mở
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 10  # Thưởng cho đường 2 mở
    if window.count(opp_piece) == 3 and window.count(0) == 1:
        score -= 80  # Phạt nặng khi đối thủ có đường 3
    if window.count(opp_piece) == 2 and window.count(0) == 2:
        score -= 10  # Phạt khi đối thủ có đường 2 mở
    return score

# Hàm đánh giá toàn bộ bảng
def evaluate_board(board, piece):
    score = 0
    opp_piece = 1 if piece == 2 else 2

    # Ưu tiên cột giữa
    center_col = COLUMN_COUNT // 2
    center_array = [board[r][center_col] for r in range(ROW_COUNT)]
    score += center_array.count(piece) * 12  # Thưởng cho việc chiếm cột giữa

    # Đánh giá các window ngang
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, opp_piece)

    # Đánh giá các window dọc
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            window = [board[r + i][c] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, opp_piece)

    # Đánh giá các window chéo tăng (/)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, opp_piece)

    # Đánh giá các window chéo giảm (\)
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, opp_piece)

    return score

# Hàm kiểm tra trạng thái kết thúc trò chơi
def terminal_node(board):
    return any(winning_move(board, p, r, c) 
               for p in [1, 2] 
               for r in range(ROW_COUNT) 
               for c in range(COLUMN_COUNT) 
               if board[r][c] != 0) or len(get_valid_moves(board)) == 0

# Thuật toán minimax với alpha-beta pruning
def minimax(board, depth, alpha, beta, maximizingPlayer, piece):
    valid_locations = get_valid_moves(board)
    is_terminal = terminal_node(board)
    opp_piece = 1 if piece == 2 else 2

    if depth == 0 or is_terminal:
        if is_terminal:
            for r in range(ROW_COUNT):
                for c in range(COLUMN_COUNT):
                    if board[r][c] == piece and winning_move(board, piece, r, c):
                        return (None, math.inf)
                    elif board[r][c] == opp_piece and winning_move(board, opp_piece, r, c):
                        return (None, -math.inf)
            return (None, 0)  # Hòa
        return (None, evaluate_board(board, piece))

    # Sắp xếp nước đi: ưu tiên cột giữa
    valid_locations.sort(key=lambda x: abs(x - COLUMN_COUNT // 2))

    if maximizingPlayer:
        value = -math.inf
        best_col = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            temp_board = [row[:] for row in board]  # Sao chép bảng
            temp_board[row][col] = piece
            new_score = minimax(temp_board, depth - 1, alpha, beta, False, piece)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = math.inf
        best_col = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            temp_board = [row[:] for row in board]  # Sao chép bảng
            temp_board[row][col] = opp_piece
            new_score = minimax(temp_board, depth - 1, alpha, beta, True, piece)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

# API endpoint để AI thực hiện nước đi
@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")

        board = game_state.board
        current_player = game_state.current_player

        # Gọi minimax để chọn nước đi
        move, _ = minimax(board, AI_DEPTH, -math.inf, math.inf, True, current_player)

        # Nếu minimax thất bại, chọn nước đi hợp lệ đầu tiên
        if move is None or move not in game_state.valid_moves:
            move = game_state.valid_moves[0]

        return AIResponse(move=move)
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

# Chạy ứng dụng
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)