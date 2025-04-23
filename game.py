import numpy as np
import pygame
import sys
import math
import time
import logging
import asyncio
import platform
from ai1 import process_request as ai1_process_request
from ai2 import process_request as ai2_process_request

# --- Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
WIN_COUNT = 4
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
FPS = 60
MOVE_DELAY = 0.5

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# --- Helper Functions ---
def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
    return None

def get_valid_columns(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def winning_move(board, piece):
    for c in range(COLUMN_COUNT - (WIN_COUNT - 1)):
        for r in range(ROW_COUNT):
            if all(board[r][c + i] == piece for i in range(WIN_COUNT)):
                return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - (WIN_COUNT - 1)):
            if all(board[r + i][c] == piece for i in range(WIN_COUNT)):
                return True
    for c in range(COLUMN_COUNT - (WIN_COUNT - 1)):
        for r in range(ROW_COUNT - (WIN_COUNT - 1)):
            if all(board[r + i][c + i] == piece for i in range(WIN_COUNT)):
                return True
    for c in range(COLUMN_COUNT - (WIN_COUNT - 1)):
        for r in range(WIN_COUNT - 1, ROW_COUNT):
            if all(board[r - i][c + i] == piece for i in range(WIN_COUNT)):
                return True
    return False

def is_board_full(board):
    return all(board[ROW_COUNT - 1][col] != 0 for col in range(COLUMN_COUNT))

def draw_board(board, screen):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2), int((r + 1) * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), int((ROW_COUNT - r) * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2), int((ROW_COUNT - r) * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()

async def animate_drop(board, screen, col, row, piece):
    x = int(col * SQUARESIZE + SQUARESIZE / 2)
    color = RED if piece == 1 else YELLOW
    drop_speed = 30
    y_start = SQUARESIZE // 2
    y_end = int((ROW_COUNT - row) * SQUARESIZE + SQUARESIZE / 2)

    y = y_start
    while y < y_end:
        draw_board(board, screen)
        pygame.draw.circle(screen, color, (x, y), RADIUS)
        pygame.display.update()
        y += drop_speed
        await asyncio.sleep(1.0 / FPS)

    board[row][col] = piece
    draw_board(board, screen)

def clear_message_area(screen):
    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
    pygame.display.update()

def log_board_state(board):
    board_str = '\n'.join([' '.join(map(str, row)) for row in board])
    logging.info(f"Current board state:\n{board_str}")

def create_request(board, current_player, valid_moves):
    # Convert NumPy array to list for JSON serialization
    board_list = board.tolist()
    return {
        "board": board_list,
        "current_player": current_player,
        "valid_moves": valid_moves
    }

# --- Game Simulation ---
def setup():
    global screen, board, game_over, turn, myfont, button_font, status_font
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Connect 4 - AI vs AI")
    board = create_board()
    game_over = False
    turn = 0
    myfont = pygame.font.SysFont("monospace", 75)
    button_font = pygame.font.SysFont("monospace", 40)
    status_font = pygame.font.SysFont("monospace", 30)
    draw_board(board, screen)

def reset_game():
    global board, game_over, turn
    board = create_board()
    game_over = False
    turn = 0
    draw_board(board, screen)

async def update_loop():
    global game_over, turn
    if not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        piece = 1 if turn == 0 else 2
        ai_name = "AI 2 (Smarter)" if turn == 0 else "AI 1 (Simple)"
        ai_func = ai2_process_request if turn == 0 else ai1_process_request
        logging.info(f"Turn: {ai_name} (Player {piece})")

        # Create request for the AI
        valid_moves = get_valid_columns(board)
        request = create_request(board, piece, valid_moves)
        logging.info(f"Sending request to {ai_name}: {request}")

        clear_message_area(screen)
        screen.blit(myfont.render(f"{ai_name} thinking...", 1, WHITE), (40, 10))
        pygame.display.update()

        try:
            col = ai_func(request)
            logging.info(f"{ai_name} selected column: {col}")
        except Exception as e:
            logging.error(f"Error in {ai_name}: {str(e)}")
            clear_message_area(screen)
            screen.blit(myfont.render(f"Error in {ai_name}!", 1, WHITE), (40, 10))
            pygame.display.update()
            game_over = True
            await asyncio.sleep(1)
            return

        valid_cols = get_valid_columns(board)
        if col is None or col not in valid_cols:
            logging.error(f"{ai_name} returned invalid column {col}. Valid columns: {valid_cols}")
            clear_message_area(screen)
            screen.blit(myfont.render(f"Invalid move by {ai_name}!", 1, WHITE), (40, 10))
            pygame.display.update()
            game_over = True
            await asyncio.sleep(1)
            return

        clear_message_area(screen)
        screen.blit(status_font.render(f"{ai_name} chooses column {col}", 1, WHITE), (40, 10))
        pygame.display.update()
        await asyncio.sleep(0.5)

        row = get_next_open_row(board, col)
        await animate_drop(board, screen, col, row, piece)
        logging.info(f"{ai_name} (Player {piece}) drops in column {col}")

        log_board_state(board)

        if winning_move(board, piece):
            logging.info(f"{ai_name} wins!")
            clear_message_area(screen)
            label = myfont.render(f"{ai_name} wins!", 1, RED if piece == 1 else YELLOW)
            screen.blit(label, (40, 10))
            pygame.display.update()
            game_over = True
            await asyncio.sleep(3)
        elif is_board_full(board):
            logging.info("Game ended in a draw!")
            clear_message_area(screen)
            label = myfont.render("Draw!", 1, WHITE)
            screen.blit(label, (40, 10))
            pygame.display.update()
            game_over = True
            await asyncio.sleep(3)
        else:
            turn = 1 - turn
            await asyncio.sleep(MOVE_DELAY)

    if game_over:
        draw_board(board, screen)
        clear_message_area(screen)
        button_rect = pygame.Rect(width // 2 - 100, SQUARESIZE // 2 - 25, 200, 50)
        pygame.draw.rect(screen, GREEN, button_rect)
        label = button_font.render("Replay", 1, BLACK)
        screen.blit(label, (width // 2 - label.get_width() // 2, SQUARESIZE // 2 - label.get_height() // 2))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    reset_game()

async def main():
    setup()
    while True:
        await update_loop()
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())