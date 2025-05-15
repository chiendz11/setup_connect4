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
MOVE_DELAY = 0.01
AUTO_RUN = True
TOTAL_GAMES = 20
AI_TIMEOUT = 10.0

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('game_log.txt'),
        logging.StreamHandler()
    ]
)

# --- Game Statistics ---
class GameStats:
    def __init__(self):
        self.wins_ai1 = 0
        self.wins_ai2 = 0
        self.draws = 0
        self.losses_ai2 = 0
        self.moves_count = []  # Số nước mỗi ván
        self.thinking_times = {1: [], 2: []}  # Thời gian suy nghĩ của mỗi AI
        self.first_player_wins = 0  # Số ván thắng khi đi trước
        self.second_player_wins = 0  # Số ván thắng khi đi sau
        self.winning_patterns = []  # Lưu pattern thắng
        self.game_replays = []  # Lưu replay các ván đấu

    def update(self, winner, last_move, current_player, is_first_player, moves_in_game, thinking_time):
        if winner == "AI 2 (Smarter)":
            self.wins_ai2 += 1
            if is_first_player:
                self.first_player_wins += 1
            else:
                self.second_player_wins += 1
        elif winner == "AI 1 (Simple)":
            self.wins_ai1 += 1
            self.losses_ai2 += 1
        else:
            self.draws += 1
        
        self.moves_count.append(moves_in_game)
        self.thinking_times[current_player].append(thinking_time)

    def get_stats_summary(self):
        total_games = self.wins_ai1 + self.wins_ai2 + self.draws
        avg_moves = sum(self.moves_count) / len(self.moves_count) if self.moves_count else 0
        avg_time_ai1 = sum(self.thinking_times[1]) / len(self.thinking_times[1]) if self.thinking_times[1] else 0
        avg_time_ai2 = sum(self.thinking_times[2]) / len(self.thinking_times[2]) if self.thinking_times[2] else 0
        
        return (
            f"Games played: {total_games}\n"
            f"AI2 Wins: {self.wins_ai2} ({self.wins_ai2/total_games*100:.1f}%)\n"
            f"AI1 Wins: {self.wins_ai1} ({self.wins_ai1/total_games*100:.1f}%)\n"
            f"Draws: {self.draws} ({self.draws/total_games*100:.1f}%)\n"
            f"First player wins: {self.first_player_wins} ({self.first_player_wins/total_games*100:.1f}%)\n"
            f"Average moves per game: {avg_moves:.1f}\n"
            f"Average thinking time - AI1: {avg_time_ai1:.3f}s, AI2: {avg_time_ai2:.3f}s"
        )

# --- Game Replay ---
class GameReplay:
    def __init__(self):
        self.moves = []
        self.thinking_times = []
        self.board_states = []
        self.winner = None
        self.first_player = None

    def add_move(self, player, move, thinking_time, board_state):
        self.moves.append((player, move))
        self.thinking_times.append(thinking_time)
        self.board_states.append(board_state.copy())

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(f"Winner: {self.winner}\n")
            f.write(f"First player: {self.first_player}\n")
            for i, (player, move) in enumerate(self.moves):
                f.write(f"Move {i+1}: Player {player} -> {move} ({self.thinking_times[i]:.3f}s)\n")
            f.write("\nFinal board state:\n")
            f.write(str(self.board_states[-1]))

# --- Move Validation ---
def validate_move(board, move, piece, valid_moves):
    if move is None:
        return False, "Move cannot be None"
    
    if isinstance(move, tuple):
        if len(move) != 2:
            return False, "Invalid remove move format"
        row, col = move
        if not (0 <= row < ROW_COUNT and 0 <= col < COLUMN_COUNT):
            return False, "Remove move out of bounds"
        if board[row][col] != 3 - piece:
            return False, "Cannot remove opponent's piece"
    else:
        if move not in valid_moves:
            return False, f"Invalid column {move}. Valid columns: {valid_moves}"
        
    return True, None

# --- Tournament Mode ---
def run_tournament(num_rounds=10):
    stats = GameStats()
    for round in range(num_rounds):
        logging.info(f"Starting tournament round {round+1}/{num_rounds}")
        # Each AI plays both first and second
        for first_player in [0, 1]:
            replay = GameReplay()
            replay.first_player = first_player
            game_result = play_game(first_player, replay)
            stats.game_replays.append(replay)
            logging.info(f"Round {round+1} Game {2*round + first_player + 1} result: {game_result}")
    
    logging.info("\nTournament Results:\n" + stats.get_stats_summary())
    return stats

# --- Time Control ---
class TimeControl:
    def __init__(self, base_time=5.0, increment=0.1):
        self.base_time = base_time
        self.increment = increment
        self.time_left = {1: base_time, 2: base_time}
        
    def update(self, player, used_time):
        self.time_left[player] -= used_time
        self.time_left[player] += self.increment
        if self.time_left[player] < 0:
            raise TimeoutError(f"Player {player} ran out of time")
        
    def get_time_left(self, player):
        return self.time_left[player]

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

def remove_piece(board, row, col, piece):
    if 0 <= row < ROW_COUNT and 0 <= col < COLUMN_COUNT and board[row][col] == (3 - piece):
        board[row][col] = 0
        return True
    return False

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
    if not AUTO_RUN:
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

def create_request(board, current_player, valid_moves, is_first_player):
    board_list = board.tolist()
    return {
        "board": board_list,
        "current_player": current_player,
        "valid_moves": valid_moves,
        "is_first_player": is_first_player
    }

def update_stats(winner, last_move, current_player, is_first_player):
    global wins_ai1, wins_ai2, draws, losses_ai2, last_games_log
    if winner == "AI 2 (Smarter)":
        wins_ai2 += 1
    elif winner == "AI 1 (Simple)":
        wins_ai1 += 1
        losses_ai2 += 1
    else:
        draws += 1
    stats = f"Stats: AI 2 Wins: {wins_ai2}, AI 1 Wins: {wins_ai1}, Draws: {draws}, AI 2 Losses: {losses_ai2}"
    logging.info(stats)
    
    if current_game >= TOTAL_GAMES - 10:
        last_games_log.append({
            "game": current_game,
            "winner": winner,
            "board": board.copy(),
            "last_move": last_move,
            "current_player": current_player,
            "is_first_player": is_first_player,
            "stats": stats
        })

# --- Game State ---
class GameState:
    def __init__(self):
        self.wins_ai1 = 0
        self.wins_ai2 = 0
        self.draws = 0
        self.current_game = 0
        self.first_player = 0  # 0 for AI2, 1 for AI1
        self.turn = 0
        self.board = None
        self.game_over = False
        self.moves_count = 0
        self.last_move = None

    def reset(self):
        self.board = create_board()
        self.game_over = False
        self.first_player = 1 - self.first_player  # Switch first player
        self.turn = self.first_player
        self.moves_count = 0
        self.last_move = None

    def get_current_piece(self):
        # First player (turn == first_player) always gets red (1)
        # Second player always gets yellow (2)
        return 1 if self.turn == self.first_player else 2

    def get_current_ai(self):
        # Return the current AI name and function based on turn
        if self.turn == 0:
            return "AI 2 (Smarter)", ai2_process_request
        else:
            return "AI 1 (Simple)", ai1_process_request

    def update_stats(self, winner):
        if winner == "AI 2 (Smarter)":
            self.wins_ai2 += 1
        elif winner == "AI 1 (Simple)":
            self.wins_ai1 += 1
        else:
            self.draws += 1
        
        stats = (
            f"Stats: AI2 Wins: {self.wins_ai2}, "
            f"AI1 Wins: {self.wins_ai1}, "
            f"Draws: {self.draws}, "
            f"Games: {self.current_game + 1}/{TOTAL_GAMES}"
        )
        logging.info(stats)

# Initialize game state
game_state = GameState()

def setup():
    global screen, myfont, button_font, status_font
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Connect 4 - AI vs AI")
    game_state.reset()
    myfont = pygame.font.SysFont("monospace", 75)
    button_font = pygame.font.SysFont("monospace", 40)
    status_font = pygame.font.SysFont("monospace", 30)
    draw_board(game_state.board, screen)
    logging.info(f"Game setup: first_player={game_state.first_player}, turn={game_state.turn}")

def reset_game():
    game_state.current_game += 1
    game_state.reset()
    draw_board(game_state.board, screen)
    logging.info(f"Starting game {game_state.current_game + 1}: first_player={game_state.first_player}, turn={game_state.turn}")

async def update_loop():
    if not game_state.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        piece = game_state.get_current_piece()
        ai_name, ai_func = game_state.get_current_ai()
        is_first_player = (game_state.turn == game_state.first_player)

        valid_moves = get_valid_columns(game_state.board)
        request = create_request(game_state.board, piece, valid_moves, is_first_player)
        logging.info(f"Turn: {ai_name} (Player {piece}), is_first_player: {is_first_player}")

        clear_message_area(screen)
        screen.blit(myfont.render(f"{ai_name} thinking...", 1, WHITE), (40, 10))
        pygame.display.update()

        try:
            start_time = time.time()
            move = ai_func(request)
            elapsed_time = time.time() - start_time
            
            if elapsed_time > AI_TIMEOUT:
                raise TimeoutError(f"{ai_name} exceeded time limit")
            
            game_state.last_move = move
            game_state.moves_count += 1
            logging.info(f"{ai_name} returned move: {move}, Time: {elapsed_time:.3f}s")

        except Exception as e:
            logging.error(f"Error in {ai_name}: {str(e)}")
            clear_message_area(screen)
            screen.blit(myfont.render(f"Error in {ai_name}!", 1, WHITE), (40, 10))
            pygame.display.update()
            game_state.game_over = True
            await asyncio.sleep(1)
            return

        if isinstance(move, tuple):
            col, row = move
            if not remove_piece(game_state.board, row, col, piece):
                logging.error(f"{ai_name} returned invalid remove move ({row}, {col})")
                clear_message_area(screen)
                screen.blit(myfont.render(f"Invalid remove by {ai_name}!", 1, WHITE), (40, 10))
                pygame.display.update()
                game_state.game_over = True
                await asyncio.sleep(1)
                return
            logging.info(f"{ai_name} removes piece at ({row}, {col})")
            draw_board(game_state.board, screen)
            log_board_state(game_state.board)
        else:
            col = move
            if col not in valid_moves:
                logging.error(f"{ai_name} returned invalid column {col}. Valid columns: {valid_moves}")
                clear_message_area(screen)
                screen.blit(myfont.render(f"Invalid move by {ai_name}!", 1, WHITE), (40, 10))
                pygame.display.update()
                game_state.game_over = True
                await asyncio.sleep(1)
                return

            clear_message_area(screen)
            screen.blit(status_font.render(f"{ai_name} chooses column {col}", 1, WHITE), (40, 10))
            pygame.display.update()
            await asyncio.sleep(0.1)

            row = get_next_open_row(game_state.board, col)
            await animate_drop(game_state.board, screen, col, row, piece)
            logging.info(f"{ai_name} (Player {piece}) drops in column {col}")
            log_board_state(game_state.board)

        if winning_move(game_state.board, piece):
            logging.info(f"{ai_name} wins!")
            clear_message_area(screen)
            label = myfont.render(f"{ai_name} wins!", 1, RED if piece == 1 else YELLOW)
            screen.blit(label, (40, 10))
            pygame.display.update()
            game_state.update_stats(ai_name)
            game_state.game_over = True
            await asyncio.sleep(1 if AUTO_RUN else 3)
        elif is_board_full(game_state.board):
            logging.info("Game ended in a draw!")
            clear_message_area(screen)
            label = myfont.render("Draw!", 1, WHITE)
            screen.blit(label, (40, 10))
            pygame.display.update()
            game_state.update_stats("Draw")
            game_state.game_over = True
            await asyncio.sleep(1 if AUTO_RUN else 3)
        else:
            game_state.turn = 1 - game_state.turn
            logging.info(f"Switching turn to: {'AI 2 (Smarter)' if game_state.turn == 0 else 'AI 1 (Simple)'}")
            await asyncio.sleep(MOVE_DELAY)

    if game_state.game_over:
        if AUTO_RUN and game_state.current_game < TOTAL_GAMES - 1:
            reset_game()
        else:
            draw_board(game_state.board, screen)
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