import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math
import os
from scipy.stats import rankdata

# Các hằng số
ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4
AI_PLAYER = 1
OPPONENT = 2

# Hàm từ mã FastAPI
def is_valid_location(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return None

def get_valid_moves(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]

def winning_move(board, piece, row, col):
    for c in range(max(0, col - 3), min(COLUMN_COUNT - 3, col + 1)):
        if all(board[row][c + i] == piece for i in range(4)):
            return True
    if row <= ROW_COUNT - 4:
        if all(board[row + i][col] == piece for i in range(4)):
            return True
    for offset in range(-3, 1):
        r, c = row + offset, col - offset
        if 0 <= r <= ROW_COUNT - 4 and 0 <= c <= COLUMN_COUNT - 4:
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True
    for offset in range(-3, 1):
        r, c = row - offset, col - offset
        if 3 <= r < ROW_COUNT and 0 <= c <= COLUMN_COUNT - 4:
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True
    return False

def evaluate_window(window, piece, opp_piece):
    score = 0
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 20
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 10
    if window.count(opp_piece) == 3 and window.count(0) == 1:
        score -= 80
    if window.count(opp_piece) == 2 and window.count(0) == 2:
        score -= 10
    return score

def evaluate_board(board, piece):
    score = 0
    opp_piece = 1 if piece == 2 else 2
    center_col = COLUMN_COUNT // 2
    center_array = [board[r][center_col] for r in range(ROW_COUNT)]
    score += center_array.count(piece) * 12
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, opp_piece)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            window = [board[r + i][c] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, opp_piece)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, opp_piece)
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, opp_piece)
    return score

def terminal_node(board):
    return any(winning_move(board, p, r, c) 
               for p in [1, 2] 
               for r in range(ROW_COUNT) 
               for c in range(COLUMN_COUNT) 
               if board[r][c] != 0) or len(get_valid_moves(board)) == 0

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
            return (None, 0)
        return (None, evaluate_board(board, piece))
    valid_locations.sort(key=lambda x: abs(x - COLUMN_COUNT // 2))
    if maximizingPlayer:
        value = -math.inf
        best_col = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            temp_board = [row[:] for row in board]
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
            temp_board = [row[:] for row in board]
            temp_board[row][col] = opp_piece
            new_score = minimax(temp_board, depth - 1, alpha, beta, True, piece)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

# Môi trường Connect Four
class ConnectFourEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[0 for _ in range(COLUMN_COUNT)] for _ in range(ROW_COUNT)]
        self.current_player = random.choice([AI_PLAYER, OPPONENT])
        return self.get_state()

    def get_state(self):
        state = np.array(self.board, dtype=np.float32)
        return state  # Shape: (6, 7)

    def step(self, action, player):
        if not is_valid_location(self.board, action):
            return self.get_state(), -10, True
        row = get_next_open_row(self.board, action)
        self.board[row][action] = player
        if winning_move(self.board, player, row, action):
            reward = 1 if player == AI_PLAYER else -1
            return self.get_state(), reward, True
        if len(get_valid_moves(self.board)) == 0:
            return self.get_state(), 0, True
        reward = evaluate_board(self.board, AI_PLAYER) * 0.005
        return self.get_state(), reward, False

    def switch_player(self):
        self.current_player = OPPONENT if self.current_player == AI_PLAYER else AI_PLAYER

# Prioritized Experience Replay
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def push(self, state, action, reward, next_state, done, error=1.0):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(error ** self.alpha)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)

# Mạng DQN
class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * ROW_COUNT * COLUMN_COUNT, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, ROW_COUNT, COLUMN_COUNT)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * ROW_COUNT * COLUMN_COUNT)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Agent DQN
class DQNAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.memory = PrioritizedReplayBuffer(capacity=100000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0003
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DQN(action_dim).to(self.device)
        self.target_model = DQN(action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')
        self.beta = 0.4
        self.beta_anneal = 0.001

    def remember(self, state, action, reward, next_state, done, error=1.0):
        self.memory.push(state, action, reward, next_state, done, error)

    def act(self, state, valid_moves):
        if not valid_moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        valid_q_values = [q_values[0][a].item() for a in valid_moves]
        return valid_moves[np.argmax(valid_q_values)]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, self.beta)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Double DQN
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, target_q_values)
        loss = (loss * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Cập nhật ưu tiên
        errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.beta = min(1.0, self.beta + self.beta_anneal)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# Hàm huấn luyện
def train_dqn(episodes=200000, batch_size=64, update_target_every=1000, save_every=10000):
    env = ConnectFourEnv()
    agent = DQNAgent(action_dim=COLUMN_COUNT)
    steps_done = 0
    win_count = 0
    loss_count = 0
    draw_count = 0
    curriculum_depth = [4, 6, 8]
    depth_idx = 0
    depth = curriculum_depth[depth_idx]

    for episode in range(episodes):
        state = env.reset()
        player = env.current_player
        done = False
        while not done:
            if player == AI_PLAYER:
                valid_moves = get_valid_moves(env.board)
                action = agent.act(state, valid_moves)
                if action is None:
                    break
                next_state, reward, done = env.step(action, AI_PLAYER)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                player = OPPONENT
            else:
                valid_moves = get_valid_moves(env.board)
                if valid_moves:
                    opp_action, _ = minimax(env.board, depth, -math.inf, math.inf, True, OPPONENT)
                    if opp_action is None or opp_action not in valid_moves:
                        opp_action = random.choice(valid_moves)
                    next_state, reward, done = env.step(opp_action, OPPONENT)
                    if reward == 1:
                        reward = -1
                    elif reward == -1:
                        reward = 1
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    player = AI_PLAYER
            if done:
                if reward >= 1:
                    win_count += 1
                elif reward <= -1:
                    loss_count += 1
                else:
                    draw_count += 1
                break
            steps_done += 1
            if steps_done % batch_size == 0:
                agent.replay(batch_size)
            if steps_done % update_target_every == 0:
                agent.update_target_model()
        # Curriculum Learning
        if episode % 50000 == 0 and depth_idx < len(curriculum_depth) - 1:
            depth_idx += 1
            depth = curriculum_depth[depth_idx]
            print(f"Chuyển sang Minimax độ sâu {depth}")
        if episode % save_every == 0 and episode > 0:
            agent.save_model(f"dqn_connect4_vs_minimax_{episode}.pth")
            print(f"Saved model at episode {episode}")
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Epsilon: {agent.epsilon:.2f}, "
                  f"Depth: {depth}, Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")
            win_count = loss_count = draw_count = 0
    agent.save_model("dqn_connect4_vs_minimax_final.pth")
    return agent

# Hàm đánh giá
def evaluate_vs_minimax(agent, episodes=100, depth=8):
    env = ConnectFourEnv()
    wins = 0
    losses = 0
    draws = 0
    for _ in range(episodes):
        state = env.reset()
        player = env.current_player
        done = False
        while not done:
            if player == AI_PLAYER:
                valid_moves = get_valid_moves(env.board)
                action = agent.act(state, valid_moves)
                state, reward, done = env.step(action, AI_PLAYER)
                player = OPPONENT
            else:
                valid_moves = get_valid_moves(env.board)
                if valid_moves:
                    opp_action, _ = minimax(env.board, depth, -math.inf, math.inf, True, OPPONENT)
                    if opp_action is None or opp_action not in valid_moves:
                        opp_action = random.choice(valid_moves)
                    state, reward, done = env.step(opp_action, OPPONENT)
                    player = AI_PLAYER
            if done:
                if reward >= 1:
                    wins += 1
                elif reward <= -1:
                    losses += 1
                else:
                    draws += 1
                break
    print(f"Vs Minimax (depth {depth}) - Wins: {wins}, Losses: {losses}, Draws: {draws}")
    return wins / episodes

def evaluate_vs_random(agent, episodes=100):
    env = ConnectFourEnv()
    wins = 0
    losses = 0
    draws = 0
    for _ in range(episodes):
        state = env.reset()
        player = env.current_player
        done = False
        while not done:
            if player == AI_PLAYER:
                valid_moves = get_valid_moves(env.board)
                action = agent.act(state, valid_moves)
                state, reward, done = env.step(action, AI_PLAYER)
                player = OPPONENT
            else:
                valid_moves = get_valid_moves(env.board)
                if valid_moves:
                    opp_action = random.choice(valid_moves)
                    state, reward, done = env.step(opp_action, OPPONENT)
                    player = AI_PLAYER
            if done:
                if reward >= 1:
                    wins += 1
                elif reward <= -1:
                    losses += 1
                else:
                    draws += 1
                break
    print(f"Vs Random - Wins: {wins}, Losses: {losses}, Draws: {draws}")
    return wins / episodes

# Chạy huấn luyện và đánh giá
if __name__ == "__main__":
    print("Bắt đầu huấn luyện DQN trên GPU...")
    agent = train_dqn(episodes=200000)
    print("Đánh giá sau huấn luyện:")
    for depth in [4, 6, 8]:
        minimax_win_rate = evaluate_vs_minimax(agent, depth=depth)
        print(f"Win rate vs Minimax (depth {depth}): {minimax_win_rate:.2%}")
    random_win_rate = evaluate_vs_random(agent)
    print(f"Win rate vs Random: {random_win_rate:.2%}")