import sys
import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

CELL_SIZE = 100
WIDTH, HEIGHT = CELL_SIZE * 3, CELL_SIZE * 3
LINE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = CELL_SIZE // 4

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("vs ai")


class QNetwork(Model):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(128, activation='relu')
        self.d3 = Dense(9)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


class TicTacToe:
    def __init__(self):
        self.reset()

    def is_winner(self, player):
        board = np.array(self.board)
        return any(
            np.all(board[row,:] == player) or np.all(board[:,col] == player)
            or np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player)
            for row in range(3) for col in range(3)
        )

    def is_draw(self):
        return not any('' in row for row in self.board)

    def get_state(self):
        return self.board

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == '']

    def make_move(self, cell, player):
        self.board[cell[0]][cell[1]] = player

    def is_draw(self):
        return all(self.board[row][col] != '' for row in range(3) for col in range(3))

    def game_over(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()

    def reset(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.player_turn = 'X'


def draw_lines():
    pygame.draw.line(screen, BLACK, (CELL_SIZE, 0), (CELL_SIZE, HEIGHT), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, (CELL_SIZE * 2, 0), (CELL_SIZE * 2, HEIGHT), LINE_WIDTH)
    
    pygame.draw.line(screen, BLACK, (0, CELL_SIZE), (WIDTH, CELL_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, (0, CELL_SIZE * 2), (WIDTH, CELL_SIZE * 2), LINE_WIDTH)

def draw_figures(game):
    for row in range(3):
        for col in range(3):
            if game.board[row][col] == 'X':
                pygame.draw.line(screen, BLACK, (col * CELL_SIZE + SPACE, row * CELL_SIZE + CELL_SIZE - SPACE),
                                 (col * CELL_SIZE + CELL_SIZE - SPACE, row * CELL_SIZE + SPACE), CROSS_WIDTH)
                pygame.draw.line(screen, BLACK, (col * CELL_SIZE + SPACE, row * CELL_SIZE + SPACE),
                                 (col * CELL_SIZE + CELL_SIZE - SPACE, row * CELL_SIZE + CELL_SIZE - SPACE), CROSS_WIDTH)
            elif game.board[row][col] == 'O':
                pygame.draw.circle(screen, BLACK, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2),
                                   CELL_SIZE // 2 - SPACE, CROSS_WIDTH)

def restart(game):
    screen.fill(WHITE)
    draw_lines()
    game.reset()

class Agent:
    def __init__(self, discount_rate=0.97, learning_rate=0.01):
        self.q_network = QNetwork()
        self.q_network.compile(loss='mse', optimizer=Adam(learning_rate))
        self.discount_rate = discount_rate

    def get_q_values(self, state):
        return self.q_network.predict(state)
    
    def get_action(self, state, available_actions):
        q_values = self.get_q_values(state)
        q_values = q_values[0]
        actions_sorted_by_q_values = np.argsort(-q_values) 
        for action in actions_sorted_by_q_values:
            if action in available_actions:
                return action
        return np.random.choice(available_actions) 

    def train(self, state, action, next_state, reward, done):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        
        q_values[0][action] = reward
        if not done:
            q_values[0][action] += self.discount_rate * np.max(next_q_values)

        self.q_network.fit(state, q_values, verbose=0)

    def save_model(self, name):
        self.q_network.save(name)
        
def preprocess_state(state):
    return np.array([[1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in row] for row in state])

def calculate_reward(game):
    if game.is_winner('X'):
        return -1
    elif game.is_winner('O'):
        return 1
    else:
        return 0

agent = Agent()

def main():
    game = TicTacToe()
    clock = pygame.time.Clock()
    screen.fill(WHITE)
    draw_lines()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if game.player_turn == 'X':  
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX = event.pos[0] // CELL_SIZE
                mouseY = event.pos[1] // CELL_SIZE

                if game.board[mouseY][mouseX] == '':
                    game.make_move((mouseY, mouseX), game.player_turn)
                    if game.is_winner(game.player_turn):
                        print(f'Player {game.player_turn} wins!')
                        pygame.time.delay(3000)
                        restart(game)
                    elif game.is_draw():
                         print('The game is a draw!')
                         pygame.time.delay(3000)
                         restart(game)
                    else:
                        game.player_turn = 'O' if game.player_turn == 'X' else 'X'
                    draw_figures(game)
        else:  
            state = game.get_state()
            state = preprocess_state(state)  
            available_actions = [i * 3 + j for i, j in game.available_actions()]
            action = agent.get_action(state.reshape(1, 9), available_actions)  
            game.make_move((action // 3, action % 3), game.player_turn)
            draw_figures(game)
            reward = calculate_reward(game)  
            next_state = game.get_state()
            next_state = preprocess_state(next_state)
            done = game.game_over()
            agent.train(state.reshape(1, 9), action, next_state.reshape(1, 9), reward, done)
            if done:
                if game.is_winner(game.player_turn):
                    print(f'Player {game.player_turn} wins!')
                elif game.is_draw():
                    print('The game is a draw!')
                pygame.time.delay(3000)
                restart(game)
            else:
                game.player_turn = 'O' if game.player_turn == 'X' else 'X'


        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()              
