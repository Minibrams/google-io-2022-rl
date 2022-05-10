from random import random
import numpy as np
import tensorflow as tf
import random
from dqn.agent import Agent
from game import build_board, get_full_columns, is_board_full, make_move, print_board

if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    rows = 6
    columns = 7
    n_games = 1000

    agent = Agent(
        gamma=1,
        epsilon=1.0,
        alpha=0.0005,
        input_dims=rows * columns,
        n_actions=columns,
        memory_size=1000,
        batch_size=256,
        epsilon_end=0.01
    )

    scores = []
    eps_history = []

    wins = []
    losses = []
    illegal_moves = []

    for i in range(n_games):
        score = 0
        is_done = False
        board = build_board(rows, columns)

        while not is_done and not is_board_full(board):
            # Player 1
            action = agent.choose_action(
                board.flatten(), closed_columns=get_full_columns(board))

            new_board, reward, is_done = make_move(board, action, 1)
            score += reward
            agent.remember(board.flatten(), action, reward,
                           new_board.flatten(), is_done)
            board = new_board
            loss = agent.learn()
            if loss:
                losses.append(loss)

            if is_done or is_board_full(board):
                continue

            # Player 2
            closed_columns = get_full_columns(board)
            action = np.random.choice([column for column in range(
                columns) if column not in closed_columns])
            new_board, reward, is_done = make_move(board, action, -1)
            board = new_board

        eps_history.append(agent.epsilon)
        scores.append(score)

        if agent.memory.memory_counter > agent.batch_size:
            avg_score = np.mean(scores[max(0, i-100):(i+1)])
            avg_loss = np.mean(losses[max(0, i-100):(i+1)])
            print(
                f'Game {i}: Avg. Score {avg_score : .2f}, Eps: {agent.epsilon : .2f}, Loss: {avg_loss}')
        else:
            print(
                f'Collecting initial transitions... ({agent.memory.memory_counter} / {agent.batch_size})')

        if i % 10 == 0 and i > 0:
            agent.save()
