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
    losses = []

    for i in range(n_games):
        score = 0
        is_done = False
        board = build_board(rows, columns)

        player_one = 'Model'
        player_two = 'Random'

        player = player_one

        while not is_done and not is_board_full(board):

            if player == player_one:
                # Let the model play
                column = agent.choose_action(
                    board.flatten(),
                    get_full_columns(board)
                )

                # Make the move, collect new state and reward
                new_board, reward, is_done = make_move(board, column, 1)

                # Store the transition to learn from later
                agent.remember(
                    board.flatten(),
                    column,
                    reward,
                    new_board.flatten(),
                    is_done
                )

                # Continue the game using the new board
                board = new_board
                score += reward

                # Let the model learn from the transitions it stored
                loss = agent.learn()
                if loss:
                    losses.append(loss)

                player = player_two

            else:
                # Let the random agent play
                # TODO: Maybe let another DQN model play here, let them improve against each other?
                full_columns = get_full_columns(board)

                column = np.random.choice([
                    column for column in range(columns)
                    if column not in full_columns  # Don't let the agent choose full columns
                ])

                new_board, reward, is_done = make_move(board, column, -1)
                board = new_board

                player = player_one

        scores.append(score)

        if agent.memory.memory_counter > agent.batch_size:
            avg_score = np.mean(scores[max(0, i-100):(i+1)])
            avg_loss = np.mean(losses[max(0, i-100):(i+1)])

            print(
                f'Game {i}: Avg. win rate {avg_score / 50 : .2f}, Eps: {agent.epsilon : .2f}, Loss: {avg_loss}')
        else:
            print(
                f'Collecting initial transitions... ({agent.memory.memory_counter} / {agent.batch_size})')

        if i % 10 == 0 and i > 0:
            agent.save()
