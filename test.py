
import numpy as np
from game import _get_winner, build_board, get_full_columns, is_board_full, make_move
from dqn.agent import Agent
from tqdm import tqdm


if __name__ == '__main__':
    rows = 6
    columns = 7
    n_games = 100

    wins = 0

    agent = Agent(
        gamma=1,
        epsilon=0.0,  # Never explore, only exploit
        alpha=0.0005,
        input_dims=rows * columns,
        n_actions=columns,
        memory_size=1000,
        batch_size=256,
        epsilon_end=0.0
    )

    # agent.load()

    for i in tqdm(list(range(n_games)), desc='[Playing games...]'):
        player_one = 'Model'
        player_two = 'Random'
        player = player_one

        board = build_board(rows, columns)

        is_done = False

        while not is_done and not is_board_full(board):
            if player == player_one:
                # Let the model play
                column = agent.choose_action(
                    board.flatten(),
                    get_full_columns(board)
                )

                new_board, _, is_done = make_move(board, column, 1)

                board = new_board
                player = player_two

            else:
                # Let the random agent play
                full_columns = get_full_columns(board)

                column = np.random.choice([
                    column for column in range(columns)
                    if column not in full_columns  # Don't let the agent choose full columns
                ])

                new_board, reward, is_done = make_move(board, column, -1)

                board = new_board
                player = player_one

        winner = _get_winner(board)
        if winner == 1:
            wins += 1

    print(f'Finished playing {n_games} games.')
    print(
        f'The model won {(wins / n_games) * 100 : .2f}% ({wins}) of the games played.')
