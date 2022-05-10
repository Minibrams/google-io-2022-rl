import numpy as np
from itertools import groupby
from typing import Iterator, List


def _get_sequences(row: np.ndarray) -> Iterator[np.ndarray]:
    """
    Yields all contiguous non-zero sequences in a row.
    """
    for player, cells in groupby(row):
        if player:
            yield list(cells)


def _rotate(mat: np.ndarray) -> np.ndarray:
    """
    Rotates (with padding) a matrix by 45 degrees, e.g.:

    ```
    [0, 0, 0, 0, 1]
    [0, 0, 0, 1, 0]
    [0, 0, 1, 0, 0]
    [0, 1, 0, 0, 0]
    ```

    ... turns into:
    ```
    [0, 0, 0, 0]
    [0, 0, 0, 0]
    [0, 0, 0, 0]
    [0, 0, 0, 0]
    [1, 1, 1, 1]
    [0, 0, 0, 0]
    [0, 0, 0, 0]
    [0, 0, 0, 0]
    [0, 0, 0, 0]
    ```

    """

    def pad(n):
        return [0 for _ in range(n)]

    return np.array([
        pad(r) + list(row) + pad(len(row) - r - 1)
        for r, row in enumerate(mat)
    ]).T


def _get_row_winner(board: np.ndarray) -> int:
    for row in board:

        sequences = _get_sequences(row)
        sequences = sorted(sequences, key=len, reverse=True)

        if not sequences:
            continue

        longest = sequences[0]
        player = longest[0]

        if len(longest) >= 4:
            return int(player)

    return 0


def _get_winner(board: np.ndarray) -> int:
    row_winner = _get_row_winner(board)
    col_winner = _get_row_winner(board.T)
    forward_diagonal_winner = _get_row_winner(_rotate(board))
    backward_diagonal_winner = _get_row_winner(_rotate(reversed(board)))

    return (
        row_winner if row_winner else
        col_winner if col_winner else
        forward_diagonal_winner if forward_diagonal_winner else
        backward_diagonal_winner if backward_diagonal_winner else
        0
    )


def _drop_token(board: np.ndarray, column: int, token: int) -> np.ndarray:
    board_copy = board.copy()
    empty_row_indexes, = np.where(board_copy[:, column] == 0.0)
    drop_at_row = max(empty_row_indexes)
    board_copy[drop_at_row, column] = token
    return board_copy


def _is_column_full(board: np.ndarray, column) -> bool:
    for token in board[:, column]:
        if token == 0:
            return False

    return True


def make_move(board: np.ndarray, column: int, token: int):
    new_board = _drop_token(board, column, token)
    winner = _get_winner(new_board)

    is_done = bool(winner)

    if is_done:
        reward = 50 if winner == token else -50
    else:
        reward = 0

    return new_board, reward, is_done


def is_board_full(board: np.ndarray) -> bool:
    _, columns = board.shape
    return len(get_full_columns(board)) == columns


def get_full_columns(board: np.ndarray) -> List[int]:
    _, columns = board.shape
    return [column for column in range(columns) if _is_column_full(board, column)]


def build_board(rows: int, columns: int) -> np.ndarray:
    return np.zeros((rows, columns))


def print_board(board_matrix: np.ndarray):
    n_rows, n_columns = board_matrix.shape

    board = [
        ' '.join([
            'o' if board_matrix[n, m] == -1 else
            'x' if board_matrix[n, m] == 1 else
            '.'
            for m in range(n_columns)
        ])

        for n in range(n_rows)
    ]

    numbers = ' '.join([str(i) for i in range(n_rows + 1)])

    print('\n'.join([numbers] + board))


def game_loop():
    n = 6
    m = 7
    board = np.zeros((n, m), dtype=np.int)

    player_one = 'Player 1 (x)'
    player_two = 'Player 2 (o)'

    player = player_one

    while True:
        print_board(board)
        print()

        column = int(input(f'{player} to play: '))
        board = _drop_token(
            board,
            column,
            1 if player == player_one else -1
        )

        winner = _get_winner(board)

        if winner:
            print(f'Winner: {player}!')
            exit()

        player = player_two if player == player_one else player_one


if __name__ == '__main__':
    game_loop()
