from tensorflow import keras
from tensorflow.keras.optimizers import Adam


def build_dqn(lr, n_actions: int, input_dims, fc1_dims, fc2_dims):
    """
    Builds a simple feed-forward DNN for predicting action rewards.
    Feel free to change this :-)
    """
    model = keras.Sequential([
        keras.layers.Dense(
            fc1_dims,
            input_shape=(input_dims, ),
            activation='relu'
        ),
        keras.layers.Dense(
            fc2_dims,
            activation='relu'
        ),
        keras.layers.Dense(n_actions)
    ])

    model.compile(
        loss='mse',
        optimizer=Adam(lr=lr),
    )

    return model
