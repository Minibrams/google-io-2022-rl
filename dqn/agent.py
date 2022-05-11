from random import random
from typing import List
import numpy as np
from dqn.model import build_dqn
from dqn.replay import ReplayBuffer
from tensorflow.keras.models import load_model


class Agent:
    def __init__(
        self,
        alpha: float,    # Learning rate
        gamma: float,    # Discount rewards
        n_actions: int,  # Number of actions
        epsilon: float,  # Starting value of epsilon
        batch_size: int,  # Number of transitions in each batch
        input_dims: int,  # Size of a game state (rows * columns)
        epsilon_dec=0.999,  # Decay coefficient for epsilon
        epsilon_end=0.01,  # Minimum value of epsilon.
        memory_size=1000,  # Max number of transitions to store in memory.
        save_to='dqn_model.h5'  # Filename of model saved with Agent.save()
    ):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.save_to = save_to

        self.memory = ReplayBuffer(memory_size, input_dims, n_actions)
        self.Q_eval = build_dqn(alpha, n_actions, input_dims, 256, 64)

    def remember(self, state, action, reward, new_state, is_done):
        self.memory.store_transition(state, action, reward, new_state, is_done)

    def choose_action(self, state: np.ndarray, closed_columns: List[int]):
        """
        Explores or exploits the game state and returns the index
        of the column to place a token in on the game board.
        """
        state = state[np.newaxis, :]

        # Explore or exploit?
        if random() < self.epsilon:
            # Explore!
            return np.random.choice([
                column for column in self.action_space
                if column not in closed_columns
            ])
        else:
            # Exploit!
            actions = self.Q_eval.predict(state)
            actions[0][closed_columns] = -np.inf  # Do not allow illegal moves
            return np.argmax(actions)

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            # Not enough transitions to fill a batch - let's wait a bit.
            return

        # Sample a batch of transitions we have gotten from playing the game
        # For refence, in the presentation this is:
        # - State we saw
        # - Action we took
        # - Reward we got
        # - State we got
        # - Was the next state final? (e.g. the end of a game)
        state, action, reward, new_state, is_done = self.memory.sample_batch(
            self.batch_size)

        # Use the one-hot action vector to extract action indices
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        # Batch index for addressing each transition individually
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Compute the predicted rewards we need to approximate the Bellman equation:
        # - Q_eval:   the predicted rewards for every action from the first state
        # - Q_next: the (sum of) predicted rewards for every action from the following state
        Q_eval = self.Q_eval.predict(state)
        Q_next = self.Q_eval.predict(new_state)
        Q_target = Q_eval.copy()

        # To satisfy the Bellman equation, we must fit our model such that:
        #
        #    R(s, a) = r + max R(s, a)
        #
        #    ... which is:
        #
        #    Q_eval = r + max Q_next
        #            |---------------|
        #           this is the Q_target
        max_Q_next = np.max(Q_next, axis=1)
        Q_target[batch_index, action_indices] = (
            reward + (
                self.gamma *  # Discount rewards far into the future
                max_Q_next *  # The highest possible reward we can obtain
                is_done       # Nullify future reward in terminal states
            )
        )

        # Finally, tell Keras to process the states through the model, get
        # the predicted rewards for all actions in each state, and minimize
        # the difference between what the model predicts and what our Q_target is.
        summary = self.Q_eval.fit(state, Q_target, verbose=0)

        # Let epsilon decrease so our model will start making decisions on its own.
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_end)

        return summary.history['loss']

    def save(self):
        self.Q_eval.save(self.save_to)

    def load(self):
        self.Q_eval = load_model(self.save_to)
