import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class QLearningAgent:
    def __init__(self, state_size, action_size, decay_rate=0.95, batch_mode=True, batch_size=100, model_name='model.h5', learning_rate = 0.001, queue_size=10000,
                 eps_start = 1.0, eps_min = 0.01, eps_decay = 0.999):
        self.decay_rate = decay_rate
        self.state_size = state_size
        self.action_size = action_size

        self.model_name = model_name

        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate

        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.data_batch = deque(maxlen=queue_size)

        self.model = self.get_model()

    def get_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def save_model(self):
        self.model.save(self.model_name)

    def load_model(self):
        self.model.load_weights(self.model_name)

    def act(self, state):

        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)

        return np.argmax(self.model.predict(state))

    def decay_epsilon(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def train(self, state, next_state, reward, action, done):
        if self.batch_mode:
            self.data_batch.append([state, next_state, reward, action, done])

            if len(self.data_batch) >= self.batch_size:
                self.decay_epsilon()
                return self.train_batch()

        else:
            self.decay_epsilon()
            return self.train_sample(state, next_state, reward, action, done)

    def train_batch(self):

        tmp_batch = random.sample(self.data_batch, self.batch_size)

        states = []
        targets = []

        for state, next_state, reward, action, done in tmp_batch:

            # q(a,s)
            target = self.model.predict(state)

            if done:
                target[0][action] = reward
            else:
                # q(a', s')
                q_future = self.model.predict(next_state)[0]
                target[0][action] = reward + self.decay_rate * np.amax(q_future)

            states.append(state[0])
            targets.append(target[0])

        history = self.model.fit(np.array(states), np.array(targets), verbose=0, epochs=1)

        return history

    def train_sample(self, state, next_state, reward, action, done):

        # q(a', s')
        q_future = self.model.predict(next_state)

        target = q_future

        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.decay_rate * np.argmax(q_future)

        history = self.model.fit(state, target, verbose=0, epochs=1)

        return history

