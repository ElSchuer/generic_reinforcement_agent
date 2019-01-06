import numpy as np
import random
from collections import deque
from keras.optimizers import Adam


class SimpleDeepQAgent:
    def __init__(self, state_size, action_size, model, decay_rate=0.95, batch_mode=True, batch_size=100, model_name='model.h5', learning_rate = 0.001, queue_size=10000,
                 eps_start = 1.0, eps_min = 0.01, eps_decay = 0.999):
        self.decay_rate = decay_rate
        self.state_size = state_size
        self.action_size = action_size

        self.model_name = model_name

        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.data_batch = deque(maxlen=queue_size)

        self.model = model
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))


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
        self.data_batch.append([state, next_state, reward, action, done])

        if len(self.data_batch) >= self.batch_size:
            self.decay_epsilon()
            return self.train_batch()


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


class TargetDeepQAgent(SimpleDeepQAgent):

    def __init__(self, state_size, action_size, model, decay_rate=0.95, batch_mode=True, batch_size=100, model_name='model.h5', learning_rate = 0.001, queue_size=10000,
                 eps_start = 1.0, eps_min = 0.01, eps_decay = 0.999, update_steps = 5000):

        super().__init__(self, action_size=action_size, model=model, decay_rate=decay_rate,
                         batch_mode=batch_mode, batch_size=batch_size, model_name=model_name, learning_rate=learning_rate,
                         queue_size=queue_size, eps_start=eps_start, eps_min=eps_min, eps_decay=eps_decay)

        self.target_model = model
        self.update_steps = update_steps
        self.step = 0

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("Update target model")

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
                q_future = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.decay_rate * np.amax(q_future)

            states.append(state[0])
            targets.append(target[0])

        history = self.model.fit(np.array(states), np.array(targets), verbose=0, epochs=1)

        return history

    def train(self, state, next_state, reward, action, done):
        self.data_batch.append([state, next_state, reward, action, done])

        if self.step % self.update_steps == 0:
            self.update_target_model()

        self.step += 1

        if len(self.data_batch) >= self.batch_size:
            self.decay_epsilon()
            return self.train_batch()
