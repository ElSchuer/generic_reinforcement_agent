import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras.layers import Dense
import dqn_agent

class RLEvaluation:
    def __init__(self, episode_ticks = 1):
        self.episodes, self.loss_values, self.score_values = [], [], []
        self.episode_ticks = episode_ticks

        #init plots
        sns.set(style="whitegrid")
        plt.ion()
        self.loss_plot = plt.subplot(211)
        self.score_plot = plt.subplot(212)
        self.loss_plot.set_ylabel("loss values")
        self.loss_plot.set_xlabel("episodes")
        self.score_plot.set_ylabel("score values")
        self.score_plot.set_xlabel("episodes")
        plt.show(block=False)

    def plot_train_loss(self):
        self.loss_plot.plot(self.episodes, self.loss_values)
        plt.draw()
        plt.pause(0.001)

    def plot_score(self):
        self.score_plot.plot(self.episodes, self.score_values)
        plt.draw()
        plt.pause(0.001)

    def visualize_data(self, episode, loss, score):
        self.episodes.append(episode)
        self.loss_values.append(loss)
        self.score_values.append(score)

        if episode % self.episode_ticks == 0:
            self.plot_score()
            self.plot_train_loss()



load_model = False

env = gym.make('CartPole-v1')

# states: consists of sin and cos of the two joint angles and the angular velocities of the joints
# [cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]
state_size = env.observation_space.shape[0]
print('state size',state_size)

# actions +1, 0 or -1 torque on the middle joint
action_size = env.action_space.n
print('action space', env.action_space)
print('action size', env.action_space.n)

# model
model = Sequential()
model.add(Dense(64, input_dim=state_size, activation='relu'))
model.add(Dense(64, activation='relu'))
#model.add(Dense(16, activation='relu'))
model.add(Dense(action_size, activation='linear'))

episodes, loss_values, time_values = [], [], []

agent = dqn_agent.QLearningAgent(state_size=state_size, action_size=action_size, model=model, learning_rate=0.0001,
                                 queue_size=100000, batch_mode=True, batch_size=500, eps_decay=0.995, eps_min=0.01)

rl_eval = RLEvaluation()

if load_model:
    agent.load_model()

for e in range(10000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        total_reward = 0

        for time in range(500):
            env.render()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            reward = reward if not done else -1
            total_reward += reward

            hist = agent.train(state, next_state, reward, action, done)

            state = next_state

            if done:
                agent.save_model()
                if hist is not None:
                    print("Episode {}, time {}, loss {:.2}, eps {:.4}, reward {}".format(e, time, hist.history.get("loss")[0], agent.eps, total_reward))

                rl_eval.visualize_data(e, hist.history.get("loss")[0] if hist is not None else 0, time)

                break
