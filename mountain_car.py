import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras.layers import Dense
import dqn_agent

sns.set(style="whitegrid")

def plot_time_loss(episodes, time_values, loss_values):
    plt.ion()
    plt.figure(1)
    plt.subplot(211)
    plt.plot(episodes, time_values)
    plt.draw()
    plt.pause(0.001)
    plt.ylabel("time values")
    plt.xlabel("episodes")

    plt.subplot(212)
    plt.plot(episodes, loss_values)
    plt.draw()
    plt.pause(0.001)
    plt.ylabel("loss values")
    plt.xlabel("episodes")

    plt.show(block=False)

load_model = False

env = gym.make('MountainCar-v0')

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
model.add(Dense(64, activation='relu'))
model.add(Dense(action_size, activation='linear'))

episodes, loss_values, time_values = [], [], []

agent = dqn_agent.QLearningAgent(state_size=state_size, action_size=action_size, model=model, learning_rate=0.001,
                                 queue_size=2000, batch_mode=True, batch_size=50, eps_decay=0.995)

if load_model:
    agent.load_model()

for e in range(10000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(200):
            env.render()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            hist = agent.train(state, next_state, reward, action, done)

            # Adjust reward for task completion
            #if state[0][0] >= -0.3:
            #    reward += abs(state[0][0])

            reward = abs(state[0][0] + 0.5)

            #print('reward', reward, ' state ', state[0][0])

            state = next_state

            if done:
                agent.save_model()

                if hist is not None:
                    print("Episode {}, time {}, loss {:.2}, eps {:.4}".format(e, time, hist.history.get("loss")[0], agent.eps))

                if e % 10 == 0:
                    episodes.append(e)
                    time_values.append(time)
                    loss_values.append(hist.history.get("loss")[0]) if hist is not None else loss_values.append(0)

                    plot_time_loss(episodes, time_values, loss_values)

                break