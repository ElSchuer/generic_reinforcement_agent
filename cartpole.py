import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
import dqn_agent
import eval

load_model = False

env = gym.make('CartPole-v1')
max_score = 499

np.random.seed(123)
env.seed(123)

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
model.add(Dense(256, input_dim=state_size, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(action_size, activation='linear'))

episodes, loss_values, time_values = [], [], []

agent = dqn_agent.TargetDeepQAgent(state_size=state_size, action_size=action_size, model=model, learning_rate=0.0001,
                                 queue_size=50000, batch_size=1500, eps_decay=0.999, eps_min=0.02, decay_rate=0.95)

rl_eval = eval.RLEvaluation()

if load_model:
    agent.load_model()

for e in range(10000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        total_reward = 0

        for time in range(max_score+1):
            env.render()

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            #reward = reward if not done else -10
            reward = -10 if done and time < max_score else reward
            total_reward += reward

            hist = agent.train(state, next_state, reward, action, done)

            state = next_state

            if done:
                agent.save_model()
                if hist is not None:
                    print("Episode {}, score {}, loss {:.2}, eps {:.4}, reward {}".format(e, time, hist.history.get("loss")[0], agent.eps, total_reward))

                if time == 499:
                    print("Done Training")
                    train_weights = False

                rl_eval.visualize_data(e, hist.history.get("loss")[0] if hist is not None else 0, time)

                break
