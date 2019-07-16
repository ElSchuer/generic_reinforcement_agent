from keras import Sequential
from keras.layers import Dense
from keras import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras.layers import Lambda
from keras.regularizers import l2
import dqn_agent
import eval
import environment

def reward_function(state, done, score, max_score, reward):

    if state[0] >= -0.5:
        reward = (state[0] + 0.5) / 1.1
    else:
        return 0

    return reward

def get_model(state_size, action_size):
    init = 'glorot_uniform'

    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=state_size))

    model.add(Conv2D(24, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init,
                     kernel_regularizer=l2(0.001)))
    model.add(Conv2D(36, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init,
                     kernel_regularizer=l2(0.001)))
    model.add(Conv2D(48, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init,
                     kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init,
                     kernel_regularizer=l2(0.001)))
    # model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(units=1164, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=100, kernel_regularizer=l2(0.001)))
    # model.add(Dense(units=50, kernel_regularizer=l2(0.001)))
    # model.add(Dense(units=10, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=action_size, activation='linear'))

    return model


eval_inst = eval.RLEvaluation(mean_subset=25)
env = environment.GymEnvironment('SpaceInvaders-v0', eval_inst=eval_inst, max_score=200, render_env=True)

print(env.state_size)

agent = dqn_agent.DQNAgent(state_size=env.state_size, action_size=env.action_size,
                           model=get_model(env.state_size, env.action_size),
                           learning_rate=0.001,
                           queue_size=500000, batch_size=64, decay_rate=0.95, loss='mse')
agent.enable_target_network(update_steps=10000)
agent.enable_double_dqn()
agent.enable_epsilon_greedy(eps_decay=0.999, eps_min=0.1, eps_start=1.0)
agent.enable_dueling_dqn(dueling_type='mean')

env.set_agent(agent)
env.set_reward_func(reward_function)

env.learn()