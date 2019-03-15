from keras import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import dqn_agent
import eval
import environment

def reward_function(state, done, score, max_score, reward):

    if state[0] >= -0.5:
        reward = (state[0] + 0.5) / 1.1
    else:
        return 0

    return reward

def create_states(state_space, state_size, num_of_states=20):
    states = []
    [states.append(np.linspace(state_space[0][s], state_space[1][s], num_of_states)) for s in range(state_size)]
    return np.array(states).T

def get_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(128, input_dim=state_size, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(action_size, activation='linear'))

    return model


eval_inst = eval.RLEvaluation(mean_subset=25)
env = environment.GymEnvironment('MountainCar-v0', eval_inst=eval_inst, max_score=200, render_env=True)

agent = dqn_agent.DQNAgent(state_size=env.state_size, action_size=env.action_size,
                           model=get_model(env.state_size, env.action_size),
                           learning_rate=0.001,
                           queue_size=500000, batch_size=64, decay_rate=0.95, loss='mse')
agent.enable_target_network(update_steps=10000)
agent.enable_double_dqn()
agent.enable_epsilon_greedy(eps_decay=0.999, eps_min=0.1, eps_start=1.0)
agent.enable_dueling_dqn(dueling_type='mean')

state_space = env.get_state_space()
states = create_states(state_space, 2, 50)

rewards = []
[rewards.append(reward_function(s, False, 0, 200, -1)) for s in states]

eval_inst.plot_reward(rewards, states)

env.set_agent(agent)
env.set_reward_func(reward_function)

env.learn()
