from keras import Sequential
from keras.layers import Dense
import dqn_agent
import eval
import environment

def reward_function(state, done, score, max_score, reward):

    state = state[0]

    # Adjust reward based on car position
    #reward = state[0] + 0.5

    # Adjust reward for task completion
    if state[0] >= 0.5:
        reward += 1

    return reward

def get_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation='relu'))
    model.add(Dense(64, activation='relu'))
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
#agent.enable_dueling_dqn(dueling_type='mean')

env.set_agent(agent)
env.set_reward_func(reward_function)

env.learn()
