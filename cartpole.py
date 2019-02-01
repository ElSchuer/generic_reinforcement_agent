from keras import Sequential
from keras.layers import Dense
import dqn_agent
import eval
import environment


def reward_function(state, done, score, max_score, reward):
    reward = -10 if done and score < max_score else score
    return reward


eval_inst = eval.RLEvaluation()
env = environment.GymEnvironment('CartPole-v1', eval_inst=eval_inst, max_score=500, render_env=True, )

# model
model = Sequential()
model.add(Dense(64, input_dim=env.state_size, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(env.action_size, activation='linear'))

agent = dqn_agent.DuelingDDQNAgent(state_size=env.state_size, action_size=env.action_size, model=model,
                                   learning_rate=0.001,
                                   queue_size=500000, batch_size=256, eps_decay=0.999, eps_min=0.05, decay_rate=0.95,
                                   update_steps=10000, loss='mse', dueling_type='mean')

env.set_agent(agent)
env.set_reward_func(reward_function)

env.learn()
