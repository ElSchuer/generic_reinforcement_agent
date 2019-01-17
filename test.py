from keras import Sequential
from keras.layers import Dense
import dqn_agent
import eval
import environment

env = environment.GymEnvironment('CartPole-v1', max_score=500, render_env=True, eval = eval.RLEvaluation())

# model
model = Sequential()
model.add(Dense(64, input_dim=env.state_size, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(env.action_size, activation='linear'))

agent = dqn_agent.DoubleDQNAgent(state_size=env.state_size, action_size=env.action_size, model=model, learning_rate=0.0001,
                                 queue_size=50000, batch_size=64, eps_decay=0.999, eps_min=0.1, decay_rate=0.95)

env.set_agent(agent)

env.learn()
