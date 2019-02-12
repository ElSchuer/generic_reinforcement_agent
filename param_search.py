from keras import Sequential
from keras.layers import Dense
import dqn_agent
import eval
import environment

batch_space = [64, 256, 512]
learning_rate_space = [0.001, 0.0005, 0.0001]
eps_min_space = [0.01, 0.1, 0.2]
max_episodes = 150
env_name = 'CartPole-v1'


def get_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))

    return model


def reward_function(state, done, score, max_score, reward):
    reward = -10 if done and score < max_score else score
    return reward


eval_inst = eval.RLEvaluation()
env = environment.GymEnvironment(env_name, eval_inst=eval_inst, max_score=500, render_env=False,
                                 max_episodes=max_episodes)

for batch in batch_space:
    for lr in learning_rate_space:
        for eps_min in eps_min_space:
            agent = dqn_agent.DQNAgent(state_size=env.state_size, action_size=env.action_size,
                                       model=get_model(env.state_size, env.action_size),
                                       learning_rate=lr, queue_size=500000, batch_size=batch,
                                       decay_rate=0.95, loss='mse')

            agent.enable_target_network(update_steps=10000)
            agent.enable_double_dqn()
            agent.enable_epsilon_greedy(eps_decay=0.999, eps_min=eps_min, eps_start=1.0)
            agent.enable_dueling_dqn(dueling_type='mean')

            env.set_agent(agent)
            env.set_reward_func(reward_function)

            env.learn()

            plot_filename = env_name + '_' + 'b' + str(batch) + '_l' + str(lr) + '_e' + str(eps_min) + '.jpeg'
            eval_inst.save_plot('./log/', plot_filename)
            eval_inst.reset()
