import gym
import numpy as np

class Environment:

    def __init__(self, eval, max_episodes = 10000):
        self.max_episodes = max_episodes
        self.agent = None
        self.eval = eval

    def step(self):
        pass

    def set_agent(self, agent):
        self.agent = agent

    def get_reward(self):
        pass

class GymEnvironment(Environment):

    def __init__(self, env_name, max_score = 500, render_env = True):
        super.__init__(self)

        self.env = gym.make(env_name)
        self.render_env = render_env
        self.max_score = max_score

        self.state_size = self.env.observation_space.shape[0]
        print('state size', self.state_size)

        self.action_size = self.env.action_space.n
        print('action space', self.action_space)
        print('action size', self.action_space.n)

    def step(self, action):
        return self.env.step(action)

    def learn(self):
        for e in range(10000):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            total_reward = 0

            for time in range(self.max_score + 1):
                if self.render_env == True:
                    self.env.render()

                action = self.agent.act(state)

                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                reward = self.get_reward()
                total_reward += reward

                hist = self.agent.train(state, next_state, reward, action, done)

                state = next_state

                if done:
                    self.agent.save_model()
                    if hist is not None:
                        print("Episode {}, score {}, loss {:.2}, eps {:.4}, reward {}".format(e, time,
                            hist.history.get("loss")[0], self.agent.eps, total_reward))

                    eval.visualize_data(e, hist.history.get("loss")[0] if hist is not None else 0, time)

                    break

