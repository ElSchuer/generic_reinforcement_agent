import gym
import numpy as np

class Environment:

    def __init__(self, max_episodes=10000, seed=10):
        self.max_episodes = max_episodes

        self.agent = None
        self.reward_func = None
        self.seed = seed

        np.random.seed(seed)

    def step(self, action):
        pass

    def set_agent(self, agent):
        self.agent = agent

    def train_agent(self):
        self.agent.train()

    def set_reward_func(self, reward_func):
        self.reward_func = reward_func

    def get_reward(self, state, done, score, max_score, reward):
        return self.reward_func(state[0], done, score, max_score, reward)

class GymEnvironment(Environment):

    def __init__(self, env_name, eval_inst, seed=10, max_score=500, render_env=True, max_episodes=10000):
        super().__init__(seed=seed, max_episodes=max_episodes)

        self.env = gym.make(env_name)
        self.env.seed(self.seed)
        self.render_env = render_env
        self.max_score = max_score

        self.eval_inst = eval_inst

        self.state_size = self.env.observation_space.shape[0]
        print('state size', self.state_size)

        self.action_size = self.env.action_space.n
        print('action space', self.env.action_space)
        print('action size', self.env.action_space.n)

    def get_state_space(self):
        return [self.env.observation_space.low, self.env.observation_space.high]

    def step(self, action):
        return self.env.step(action)

    def train_agent(self):
        if len(self.agent.data_batch) >= self.agent.batch_size:
            return self.agent.train()

    def learn(self):
        for e in range(self.max_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            total_reward = 0

            loss_values = []

            for s in range(self.max_score + 1):
                if self.render_env == True:
                    self.env.render()

                action = self.agent.act(state)

                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                reward = self.get_reward(state, done, s, self.max_score, reward)
                total_reward += reward

                self.agent.remember(state, next_state, reward, action, done)

                hist = self.train_agent()

                state = next_state

                if hist is not None:
                    loss_values.append(hist.history.get("loss")[0])

                if done:
                    self.agent.save_model()

                    print("Episode {}, score {}, loss {:.2}, eps {:.4}, reward {}".format(e, s,
                           np.mean(loss_values), float(self.agent.eps), total_reward))

                    self.eval_inst.visualize_data(e, np.mean(loss_values), s)

                    break

