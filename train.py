import gym
# import atari_py as ap
import time

class GamePlay(object):
    def __init__(self, gym_name, agent):
        self.env = gym.make(gym_name)
        self.agent = agent
        self.agent.set_env(self.env)
        self.score = 0

    def load(self, replay):
        self.agent.load(replay)

    def train(self):
        self.agent.train()


    def play(self):
        state = self.env.reset()
        done = False
        while not done:
            #action_n = self.agent.get_state(self.env)
            action_n = self.agent.get_state(state)
            # print action_n
            state, reward, done, info = self.env.step(action_n)
            self.env.render()
            time.sleep(0.02)
            if reward > 0:
                self.score += reward

    def get_score(self):
        return self.score
