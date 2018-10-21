import gym
# import atari_py as ap
import time
from recorder import Recorder


# game_list = ap.list_games()
# print(sorted(game_list))


# env = gym.make('SpaceInvaders-v0')
# env = gym.make('Zaxxon-v0')
# env = gym.make('MontezumaRevenge-v0')
# env = gym.make('Riverraid-v0')
# env = gym.make('AirRaid-v0')
# env.reset()


class Gameplay(object):
    def __init__(self, gym_name, agent):
        self.agent = agent
        self.env = gym.make(gym_name)
        self.score = 0
        self.recorder = Recorder()
        self.lives = 0
        self.agent.learn_how_to_play(self.env)

    def play(self, playback=None):
        self.score = 0
        self.env.reset()
        if playback is not None:
            playback.rewind()
            self.replay(playback)
            self.recorder = playback

        done = False
        while not done:
            action_n = self.agent.get_state(self.env)
            # print action_n
            state, reward, done, info = self.env.step(action_n)
            self.lives = info['ale.lives']
            if not done:
                self.recorder.record(action_n)
            self.env.render()
            time.sleep(0.02)

            if reward > 0:
                self.score += reward

    def replay(self, playback):
        while playback.has_playback():
            action_n = playback.play()
            state, reward, done, info = self.env.step(action_n)
            if done:
                print "Playback not finished"
                return False
            self.score += reward
            self.env.render()
        return True

    def get_score(self):
        return self.score

    def get_playback(self):
        return self.recorder

    def get_lives(self):
        return self.lives
