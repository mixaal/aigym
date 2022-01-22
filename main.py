from train import GamePlay
from agents import RandomAgent, MarkovDecisionModelAgent, FilipkuvAgent, PPOAgent
import atari_py as ap
from gym import envs

#print sorted(ap.list_games())
#for e in sorted(envs.registry.env_specs.keys()):
#    print e

#agent = MarkovDecisionModelAgent('riverraid.model')
#agent = RandomAgent()
agent = FilipkuvAgent()

game = GamePlay('Riverraid-v0', PPOAgent())
game.train()
game.play()
print(game.get_score())
