from gameplay import Gameplay
from agents import RandomAgent, MarkovDecision
import atari_py as ap
from gym import envs

#print sorted(ap.list_games())
#for e in sorted(envs.registry.env_specs.keys()):
#    print e

agent = MarkovDecision('riverraid.model')
#agent = RandomAgent()

game = Gameplay('Riverraid-v0', agent)
game.play()
print game.get_score()
print game.get_lives()

# playback = game.get_playback()
#
# for i in xrange(100):
#     print "Playback len: %d" % len(playback._states)
#     game.play(playback)
#     print "Score: %d" % game.get_score()
#     #print game.get_lives()
#     playback = game.get_playback()


