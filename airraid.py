import gym
import atari_py as ap
import time
game_list = ap.list_games()
print(sorted(game_list))
env = gym.make('SpaceInvaders-v0')
#env = gym.make('Zaxxon-v0')
#env = gym.make('MontezumaRevenge-v0')
#env = gym.make('Riverraid-v0')
#env = gym.make('AirRaid-v0')
env.reset()
env
game_count = 1
for i in range(50000):
  action_n = env.action_space.sample()
  state, _, done, info = env.step(action_n)
  env.render()
  time.sleep(0.1)
  if done:
    env.reset()
