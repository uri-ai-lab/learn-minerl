import minerl
import gym
#import logging

# do your main minerl code
#logging.basicConfig(level=logging.DEBUG)
env = gym.make('MineRLNavigateDense-v0')

obs = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)