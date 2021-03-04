import minerl
import gym

env = gym.make('MineRLNavigateDense-v0')

# Type in another terminal to connect to miecraft server: python -m minerl.interactor 6666

# set the environment to allow interactive connections on port 6666
# and slow the tick speed to 6666.
env.make_interactive(port=6666, realtime=True)

obs = env.reset()

done = False
net_reward = 0

# Code bellow  is from the no-op example in 'https://minerl.io/docs/tutorials/first_agent.html#no-op-actions-and-a-better-policy'

while not done:
    action = env.action_space.noop()

    action['camera'] = [0, 0.03*obs["compassAngle"]]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(
        action)

    net_reward += reward
    print("Total reward: ", net_reward)

