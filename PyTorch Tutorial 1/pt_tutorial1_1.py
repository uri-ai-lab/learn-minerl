import gym
from pt_tutorial1 import Agent
from utils import plotLearning
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, e[silon=1.0, batch_sizw=64, n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.002)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        obeservation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(obervation, action, reward,
                                                obeservation_, done)
        agent.learn()
        obeservation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(score[-100:])

        print('episode ', i, 'score %.2f' % score,
                'avarage score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander_2020.png'
    plotLearning(x, scores, eps_history, filename)