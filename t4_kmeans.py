import gym
import tqdm
import minerl
import numpy as np

from sklearn.cluster import KMeans

dat = minerl.data.make('MineRLTreechopVectorObf-v0')

act_vectors = []
NUM_CLUSTERS = 30

# Load the dataset storing 1000 batches of actions
for _, act, _, _, _ in tqdm.tqdm(dat.batch_iter(16, 32, 2, preload_buffer_size=20)):
    act_vectors.append(act['vector'])
    if len(act_vectors) > 1000:
        break

# Reshape these the action batches
acts = np.concatenate(act_vectors).reshape(-1, 64)
kmeans_acts = acts[:100000]

# Use sklearn to cluster the demonstrated actions
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(kmeans_acts)

i, net_reward, done, env = 0, 0, False, gym.make('MineRLTreechopVectorObf-v0')
obs = env.reset()

while not done:
    # Let's use a frame skip of 4 (could you do better than a hard-coded frame skip?)
    if i % 4 == 0:
        action = {
            'vector': kmeans.cluster_centers_[np.random.choice(NUM_CLUSTERS)]
        }

    obs, reward, done, info = env.step(action)
    env.render()

    if reward > 0:
        print("+{} reward!".format(reward))
    net_reward += reward
    i += 1

print("Total reward: ", net_reward)