import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from sklearn.preprocessing import StandardScaler


# PRODUCE SOME DATA -----------------------------------
def compute_range(v, g, theta):
    return (v**2 / g)*np.sin(2*np.deg2rad(theta))

velocities = range(0,200,5)
g = [10]
theta = range(-45,90,5)

traj = []
for x in itertools.product(velocities, g, theta):
    v, g, theta = x[0],x[1],x[2]
    d = compute_range(x[0],x[1],x[2])
    traj.append([v, g, theta, d])

traj = pd.DataFrame(traj, columns=['v', 'g', 'theta', 'd'])
traj['c'] = traj.d.apply(lambda x: 1 if x > 500 else 0)

plt.scatter(traj.v, traj.theta, c=traj.c)
sc = StandardScaler()
traj[['v', 'theta']] = sc.fit_transform(traj[['v', 'theta']])


# MODELLING WITH MACHINE LEARNING -------------------
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target*scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(features, target, num_steps, learning_rate, sleepiness=0.5):

    sleep(sleepiness)

    intercept = np.ones((features.shape[0], 1))
    features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        loss = log_likelihood(features, target, weights)
        print(loss)

        scr = np.dot(features, weights)

        if step%200 == 0:

            plt.scatter(features[:, 1], features[:, 2], c=np.round(sigmoid(scr)) == target, alpha=.8, s=10, cmap='GnBu')
            plt.title('loss: {}'.format(loss))
            plt.pause(sleepiness)
            sleep(0.5)
            print('plot')

    return weights

plt.ion()
weights = logistic_regression(traj[['v', 'theta']], traj['c'], num_steps=5000, learning_rate=5e-6, sleepiness=0.1)

final_scores = np.dot(traj[['v', 'theta']], weights[1:]) + weights[0]
preds = np.round(sigmoid(final_scores))

print('Accuracy: {}'.format((preds == traj['c']).sum().astype(float) / len(preds)))