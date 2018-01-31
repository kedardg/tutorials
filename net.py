import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from sklearn.preprocessing import StandardScaler


# PRODUCE SOME DATA -----------------------------------
# produce a set of points with velovity, gravitational accelleration and angle 
# compute the range with the equation below
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

# define the output class: if range is over 500m
traj['c'] = traj.d.apply(lambda x: 1 if x > 500 else 0)

# plot data
# plt.scatter(traj.v, traj.theta, c=traj.c)

# scale data between 1 and 0
sc = StandardScaler()
traj[['v', 'theta']] = sc.fit_transform(traj[['v', 'theta']])


# MODELLING WITH MACHINE LEARNING -------------------
def sigmoid(scores):
	"""funciton to compute the sigmoid to be used as the activation function"""
	return 1 / (1 + np.exp(-scores))


def compute_cost(A, Y):
    """
    Computes the cross-entropy cost

    Arguments:
    A2 -- The sigmoid output of the activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cross-entropy 
    """

    m = len(Y)  # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A), Y) + np.multiply((1 - Y), np.log(1 - A))
    cost = - np.sum(logprobs) / m
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.

    return cost


def nn_model(X, Y, n_h, learning_rate, num_iterations=10000, print_cost=False):

    n_x = 2 # number of input features
    n_y = 1 # number of classes
    m = X.shape[1] # number of samples

    W1 = np.random.randn(n_h, n_x) * 0.01 	# l1 wights, neurons x features
    b1 = np.zeros(shape=(n_h, 1))			# l1 bias, neurons x 1
    W2 = np.random.randn(n_y, n_h) * 0.01	# l2 wights, neurons x features
    b2 = np.zeros(shape=(n_y, 1))			# l2 bias, neurons x 1

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Implement Forward Propagation to calculate A2 (probabilities)
		# multiply weights by samples
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        # Cost function
        cost = compute_cost(A2, Y)
        #cost = np.squeeze(cost)

        # Backward propagation: calculate dW1, db1, dW2, db2.
        dZ2 = A2 - Y 
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        # Print the cost every 1000 iterations
        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

        # Print the cost every 1000 iterations
        if i % 5000 == 0:
            Z1 = np.dot(W1, traj[['v', 'theta']].T) + b1
            A1 = np.tanh(Z1)
            Z2 = np.dot(W2, A1) + b2
            A2 = sigmoid(Z2)
            yhat = np.round(A2)
            plt.scatter(X.iloc[0,:], X.iloc[1,:], c=yhat.ravel()==Y.ravel(), alpha=.8, s=10)
            plt.title('loss: {}'.format(cost))
            plt.pause(0.5)
            sleep(0.5)

    return W1, W2, b1, b2


W1, W2, b1, b2 = nn_model(traj[['v', 'theta']].T, traj['c'].reshape(1,-1), n_h=3, learning_rate=0.01, num_iterations=50000)

Z1 = np.dot(W1, traj[['v', 'theta']].T) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)
results = np.round(A2)

print('Accuracy: {}'.format((pd.Series(results.ravel()) == traj['c']).sum().astype(float) / len(traj)))
