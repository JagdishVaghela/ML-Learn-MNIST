# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Acessing the MNIST data using pandas (column n is 1+data due to label column), Cleaning test and train data into nympy arrays

# %%
data1 = pd.read_csv("mnist_test.csv")
# data.head()
data1 = np.array(data1)
m, n = data1.shape
np.random.shuffle(data1)

data_test = data1[0:10000].T
Y_test = data_test[0]
X_test = data_test[1:n]

# %%
data2 = pd.read_csv("mnist_train.csv")
# data.head()
data2 = np.array(data2)
m, n = data2.shape
np.random.shuffle(data2)

data_train = data2[0:60000].T
Y_train = data_train[0]
X_train = data_train[1:n]

# %%
# X_train[:,0].shape

# %% [markdown]
# Neural network code

# %%
# Initialize parameters
def init_parameters():
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2

# Machine learning Forward propagation

# Activation function, returns 0 if Z < 0, else returns Z
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_propagation(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + B2
    A2 = softmax(A1)
    return Z1, A1, Z2, A2

# Machine learning backwards propagation
def one_hot_encode(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size),Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def ReLU_derivative(Z):
    return Z > 0

def back_propagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_y = one_hot_encode(Y)
    dZ2 = A2 - one_hot_y
    dW2 = (1/m) * dZ2.dot(A1.T)
    dB2 = (1/m) * np.sum(dZ2, 1)
    dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)

    dW1 = (1/m) * dZ1.dot(X.T)
    dB1 = (1/m) * np.sum(dZ1, 1)
    
    return dW1, dB1, dW2, dB2

def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
    W1 -= learning_rate * dW1
    B1 -= learning_rate * dB1
    W2 -= learning_rate * dW2
    B2 -= learning_rate * dB2
    return W1, B1, W2, B2 

# %% [markdown]
# Code for Gradient Descent

# %%
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.mean(predictions == Y)


def gradient_descent(X, Y, iterations, learning_rate):
    W1, B1, W2, B2 = init_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = back_propagation(Z1, A1, Z2, A2, W2, X, Y)
        W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate)

        if i % 100 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}, Accuracy: {accuracy}")
    return W1,B1,W2,B2   


# %% [markdown]
# Run network and train with  data

# %%
W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 500, 0.01)


