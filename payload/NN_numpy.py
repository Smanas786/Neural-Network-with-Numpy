# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 19:10:19 2020

@author: ashah
"""

# # Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Generating Random DataSet for the Neural Network")

X = np.vstack([(np.random.rand(10,2)*5), (np.random.rand(10,2)*10)]) # Vertical Stack of 2 arrays 2x(10x2) = (20x2)
Y = np.hstack([[0]*10,[1]*10])                                       # Horizontal Stack of 0 and 1s
dataset = pd.DataFrame(X, columns={"X1", "X2"})
dataset['Y'] = Y

print("X Matrix of {}".format(X.shape), end="")
print(" and Y Matrix of {}".format(Y.shape))
print(dataset.head())
print(dataset.tail())

# # Lets Visualize our DataSet

plt.plot(dataset)

print("A NN model to show the Probability of Y i.e. 0 and 1 on the random dataset of X")
print("for Lower Value of X, Y is 0")
print("for higher Value of X, Y is 1")

# # One-hot encoding

v_Actual = np.zeros((20,2))
for i in range(20):
    v_Actual[i, Y[i]] = 1

# # Assign weights and bias for our Neural Network
print("Creating 2 layer deep network with 3 and 2 neuron")

W1 = np.random.rand(3,2)   # First Layer 
B1 = np.random.rand(3)     # with 3 Neuron
W2 = np.random.rand(3,2)   # Second Layer
B2 = np.random.rand(2)     # with 2 Neuron


# # Forward Propogation

def func_FP(X, W1, B1, W2, B2):
    
    # Simple first Layer
    Z = X.dot(W1.T) + B1              # Z(20x3) = W.T(2x3) * X(20x2) + B #20x3 Matrix
    v_Sigmoid = 1 / (1 + np.exp(-Z))  # v_Sigmoid = Sigmoid(Z)
    
    #Second Layer
    N = v_Sigmoid.dot(W2) + B2      # N(20x2) = W(3x2) * v_Sigmoid(20x3) + B #2nd Layer with 20x2 Matrix
    
    # SoftMax to Output Layer
    A = np.exp(N)                                 # Probability Output using softmax formula (e^z11/e^z11 + e^z21+ e^z31)
    v_Softmax = A / A.sum(axis=1, keepdims=True)  # keep Dimension to avoid ValueError of shapes (20,2) (20,) 
   
    return v_Softmax, v_Sigmoid


# func_FP(X, W1, B1, W2, B2)  # Return v_Softmax[0] = Class Probability of 0 & v_Softmax[1] = Class Probability of 1

# # Time to go back : Gradient Descent - finding a local minima

# ### Differentiate the cost function with rspt to Weight and Biases

# Gradient with rspt to W - 1st Differenciation
def func_W2(H_out, v_Actual, v_Pred):
    return H_out.T.dot(v_Actual - v_Pred)

# Gradient with rspt to W - 2nd Differenciation
def func_W1(X, H_out, v_Actual, v_Pred, W2):
    return X.T.dot((v_Actual - v_Pred).dot(W2.T) * H_out * (1-H_out))

# Gradient with rspt to B - 1st Differenciation
def func_B2(v_Actual, v_Pred):
    return (v_Actual - v_Pred).sum(axis=0)

# Gradient with rspt to B - 2nd Differenciation
def func_B1(v_Actual, v_Pred, W2, H_out):
    return ((v_Actual - v_Pred).dot(W2.T) * H_out * (1-H_out)).sum(axis=0)


lr = 1e-3 # Learning Rate default value set to 0.0001
epochs = 5000 # For GD, epochs Cycles should be enough to avoid underfitting and overfitting

print("Training Model...", end='')
for iteration in range(epochs):    
    v_Pred, H_out = func_FP(X, W1, B1, W2, B2)
    W2 += lr * func_W2(H_out, v_Actual, v_Pred)
    B2 += lr * func_B2(v_Actual, v_Pred)
    W1 += lr * func_W1(X, H_out, v_Actual, v_Pred, W2).T
    B1 += lr * func_B1(v_Actual, v_Pred, W2, H_out)

print("Done")
print("Lets Test our Model")

def func_Testing(Test_X):
    Test_Hidden = 1 / (1 + np.exp(-Test_X.dot(W1.T) - B1))
    Test_Output = Test_Hidden.dot(W2) + B2
    val_Y = np.exp(Test_Output)
    Test_Y = val_Y / val_Y.sum()

    print("Testing the model wtih new Weight and Bias for Input: ", Test_X)
    print("Probability of class 0 with Input from Test_X: ", Test_Y[0])
    print("Probability of class 1 with Input from Test_X: ", Test_Y[1])

# # Some Test Cases to Run

func_Testing(np.array([9,9])) # Provided Input

func_Testing(np.array([5,7])) # Provided Input

func_Testing(np.array([1,2])) # Provided Input

func_Testing(np.array([3,5])) # Provided Input