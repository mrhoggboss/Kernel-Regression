# Implementation of local network regression according to Muller 2022
# Yifan Xu, May 2024

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def global_network_regression(L_list: list, X_list: list, x: float):
    '''
    performs global network regression based on the given dataset and the input predictor

    L_list: a list of np.arrays containing graph laplacians of data
    X_list: a list of floats containing the corresponding covariates
    x: a scalar which indicates the input predictor.

    returns the graph laplacian in the form of a np.array that is the result of global network regression corresponding to x
    '''
    # find sample mean and covariance
    mu = np.mean(X_list)
    sigma = np.var(np.array(X_list))
    sigma_inv = 1/sigma
    
    # find weights and B
    def weights_global(X: float, x:float):
        return 1 + (X - mu) * sigma_inv * (x - mu)
    m = L_list[0].shape[0] # the number of nodes - also the dimension of the graph laplacians
    n = len(L_list) # the number (X, L), i.e., the number of datapoints
    weights = [weights_global(X_list[k], x) for k in range(n)]
    B = sum((weights[k] * L_list[k]) for k in range(n)) / sum(weights) # this is a m by m matrix

    # now we solve the optimization problem
    # minimizer L
    L = cp.Variable((m, m), symmetric=True)

    # the set of linear constraints
    constraints = [L[i, j] == L[j, i] for i in range(m) for j in range(m)]  # Symmetry
    constraints += [sum(L[i, :]) == 0 for i in range(m)]  # Row sum to zero
    constraints += [L[i, j] <= 0 for i in range(m) for j in range(m) if i != j]  # edge weights must be non-neg
    # constraints += [L[i, j] >= -W for i in range(m) for j in range(m) if i != j]  # W?
    
    # the objective function
    objective = cp.Minimize(cp.sum_squares(L - B))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    
    return L.value

def local_network_regression(L_list: list, X_list: list, x: float, kernel, bandwidth: float):
    '''
    performs local network regression based on the given dataset, kernel, bandwidth and the input predictor

    L_list: a list of np.arrays containing graph laplacians of data
    X_list: a list of floats containing the corresponding covariates
    x: a scalar which indicates the input predictor
    kernel: a mathematical function representing the kernel
    bandwidth: a number representing the bandwidth used

    returns the graph laplacian in the form of a np.array that is the result of local network regression corresponding to x
    '''

    m = L_list[0].shape[0] # the number of nodes - also the dimension of the graph laplacians
    n = len(L_list) # the number (X, L), i.e., the number of datapoints

    # find mu's
    def K_h(X):
        return kernel((X - x)/bandwidth)/bandwidth
    mu0 = sum(K_h(X_list[k] - x) for k in range(n)) / n
    mu1 = sum(K_h(X_list[k] - x) * (X_list[k] - x) for k in range(n)) / n
    mu2 = sum(K_h(X_list[k] - x) * (X_list[k] - x)**2 for k in range(n)) / n
    sigma_squared = mu0*mu2 - mu1**2

    # find weights and B
    def weights_local(X: float, x:float):
        return K_h(X - x) * (mu2 - mu1 * (X - x)) / sigma_squared
    
    weights = np.array([weights_local(X_list[k], x) for k in range(n)])
    B = sum((weights[k] * L_list[k]) for k in range(n)) / sum(weights) # this is a m by m matrix

    # now we solve the optimization problem
    # minimizer L
    L = cp.Variable((m, m), symmetric=True)

    # the set of linear constraints
    constraints = [L[i, j] == L[j, i] for i in range(m) for j in range(m)]  # Symmetry
    constraints += [sum(L[i, :]) == 0 for i in range(m)]  # Row sum to zero
    constraints += [L[i, j] <= 0 for i in range(m) for j in range(m) if i != j]  # edge weights must be non-neg
    # constraints += [L[i, j] >= -W for i in range(m) for j in range(m) if i != j]  # W?
    
    # the objective function
    objective = cp.Minimize(cp.sum_squares(L - B))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    
    return L.value

# Toy Example in the paper

L1 = np.array([[1, -0.5, -0.5], 
               [-0.5, 1.5, -1],
               [-0.5, -1, 1.5]])
L2 = np.array([[0.75, -0.25, -0.25], 
               [-0.25, 0.5, -0.5], 
               [-0.25, -0.5, 0.75]])
L3 = np.array([[1/3, -1/6, -1/6],
                [-1/6, 0.5, -1/3],
                [-1/6, -1/3, 0.5]])
L4 = np.array([[0.25, -0.125, -0.125],
               [-0.125, 0.375, -0.25], 
               [-0.125, -0.25, 0.375]])

L_list = [L1, L2, L3, L4]
X_list = [2, 4, 6, 8]

# print(global_network_regression(L_list, X_list, 5))
# print(local_network_regression(L_list, X_list, 5, lambda x : 3 * (1 - x**2) / 4, 2)*24)

def frobenius_norm(B, L):
    sum = 0
    for i in range(len(B)):
        for j in range(len(B)):
            sum += (B[i, j] - L[i, j])**2
    return sum

def plot_error(L_list, X_list, x_values, true_graphs, kernel, bandwidth=1.0):
    global_frobenius_values = []
    local_frobenius_values = []
    
    for i in range(len(x_values)):
        global_frobenius_values.append(frobenius_norm(global_network_regression(L_list, X_list, x_values[i]), true_graphs[i]))
        local_frobenius_values.append(frobenius_norm(local_network_regression(L_list, X_list, x_values[i], kernel, bandwidth), true_graphs[i]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, global_frobenius_values, label='Global Network Regression', marker='o')
    plt.plot(x_values, local_frobenius_values, label='Local Network Regression', marker='x')
    plt.xlabel('Predictor X')
    plt.ylabel('Frobenius Norm')
    plt.title('Frobenius Norm vs Predictor X for Global and Local Network Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

x_values = [2, 3, 4, 5, 6, 7, 8]
true_graphs = [np.array([[2/k, -1/k, -1/k],
               [-1/k, 3/k, -2/k],
               [-1/k, -2/k, 3/k]]) for k in range(2, 9)]
# print(true_graphs)
plot_error(L_list, X_list, x_values, true_graphs, lambda x : 3 * (1 - x**2) / 4, 2)