# Second attempt at implementing network regression according to Muller 2022
# Yifan Xu, October 1st

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math

W = 2e2

def frobenius_norm(B, L):
    sum = 0
    for i in range(len(B)):
        for j in range(len(B)):
            sum += (B[i, j] - L[i, j])**2
    return sum

def global_network_regression(L_list: list, X_list: list, x: float):
    '''
    performs global network regression based on the given dataset and the input predictor

    L_list: a list of np.arrays containing graph laplacians of data
    X_list: a list of floats containing the corresponding covariates
    x: a float indicating the input

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
    constraints += [L[i, j] >= -W for i in range(m) for j in range(m) if i != j]  # W?
    
    # the objective function
    objective = cp.Minimize(cp.sum_squares(L - B))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    sol = L.value

    # print(sol)

    # keep only the positive eig vectors
    eval, evec = np.linalg.eig(sol)
    eval = np.where(eval > 0, eval, 0)
    sol = evec @ np.diag(eval) @ evec.T

    return sol

def polynomial_network_regression(L_list: list, X_list: list, x):
    '''
    performs polynomial global network regression based on the given dataset and the input predictor

    L_list: a list of np.arrays containing graph laplacians of data
    X_list: a list of numpy n by 1 matrices (ndarrays) containing the corresponding covariates
    x: a numpy n by 1 matrix indicating the input

    returns the graph laplacian in the form of a np.array that is the result of polynomial (of degree n) global network regression corresponding to x
    '''

    # find sample mean and covariance
    m = L_list[0].shape[0] # the number of nodes - also the dimension of the graph laplacians
    n = len(L_list) # the number (X, L), i.e., the number of datapoints
    mu = np.array(sum(X_list) / n)
    sigma = np.array(sum([(X_list[k] - mu) @ ((X_list[k] - mu).T) for k in range(n)]) / n)
    sigma_inv = np.linalg.inv(sigma)
    # sigma_inv = sigma**(-1)
    
    # find weights and B
    def weights_poly(X, x):
        print("Xmu")
        print(str(X) + ' '+str(mu))
        print(str((X-mu).T))
        print(sigma_inv @ (x - mu))
        weight = 1 + (((X - mu).T) @ (sigma_inv @ (x - mu)))
        # print(np.size(weight))
        return weight[0, 0]
    
    weights = [weights_poly(np.array(X_list[k]), np.array(x)) for k in range(n)]

    if (not (3.99 < sum(weights) < 4.01)):
        print("sum of weights is not 4, it is "+str(sum(weights)))
    degree = np.size(x)
    print('the weights to order ' + str(degree))
    print(weights)

    B = sum((weights[k] * L_list[k]) for k in range(n)) / sum(weights) # this is a m by m matrix
    print('barycenter')
    print(B)
    # now we solve the optimization problem
    # minimizer L
    L = cp.Variable((m, m), symmetric=True)

    # the set of linear constraints
    constraints = [L[i, j] == L[j, i] for i in range(m) for j in range(m)]  # Symmetry
    constraints += [sum(L[i, :]) == 0 for i in range(m)]  # Row sum to zero
    constraints += [L[i, j] <= 0 for i in range(m) for j in range(m) if i != j]  # edge weights must be non-neg
    constraints += [L[i, j] >= -W for i in range(m) for j in range(m) if i != j]  # W?
    
    # the objective function
    objective = cp.Minimize(cp.sum_squares(L - B))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    sol = L.value

    # print(sol - B)  

    # keep only the positive eig vectors
    eval, evec = np.linalg.eig(sol)
    eval = np.where(eval > 0, eval, 0)
    sol = evec @ np.diag(eval) @ evec.T

    return sol

def performance_plots(L_list, X_list, x_values, true_graphs, degrees):
    # global_reg = []
    means = []
    stds = []
    # for i in range(len(x_values)):
    #     global_reg.append(frobenius_norm(global_network_regression(L_list, X_list, x_values[i]), true_graphs[i]))
    # means.append(np.mean(global_reg))
    # stds.append(np.std(global_reg))
    
    
    for degree in degrees:
        # if (degree == 4):
        #     continue
        poly_reg = []
        X_list_poly = [np.array([[X_list[k]**deg] for deg in range(1, degree+1)]) for k in range(len(X_list))]
        x_poly = [np.array([[x_values[k]**deg] for deg in range(1, degree+1)]) for k in range(len(x_values))]
        # print("xlistpoly")
        # print(X_list_poly)
        # print(x_poly)
        for i in range(len(x_values)):
            # if (degree == 1):
            #     print(polynomial_network_regression(L_list, X_list_poly, x_poly[i]))
            #     print(global_network_regression(L_list, X_list, x_values[i]))
            poly_reg.append(frobenius_norm(polynomial_network_regression(L_list, X_list_poly, x_poly[i]), true_graphs[i]))
        means.append(np.mean(poly_reg))
        stds.append(np.std(poly_reg))

    plt.figure(figsize=(10, 6))
    plt.errorbar([i+1 for i in range(degree)], means,yerr=stds, capsize=5, capthick=1) 
    plt.xlabel('Degree of Polynomial Regression n')
    plt.ylabel('Average Error Across inputs 2 to 8')
    plt.title('Performance of Polynomial network regression on the Toy Example for X between 2 to 8')
    plt.legend()
    plt.grid(True)
    plt.show()

L1 = np.array([[1, -0.5, -0.5], 
               [-0.5, 1.5, -1],
               [-0.5, -1, 1.5]])
L2 = np.array([[0.5, -0.25, -0.25], 
               [-0.25, 0.75, -0.5], 
               [-0.25, -0.5, 0.75]])
L3 = np.array([[1/3, -1/6, -1/6],
                [-1/6, 0.5, -1/3],
                [-1/6, -1/3, 0.5]])
L4 = np.array([[0.25, -0.125, -0.125],
               [-0.125, 0.375, -0.25], 
               [-0.125, -0.25, 0.375]])

L_list = [L1, L2, L3, L4]
X_list = [2, 4, 6, 8]

x_values = [k for k in range(3, 8)]
true_graphs = [np.array([[2, -1, -1],
               [-1, 3, -2],
               [-1, -2, 3]])/k for k in range(3, 8)]

# print(polynomial_network_regression(L_list, np.array([[X_list[k]**deg] for deg in range(1, 8)]), ))
performance_plots(L_list, X_list, x_values, true_graphs, list(range(1, 4)))