# Implementation of local network regression according to Muller 2022
# Yifan Xu, May 2024

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
import copy

W = 2e10

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

def local_network_regression(L_list: list, X_list: list, x: float, kernel, bandwidth: float):
    '''
    performs local network regression based on the given dataset, kernel, bandwidth and the input predictor

    L_list: a list of np.arrays containing graph laplacians of data
    X_list: a list of floats containing the corresponding covariates
    x: a scalar which indicates the input predictor
    kernel: a mathematical function representing the kernel, it takes in two inputs
    bandwidth: a number representing the bandwidth used

    returns the graph laplacian in the form of a np.array that is the result of local network regression corresponding to x
    '''

    m = L_list[0].shape[0] # the number of nodes - also the dimension of the graph laplacians
    n = len(L_list) # the number (X, L), i.e., the number of datapoints

    # find mu's
    def K_h(X):
        return kernel(X - x, bandwidth)
    mu0 = sum(K_h(X_list[k] - x) for k in range(n)) / n
    mu1 = sum(K_h(X_list[k] - x) * (X_list[k] - x) for k in range(n)) / n
    mu2 = sum(K_h(X_list[k] - x) * (X_list[k] - x)**2 for k in range(n)) / n
    sigma_squared = mu0*mu2 - mu1**2
    
    # # sometimes these get too small
    # mu0 += 1e-6
    # mu1 += 1e-6
    # mu2 += 1e-6
    # sigma_squared += 1e-6

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
    constraints += [L[i, j] >= -W for i in range(m) for j in range(m) if i != j]  # W?
    
    # the objective function
    objective = cp.Minimize(cp.sum_squares(L - B))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    sol = L.value

    # keep only the positive eig vectors
    eval, evec = np.linalg.eig(sol)
    eval = np.where(eval > 0, eval, 0)
    sol = evec @ np.diag(eval) @ evec.T

    return sol

def frobenius_norm(B, L):
    sum = 0
    for i in range(len(B)):
        for j in range(len(B)):
            sum += (B[i, j] - L[i, j])**2
    return sum

def plot_error(L_list, X_list, x_values, true_graphs, kernel, bandwidth=1.0, just_global = False):
    global_frobenius_values = []
    local_frobenius_values = []
    
    for i in range(len(x_values)):
        glob = global_network_regression(L_list, X_list, x_values[i])
        global_frobenius_values.append(frobenius_norm(glob, true_graphs[i]))
        # print('the ' + str(i) + 'th value is fine')
        if not just_global:
            loc = local_network_regression(L_list, X_list, x_values[i], kernel, bandwidth)
            local_frobenius_values.append(frobenius_norm(loc, true_graphs[i]))

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, global_frobenius_values, label='Global Network Regression', marker='o')
    if not just_global:
        plt.plot(x_values, local_frobenius_values, label='Local Network Regression', marker='x')
    plt.xlabel('Predictor X')
    plt.ylabel('Frobenius Norm')
    plt.title('Frobenius Norm vs Predictor X for Network Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_kernels_plot(L_list, X_list, x_values, true_graphs, kernels: dict, bandwidth=1.0):
    '''
    kernels: a dictionary with names(strings) as keys and the function itself as values
    '''
    y_values = dict()

    for label in kernels:
        local_frobenius_values = []
        for i in range(len(x_values)):
            local_frobenius_values.append(frobenius_norm(local_network_regression(L_list, X_list, x_values[i], kernels[label], bandwidth), true_graphs[i]))
        y_values[label] = local_frobenius_values
    plt.figure(figsize=(10, 6))
    for label in y_values:
        plt.plot(x_values, y_values[label], label=label)
    plt.xlabel('Predictor X')
    plt.ylabel('Frobenius Norm')
    plt.title('Frobenius Norm vs Predictor X for Various Kernels for Local Network Regression')
    plt.legend()
    plt.grid(True)
    plt.show()


# # -----------------------------------------------------------------------------------------------------------------
# # Toy Example in the paper

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

# print(global_network_regression(L_list, X_list, 7))
# # print(global_network_regression(L_list, X_list, 5))

# # res = global_network_regression(L_list, X_list, 10)
# # max_entry = res.max()
# # res = res / abs(max_entry)
# # k = 10
# # print(res)
# # # print(print(np.array([[2/k, -1/k, -1/k],
# # #                [-1/k, 3/k, -2/k],
# # #                [-1/k, -2/k, 3/k]])))
# # # print('---------')
# # print(res)
# # # print(local_network_regression(L_list, X_list, 5, lambda x, bandwidth : 3 * (1 - (x / bandwidth)**2) / 4, 2)*24)

# # # print(np.array([[2/k, -1/k, -1/k],
# # #                [-1/k, 3/k, -2/k],
# # #                [-1/k, -2/k, 3/k]]) / (frobenius_norm(np.array([[2/k, -1/k, -1/k],
# # #                [-1/k, 3/k, -2/k],
# # #                [-1/k, -2/k, 3/k]]), np.zeros((3, 3)))))

# # print(np.array([[2/k, -1/k, -1/k],
# #                [-1/k, 3/k, -2/k],
# #                [-1/k, -2/k, 3/k]]) / (3 / k))

# x_values = [k for k in range(2, 31)]
# true_graphs = [np.array([[2, -1, -1],
#                [-1, 3, -2],
#                [-1, -2, 3]])/k for k in range(2, 31)]

# x_values = [k/100 for k in range(430, 451)]
# true_graphs = [np.array([[20, -10, -10],
#                [-10, 30, -20],
#                [-10, -20, 30]])/(k/10) for k in range(430, 451)]

# plot_error(L_list, X_list, x_values, true_graphs, lambda x, bandwidth : 3 * (1 - (x / bandwidth)**2) / 4, 2)

# print(local_network_regression(L_list, X_list, 4.4, ))

# plot_error(L_list, X_list, x_values, true_graphs, lambda x, bandwidth: 1 - abs(x) / bandwidth, 2)
# # k=15
# # print(global_network_regression(L_list, X_list, 15))
# # print([[2/k, -1/k, -1/k],
# #                [-1/k, 3/k, -2/k],
# #                [-1/k, -2/k, 3/k]])


kernels = {
    # 'Gaussian': lambda x, bandwidth : math.exp(-(x / bandwidth)**2),
    # 'Epanechnikov': lambda x, bandwidth : 3 * (1 - (x / bandwidth)**2) / 4,
    # 'Uniform': lambda x, bandwidth : 0.5,
    'Triangular': lambda x, bandwidth: 1 - abs(x) / bandwidth,
    'RBF': lambda x, bandwidth: math.exp(-bandwidth * x ** 2)
}

# compare_kernels_plot(L_list, X_list, x_values, true_graphs, kernels, 2)
# print(local_network_regression(L_list, X_list, 31, lambda x, bandwidth : math.exp(-(x / bandwidth)**2), 2))

# # -----------------------------------------------------------------------------------------------------------------
# # Constructing our own example using given eigenvalues

def eigenvalue_plot(L_list, X_list, x_values, kernels, bandwidth):
    '''
    plots the eigenvalues of the outputs of network regression for x_values.
    '''
    fig, ax = plt.subplots()
    
    # plotting the true values
    x_vals = np.array(x_values)
    x_vals = x_vals[0 <= x_vals]
    x_vals = x_vals[x_vals <= 1]
    y_values = []
    for x in x_vals:
        y_values.append(math.sqrt(1 - x**2))
    plt.scatter(x_vals, y_values, label='true values', color = 'r')
    
    # plotting the predictions
    for kernel in kernels:
        lambda_1s = []
        lambda_2s = []

        for i in range(len(x_values)):
            pred = local_network_regression(L_list, X_list, x_values[i], kernels[kernel], bandwidth)
            eigenvalues, eigenvectors = np.linalg.eig(pred)
            non_zero = []
            # to make sure we only exclude one zero from the eigenvalues
            flag = True
            for eval in eigenvalues:
                if abs(eval) < 1e-15 and flag:
                    flag = False
                else:
                    non_zero.append(eval)

            # non_zero.sort()
            if 5.77e-01 < abs(eigenvectors[0, 0]) < 5.78e-01:
                # column corresponding to zero goes first
                lambda_1s.append(non_zero[0])
                lambda_2s.append(non_zero[1])
            elif 4.08e-01 < abs(eigenvectors[0, 0]) < 4.09e-01:
                # column corresponding to y goes first
                lambda_2s.append(non_zero[0])
                lambda_1s.append(non_zero[1])
            elif 7.07e-01 < abs(eigenvectors[0, 0])< 7.08e-01:
                # column corresponding to x goes first
                lambda_1s.append(non_zero[0])
                lambda_2s.append(non_zero[1])
            elif eigenvectors[0, 0] == 1 and eigenvectors[1, 1] == 1 and eigenvectors[2, 2] == 1:
                lambda_1s.append(0)
                lambda_2s.append(0)
            else:
                print('error!!')
        ax.scatter(lambda_1s, lambda_2s, label=kernel, alpha = 0.3)
    
        print(lambda_1s)
        print(x_vals)
        print(y_values)
        print([x_vals[0], y_values[0]])
        
        # establish the connections between the true values and the predictions
        for idx in range(len(x_values)):
            plt.plot([lambda_1s[idx], x_vals[idx]], [lambda_2s[idx], y_values[idx]], 'r-')
    
    plt.xlabel('lambda1')
    plt.ylabel('lambda2')
    plt.title('A plot of non-zero eigenvalues of local regression results')
    plt.legend()
    plt.grid(True)
    plt.show()

# # the eigenvector matrix we use (basis) is
# U = np.matrix([
#     [-math.sqrt(2)/2, -math.sqrt(6)/6, math.sqrt(3)/3],
#     [math.sqrt(2)/2, -math.sqrt(6)/6, math.sqrt(3)/3],
#     [0, math.sqrt(6)/3, math.sqrt(3)/3]
# ])

# # generate the x's, and construct the graphs using reverse SVD
# X_list = [x/10 for x in range(0, 11)]
# L_list = [U @ np.matrix(np.diag([x/10, math.sqrt(1 - (x/10)**2), 0])) @ U.T for x in range(0, 11)]
# # find eigenvalues for the outputs and plot the two non-zero eigvals
# x_values = [x/10 for x in range(0, 11)]
# # print(local_network_regression(L_list, X_list, -0.9, kernels['Epanechnikov'], 2))
# eigenvalue_plot(L_list, X_list, x_values, kernels, 2)
# # pred = local_network_regression(L_list, X_list, 0, kernels['Uniform'], 2)
# # eigenvalues, eigenvectors = np.linalg.eig(pred)
# # print(pred)
# # print(eigenvalues)
# # eigenvalues, eigenvectors = np.linalg.eig(L_list[0])
# # print(eigenvalues)

# # -----------------------------------------------------------------------------------------------------------------
# # Polynomial Regression with the toy example and the eigenvalue plot

def polynomial_network_regression(L_list: list, X_list: list, x):
    '''
    performs polynomial global network regression based on the given dataset and the input predictor

    L_list: a list of np.arrays containing graph laplacians of data
    X_list: a list of numpy n by 1 matrices (ndarrays) containing the corresponding covariates
    x: a numpy n by 1 matrix indicating the input

    returns the graph laplacian in the form of a np.array that is the result of polynomial (of degree n) global network regression corresponding to x
    '''

    # find sample mean and covariance
    
    data_merged = np.hstack(X_list)
    mu = np.mean(data_merged, axis=1, keepdims=True)
    sigma = np.cov(data_merged)
    sigma_inv = np.linalg.inv(sigma)
    
    # find weights and B
    def weights_global(X, x):
        weight = 1 + (X - mu).T @ sigma_inv @ (x - mu)
        return weight[0, 0]
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

# # the toy example with quadratic covariates
# L1 = np.array([[1, -0.5, -0.5], 
#                [-0.5, 1.5, -1],
#                [-0.5, -1, 1.5]])
# L2 = np.array([[0.5, -0.25, -0.25], 
#                [-0.25, 0.75, -0.5], 
#                [-0.25, -0.5, 0.75]])
# L3 = np.array([[1/3, -1/6, -1/6],
#                 [-1/6, 0.5, -1/3],
#                 [-1/6, -1/3, 0.5]])
# L4 = np.array([[0.25, -0.125, -0.125],
#                [-0.125, 0.375, -0.25], 
#                [-0.125, -0.25, 0.375]])

# L_list = [L1, L2, L3, L4]
# X_list = [np.array([[2],
#                     [4]]), 
#           np.array([[4],
#                     [16]]), 
#           np.array([[6],
#                    [36]]),
#           np.array([[8],
#                    [64]])]

# print(polynomial_network_regression(L_list, X_list, np.array([[7], [49]])))
def compare_polynomial_plot(L_list, X_list, x_values, true_graphs, degrees):
    y_values = dict()
    global_reg = []
    
    for i in range(len(x_values)):
        global_reg.append(frobenius_norm(global_network_regression(L_list, X_list, x_values[i]), true_graphs[i]))
    y_values['global'] = global_reg
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values['global'], label='global')
    
    for degree in degrees:
        poly_reg = []
        X_list_poly = [np.array([[X_list[k]**deg] for deg in range(1, degree+1)]) for k in range(len(X_list))]
        x_poly = [np.array([[x_values[k]**deg] for deg in range(1, degree+1)]) for k in range(len(x_values))]
        for i in range(len(x_values)):
            poly_reg.append(frobenius_norm(polynomial_network_regression(L_list, X_list_poly, x_poly[i]), true_graphs[i]))
        print('----------------------')
        print('degree '+str(degree))
        print('mean error: '+str(np.average(poly_reg)))
        print('std of error: '+str(np.std(poly_reg)))
        y_values[str(degree)] = poly_reg
    # print(y_values)
        plt.plot(x_values, y_values[str(degree)], label='degree '+str(degree))
    plt.xlabel('Predictor X')
    plt.ylabel('Frobenius Norm')
    plt.title('Frobenius Norm vs Predictor X for global and quadratic Network Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

def performance_plots(L_list, X_list, x_values, true_graphs, degrees):
    y_values = dict()
    global_reg = []
    means = []
    stds = []
    for i in range(len(x_values)):
        global_reg.append(frobenius_norm(global_network_regression(L_list, X_list, x_values[i]), true_graphs[i]))
    means.append(np.average(global_reg))
    stds.append(np.std(global_reg))
    plt.figure(figsize=(10, 6))
    
    for degree in degrees:
        poly_reg = []
        X_list_poly = [np.array([[X_list[k]**deg] for deg in range(1, degree+1)]) for k in range(len(X_list))]
        x_poly = [np.array([[x_values[k]**deg] for deg in range(1, degree+1)]) for k in range(len(x_values))]
        for i in range(len(x_values)):
            poly_reg.append(frobenius_norm(polynomial_network_regression(L_list, X_list_poly, x_poly[i]), true_graphs[i]))
        means.append(np.average(poly_reg))
        stds.append(np.std(poly_reg))

    normalized_means = [max(means) - error for error in means]
    normalized_means.pop(-2)
    stds.pop(-2)
    plt.errorbar([i+1 for i in range(degree) if i != 7], normalized_means,yerr=stds, capsize=5, capthick=1)
    plt.xlabel('Degree of Polynomial Regression n')
    plt.ylabel('Performance (normalized error)')
    plt.title('Performance of Polynomial network regression on the Toy Example for X between 2 to 8')
    plt.legend()
    plt.grid(True)
    plt.show()

x_values = [k for k in range(2, 9)]
true_graphs = [np.array([[2, -1, -1],
               [-1, 3, -2],
               [-1, -2, 3]])/k for k in range(2, 9)]

# compare_polynomial_plot(L_list, X_list, x_values, true_graphs, list(range(2, 10)))
performance_plots(L_list, X_list, x_values, true_graphs, list(range(2, 10)))

# # ----------------------------------------------------------------------------------------------------
# # comparing eigvals for global and polynomial reg

def global_poly_eigenvalue_plot(L_list, X_list, x_values, degree: int):
    '''
    X_list: a list of floats representing the covariates
    x_values: a list of floats representing the inputs to the regression algorithm

    plots the eigenvalues of the outputs of network regression for x_values.
    '''
    fig, ax = plt.subplots()
    
    # plotting the true values
    x_vals = np.array(x_values)
    x_vals = x_vals[0 <= x_vals]
    x_vals = x_vals[x_vals <= 1]
    y_values = []
    for x in x_vals:
        y_values.append(math.sqrt(1 - x**2))
    plt.scatter(x_vals, y_values, label='true values', color = 'r')
    
    # plotting the predictions for global
    lambda_1s = []
    lambda_2s = []

    for i in range(len(x_values)):
        pred = global_network_regression(L_list, X_list, x_values[i])
        eigenvalues, eigenvectors = np.linalg.eig(pred)
        # if i < 5:
        #     print(eigenvalues)
        #     print(pred)
        #     print('---------------------------')
        non_zero = []
        # to make sure we only exclude one zero from the eigenvalues
        flag = True
        for eval in eigenvalues:
            if abs(eval) < 1e-15 and flag:
                flag = False
            else:
                non_zero.append(eval)
        # non_zero.sort()
        if 5.77e-01 < abs(eigenvectors[0, 0]) < 5.78e-01 and 7.07e-01 < abs(eigenvectors[0, 1])< 7.08e-01:
            # column corresponding to zero goes first
            lambda_1s.append(non_zero[0])
            lambda_2s.append(non_zero[1])
        elif 4.08e-01 < abs(eigenvectors[0, 0]) < 4.09e-01 and 7.07e-01 < abs(eigenvectors[0, 1])< 7.08e-01:
            # column corresponding to y goes first
            lambda_2s.append(non_zero[0])
            lambda_1s.append(non_zero[1])
        elif 7.07e-01 < abs(eigenvectors[0, 0])< 7.08e-01 and 5.77e-01 < abs(eigenvectors[0, 1]) < 5.78e-01:
            # column corresponding to x goes first
            lambda_1s.append(non_zero[0])
            lambda_2s.append(non_zero[1])
        elif eigenvectors[0, 0] == 1 and eigenvectors[1, 1] == 1 and eigenvectors[2, 2] == 1:
            lambda_1s.append(0)
            lambda_2s.append(0)
        else:
            print('error!!')
    ax.scatter(lambda_1s, lambda_2s, label='global', alpha = 0.3)

    # establish the connections between the true values and the predictions
    for idx in range(len(x_values)):
        plt.plot([lambda_1s[idx], x_vals[idx]], [lambda_2s[idx], y_values[idx]], 'b-')

    # plotting the predictions for quadratic
    lambda_1s = []
    lambda_2s = []
    X_list_poly = [np.array([[X_list[k]**deg] for deg in range(1, degree+1)]) for k in range(len(X_list))]
    x_poly = [np.array([[x_values[k]**deg] for deg in range(1, degree+1)]) for k in range(len(x_values))]
    for i in range(len(x_values)):
        pred = polynomial_network_regression(L_list, X_list_poly, x_poly[i])
        eigenvalues, eigenvectors = np.linalg.eig(pred)
        non_zero = []
        # to make sure we only exclude one zero from the eigenvalues
        flag = True
        for eval in eigenvalues:
            if abs(eval) < 1e-15 and flag:
                flag = False
            else:
                non_zero.append(eval)

        # non_zero.sort()
        if 5.77e-01 < abs(eigenvectors[0, 0]) < 5.78e-01:
            # column corresponding to zero goes first
            lambda_1s.append(non_zero[0])
            lambda_2s.append(non_zero[1])
        elif 4.08e-01 < abs(eigenvectors[0, 0]) < 4.09e-01:
            # column corresponding to y goes first
            lambda_2s.append(non_zero[0])
            lambda_1s.append(non_zero[1])
        elif 7.07e-01 < abs(eigenvectors[0, 0])< 7.08e-01:
            # column corresponding to x goes first
            lambda_1s.append(non_zero[0])
            lambda_2s.append(non_zero[1])
        elif eigenvectors[0, 0] == 1 and eigenvectors[1, 1] == 1 and eigenvectors[2, 2] == 1:
            lambda_1s.append(0)
            lambda_2s.append(0)
        else:
            print('error!!')
    ax.scatter(lambda_1s, lambda_2s, label='degree '+str(degree), alpha = 0.3)
    
    # establish the connections between the true values and the predictions
    for idx in range(len(x_values)):
        plt.plot([lambda_1s[idx], x_vals[idx]], [lambda_2s[idx], y_values[idx]], 'r-')
    
    plt.xlabel('lambda1')
    plt.ylabel('lambda2')
    plt.title('A plot of non-zero eigenvalues of global and quadratic regression results')
    plt.legend()
    plt.grid(True)
    plt.show()

# # the eigenvector matrix we use (basis) is
U = np.matrix([
    [-math.sqrt(2)/2, -math.sqrt(6)/6, math.sqrt(3)/3],
    [math.sqrt(2)/2, -math.sqrt(6)/6, math.sqrt(3)/3],
    [0, math.sqrt(6)/3, math.sqrt(3)/3]
])

# generate the x's, and construct the graphs using reverse SVD
X_list = [x/10 for x in range(0, 11)]
L_list = [U @ np.matrix(np.diag([x/10, math.sqrt(1 - (x/10)**2), 0])) @ U.T for x in range(0, 11)]
# for gl in L_list:
#     if gl[0, 1] >= 0:
#         print(gl)
# find eigenvalues for the outputs and plot the two non-zero eigvals
x_values = [x/50 for x in range(0, 51)]
# global_poly_eigenvalue_plot(L_list, X_list, x_values, 2)