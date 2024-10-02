# import numpy as np
# import cvxpy as cp

# L = cp.Variable((3, 1))

# # the set of linear constraints
# constraints = [sum(L[:, 0]) == 0.5]  # sum of weights constant
# constraints += [L[i, 0] >= 0 for i in range(2)]  # weights nonneg

# # the objective function
# objective = cp.Minimize(max([min(L[0, 0], L[1, 0] + L[2, 0]), min(L[1, 0], L[0, 0] + L[2, 0]), min(L[2, 0], L[1, 0] + L[0, 0])]))
# prob = cp.Problem(objective, constraints)
# prob.solve()
# sol = L.value

# print(sol)

import numpy as np
from scipy.optimize import minimize

def objective(weights):
    # Objective is to maximize the diameter, which is the maximum of the weights
    w_1 = weights[0]
    w_2 = weights[1]
    w_3 = weights[2]
    return -max(min(w_1, w_2 + w_3), min(w_2, w_1 + w_3), min(w_3, w_2 + w_1))  # Minimize the negative to maximize the positive

def constraint_sum(weights, S):
    # Sum of the weights should be S
    return np.sum(weights) - S

def optimize_weights(S):
    # Initial guess for the weights
    initial_guess = np.array([S/3, S/3, S/3])
    
    # Define bounds for each weight (nonnegative)
    bounds = [(0, S), (0, S), (0, S)]
    
    # Define the constraint
    constraints = {'type': 'eq', 'fun': lambda w: constraint_sum(w, S)}
    
    # Perform the optimization
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    
    if result.success:
        optimized_weights = result.x
        max_diameter = -result.fun
        return optimized_weights, max_diameter
    else:
        raise ValueError("Optimization failed")

# Example usage
S = 0.5  # Example sum of the weights
weights, max_diameter = optimize_weights(S)
print(f"Optimized weights: {weights}")
print(f"Maximum diameter: {max_diameter}")
