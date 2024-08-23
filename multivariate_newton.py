import numpy as np
from scipy.optimize import approx_fprime

def numerical_hessian(f, x, epsilon=1e-5):
    n = x.shape[0]
    hess = np.zeros((n, n))
    grad = approx_fprime(x, f, epsilon)
    
    for i in range(n):
        x1 = np.array(x, dtype=float)
        x1[i] += epsilon
        grad1 = approx_fprime(x1, f, epsilon)
        hess[:, i] = (grad1 - grad) / epsilon
    
    return hess

def multivariate_newtons_method_simple(f, x0, tol=1e-6, max_iter=100):
    x = x0
    epsilon = np.sqrt(np.finfo(float).eps)

    for _ in range(max_iter):
        grad = approx_fprime(x, f, epsilon) 
        hess = numerical_hessian(f,x)

        hessian_inv = np.linalg.inv(hess)
        delta_x = -hessian_inv @ grad

        x_new = x + delta_x

        if np.linalg.norm(delta_x) < tol:
            break

        x = x_new

    return x

def example_f(x):
    # Example function: f(x) = x1^2 + x2^2
    return x[0]**2 + x[1]**2

x0 = np.array([10.0, -10.0])  # Initial guess
minimum = multivariate_newtons_method_simple(example_f, x0)

print("Found minimum:", minimum)
