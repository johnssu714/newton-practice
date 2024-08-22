import math
import warnings

def derivative(f, x, epsilon=1e-6):
    """Calculate the first derivative of f at x using a finite difference."""
    return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

def second_derivative(f, x, epsilon=1e-6):
    """Calculate the second derivative of f at x using a finite difference."""
    return (f(x + epsilon) - 2 * f(x) + f(x - epsilon)) / (epsilon**2)

def newtons_method(f, x0, epsilon=1e-6, max_iter=100):
    """Implement Newton's method to find the root of f starting from x0."""
    
    # Input validation
    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")
    
    if not isinstance(x0, (int, float)):
        raise TypeError(f"Initial guess x0 must be a number, but got {type(x0)}")
    
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError(f"epsilon must be a positive number, but got {epsilon}")
    
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer, but got {max_iter}")

    x = x0
    for iter in range(max_iter):
        f_prime = derivative(f, x, epsilon)
        f_double_prime = second_derivative(f, x, epsilon)

        if abs(f_double_prime) < epsilon:
            warnings.warn(f"Second derivative near zero at x = {x}. Results may be inaccurate.")
            return x, False  # Return a failure flag

        x_new = x - f_prime / f_double_prime

        # Warn if the new step is significantly worse
        if abs(x_new - x) > 1e6:  # Arbitrary large number for demonstration
            warnings.warn(f"Large step detected at iteration {iter}: x_new = {x_new} from x = {x}")

        if abs(x_new - x) < epsilon:
            return x_new, True  # Return success

        x = x_new
    
    # If the method didn't converge
    warnings.warn("Newton's method did not converge within the maximum number of iterations.")
    return x, False  # Return a failure flag

