import numpy as np
import math

def derivative(f, x, epsilon=1e-6):
    return(f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

def second_derivative(f, x, epsilon=1e-6): 
    return (f(x + epsilon) - 2*f(x) + f(x - epsilon))/ (epsilon **2)

def newtons_method(f, x0, epsilon=1e-6, max_iter=100):
        x = x0
        for _ in range(max_iter):
            f_prime = derivative(f, x, epsilon)
            f_double_prime = second_derivative(f, x, epsilon)

            x_new= x - f_prime / f_double_prime

            if abs(x_new - x) < epsilon:
                return x_new

            x = x_new
        return x

fx = lambda x: math.sin(x) * 2 + (x -1) ** 2 + x 

print(newtons_method(fx, 0, epsilon=1e-6))
