import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define variables
t = sp.Symbol('t')
f = sp.Function('f')

y = sp.Function('y')(t)

# Define the Cauchy Problem
# The null ode means that its expression is one side of the ode while the other side is zer
null_ode = sp.Derivative(y) + 10 * y**2

# Initial Condition
y_neg_100 = 1

# Solve the ode
sol = sp.dsolve(null_ode, y, ics={y.subs(t, -100): y_neg_100})
print(sol)

""" Getting the solution by using rhs which stands for right hand side,
 because sol is an equation with y(t) on the left side and its expression in the right side"""
true_y = sol.rhs

# Solving Euler's Method

# The step to take
step = 0.01

# If we consider [a,b] to be the interval to work with
a, b = -100, 100

""" Setting up the results table inside a numpy array for optimization
 the first row for the t values are a subdivision of [-1000, 1000] with 0.01 spacing
 second row for the approximate y values and third row for the errors
"""
method_results = np.zeros((4, int((b-a)/step)))
print(method_results.shape)

# Define y as a symbol which means a variable
y = sp.Symbol('y')
cauchy_function = - 10 * y**2
print(f"cauchy_function: {cauchy_function}")

# Initial t_value
t_value = a

# Applying the euler's method
for i in range(int((b-a)/step)):
    print(f"Iteration {i+1}/"+str(int((b-a)/step)))
    true_y_t = true_y.subs(t, t_value)

    if i == 0:
        pred_y_t = y_neg_100
    else:
        pred_y_t = method_results[1][i-1] + step * cauchy_function.subs(y, method_results[1][i-1])

    method_results[0][i] = t_value
    method_results[1][i] = pred_y_t
    method_results[2][i] = true_y_t
    method_results[3][i] = abs(true_y_t - pred_y_t)

    t_value += step

# TODO: Fix this code tommorow inchallah

# display the method_results matrix using matplot lib and of course add labels so we understand what those values are
import pandas as pd

df = pd.DataFrame(method_results.T, columns=['t', 'y_approx', 'y_true', 'error'])

# save dataframe to csv file
df.to_csv('euler_method_results.csv', index=False)


# plot the results
plt.plot(df['t'][:100], df['y_approx'][:100], label='approximate', color='red')
plt.plot(df['t'][:100], df['y_true'][:100], label='true', color='blue')
plt.plot(df['t'][:100], df['error'][:100], label='error', color='green')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()

# TODO: Implement a convergence rate analysis for Euler's method and plot it.