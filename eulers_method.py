import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import gc
import pandas as pd


def main():
    gc.disable()

    # Define variables
    t = sp.Symbol('t')
    f = sp.Function('f')

    y = sp.Function('y')(t)

    # Define the Cauchy Problem
    # The null ode means that its expression is one side of the ode while the other side is zer
    null_ode = sp.Derivative(y) - t*y

    # Initial Condition
    y_neg_10 = 1

    ode_equation = sp.Eq(null_ode, 0)

    # Solve the ode
    sol = sp.dsolve(ode_equation, y, ics={y.subs(t, -10): y_neg_10})
    print(sol)

    """ Getting the solution by using rhs which stands for right hand side,
     because sol is an equation with y(t) on the left side and its expression in the right side"""
    true_y = sol.rhs

    # Solving Euler's Method

    # The step to take
    step = 0.01

    # If we consider [a,b] to be the interval to work with
    a, b = -10, 10

    method_results = np.zeros((6, int((b-a)/step)))

    # Define y as a symbol which means a variable
    y = sp.Symbol('y')
    cauchy_function = t*y
    print(f"cauchy_function: {cauchy_function}")

    # Initial t_value
    t_value = a

    # Define the derivatives outside the loop for efficiency
    df_dt = sp.Derivative(cauchy_function, t).doit()  # Partial derivative of f w.r.t t
    print(df_dt)
    df_dy = sp.Derivative(cauchy_function, y).doit()  # Partial derivative of f w.r.t y
    print(df_dy)

    # Applying the euler's method
    for i in tqdm(range(int((b-a)/step)), desc="Applying Euler's method", ncols=100):
        true_y_t = true_y.subs(t, t_value)

        if i == 0:
            pred_y_t = y_neg_10
            pred_taylor_y_t = y_neg_10  # Taylor approximation is the same as the exact value for the first step
        else:
            pred_y_t = method_results[1][i-1] + step * cauchy_function.subs({y: method_results[1][i-1], t: t_value})

            # Inside the loop, update pred_taylor_y_t
            pred_taylor_y_t = (
                    method_results[2][i-1] + step * cauchy_function.subs({y: method_results[2][i-1], t: t_value})
                    + (step**2 / 2) * df_dt.subs({y: method_results[2][i-1], t: t_value})
                    + (step**2 / 2) * df_dy.subs({y: method_results[2][i-1], t: t_value}) * cauchy_function.subs({y: method_results[2][i-1], t: t_value})
                                )
        method_results[0][i] = t_value
        method_results[1][i] = true_y_t
        method_results[2][i] = pred_y_t
        method_results[3][i] = pred_taylor_y_t
        method_results[4][i] = abs(true_y_t - pred_y_t)
        method_results[5][i] = abs(true_y_t - pred_taylor_y_t)

        t_value += step

    df = pd.DataFrame(method_results.T, columns=['t', 'y_true', 'y_approx_euler', 'y_approx_taylor', 'error_euler', 'error_taylor'])

    # save dataframe to csv file
    df.to_csv('euler_method_results.csv', index=False)


    # plot the results
    plt.plot(df['t'][:1000], df['y_approx_euler'][:1000], label='approximate euler', color='blue')
    plt.plot(df['t'][:1000], df['y_approx_taylor'][:1000], label='approximate taylor', color='green')
    plt.plot(df['t'][:1000], df['y_true'][:1000], label='true', color='red')
    plt.plot(df['t'][:1000], df['error_euler'][:1000], label='error euler', color='yellow')
    plt.plot(df['t'][:1000], df['error_taylor'][:1000], label='error taylor', color='red')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    gc.enable()


if __name__ == "__main__":
    main()













