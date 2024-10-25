import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
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

    method_results = np.zeros((10, int((b-a)/step)))

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
        true_y_t = float(true_y.subs(t, t_value))

        if i == 0:
            pred_y_t = pred_taylor_y_t = pred_runge_kutta_2_y_t = pred_runge_kutta_4_y_t = y_neg_10
        else:
            pred_y_t = method_results[1, i-1] + step * cauchy_function.subs({y: method_results[1, i-1], t: t_value})
            
            pred_taylor_y_t = (
                method_results[2, i-1] + step * cauchy_function.subs({y: method_results[2, i-1], t: t_value})
                + (step**2 / 2) * (df_dt.subs({y: method_results[2, i-1], t: t_value})
                + df_dy.subs({y: method_results[2, i-1], t: t_value}) * cauchy_function.subs({y: method_results[2, i-1], t: t_value}))
            )
            
            y_modified = pred_y_t
            pred_runge_kutta_2_y_t = (
                method_results[3, i-1] + step * 0.5 * (cauchy_function.subs({y: method_results[3, i-1], t: t_value})
                + cauchy_function.subs({y: y_modified, t: t_value}))
            )

            k_1 = step * cauchy_function.subs({y: method_results[3, i-1], t: t_value})
            k_2 = step * cauchy_function.subs({y: method_results[3, i-1] + 0.5*k_1, t: t_value + 0.5*step})
            k_3 = step * cauchy_function.subs({y: method_results[3, i-1] + 0.5*k_2, t: t_value + 0.5*step})
            k_4 = step * cauchy_function.subs({y: method_results[3, i-1] + k_3, t: t_value + step})
            pred_runge_kutta_4_y_t = method_results[3, i-1] + (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)

        method_results[:, i] = [
            t_value, true_y_t, pred_y_t, pred_taylor_y_t, pred_runge_kutta_2_y_t, pred_runge_kutta_4_y_t,
            abs(true_y_t - pred_y_t), abs(true_y_t - pred_taylor_y_t),
            abs(true_y_t - pred_runge_kutta_2_y_t), abs(true_y_t - pred_runge_kutta_4_y_t)
        ]

        t_value += step

    df = pd.DataFrame(method_results.T, columns=['t', 'y_true', 'y_approx_euler', 'y_approx_taylor', 'y_approx_rk2', 'y_approx_rk4', 'error_euler', 'error_taylor', 'error_rk2', 'error_rk4'])

    # save dataframe to csv file
    df.to_csv('euler_method_results.csv', index=False)

    # plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(df['t'][:20], df['y_true'][:20], label='true', color='red')
    plt.plot(df['t'][:20], df['y_approx_euler'][:20], label='approximate euler', color='blue')
    plt.plot(df['t'][:20], df['y_approx_taylor'][:20], label='approximate taylor', color='green')
    plt.plot(df['t'][:20], df['y_approx_rk2'][:20], label='approximate RK2', color='purple')
    plt.plot(df['t'][:20], df['y_approx_rk4'][:20], label='approximate RK4', color='orange')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title('Approximation Methods Comparison')
    plt.show()

    # plot the errors
    plt.figure(figsize=(12, 8))
    plt.plot(df['t'][:20], df['error_euler'][:20], label='error euler', color='blue')
    plt.plot(df['t'][:20], df['error_taylor'][:20], label='error taylor', color='green')
    plt.plot(df['t'][:20], df['error_rk2'][:20], label='error RK2', color='purple')
    plt.plot(df['t'][:20], df['error_rk4'][:20], label='error RK4', color='orange')
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Error Comparison')
    plt.show()

    gc.enable()


if __name__ == "__main__":
    main()













