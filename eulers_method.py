import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pandas as pd


class CauchyProblem:
    def __init__(self, f, initial_conditions):
        t = sp.Symbol('t')
        y = sp.Function("y")(t)
        self.f = sp.sympify(f, locals={'t': t, 'y': y})
        print(sp.sympify(f, locals={'t': t, 'y': y}))
        print(type(sp.sympify(f, locals={'t': t, 'y': y})))
        self.initial_conditions = initial_conditions
        self.solution = None
        print(f"Initialized CauchyProblem with function: {self.f} and initial conditions: {self.initial_conditions}")

    def solve(self):
        t = sp.Symbol('t')
        y = sp.Function("y")(t)
        ode_equation = sp.Eq(self.f, sp.Derivative(y, t))
        t_initial = float(list(self.initial_conditions.keys())[0])
        y_initial_value = self.initial_conditions[str(t_initial)]
        self.solution = sp.dsolve(ode_equation, y, ics={y.subs(t, t_initial): y_initial_value})
        print(f"Solved ODE: {ode_equation}, Solution: {self.solution}")
        return self.solution.rhs


class NumericalMethods:
    def __init__(self, f, initial_conditions, step, interval, true_solution=None):
        if true_solution is None:
            self.true_solution = CauchyProblem(f, initial_conditions).solve()
        else:
            exact_solution = CauchyProblem(f, initial_conditions).solve()
            if exact_solution != true_solution:
                raise ValueError(
                    'The true solution does not match the computed solution for the cauchy problem you gave\n'
                    'Please make sure to use the CauchyProblem class to solve the problem and then implement'
                    'the result of .solve() method in the CauchyProblem class')
            else:
                self.true_solution = true_solution

        if float(list(initial_conditions.keys())[0]) != float(interval[0]):
            raise ValueError("the initial value should be equal to left of the interval")
        else:
            self.initial_value = float(list(initial_conditions.values())[0])
            self.interval = interval

        self.step = step
        self.results = None
        self.f = sp.sympify(f)
        print(f"Initialized NumericalMethods with function: {self.f}, step: {self.step}, interval: {self.interval}")

    def apply_methods(self):
        a, b = self.interval
        method_results = np.zeros((10, int((b - a) / self.step)))
        t = sp.Symbol('t')
        y = sp.Symbol('y')
        cauchy_function = self.f
        t_value = a
        df_dt = sp.Derivative(cauchy_function, t).doit()
        df_dy = sp.Derivative(cauchy_function, y).doit()

        for i in tqdm(range(int((b - a) / self.step)), desc="Applying numerical methods", ncols=100):
            true_y_t = float(self.true_solution.subs(t, t_value))

            if i == 0:
                pred_y_t = pred_taylor_y_t = pred_runge_kutta_2_y_t = pred_runge_kutta_4_y_t = self.initial_value
            else:
                pred_y_t = method_results[1, i - 1] + self.step * cauchy_function.subs(
                    {y: method_results[1, i - 1], sp.Symbol('t'): t_value})

                pred_taylor_y_t = (
                        method_results[2, i - 1] + self.step * cauchy_function.subs(
                        {y: method_results[2, i - 1], sp.Symbol('t'): t_value})
                        + (self.step ** 2 / 2) * (df_dt.subs({y: method_results[2, i - 1], sp.Symbol('t'): t_value})
                                                  + df_dy.subs(
                            {y: method_results[2, i - 1], sp.Symbol('t'): t_value}) * cauchy_function.subs(
                            {y: method_results[2, i - 1], sp.Symbol('t'): t_value}))
                )

                y_modified = pred_y_t
                pred_runge_kutta_2_y_t = (
                        method_results[3, i - 1] + self.step * 0.5 * (
                            cauchy_function.subs({y: method_results[3, i - 1], sp.Symbol('t'): t_value})
                            + cauchy_function.subs({y: y_modified, sp.Symbol('t'): t_value}))
                )

                k_1 = self.step * cauchy_function.subs({y: method_results[3, i - 1], sp.Symbol('t'): t_value})
                k_2 = self.step * cauchy_function.subs(
                    {y: method_results[3, i - 1] + 0.5 * k_1, sp.Symbol('t'): t_value + 0.5 * self.step})
                k_3 = self.step * cauchy_function.subs(
                    {y: method_results[3, i - 1] + 0.5 * k_2, sp.Symbol('t'): t_value + 0.5 * self.step})
                k_4 = self.step * cauchy_function.subs(
                    {y: method_results[3, i - 1] + k_3, sp.Symbol('t'): t_value + self.step})
                pred_runge_kutta_4_y_t = method_results[3, i - 1] + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

            method_results[:, i] = [
                t_value, true_y_t, pred_y_t, pred_taylor_y_t, pred_runge_kutta_2_y_t, pred_runge_kutta_4_y_t,
                abs(true_y_t - pred_y_t) / (abs(true_y_t) + 10**-6), abs(true_y_t - pred_taylor_y_t) / (abs(true_y_t) + 10**-6),
                abs(true_y_t - pred_runge_kutta_2_y_t) / (abs(true_y_t) + 10**-6),
                abs(true_y_t - pred_runge_kutta_4_y_t) / (abs(true_y_t) + 10**-6)
            ]

            t_value += self.step

        self.results = method_results.T
        print("Completed applying numerical methods.")
        return self.results

    def save_results(self, filename):
        df = pd.DataFrame(self.results, columns=['t', 'y_true', 'y_approx_euler', 'y_approx_taylor', 'y_approx_rk2',
                                                 'y_approx_rk4', 'rel_error_euler', 'rel_error_taylor',
                                                 'rel_error_rk2', 'rel_error_rk4'])
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def plot_results(self):
        df = pd.DataFrame(self.results, columns=['t', 'y_true', 'y_approx_euler', 'y_approx_taylor', 'y_approx_rk2',
                                                 'y_approx_rk4', 'rel_error_euler', 'rel_error_taylor',
                                                 'rel_error_rk2', 'rel_error_rk4'])

        plt.figure(figsize=(12, 8))
        plt.plot(df['t'][:], df['y_true'][:], label='true', color='red')
        plt.plot(df['t'][:], df['y_approx_euler'][:], label='approximate euler', color='blue')
        plt.plot(df['t'][:], df['y_approx_taylor'][:], label='approximate taylor', color='green')
        plt.plot(df['t'][:], df['y_approx_rk2'][:], label='approximate RK2', color='purple')
        plt.plot(df['t'][:], df['y_approx_rk4'][:], label='approximate RK4', color='orange')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.legend()
        plt.title('Approximation Methods Comparison')
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(df['t'][:100], df['rel_error_euler'][:100], label='relative error euler', color='blue')
        plt.plot(df['t'][:100], df['rel_error_taylor'][:100], label='relative error taylor', color='green')
        plt.plot(df['t'][:100], df['rel_error_rk2'][:100], label='relative error RK2', color='purple')
        plt.plot(df['t'][:100], df['rel_error_rk4'][:100], label='relative error RK4', color='orange')
        plt.xlabel('t')
        plt.ylabel('Relative Error')
        plt.legend()
        plt.title('Error Comparison')
        plt.show()


def main():
    gc.disable()

    """
    Example First-Order ODEs
Linear Cauchy Function

Equation: y' = t + 2y
Description: The first derivative of y with respect to t is equal to t plus twice y.
Exponential Cauchy Function

Equation: y' = exp(t) + y
Description: The first derivative of y is equal to the exponential function of t plus y.
Trigonometric Cauchy Function

Equation: y' = sin(t) + cos(y)
Description: The first derivative of y is equal to the sine of t plus the cosine of y.
Polynomial Cauchy Function

Equation: y' = t2 + y2
Description: The first derivative of y is equal to the square of t plus the square of y.
Logarithmic Cauchy Function

Equation: y' = log(t + 1) + y
Description: The first derivative of y is equal to the natural logarithm of t + 1 plus y.
Rational Cauchy Function

Equation: y' = t / (1 + y)
Description: The first derivative of y is equal to t divided by (1 + y).
Mixed Function

Equation: y' = t * exp(-y) + cos(t)
Description: The first derivative of y is equal to t multiplied by the exponential of -y plus the cosine of t.
Absolute Value Function

Equation: y' = |t - y| + t
Description: The first derivative of y is equal to the absolute value of t - y plus t.
Operators Used
y': Represents the first derivative of y with respect to t.
+: Addition operator, used to sum two expressions.
-: Subtraction operator, used to find the difference between two expressions.
*: Multiplication operator, used to multiply two expressions.
/: Division operator, used to divide one expression by another.
(): Parentheses are used to group expressions and dictate the order of operations.
exp(): The base of the natural logarithm, approximately equal to 2.71828 if it had 1 as argument
log(): Natural logarithm function, which gives the logarithm to the base e.
| |: Absolute value, which represents the distance of a number from zero, disregarding its sign.
t: Independent variable, often representing time or another parameter in ODEs.
y: Dependent variable, whose behavior we are studying in relation to t.
    """

    # User input for initial conditions and interval
    cauchy_function = input("Enter the cauchy function f expression: ")
    t_initial = float(input("Enter the initial time (t0): "))
    y_initial_value = float(input("Enter the initial value of y at t0: "))
    a = int(t_initial)
    b = int(input("Enter the end time (b): "))
    step = float(input("Enter the step size: "))

    # Define the Cauchy Problem
    f = cauchy_function

    y_initial = {str(t_initial): y_initial_value}
    cauchy_problem = CauchyProblem(f, y_initial)
    true_y = cauchy_problem.solve()

    numerical_methods = NumericalMethods(f=f, initial_conditions=y_initial,
                                         step=step, interval=(a, b), true_solution=true_y)
    numerical_methods.apply_methods()
    numerical_methods.save_results('euler_method_results.csv')
    numerical_methods.plot_results()

    gc.enable()


if __name__ == "__main__":
    main()
