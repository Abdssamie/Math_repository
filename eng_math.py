import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import re


def rearrange_ode(ode):
    # Pattern to find y' and everything else
    pattern = r"^(.*?)(y')\s*([+\-].*)$"  # Matches y' and captures the rest
    match = re.match(pattern, str(ode))

    if match:
        lhs = match.group(2)  # This is y'
        rhs = match.group(3).strip()  # This is the RHS part
        return f"{lhs} = {rhs}"
    else:
        return ode  # Return unchanged if no match


class NumericalMethod:
    """ A base class for numerical methods to solve Ordinary Differential Equations (ODEs).

        This class provides the foundation for implementing various numerical methods
        for solving ODEs. It includes functionality for parsing and solving ODEs using
        SymPy's symbolic mathematics capabilities.

        Attributes:
        base_method (str): The name of the base numerical method.
        """
    base_method = "Euler's method explicit"

    def __init__(self, ode, initial_parameter, cauchy_function, a, b, step):
        """
        Initialize the NumericalMethod instance.

        Args:
            ode (str): The Ordinary Differential Equation (ODE) in string format.
            cauchy_function (function): The function that represents the right-hand side of the ODE.
            a (float): The lower limit of the interval.
            b (float): The upper limit of the interval.
            step (float): The step size for the numerical method.
            initial_parameter (dict): The initial conditions for the ODE.

        Note:
            The ODE should be in the format: a(t) + b(t)*y' + c(t)*y" + ... = 0,
            where a(t), b(t), c(t), etc., are functions of t.
        """
        self.a = a
        self.b = b
        self.step = step
        self.initial_parameter = initial_parameter
        self.cauchy_function = cauchy_function
        """
        The class expects the the ode to be in the following format so we can better handle
         it to turn into a sympy equation using sp.Eq()
         for example: a(t) + b(t)*y' + c(t)*y" + y**2 + sqrt(y) + sin(y) + cos(y) + 455 = 0 where a(t), b(t) and c(t)
         could be any type of functions that depend on t
        """
        self.ode = self.parse_ode(ode)
        self.lhs_expr = self.parse_ode(ode).lhs
        self.rhs_expr = self.parse_ode(ode).rhs

    def parse_ode(self, ode=None):
        """
        Parse the ODE string into SymPy expressions.

        This method converts the ODE string into SymPy expressions for the left-hand
        side (LHS) and right-hand side (RHS) of the equation.

        Args:
            ode (str, optional): The ODE to parse. If None, uses the instance's ode.

        Returns:
            sympy.Eq: A SymPy equation representing the parsed ODE.

        Raises:
            ValueError: If the ODE is not in the correct format.
        """
        try:
            if ode is None:
                # Remove whitespace and split the input at '='
                self.ode = sp.sympify(str(rearrange_ode(self.ode)).replace(" ", ""))
                print(self.ode)
                lhs, rhs = str(self.ode).split("=")
            else:
                ode = rearrange_ode(ode).replace(" ", "")
                lhs, rhs = ode.split("=")

        except ValueError:
            raise ValueError("The ODE must be in the form of 'left_side = right_side'")

        # Set up the variables
        t = sp.symbols('t')
        y = sp.Function('y')(t)

        # Define derivatives using SymPy
        y_diff = y.diff(t)  # y'
        y_diff2 = y.diff(t, 2)  # y"

        # Prepare a dictionary to store derivatives
        derivatives = {}
        max_order = 10  # Set this to the maximum order you expect

        # Generate derivative symbols and replacements
        for i in range(max_order):
            if i == 0:
                derivatives[f"y\'"] = y  # y
            elif i == 1:
                derivatives[f'y'''] = y_diff  # y'
                derivatives[f"y\""] = y_diff2  # y"
            else:
                derivatives[f'y({i + 1})'] = y.diff(t, i)  # y', y'', etc.

        # Replace all derivatives in the left-hand side expression
        for i in range(max_order + 1):
            if derivatives.get(f'y({i}') is not None:
                lhs = lhs.replace(f'y({i})', str(derivatives[f'y({i})']))
            else:
                continue

        lhs = lhs.replace("y'", str(derivatives[f"y\'"])).replace("y''", str(derivatives[f"y\""]))

        # Convert the left-hand side to a SymPy expression
        self.lhs_expr = sp.sympify(lhs)

        # Convert the right-hand side (should be 0)
        self.rhs_expr = sp.sympify(rhs)

        # Construct the equation
        equation = sp.Eq(self.lhs_expr, self.rhs_expr)

        return equation

    def solve_ode(self):
        """
       Solve the ODE using SymPy's dsolve function.

       This method prepares the initial conditions and solves the ODE using
       SymPy's symbolic solver.

       Returns:
           sympy.Eq: The symbolic solution of the ODE.

       Raises:
           ValueError: If there's an invalid initial parameter format.
           KeyError: If a required initial parameter is missing.
           Exception: For any other error in processing initial parameters.
       """
        # Solve the ODE using SymPy's dsolve function
        t = sp.Symbol('t')
        y = sp.Function('y')(t)
        # Prepare initial conditions
        ics = {}
        try:
            for order, init_param in self.initial_parameter.items():
                if order == 'y':
                    t_0, y_0 = next(iter(init_param.items()))
                    ics[y.subs(t, t_0)] = y_0
                elif order == 'y\'':
                    t_diff1_0, y_diff1_0 = next(iter(init_param.items()))
                    ics[y.diff(t).subs(t, t_diff1_0)] = y_diff1_0
                else:
                    t_diff_i_0, y_diff_i_0 = next(iter(init_param.items()))

                    if "'" in order:
                        derivative_order = order.count("'")
                    elif '"' in order:
                        derivative_order = 2
                    else:
                        # Extract order from parentheses
                        derivative_order = int(order[order.index('(') + 1: order.index(')')])

                    ics[y.diff(t, derivative_order).subs(t, t_diff_i_0)] = y_diff_i_0

        except ValueError as e:
            raise ValueError(f"Invalid initial parameter format: {e}")
        except KeyError as e:
            raise KeyError(f"Missing required initial parameter: {e}")
        except Exception as e:
            raise Exception(f"Error processing initial parameters: {e}")

        # Solve the ODE with initial conditions
        solution = sp.dsolve(self.ode, y, ics=ics)

        # Return the solution
        return solution


class EulerMethod(NumericalMethod):
    """
    A class implementing Euler's method for solving first-order ordinary differential equations (ODEs).

    This class extends the NumericalMethod class and provides specific implementation
    for Euler's method to solve ODEs numerically.

    Attributes:
        t (sympy.Symbol): Symbol representing the independent variable (typically time).
        y (sympy.Function): Function representing the dependent variable.
    """

    t = sp.Symbol('t')
    y = sp.Function('y')(t)

    def __init__(self, ode, initial_parameter, cauchy_function, a, b, step):
        """
        Initialize the EulerMethod instance.

        Args:
            ode (str): The ordinary differential equation in string format.
            initial_parameter (dict): Initial conditions for the ODE.
            cauchy_function (function): The function representing the right-hand side of the ODE.
            a (float): The lower bound of the interval.
            b (float): The upper bound of the interval.
            step (float): The step size for the Euler method.
        """
        super().__init__(ode, initial_parameter, cauchy_function, a, b, step)

        # Assign lhs_expr and rhs_expr after initializing the parent class
        self.lhs_expr = self.parse_ode().lhs
        self.rhs_expr = self.parse_ode().rhs

        self.cauchy_function = self.rhs_expr
        import re
        # Find the index of the y term in the left-hand side
        self.y_index = re.search(r'y', str(self.lhs_expr)).start()

        self.f = - self.lhs_expr + self.lhs_expr.ind

    def exp_method_results(self):
        """
        Apply Euler's method to solve the ODE and compute results.

        This method is only valid for first-order ODEs. It applies Euler's method
        to numerically solve the ODE and computes the approximate solution,
        true solution, and error at each step.

        Returns:
            numpy.ndarray or None: A 2D numpy array containing the results of Euler's method.
                The array has 4 rows:
                - Row 0: t values
                - Row 1: Approximate y values (predicted by Euler's method)
                - Row 2: True y values (if available)
                - Row 3: Absolute error between true and approximate values
                Returns None if an error occurs during computation.
        """
        method_results = np.zeros((4, int((self.b-self.a)/self.step)))
        try:
            t = int(self.initial_parameter['y'].keys())
            y_0 = int(self.initial_parameter['y'].values())
            for i in range(int((self.b-self.a)/self.step)):
                if i % int((self.b-self.a)/self.step)/10 == 0:
                    print("||")
                    print(f"Iteration {i+1}/"+str(int((self.b-self.a)/self.step)))
                    print("||", end="")
                elif i % int((self.b-self.a)/self.step)/300 == 0:
                    print("_", end="")

                true_y_t = self.lhs_expr.subs(t, t)

                if i == 0:
                    pred_y_t = y_0
                else:
                    pred_y_t = (method_results[1][i-1] +
                                self.step * self.cauchy_function.subs(self.y, method_results[1][i-1]))

                method_results[0][i] = t
                method_results[1][i] = pred_y_t
                method_results[2][i] = true_y_t
                method_results[3][i] = abs(true_y_t - pred_y_t)

                t += self.step
        except ValueError:
            print("The ODE could not be solved. t"
                  "Please make sure the a and b define the limits of the interval at which this function is defined at:"
                  f"{self.lhs_expr}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
