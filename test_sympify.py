from sympy import symbols, Function, Eq, Derivative, sympify, sin, cos, exp, log, Abs

# Define the independent variable
t = symbols('t')

# Define y as a function of t
y = Function('y')(t)

# Define different right-hand side expressions as strings
equations = {
    'Linear Cauchy Function': 't + 2*y',
    'Exponential Cauchy Function': 'exp(t) + y',
    'Trigonometric Cauchy Function': 'sin(t) + cos(y)',
    'Polynomial Cauchy Function': 't**2 + y**2',
    'Logarithmic Cauchy Function': 'log(t + 1) + y',
    'Rational Cauchy Function': 't / (1 + y)',
    'Mixed Function': 't * exp(-y) + cos(t)',
    'Absolute Value Function': 'Abs(t - y) + t'
}

# Sympify the right-hand sides and construct ODEs
for name, expr_str in equations.items():
    rhs_expr = sympify(expr_str, locals={'t': t, 'y': y})  # Sympify the RHS with y as a function of t
    ode_equation = Eq(Derivative(y, t), rhs_expr)  # Create the ODE
    print(f"{name}: {ode_equation}")
