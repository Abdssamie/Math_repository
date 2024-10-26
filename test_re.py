import re

# Define a function to rearrange the ODE
def rearrange_ode(ode):
    # Pattern to find y' and everything else
    pattern = r"^(.*?)(y')\s*([+\-].*)$"  # Matches y' and captures the rest
    match = re.match(pattern, ode)

    if match:
        lhs = match.group(2)  # This is y'
        rhs = match.group(3).strip()  # This is the RHS part
        return f"{lhs} = {rhs}"
    else:
        return ode  # Return unchanged if no match

# Example ODE strings
ode_strings = [
    "y' + 2*y + 3 = 0",
    "5*y' - 4 = 0",
    "y' * f(t) + y = 1",
    "x * y' + sin(x) = 0",
    "y' + 3y' + 4 = 0",
    "c * y' + y'' = 0",
    "5 * g(t) + y' = 7",
    "c*y' + d*y'' + e*y = 0"
]

# Rearranging each ODE
for ode in ode_strings:
    rearranged_ode = rearrange_ode(ode)
    print(f"Original: {ode} --> Rearranged: {rearranged_ode}")
