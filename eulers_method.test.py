import unittest
from eulers_method import EulerMethod, NumericalMethod
import math

class TestEulerMethod(unittest.TestCase):
    def setUp(self):
        self.ode = "y' = y"
        self.initial_parameter = {'y': {0: 1}}
        self.f = lambda t, y: y
        self.a = 0
        self.b = 1
        self.step = 0.1
        
    def test_large_number_of_steps(self):
        ode = "y' = y"
        initial_parameter = {'y': {0: 1}}
        cauchy_function = lambda t, y: y
        a = 0
        b = 1
        step = 0.0001  # Very small step size
    
        euler = EulerMethod(ode, initial_parameter, cauchy_function, a, b, step)
        results = euler.exp_method_results()
    
        self.assertIsNotNone(results)
        self.assertEqual(results.shape[1], int((b-a)/step))
        self.assertAlmostEqual(results[1][-1], 2.718281828, places=3)  # Check if final value is close to e
        
    def test_incorrect_ode_format(self):
        with self.assertRaises(ValueError):
            NumericalMethod("y' + y", {'y': {0: 1}}, lambda t, y: y, 0, 1, 0.1)
        
    def test_stiff_ode(self):
        # Define a stiff ODE: y' = -1000y + 3000 - 2000e^(-t)
        ode = "y' = -1000y + 3000 - 2000*exp(-t)"
        initial_parameter = {'y': {0: 0}}
        cauchy_function = lambda t, y: -1000 * y + 3000 - 2000*math.exp(-t)
        a = 0
        b = 1
        step = 0.0001  # Small step size for stiff ODE
    
        euler = EulerMethod(ode, initial_parameter, cauchy_function, a, b, step)
        results = euler.exp_method_results()
    
        # Check if results are not None (computation successful)
        self.assertIsNotNone(results)
    
        # Exact solution: y(t) = 3 - 2e^(-t) - e^(-1000t)
        exact_solution = lambda t: 3 - 2*math.exp(-t) - math.exp(-1000*t)
    
        # Check accuracy at t = 1 (end of interval)
        t_end = 1
        y_approx = results[1][-1]
        y_exact = exact_solution(t_end)
    
        # Allow for some error due to the stiffness of the ODE
        tolerance = 1e-3
        self.assertAlmostEqual(y_approx, y_exact, delta=tolerance)

    def test_ode_with_discontinuity(self):
        ode = "y' = 1/x"
        initial_parameter = {'y': {1: 0}}  # y(1) = 0
        cauchy_function = lambda x, y: 1/x
        a = 1
        b = 3
        step = 0.1
        
        euler = EulerMethod(ode, initial_parameter, cauchy_function, a, b, step)
        results = euler.exp_method_results()
        
        self.assertIsNotNone(results)
        self.assertEqual(results.shape, (4, 20))  # 20 steps from 1 to 3 with step 0.1
        
        # Check if the solution is continuous despite the discontinuity in the ODE
        for i in range(1, len(results[1])):
            self.assertLess(abs(results[1][i] - results[1][i-1]), 0.5)  # Allow for some numerical error
if __name__ == '__main__':
    unittest.main()
