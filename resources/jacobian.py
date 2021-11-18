from sympy import symbols, Symbol, Matrix, diff, simplify, sin, cos
from pprint import pprint

l1, l2, l3 = symbols('l1, l2, l3', real=True, positive=True)
q0, q1 = symbols('q0, q1', real=True)

f = Matrix([
    l2*sin(q0) + l3*sin(q0+q1),
    l1 + l2*cos(q0) + l3*cos(q0+q1)
])

J = simplify(f.jacobian([q0, q1]))
# Jx = diff(f[0], q0)
# Jy =

pprint(J)
