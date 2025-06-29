# main.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from parametros import a0, A0, dA0
from modelo import deriv, H_squared_naive, Mpl, mA, xi

# Integração
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
y0 = [a0, A0, dA0]

sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval, rtol=1e-8)

# Resultados
a, A, dA = sol.y
t = sol.t
H = np.sqrt(H_squared_naive(a, A, dA))
rhoA = 0.5 * (dA**2 + mA**2 * A**2) + 3 * xi * H**2 * A**2
pA = 0.5 * (dA**2 - mA**2 * A**2) - 3 * xi * H**2 * A**2
wA = pA / rhoA

# Plot
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, a, label="Fator de escala $a(t)$")
plt.plot(t, A, label="Campo $A_0(t)$")
plt.legend(); plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, rhoA, label="Energia vetorial $\\rho_A$")
plt.plot(t, wA, label="Parâmetro de estado $w_A$")
plt.grid(); plt.legend()
plt.xlabel("Tempo (normalizado)")
plt.tight_layout()
plt.show()
