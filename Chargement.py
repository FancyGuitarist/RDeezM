import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly
from scipy import integrate
from scipy.integrate import cumtrapz

title = 'lift_2'
data = pd.read_csv(f'{title}.csv', delimiter= ',')

pos = data['Position (m)'].values
w = data['   Chargement (N/m)'].values

V = integrate.cumtrapz(w,pos,initial = 0)
M = integrate.cumtrapz(V,pos,initial=0)


"""charge = []
M = []
fit_charge = poly.polyfit(pos,charge_m,20)"""
"""for index, value in enumerate(charge_m):
    charge.append(pos[index]*value)
    M.append((pos[index]**2)*value)"""



"""fit_M = poly.polyfit(pos,M,120)
fit_charge = poly.polyfit(pos,charge,120)"""
#x_fit = np.linspace(0,1.51,2000)
def make_poly(fit_coeffs, symbol):
    y_fit = 0
    for index, value in enumerate(fit_coeffs):
        y_fit += value*(symbol**index)
    return y_fit

#y_fit_charge = make_poly(fit_charge, x_fit)
#y_fit_M = make_poly(fit_M,x_fit)
#M = np.trapz(y_fit_charge,x_fit)
#M_max = np.max(y_fit_M)
#V_max = np.max(y_fit_charge)
"""x = Symbol('x')
sigma = make_poly(fit_charge,x)
V = integrate(sigma,x)
M = integrate(V, x)"""

#area_M = np.trapz(y_fit_M, x_fit)
#area_charge = np.trapz(y_fit_charge,x_fit)
#print(f'M = {M}, V = {V_max}')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pos, V, color = 'Black')
ax.plot(pos, M, color = 'Orange')
ax.plot(pos, w, color = 'Blue')
#ax.plot(x_fit, y_fit_charge, color = 'Red')
#ax.plot(x_fit, y_fit_M, color = 'Blue')
fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.xlabel("position")
plt.ylabel('charge')
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.set_ylim([0,65])
ax.set_xlim([0,1.6])
plt.show()
