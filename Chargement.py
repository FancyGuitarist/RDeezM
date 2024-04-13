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
M_inv = M[::-1]
V_inv = V[::-1]


def make_poly(fit_coeffs, symbol):
    y_fit = 0
    for index, value in enumerate(fit_coeffs):
        y_fit += value*(symbol**index)
    return y_fit

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pos, V_inv, color = 'Green')
#ax.plot(pos, M_inv, color = 'Orange')
#ax.plot(pos, w, color = 'Blue')
fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.xlabel("position")
plt.ylabel('charge')
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.set_ylim([0,80])
ax.set_xlim([0,1.6])
plt.show()
