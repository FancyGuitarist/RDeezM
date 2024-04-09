import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly

title = 'lift_2'
data = pd.read_csv(f'{title}.csv', delimiter= ',')

pos = data['Position (m)'].values
charge_m = data['   Chargement (N/m)'].values

charge = []
M = []

for index, value in enumerate(charge_m):
    charge.append(pos[index]*value)
    M.append((pos[index]**2)*value)


fit_M = poly.polyfit(pos,M,120)
fit_charge = poly.polyfit(pos,charge,120)
x_fit = np.linspace(0,1.51,2000)
def make_poly(fit_values, x_values):
    y_fit = 0
    for index, value in enumerate(fit_values):
        y_fit += value*(x_values**index)
    return y_fit

y_fit_charge = make_poly(fit_charge, x_fit)
y_fit_M = make_poly(fit_M,x_fit)

M_max = np.max(y_fit_M)
V_max = np.max(y_fit_charge)


#area_M = np.trapz(y_fit_M, x_fit)
#area_charge = np.trapz(y_fit_charge,x_fit)
print(f'M = {M_max}, V = {V_max}')

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(pos, charge, color = 'Orange')
ax.plot(x_fit, y_fit_charge, color = 'Red')
ax.plot(x_fit, y_fit_M, color = 'Blue')
fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.xlabel("position")
plt.ylabel('charge')
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.set_ylim([0,65])
ax.set_xlim([0,1.6])
plt.show()
