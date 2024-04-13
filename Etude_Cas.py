import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly
from scipy import integrate

# Open na noor
title = 'MH114'
data = pd.read_csv(f'{title}.dat', delim_whitespace=True)

x = data['x'].values
y = data['y'].values

# Split Wing Ding ding in two halves
x_bottom = []
x_top = []
y_bottom = []
y_top = []

for index, value in enumerate(y):
    if value < 0:
        y_top = y[:index-1:]
        y_bottom = y[index::]
        x_top = x[:index-1:]
        x_bottom = x[index::]
        break
    else:
        continue

# Fit dem curves
fit_bottom = poly.polyfit(x_bottom, y_bottom, 18)
fit_top = poly.polyfit(x_top, y_top, 18)
x_fit = np.linspace(0, 1, 2000)

# Make the polynomial with the fit


def make_poly(fit_values, x_values):
    y_fit = 0
    for index, value in enumerate(fit_values):
        y_fit += value*(x_values**index)
    return y_fit


y_bottom_fit = make_poly(fit_bottom, x_fit)
y_top_fit = make_poly(fit_top, x_fit)

y_cen = 0.053017
x_cen = 0.40575
# Calculate Area of Wing Ding ding
area_bottom = np.trapz(y_bottom_fit, x_fit)
cutoff = []
prev_val = 0
for index, value in enumerate(y_top_fit):
    if index > 0:
        prev_val = y_top_fit[index - 1]
        if value >= y_cen >= prev_val:
            cutoff.append(index)
        if value <= y_cen <= prev_val:
            cutoff.append(index)
    else:
        continue

y_split_1 = y_top_fit[:cutoff[0]:]
y_split_2 = y_top_fit[cutoff[1]::]
y_top_main = y_top_fit[cutoff[0]:cutoff[1]:]
x_split_1 = x_fit[:cutoff[0]:]
x_split_2 = x_fit[cutoff[1]::]
x_top_main = x_fit[cutoff[0]:cutoff[1]:]

# Area above centroid
area_top = np.trapz(y_top_fit, x_fit)
Atot = area_top - area_bottom
A2 = np.trapz(y_split_1, x_split_1)
A3 = (x_fit[cutoff[1]]-x_fit[cutoff[0]]) * 0.053017
A4 = np.trapz(y_split_2, x_split_2)
A1_V1 = Atot - (A2 + A3 + A4) + area_bottom

print(f'Area over centroid v1 = {A1_V1} m^2/c^2')

A1_V2 = np.trapz(y_top_main - y_cen, x_top_main)
print(f'Area over centroid v2 = {A1_V2} m^2/c^2')

# Centroid of new area
x_int = np.trapz((y_top_main - y_cen)*x_top_main, x_top_main)
x_bar = x_int/A1_V2

print(f'x coords of new centroid = {x_bar} m/c')
y_int = np.trapz((y_top_main - y_cen)*(y_top_main+y_cen)/2, x_top_main)
y_bar = y_int/A1_V2

print(f'y coords of new centroid = {y_bar} m/c')

# find Q
Q = (y_bar-y_cen)*A1_V2
print(f'Q = {Q}')

# find V and M
title2 = 'lift_2'
data = pd.read_csv(f'{title2}.csv', delimiter=',')

pos = data['Position (m)'].values
w = data['   Chargement (N/m)'].values

v = integrate.cumtrapz(w[::-1], pos, initial=0)
V = np.trapz(w[::-1], pos)
M = np.trapz(v, pos)

print(f'V = {V}, M = {M}')
# define what is missing for tau

t = np.max(x_top_main) - np.min(x_top_main)
print(f'x_max = {np.max(x_top_main)}, ')
Ix = 9.8490*10**(-5)


l = np.linspace(0, 1.51, len(v))
c_p = 0.340
c_r = 0.606
c = (((c_r-c_p)/1.51)*l)+c_p
tau = (v*Q)/(Ix*t*(c**2))
tau_max = np.max(tau)
i = 0
for index, value in enumerate(tau):
    if value == tau_max:
        i = index
        break
print(f'tau_max = {tau_max/10**3} kPa with c = {c[i]} m')

tau_rup = 600*10**3

# print(f'Contrainte de cisaillement = {tau/10**(3)} kPa')
FS = tau_rup/tau_max

# print(f'Facteur de sécurité en cisaillement : {FS}')


# Plot this shit
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_split_1, y_split_1, color='Orange')
ax.plot(x_split_2, y_split_2, color='Orange')
ax.plot(x_top_main, y_top_main, color='Blue')
ax.plot(x_fit, y_bottom_fit, color='Orange')
ax.scatter(x_bar, y_bar, color='Blue')
# ax.scatter(0.49251, 0.046128, color = 'Green') # Centroid with thickness
ax.scatter(x_cen, y_cen, color='Orange')  # Centroid of heart

# Graph Setup Mumbo-Jumbo
fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.xlabel("x/c")
plt.ylabel('y/c')
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
ax.set_ylim([-0.25, 0.25])
ax.set_xlim([0, 1])
plt.show()
# fig.savefig(f"Graph_{title}", bbox_inches='tight',dpi=600)

# Plot top only
plt.plot(x_top_main, y_top_main, color='blue')
plt.scatter(x_bar, y_bar, color='blue')
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.xlabel("x/c")
plt.ylabel('y/c')
plt.ylim([0, 0.3])
plt.xlim([0, 1])
# plt.savefig(f"Graph_top", bbox_inches='tight',dpi=600)
plt.show()

# Plot V on Wing
plt.plot(l, tau, color='blue')
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.xlabel("l")
plt.ylabel('tau')
# plt.ylim([0, 0.3])
plt.xlim([0, 1.51])
plt.show()
