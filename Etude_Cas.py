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
x_fit = np.linspace(0, 1, 100000)

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

# Calculate bottom
A_bot = Atot - A1_V2

# find Q
Q = (y_bar-y_cen)*A1_V2
y_bar_bot = y_cen - (Q/A_bot)
Q2 = (y_bar_bot - y_cen)*A_bot
print(f'A_tot = {Atot}')
print(f'Q = {Q}, Q2 = {Q2}')
print(f'y_bar_bot = {y_bar_bot}')

# find V and M
title2 = 'lift_2'
data2 = pd.read_csv(f'{title2}.csv', delimiter=',')

pos = data2['Position (m)'].values
w = data2['   Chargement (N/m)'].values

v = integrate.cumtrapz(w, pos, initial=0)
m = integrate.cumtrapz(v, pos, initial=0)
V = np.trapz(w, pos)
M = np.trapz(v, pos)

v = v[::-1]
m = m[::-1]


print(f'V = {V}, M = {M}')
# define what is missing for tau

t = np.max(x_top_main) - np.min(x_top_main)
Ix = 9.8490*10**(-5)

l = np.linspace(0, 1.51, len(v))
c_p = 0.340
c_r = 0.606
c = (((c_r-c_p)/1.51)*l)+c_p

tau = (v*Q)/(Ix*t*(c[::-1]**2))
tau_max = np.max(tau)
i = 0
for index, value in enumerate(tau):
    if value == tau_max:
        i = index
        break
print(f'tau_max = {tau_max/10**3} kPa with c = {c[i]} m and l = {l[i]} m')

tau_rup = 600*10**3

# print(f'Contrainte de cisaillement = {tau/10**(3)} kPa')
FS = tau_rup/tau_max
print(f'FS = {FS}')

# print(f'Facteur de sécurité en cisaillement : {FS}')

# Mmmmm
x_cen2 = 0.49251
y_cen2 = 0.046128
x_2 = abs(x_fit - x_cen2)
x_min = np.min(x_2)
i2 = 0
print(x_2)
for index, value in enumerate(x_2):
    if abs(value) == x_min:
        i2 = index
        break

x2 = x_fit[i2]
y2_top = y_top_fit[i2]
y2_bot = y_bottom_fit[i2]
print(f'Y_bot = {y2_bot - y_cen2}, Y_top = {y2_top - y_cen2}')
print(f'X_bar = {x2}')

t2 = 0.35*10**(-3)
Ix_rev = 0.0051752 * t2
c2 = c[::-1]
sigma_top = m*(y2_top + t2 - y_cen2)/(Ix_rev*(c2**2))
sigma_bot = m*(y2_bot - t2 - y_cen2)/(Ix_rev*(c2**2))
print(f'm = {m}')
print(f'c = {c2}')
sigma_comp = np.max(sigma_top)
sigma_trac = np.max(abs(sigma_bot))
x_max = np.where(sigma_top == sigma_comp)
x_min = np.where(abs(sigma_bot) == sigma_trac)
x_max = l[x_max]
x_min = l[x_min]
print(f'X_max = {x_max}, X_min = {x_min}')
print(f'Sigma_Comp = {sigma_comp/10**6} MPa, Sigma_Trac = {sigma_trac/10**6} MPa')
# Plot this shit
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_split_1, y_split_1, color='Orange')
ax.plot(x_split_2, y_split_2, color='Orange')
ax.plot(x_top_main, y_top_main, color='Blue')
ax.plot(x_fit, y_bottom_fit, color='Orange')
plt.scatter(x_bar, y_bar_bot, color='Green')
plt.scatter(x2, y2_top, color='Cyan')
plt.scatter(x2, y2_bot, color='Cyan')
ax.scatter(x_bar, y_bar, color='Blue')
ax.scatter(0.49251, 0.046128, color = 'Cyan') # Centroid with thickness
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

# Plot tau on Wing
# plt.plot(l, tau, color='blue')
plt.plot(c, tau, color='Red')
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.xlabel("c**2")
plt.ylabel('tau')
# plt.ylim([0, 0.3])
plt.xlim([np.min(c), np.max(c)])
plt.show()

# Plotte au bic
plt.plot(pos, sigma_top, color='Red')
plt.plot(pos, abs(sigma_bot), color='Blue')
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.xlabel("x")
plt.ylabel('Sigma')
# plt.ylim([0, 0.3])
plt.xlim([np.min(pos), np.max(pos)])
plt.show()