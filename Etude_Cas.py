import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly
from scipy import integrate

# Open na noor (read les données de l'aile)
title = 'MH114'
data = pd.read_csv(f'{title}.dat', delim_whitespace=True)

x = data['x'].values
y = data['y'].values

# Split Wing Ding ding in two halves (split le profil en deux pour faire un curve fit)
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

# Fit dem curves (curve fit)
fit_bottom = poly.polyfit(x_bottom, y_bottom, 18)
fit_top = poly.polyfit(x_top, y_top, 18)
x_fit = np.linspace(0, 1, 100000)

# Make the polynomial with the fit (Le curve fit donne les coefficients, la fonction en bas les mets en équation)


def make_poly(fit_values, x_values):
    y_fit = 0
    for index, value in enumerate(fit_values):
        y_fit += value*(x_values**index)
    return y_fit


y_bottom_fit = make_poly(fit_bottom, x_fit)
y_top_fit = make_poly(fit_top, x_fit)

# Centroide du coeur donné par la prof
y_cen = 0.053017
x_cen = 0.40575
# Calculate Area of Wing Ding ding (split le profil en deux parties, celle au dessus et celle en dessous du centroide)

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

# Area above centroid (calcul de l'aire)
area_top = np.trapz(y_top_fit, x_fit)
Atot = area_top - area_bottom  # aire totale de l'aile, donnée par la prof, mais recalculée pour être sur
A2 = np.trapz(y_split_1, x_split_1)
A3 = (x_fit[cutoff[1]]-x_fit[cutoff[0]]) * 0.053017
A4 = np.trapz(y_split_2, x_split_2)
A1_V1 = Atot - (A2 + A3 + A4) + area_bottom  # J'ai utilisé deux méthodes pour calculer l'aire au dessus du centroide pour me valider

# print(f'Area over centroid v1 = {A1_V1} m^2/c^2')

A1_V2 = np.trapz(y_top_main - y_cen, x_top_main)  # On utilise celle-là pour nos calculs
print(f'Area over centroid v2 = {A1_V2} m^2/c^2')

# Centroid of new area (faut trouver le centroide de la zone au-dessus du centroide du coeur de l'aile
x_int = np.trapz((y_top_main - y_cen)*x_top_main, x_top_main)
x_bar = x_int/A1_V2 # Coordonnée en x du nouveau centroide

print(f'x coords of new centroid = {x_bar} m/c')
y_int = np.trapz((y_top_main - y_cen)*(y_top_main+y_cen)/2, x_top_main)
y_bar = y_int/A1_V2 # Coordonnée en y du nouveau centroide

print(f'y coords of new centroid = {y_bar} m/c')

# Calculate bottom
A_bot = Atot - A1_V2 # Pour se valider on a fait l'aire en bas et au-dessus du centroide du coeur

# find Q
Q = (y_bar-y_cen)*A1_V2
y_bar_bot = y_cen - (Q/A_bot) # y du centroide de la partie inférieure, pas besoin dans le vrai calcul
Q2 = (y_bar_bot - y_cen)*A_bot
print(f'A_tot = {Atot}')
print(f'Q = {Q}, Q2 = {Q2}')
print(f'y_bar_bot = {y_bar_bot}')

# find V and M (on ouvre le dossier des chargements fournis par la prof)
title2 = 'lift_2'
data2 = pd.read_csv(f'{title2}.csv', delimiter=',')

pos = data2['Position (m)'].values
w = data2['   Chargement (N/m)'].values
# V c'est l'intégrale de w, et M est l'intégrale de V
v = integrate.cumtrapz(w, pos, initial=0)
m = integrate.cumtrapz(v, pos, initial=0)
V = np.trapz(w, pos)
M = np.trapz(v, pos)

v = v[::-1] # Faut les flip pour que la charge diminue en se rapprochant du bout de l'aile
m = m[::-1]


print(f'V = {V}, M = {M}')
# define what is missing for tau

t = np.max(x_top_main) - np.min(x_top_main) # épaisseur de la section au dessus du centroide du coeur
Ix = 9.8490*10**(-5) # Moment d'inertie en x

l = np.linspace(0, 1.51, len(v)) # largeur de l'aile
c_p = 0.340
c_r = 0.606
c = (((c_r-c_p)/1.51)*l)+c_p # droite du dessus de l'aile

tau = (v*Q)/(Ix*t*(c[::-1]**2)) # Graph de la contrainte de cisaillement
tau_max = np.max(tau) # Trouve la valeur max du cisaillement
i = 0 # trouve la position du cisaillement max sur l'aile
for index, value in enumerate(tau):
    if value == tau_max:
        i = index
        break
print(f'tau_max = {tau_max/10**3} kPa with c = {c[i]} m and l = {l[i]} m') # Cisaillement max en kPa avec largeur et hauteur de l'aile

tau_rup = 600*10**3 # Cisaillement max à la rupture (celui du coeur)

FS = tau_rup/tau_max
print(f'FS = {FS}') # FS du cisaillement

# Mmmmm (On se lance sur le M pour la traction et la compression)
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
# On trouve les y min et max alignés avec le centroide du revêtement
x2 = x_fit[i2]
y2_top = y_top_fit[i2]
y2_bot = y_bottom_fit[i2]
print(f'Y_bot = {y2_bot - y_cen2}, Y_top = {y2_top - y_cen2}') # Les y alignés avec le centroide du revêtement
print(f'X_bar = {x2}') # x du curvefit le plus proche du x du centroide

t2 = 0.35*10**(-3) # fournis par la prof
Ix_rev = 0.0051752 * t2 # Ix du revêtement
c2 = c[::-1] # Faut flip le c de bord comme les V et M

sigma_top = m*(y2_top + t2 - y_cen2)/(Ix_rev*(c2**2)) # Contrainte en haut
sigma_bot = m*(y2_bot - t2 - y_cen2)/(Ix_rev*(c2**2)) # Contrainte en bas
print(f'm = {m}')
print(f'c = {c2}')
sigma_comp = np.max(sigma_top) # Contrainte en compression
sigma_trac = np.max(abs(sigma_bot)) # Contrainte en traction
x_max = np.where(sigma_top == sigma_comp)
x_min = np.where(abs(sigma_bot) == sigma_trac)
x_max = l[x_max]
x_min = l[x_min]
print(f'X_max = {x_max}, X_min = {x_min}')
print(f'Sigma_Comp = {sigma_comp/10**6} MPa, Sigma_Trac = {sigma_trac/10**6} MPa')
# Plot this shit
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_split_1, y_split_1, color='Orange') # En orange c'est la partie en dessous du centroide du coeur
ax.plot(x_split_2, y_split_2, color='Orange')
ax.plot(x_fit, y_bottom_fit, color='Orange')
ax.scatter(x_cen, y_cen, color='Orange')  # Centroide du coeur
ax.scatter(x_bar, y_bar, color='Blue') # En bleu c'est la partie au dessus du centroide
ax.plot(x_top_main, y_top_main, color='Blue')

plt.scatter(x_bar, y_bar_bot, color='Green') # Centroide du bas, pas nécessaire
plt.scatter(x2, y2_top, color='Cyan') # En cyan c'est les centroides alignés avec le revêtement
plt.scatter(x2, y2_bot, color='Cyan')
ax.scatter(0.49251, 0.046128, color = 'Cyan') # Centroide du revêtement


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
# fig.savefig(f"Graph_{title}", bbox_inches='tight',dpi=600) # Enlève le comment si tu veux save le graph

# Plot top only (au besoin c'est juste la section au dessus du centroide du coeur)
plt.plot(x_top_main, y_top_main, color='blue')
plt.scatter(x_bar, y_bar, color='blue')
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.xlabel("x/c")
plt.ylabel('y/c')
plt.ylim([0, 0.3])
plt.xlim([0, 1])
# plt.savefig(f"Graph_top", bbox_inches='tight',dpi=600) # retire le comment pour save
plt.show()

# Plot tau on Wing (au besoin, sers à plot tau en fonction de l ou c)
# plt.plot(l, tau, color='blue')
plt.plot(c, tau, color='Red')
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.xlabel("c**2")
plt.ylabel('tau')
# plt.ylim([0, 0.3])
plt.xlim([np.min(c), np.max(c)])
plt.show()

# Plotte au bic (au besoin, graph des contraintes en traction et compression)
plt.plot(pos, sigma_top, color='Red')
plt.plot(pos, abs(sigma_bot), color='Blue')
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.xlabel("x")
plt.ylabel('Sigma')
# plt.ylim([0, 0.3])
plt.xlim([np.min(pos), np.max(pos)])
plt.show()