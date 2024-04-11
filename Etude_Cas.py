import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly
#Open na noor
title = 'MH114'
data = pd.read_csv(f'{title}.dat', delim_whitespace=True)

x = data['x'].values
y = data['y'].values

#Split Wing Ding ding in two halves
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
#Fit dem curves
fit_bottom = poly.polyfit(x_bottom,y_bottom,18)
fit_top = poly.polyfit(x_top,y_top,18)
x_fit = np.linspace(0,1,2000)
#Make the polynomial with the fit
def make_poly(fit_values, x_values):
    y_fit = 0
    for index, value in enumerate(fit_values):
        y_fit += value*(x_values**index)
    return y_fit

y_bottom_fit = make_poly(fit_bottom, x_fit)
y_top_fit = make_poly(fit_top, x_fit)

y_cen = 0.053017
#Calculate Area of Wing Ding ding
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

#area_top = np.trapz(y_top_fit, x_fit)
#area = area_top - area_bottom
#top de l'aile
"""l = np.linspace(0,1.51,1000)
c = ((0.34-0.606)/2)*l+0.606"""
c_p = 0.606
#Calculate Q baby

"""A_top = np.linspace(0,area_top*c_p**2,2000)
A_bot = np.linspace(0,area_bottom*c_p**2,2000)
Q_top = np.trapz((y_top_fit-y_cen)*c_p,A_top)
Q_bot = np.trapz((y_bottom_fit-y_cen)*c_p,A_bot)
Q_tot = Q_top - Q_bot
print(f'Q = {Q_tot}')
"""
#Plot this shit
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_split_1, y_split_1, color = 'Orange')
ax.plot(x_split_2, y_split_2, color = 'Cyan')
ax.plot(x_top_main, y_top_main, color = 'Blue')
#ax.plot(x_bottom, y_bottom, color = 'blue')
ax.plot(x_fit,y_bottom_fit, color = 'Purple')
#ax.scatter(0.49251, 0.046128, color = 'Green')
ax.scatter(0.40575, 0.053017, color = 'Red')
fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.xlabel("x/c")
plt.ylabel('y/c')
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.set_ylim([-0.25,0.25])
ax.set_xlim([0,1])
plt.show()
#fig.savefig(f"Graph_{title}", bbox_inches='tight',dpi=600)