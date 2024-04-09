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

#Calculate Area of Wing Ding ding
area_bottom = np.trapz(y_bottom_fit, x_fit)
area_top = np.trapz(y_top_fit, x_fit)
area = area_top - area_bottom

c = 0.340 #where c = c_p

#Plot this shit
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(x_top, y_top, color = 'Orange')
ax.plot(x_fit, y_bottom_fit, color = 'Purple')
#ax.plot(x_bottom, y_bottom, color = 'blue')
ax.plot(x_fit,y_top_fit, color = 'Purple')
fig.tight_layout()
fig.subplots_adjust(top=0.9)
plt.xlabel("x/c")
plt.ylabel('y/c')
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.set_ylim([-0.25,0.25])
ax.set_xlim([0,1])
#plt.show()
#fig.savefig(f"Graph_{title}", bbox_inches='tight',dpi=600)