import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline, BSpline

if not os.path.isdir("plot_UniTS"):
    os.makedirs("plot_UniTS")

x9 = [val for val in np.arange(0, np.pi * 2, 0.01)]
y9 = [np.sin(val) / 2 for val in x9]
x10 = [np.pi * 2, np.pi * 2 + 1, np.pi * 2 + 2, np.pi * 2 + 3, np.pi * 2 + 4]
y10 = [0, 0, -5, 0.25, 0.25]
x11 = [val for val in np.arange(np.pi * 2 + 4, np.pi * 2.9375 + 4, 0.01)]
y11 = [0.25 + np.sin((val - 4) * 2) / 2 for val in x11]
ax = plt.axes()
#ax.set_facecolor("lightgray")
plt.plot(x9, y9, c = "k", linewidth = 3)
plt.plot(x10, y10, c = "k", linewidth = 3)
plt.plot(x11, y11, c = "k", linewidth = 3)
plt.plot([x11[-1], x11[-1] + 1], [y11[-1], y11[-1]], c = "k", linewidth = 3)
plt.axis("off")
plt.savefig("plot_UniTS/1.png", bbox_inches = "tight")
plt.close()

xsin1 = [xv for xv in np.arange(1, 10, 0.001)]
ysin1 = []
lim1 = 1.5 * np.pi
lim2 = 1.94 * np.pi
diffmidi = 1000
midiix = 0
a1 = 1
a2 = 7
a3 = 2
f1 = 0.2
f2 = 0.25
f3 = 0.4
for ix in range(len(xsin1)):
    val = xsin1[ix]
    if val < lim1:
        ysin1.append(np.sin(val / f1) * a1)
        continue
    if val < lim2:
        ysin1.append(np.cos(val / f2 + np.pi ) * a2 + 6)
        if abs(ysin1[-1] - a2 - 6) < diffmidi:
            diffmidi = abs(ysin1[-1] - a2 - 6)
            midiix = ix
        continue
    ysin1.append(np.sin(val / f3) * a3)
ax = plt.axes()
#ax.set_facecolor("lightgray")
plt.plot(xsin1, ysin1, c = "k", linewidth = 3)
plt.scatter(xsin1[midiix], ysin1[midiix], c = "red", linewidth = 10)
plt.axis("off")
plt.savefig("plot_UniTS/2.png", bbox_inches = "tight")
plt.close()
 
x1 = [0, 1, 1.5, 2, 2.5, 3, 4, 4.5, 5.5, 6.5, 7, 8, 8.5, 9, 9.5, 10, 11]
y1 = [0, 0, 0.35, 0.5, 0.35, 0, 0, -1, 5, -1, 0, 0, 0.35, 0.5, 0.35, 0, 0]
ax = plt.axes()
#ax.set_facecolor("lightgray")
plt.plot(x1, y1, c = "k", linewidth = 3)
plt.axis("off")
plt.savefig("plot_UniTS/3.png", bbox_inches = "tight")
plt.close()

x2 = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7, 8, 9, 9.5, 10.5, 11.5, 12.5, 13.5, 14]
y2 = [0.25, 0.5, 0, 0.25, 5, 0.25, 0, 0.5, 0.25, 5, 0.25, 0.5, 0, 0.25, 5, 0.25, 0.5]
ax = plt.axes()
#ax.set_facecolor("lightgray")
plt.plot(x2, y2, c = "k", linewidth = 3)
plt.axis("off")
plt.savefig("plot_UniTS/4.png", bbox_inches = "tight")
plt.close()

x5 = [0, 1, 2, 3, 4, 5]
y5 = [1, 2, 1, 2, 2, 3]
x6 = [5, 6, 7, 8]
y6 = [3, 2, 4, 3]
x7 = [8, 9, 10, 11, 12, 13]
y7 = [3, 2, 0, 1, 1, 2]
x8 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
y8 = [1, 2, 1, 2, 2, 3, 2, 4, 3, 2, 0, 1, 1, 2]
spl8 = make_interp_spline(x8, y8, k = 3)
power_smooth5 = spl8(np.arange(0, 5, 0.01)) 
power_smooth6 = spl8(np.arange(5, 8, 0.01)) 
power_smooth7 = spl8(np.arange(8, 13, 0.01)) 
ax = plt.axes()
#ax.set_facecolor("lightgray")
plt.plot(np.arange(0, 5, 0.01), power_smooth5, c = "k", linewidth = 3)
plt.plot(np.arange(5, 8, 0.01), power_smooth6, c = "g", linewidth = 3, linestyle = "dashed")
plt.plot(np.arange(8, 13, 0.01), power_smooth7, c = "k", linewidth = 3)
plt.axis("off")
plt.savefig("plot_UniTS/5.png", bbox_inches = "tight")
plt.close()

x3 = [0, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
y3 = [0, 2.25, 1.5, 2, 1, 2.5, 1, 2, 1.5, 2.5, 2.75, 3.25, 2.75]
y3 = [y * 2 for y in y3]
ax = plt.axes()
#ax.set_facecolor("lightgray")
plt.axis("equal")
plt.plot(x3, y3, c = "k", linewidth = 3)
plt.axis("off")
plt.savefig("plot_UniTS/6.png", bbox_inches = "tight")
plt.close()

x4 = [16, 19, 20, 21, 25, 26, 27]
y4 = [2.75, 2, 2.75, 2.5, 3.25, 3.5, 3.25]
y4 = [y * 2 for y in y4]
ax = plt.axes()
#ax.set_facecolor("lightgray")
plt.axis("equal")
plt.plot(x3, y3, c = "k", linewidth = 3)
plt.plot(x4, y4, c = "g", linewidth = 3, linestyle = "dashed")
plt.axis("off")
plt.savefig("plot_UniTS/7.png", bbox_inches = "tight")
plt.close()