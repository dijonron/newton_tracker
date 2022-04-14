import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

file = open("x.csv")
csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
x = []
for row in csvreader:
    x.append(row)
file.close()
x = np.array(x)

file = open("y.csv")
csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
y = []
for row in csvreader:
    y.append(row)
file.close()
y= np.array(y)

z = np.polyfit(x[:,7], np.negative(y[:,7]), 2)
f = np.poly1d(z)
x_new = np.linspace(0, 1900, 50)
y_new = f(x_new)


plt.plot(x, np.negative(y), 'o')#, x_new, y_new, '-', markersize=3)
plt.plot()
plt.xlim(0, 1900)
plt.ylim(-1072, 0)
plt.show()