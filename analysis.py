import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.stats import ttest_ind

print('*'*64)
print('\nPerforming T-Test for Null Hypothesis:\
    \n\nThere is no statistical difference in the execution time between the baseline template matching tracker and the Newton prediction tracker.')
file = open("data/times.csv")
csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
times = []
for row in csvreader:
    times.append(row)
file.close()
times = np.array(times)

baseline_times = times[:,0]
prediction_times = times[:,1]

print('\nbaseline average: ', np.average(baseline_times))
print('prediction average: ', np.average(prediction_times))

stat, p = ttest_ind(baseline_times, prediction_times)
print('\nStatistics=%.3f, p=%.3f' % (stat, p))


alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject null hypothesis)\n')
else:
	print('Different distributions (reject null hypothesis)\n')
print('*'*64)

file = open("data/baseline.csv")
csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
baseline = []
for row in csvreader:
    baseline.append(row)
file.close()
baseline = np.array(baseline)

file = open("data/prediction.csv")
csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
prediction = []
for row in csvreader:
    prediction.append(row)
file.close()
prediction = np.array(prediction)

file = open("data/prediction_measured.csv")
csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
prediction_measured = []
for row in csvreader:
    prediction_measured.append(row)
file.close()
prediction_measured = np.array(prediction_measured)

file = open("data/prediction_predicted.csv")
csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
prediction_predicted = []
for row in csvreader:
    prediction_predicted.append(row)
file.close()
prediction_predicted = np.array(prediction_predicted)

z = np.polyfit(baseline[1:-1, 0],
                       np.negative(baseline[1:-1, 1]), 2)
f = np.poly1d(z)
x_new = np.linspace(0, 1900, 50)
y_new = f(x_new)

plt.figure()
plt.plot(x_new, y_new, '--')
plt.plot(baseline[1:-1, 0], np.negative(baseline[1:-1, 1]),
            'go', markersize=3)
plt.title('Baseline Measurement')
plt.xlabel('x pixel coordinate')
plt.ylabel('y pixel coordinate')
plt.xlim([0, 1960])
plt.ylim([-1060, 0])
plt.legend(['Calculate Trajectory', 'Measured Points'])

z_p = np.polyfit(prediction[1:-1, 0],
                       np.negative(prediction[1:-1, 1]), 2)
f_p = np.poly1d(z_p)
x_new_p = np.linspace(0, 1900, 50)
y_new_p = f_p(x_new_p)
print(f_p)

plt.figure()
plt.plot(x_new_p, y_new_p, '--')
plt.plot(prediction_measured[1:-1, 0], np.negative(prediction_measured[1:-1, 1]),
            'go', markersize=3)
plt.plot(prediction_predicted[1:-1, 0], np.negative(prediction_predicted[1:-1, 1]),
            'rx', markersize=3)
plt.title('Predictions Measurement')
plt.xlabel('x pixel coordinate')
plt.ylabel('y pixel coordinate')
plt.xlim([0, 1960])
plt.ylim([-1060, 0])
plt.legend(['Calculate Trajectory', 'Measured Points','Predicted Points'])


plt.figure()
plt.plot(prediction[1:-1, 0], np.negative(prediction[1:-1, 1]) - np.negative(baseline[1:-1, 1]), '.')
plt.xlim([0, 1960])
plt.xlabel('x pixel coordinate')
plt.ylabel('Pixel Difference')
plt.title('Difference Between Baseline and Predictions')


plt.show()

