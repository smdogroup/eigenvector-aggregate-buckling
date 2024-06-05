import numpy as np

# compute relative percentage difference
a = [12.50, 14.19, 14.78, 15.04, 15.61, 15.66]
# b = [14.86, 17.42, 18.44, 18.78, 19.02, 19.30]
# b = [14.08, 15.011, 15.205, 16.175, 16.699, 17.468]
b = [13.997, 14.776, 14.846, 15.725, 16.408, 17.188]

rpd = np.abs(np.subtract(a, b)) / np.add(a, b) * 100
print(rpd)


# compute relative percentage difference
a = [12.68, 13.47, 13.70, 13.86, 14.72, 14.90]
b = [13.213, 13.50, 13.872, 14.346, 14.617, 15.319]

rpd = np.abs(np.subtract(a, b)) / np.add(a, b) * 100
print(rpd)

# compute relative percentage difference
a = [10.89, 11.79, 12.30, 12.40, 12.88, 12.90]
b = [11.603, 11.802, 12.619, 12.764, 13.316, 13.957]

rpd = np.abs(np.subtract(a, b)) / np.add(a, b) * 100
print(rpd)


# compute relative percentage difference
a = [7.99, 9.54, 10.33, 10.85, 10.87, 11.27]
# b = [9.0289, 10.721, 11.109, 11.787, 12.267, 12.684]
b = [8.5575, 9.8488, 10.180, 10.885, 10.950, 11.272]

rpd = np.abs(np.subtract(a, b)) / np.add(a, b) * 100
print(rpd)

# compute relative percentage difference
a = [8.90, 8.90, 9.02, 9.12, 9.14, 9.43]
b = [8.8542, 8.9231, 10.022, 10.208, 10.259, 11.069]

rpd = np.abs(np.subtract(a, b)) / np.add(a, b) * 100
print(rpd)
