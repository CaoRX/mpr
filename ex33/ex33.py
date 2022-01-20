import numpy as np

a = np.array([
    [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]
], dtype = np.float64)

w, v = np.linalg.eigh(a)

print(w, v)
