import numpy as np
with np.load('calibration_data.npz') as data:
    K = data['mtx']
    dist = data['dist']
print(K)

