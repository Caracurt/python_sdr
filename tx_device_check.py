import adi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy.matlib

sdr = adi.Pluto('ip:192.168.2.2')

full_frame_rep = np.random.normal(0.0, 1.0, (10, 1)) + 1j*np.random.normal(0.0, 1.0, (10, 1))

sdr.tx(full_frame_rep[:,0] * 10024.0)

