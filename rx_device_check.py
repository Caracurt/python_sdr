import adi
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import scipy


sdr = adi.ad9361(uri='ip:192.168.1.1')
sdr.rx_enabled_channels = [0, 1]
data = sdr.rx()

print(data[:10])