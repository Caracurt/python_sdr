import pickle
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np

from scipy.signal import butter, filtfilt

def plot_psd(sigs, fs, Nfft):


    for sig in sigs:
        pxx, f_arr = plt.psd(sig, NFFT=Nfft, Fs=fs)

        plt.clf()

        scale = max(pxx)
        pxx = pxx / scale

        plt.plot(f_arr, 10.0 * np.log10(pxx) )


file_name = 'dump_tx_fs20MHZ_QAM4.pkl'
dump_file = Path(file_name)
with dump_file.open('rb') as f:
    PDin = pickle.load(f)

fs_base = float(re.findall('_fs(\d+)', file_name)[0]) * 1e6
Nfft = 1024

# apply filter
# Параметры фильтра
order = 4  # порядок фильтра
cutoff = 0.65  # критическая частота

# Проектирование фильтра
b, a = butter(order, cutoff, btype='low')

# Применение фильтра к сигналу
filtered_signal = filtfilt(b, a, PDin)

PDin = filtered_signal


plt.clf()
plot_psd((PDin,), fs=fs_base, Nfft=Nfft)
plt.grid()
plt.xlim((-fs_base/2, fs_base/2))
plt.xlabel('HZ')
plt.ylabel('dB')
plt.legend(['PDin'])
plt.title('PSD of tx baseband signal, before call sdr.tx')
plt.show()


