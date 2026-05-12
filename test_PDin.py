import pickle
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np

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



plt.clf()
plot_psd((PDin,), fs=fs_base, Nfft=Nfft)
plt.grid()
plt.xlim((-fs_base/2, fs_base/2))
plt.xlabel('HZ')
plt.ylabel('dB')
plt.legend(['PDin'])
plt.title('PSD of tx baseband signal, before call sdr.tx')
plt.show()


