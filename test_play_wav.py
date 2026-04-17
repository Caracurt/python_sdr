import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

filename = '01-We-Die-Young.wav'
data, fs = sf.read(filename)

N_plot = 4096 * 400

up_s = 5
sdr_samp_rate = fs * up_s

data_cut = data[:N_plot, 0]
plt.plot(range(N_plot), data[:N_plot, 0])
plt.grid()
plt.show()

# sd.play(data_cut, fs)
# sd.wait()  # Ожидание завершения воспроизведения

data_cut_up = signal.resample_poly(data_cut, up_s, 1)

data_cut_down0 = signal.resample_poly(data_cut_up, 1, up_s)

# sd.play(data_cut_down, fs)
# sd.wait()

# emulate fm mod
kf = 75e3
phase_tx = 2 * np.pi * kf * np.cumsum(data_cut_up) / sdr_samp_rate
iq_samples = np.exp(1j * phase_tx)

# fm demodulate
phase_rx = np.angle(iq_samples[1:] * np.conj(iq_samples[0:-1]))

data_cut_down1 = signal.resample_poly(phase_rx, 1, up_s)


sd.play(data_cut_down1, fs)
sd.wait()