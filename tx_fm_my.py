import adi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal



# get audio
filename = '01-We-Die-Young.wav'
data, fs = sf.read(filename)

#N_plot = 4096 * 400
N_plot = 4096 * 1
N_data = data.shape[0]
up_s = 10
sdr_samp_rate = fs * up_s
N_gr = N_data // N_plot

# end get audio


sample_rate = sdr_samp_rate # Hz
center_freq = 2.0e9 # Hz

sdr = adi.ad9363(uri='ip:192.168.1.1')
#sdr = adi.ad9361(uri='ip:192.168.2.2')
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
#sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB
tx_gain0 = -10
tx_gain1 = -10
#
#sdr = adi.ad9363(uri='ip:192.168.1.1')
sdr.tx_enabled_channels = list(range(1))

sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_lo = int(center_freq)
sdr.tx_cyclic_buffer = False
sdr.tx_hardwaregain_chan0 = int(tx_gain0)
sdr.tx_hardwaregain_chan1 = int(tx_gain1)

FrameSize = N_plot * up_s
sdr.tx_buffer_size = FrameSize


# debug
# N = FrameSize # number of samples to transmit at once
# t = np.arange(N)/sample_rate
# f_tx = int(0.1 * sample_rate)
# samples = 0.5*np.exp(2.0j*np.pi*f_tx*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
# samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
# pdp = np.fft.fft(samples, axis=0, norm='ortho')
# pdp_avg = np.abs(pdp)**2
# pos_max = np.argmax(pdp_avg)
# print(f'Tx Freq = {pos_max}')


while True:

    Rep_idx = 0

    for gr_idx in range(N_gr):

        sc_slice = slice(gr_idx * N_plot, (gr_idx + 1) * N_plot)

        data_cut = data[sc_slice, 0] # take mono sig
        data_cut_up = signal.resample_poly(data_cut, up_s, 1)
        kf = 75e3
        phase_tx = 2 * np.pi * kf * np.cumsum(data_cut_up) / sdr_samp_rate
        iq_samples = np.exp(1j * phase_tx)


        if gr_idx == 0:
            print(f'Tx success Rep_Idx: {Rep_idx}')

        # Transmit our batch of samples 100 times, so it should be 1 second worth of samples total, if USB can keep up
        sdr.tx(iq_samples * 2**14)

    Rep_idx = Rep_idx + 1

# print('Success tx : enter if exit')
# try:
#     ue_in = input()
#     if ue_in == 'E' and ue_in == 'e':
#         raise KeyboardInterrupt()
# except KeyboardInterrupt:
#     sdr.tx_destroy_buffer()
#     exit(1)
#     ff = 1

