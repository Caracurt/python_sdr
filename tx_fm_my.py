import adi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy.matlib

sample_rate = 1e6 # Hz
center_freq = 2.0e9 # Hz

sdr = adi.ad9363(uri='ip:192.168.1.1')
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
#sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB
tx_gain0 = 0
tx_gain1 = 0
sdr = adi.ad9363(uri='ip:192.168.1.1')
sdr.tx_enabled_channels = list(range(1))

sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_lo = int(center_freq)
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = int(tx_gain0)
sdr.tx_hardwaregain_chan1 = int(tx_gain1)

FrameSize = 128
sdr.tx_buffer_size = FrameSize


N = FrameSize # number of samples to transmit at once
t = np.arange(N)/sample_rate
f_tx = int(0.1 * sample_rate)
samples = 0.5*np.exp(2.0j*np.pi*f_tx*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

pdp = np.fft.fft(samples, axis=0, norm='ortho')

pdp_avg = np.abs(pdp)**2

pos_max = np.argmax(pdp_avg)

print(f'Tx Freq = {pos_max}')


# Transmit our batch of samples 100 times, so it should be 1 second worth of samples total, if USB can keep up
sdr.tx(samples)

print('Success tx : enter if exit')
try:
    ue_in = input()
    if ue_in == 'E' and ue_in == 'e':
        raise KeyboardInterrupt()
except KeyboardInterrupt:
    sdr.tx_destroy_buffer()
    exit(1)
    ff = 1

