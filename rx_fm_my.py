import adi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy.matlib

sample_rate = 1e6 # Hz
center_freq = 2.0e9 # Hz
FrameSize = 128

# Tx
# sdr = adi.ad9363(uri='ip:192.168.1.1')
# sdr.sample_rate = int(sample_rate)
# sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
# sdr.tx_lo = int(center_freq)
# #sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB
# tx_gain0 = 0
# tx_gain1 = 0
# sdr = adi.ad9363(uri='ip:192.168.1.1')
# sdr.tx_enabled_channels = list(range(1))
#
# sdr.tx_rf_bandwidth = int(sample_rate)
# sdr.tx_lo = int(center_freq)
# sdr.tx_cyclic_buffer = True
# sdr.tx_hardwaregain_chan0 = int(tx_gain0)
# sdr.tx_hardwaregain_chan1 = int(tx_gain1)
# sdr.tx_buffer_size = FrameSize
# Rx
sdr = adi.ad9361(uri='ip:192.168.2.2') # eth 192.168.1.10 , usb 192.168.2.2

samp_rate = sample_rate  # must be <=30.72 MHz if both channels are enabled
num_samps = FrameSize  # number of samples per buffer.  Can be different for Rx and Tx
rx_lo = center_freq
#rx_mode = "slow_attack"  # can be "manual" or "slow_attack"
rx_mode = "slow_attack"
rx_gain0 = 70
rx_gain1 = 70
tx_lo = rx_lo
tx_gain0 = 0
tx_gain1 = 0

sdr.rx_enabled_channels = [0, 1]
#sdr.rx_enabled_channels = [0]
sdr.sample_rate = int(samp_rate)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
sdr.rx_hardwaregain_chan1 = int(rx_gain1)
sdr.rx_buffer_size = int(num_samps)


# Tx message
N = FrameSize # number of samples to transmit at once
t = np.arange(N)/sample_rate
f_tx = 0.1 * sample_rate
samples = 0.5*np.exp(2.0j*np.pi*f_tx*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

# Transmit our batch of samples 100 times, so it should be 1 second worth of samples total, if USB can keep up

#while True:
Ntrial = 10
for i in range(Ntrial):
    data = sdr.rx()

    data = np.array(data) # Nrx x Nsamples

    pdp = np.fft.fft(data, axis=1, norm='ortho')

    pdp_avg = np.mean(np.abs(pdp)**2, axis=0)

    # noise est
    n_left = int(0.5 * FrameSize)
    n_right = int(0.7 * FrameSize)

    noise_win = pdp[:, n_left:n_right]
    sigma0 = np.mean(np.abs(noise_win)**2, axis=1)

    pos_max = np.argmax(pdp_avg)

    SNR_rx = 10.0 * np.log10(np.abs(pdp[:, pos_max]) ** 2 / sigma0)

    print(f'SNR_rx0: {SNR_rx[0] : .2f} SNR_rx1: {SNR_rx[1] : .2f}')

    print(f'Freq Decode = {pos_max}')


    plt.plot(range(len(pdp_avg)), pdp_avg)
    plt.grid()
    plt.show()







