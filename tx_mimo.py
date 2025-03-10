import adi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy.matlib

import tensorflow as tf
import sionna as sn

sample_rate = 5e6  # Hz
center_freq = 2.0e9 # Hz
device_type = 'ANTsdr' # Pluto, ANTsdr - for MIMO it should ANTsdr
Ntx = 1

delta_t = 1 / sample_rate
delta_f = 30e3 # ???
frac_guard = 0.5

N_sc_av = int(sample_rate / delta_f)
N_fft =  2**(int(np.log2(N_sc_av)))
N_sc_use = int( N_fft * frac_guard)
guard_length = int( 0.5 * N_fft )
CP_len = int(N_fft * 0.2 )
num_bits_sym = 2

BlockSize = N_sc_use * num_bits_sym

do_cfo_corr = 1
do_ce = 1
dc_offset = 0


def create_preamble(N_fft, CP_len, N_repeat = 2):
    preamble = 1 - 2 * np.random.randint(0, 2, size=(int(N_fft/2), 1))
    preamble = np.complex64(preamble)
    preamble_full = np.tile(preamble, (N_repeat, 1))
    preamble_full_cp = np.concatenate((preamble_full[-CP_len:], preamble_full))
    return preamble_full_cp

def create_data(N_sc, N_fft, CP_len, mod_dict_data, dc_offset = False, return_freq_data = False, return_data = False, comb_start=0, comb_step=1):

    if mod_dict_data['num_bit'] == 1:
        data_stream = 1 - 2 * np.random.randint(0, 2, size=(N_sc, 1))
        mod_sym_pilot_whole = np.complex64(data_stream)

        mod_sym_pilot = np.zeros_like(mod_sym_pilot_whole)
        mod_sym_pilot[comb_start::comb_step, :] = mod_sym_pilot_whole[comb_start::comb_step, :]

    else:
        binary_source = mod_dict_data['binary_source']
        mapper = mod_dict_data['mapper']

        bits = binary_source([1, N_sc * mod_dict_data['num_bit']])

        data_stream = bits.numpy().T

        mod_sym_pilot = mapper(bits)

        mod_sym_pilot = mod_sym_pilot.numpy().T

    tx_ofdm_sym = np.zeros((N_fft, 1), dtype = np.complex64)
    dc = int(dc_offset)
    tx_ofdm_sym[dc : N_sc//2 + dc] = mod_sym_pilot[ N_sc//2: ]
    tx_ofdm_sym[-N_sc//2: ] = mod_sym_pilot[ 0 : N_sc//2]

    time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_sym, axis = 0, norm = 'ortho')
    time_ofdm_sym_cp_pilot = np.concatenate( (time_ofdm_sym_pilot[-CP_len:], time_ofdm_sym_pilot) )

    if return_freq_data:
        return time_ofdm_sym_cp_pilot, mod_sym_pilot
    else:
        return time_ofdm_sym_cp_pilot

np.random.seed(123)
tf.random.set_seed(123)

# preambule is the same for both channels
preamble = create_preamble(N_fft, CP_len, 2)

NUM_BITS_PER_SYMBOL = num_bits_sym
binary_source = sn.utils.BinarySource()


if num_bits_sym > 1:

     # The constellation is set to be trainable
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True)
    # Mapper and demapper
    mapper = sn.mapping.Mapper(constellation=constellation)
    demapper = sn.mapping.Demapper("maxlog", constellation=constellation)
else:
    mapper = list()
    demapper = list()


pilot_tx = list()
data_tx = list()
repeated_frame_tx = []


for tx_idx in range(Ntx):

    mod_dict_pilot = dict()
    mod_dict_pilot['num_bit'] = 1

    comb_start = tx_idx
    comb_step = Ntx
    pilot, pilot_freq = create_data(N_sc_use, N_fft, CP_len, mod_dict_pilot, False, True, False, comb_start, comb_step)

    pilot_tx.append(pilot_freq)

    mod_dict_data = dict()
    mod_dict_data['binary_source'] = binary_source
    mod_dict_data['mapper'] = mapper
    mod_dict_data['demapper'] = demapper
    mod_dict_data['num_bit'] = num_bits_sym
    data = create_data(N_sc_use, N_fft, CP_len, mod_dict_data)
    data_tx.append(data)

    guard = np.zeros((guard_length, 1), dtype=np.complex64)
    frame = np.concatenate((preamble, pilot, data, guard))

    repeated_frame = np.tile(frame, reps = (3,1))

    if len(repeated_frame_tx) == 0:
        repeated_frame_tx = repeated_frame
    else:
        repeated_frame_tx = np.hstack((repeated_frame_tx, repeated_frame))


FrameSize = repeated_frame_tx.shape[0]

tx_gain0 = 0
#sdr = adi.Pluto('ip:192.168.3.3') # interfere cfg
if device_type !='ANTsdr':
    sdr = adi.Pluto('ip:192.168.1.1')
    #sdr = adi.ad9363(uri='ip:192.168.1.1')

    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 70.0  # dB
    sdr.rx_lo = int(center_freq)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(sample_rate)  # filter width, just set it to the same as sample rate for now
    sdr.rx_buffer_size = FrameSize

    '''Configure Tx properties'''
    sdr.tx_rf_bandwidth = int(sample_rate)
    sdr.tx_lo = int(center_freq)
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = int(tx_gain0)

    sdr.tx_buffer_size = FrameSize
    sdr.tx(repeated_frame[:, 0] * 10024.0)
else:
    # adi setup
    sdr = adi.ad9363(uri='ip:192.168.1.1')
    samp_rate = sample_rate  # must be <=30.72 MHz if both channels are enabled
    num_samps = len(repeated_frame)  # number of samples per buffer.  Can be different for Rx and Tx
    rx_lo = int(center_freq)
    rx_mode = "slow_attack"  # can be "manual" or "slow_attack"
    rx_gain0 = 70
    rx_gain1 = 70
    tx_lo = rx_lo
    tx_gain0 = 0
    tx_gain1 = 0


    sdr.tx_enabled_channels = list(range(Ntx))
    #sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain0)
    sdr.rx_hardwaregain_chan1 = int(rx_gain1)
    sdr.rx_buffer_size = int(num_samps)
    # end adi setup

    sdr.tx_rf_bandwidth = int(sample_rate)
    sdr.tx_lo = int(center_freq)
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = int(tx_gain0)
    sdr.tx_hardwaregain_chan1 = int(tx_gain1)

    sdr.tx_buffer_size = FrameSize

    if Ntx > 1:
        #repeated_frame_tx = np.tile(repeated_frame, reps = (1,Ntx))

        #repeated_frame_tx = repeated_frame_tx.T

        repeated_frame_tx = repeated_frame_tx.T
        #repeated_frame_tx[:, 1] = 0.0
    else:
        repeated_frame_tx = repeated_frame[:, 0]


    sdr.tx(repeated_frame_tx * 10024.0)

# data = np.arange(1, 10, 3)
# Send
# sdr.tx(data)
print('Success tx : enter if exit')
try:
    ue_in = input()
    if ue_in == 'E' and ue_in == 'e':
        raise KeyboardInterrupt()
except KeyboardInterrupt:
    sdr.tx_destroy_buffer()
    exit(1)
    ff = 1