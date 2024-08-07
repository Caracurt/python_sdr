# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:00:02 2023

@author: trefi
"""

import adi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy.matlib

# receiver params
do_cfo_corr = 1
do_ce = 1
SNR = 5.0

use_sdr = 1
do_tx_only = 1

sample_rate = 20e6 # Hz
center_freq = 2.0e9 # Hz

tti_len = 1024
num_samps = tti_len * 20 # number of samples returned per call to rx()

rng = np.random.RandomState(123)


if use_sdr:
    tx_gain0 = 0
    sdr = adi.Pluto('ip:192.168.2.2')
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 70.0 # dB
    sdr.rx_lo = int(center_freq)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
    sdr.rx_buffer_size = num_samps
    
    '''Configure Tx properties'''
    sdr.tx_rf_bandwidth = int(sample_rate)
    sdr.tx_lo = int(center_freq)
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = int(tx_gain0)
    #sdr.tx_hardwaregain_chan1 = int(tx_gain1)
    

# OFDM sig parameters
delta_t = 1 / sample_rate

delta_f = 30e3
frac_guard = 0.5

N_sc_av = int( np.fix( sample_rate / delta_f ) )
N_fft = int( 2**( np.fix(np.log2(N_sc_av)) ) )
N_sc_use = int( np.fix(N_fft * frac_guard) )
guard_length = int( np.fix( 0.5 * N_fft ) )

CP_len = int( np.fix( N_fft * 0.2 ) )
num_bits_sym = 1
num_bits_sym_d = 1
do_dc_offset = 0

BlockSize = N_sc_use * num_bits_sym
BlockSize_d = N_sc_use * num_bits_sym_d

N_repeat = 10
num_rep_pilot = 1
signal_scale = 1.0

 
preamble = 1 - 2 * rng.randint(0, 2, size=(int(N_fft/2), 1))
preamble = np.complex64(preamble)
preamble_full = np.matlib.repmat(preamble, 2, 1)
preamble_full_cp = np.concatenate((preamble_full[-CP_len:], preamble_full))

pream_len = preamble.shape[0]


data_stream = rng.randint(0, 2, size=(int(BlockSize), 1))
mod_sym_pilot = 1.0 - 2.0 * np.complex64(data_stream)

tx_ofdm_sym = np.zeros((N_fft, 1), dtype=np.complex64);
tx_ofdm_sym[0+do_dc_offset:int(N_sc_use/2) + do_dc_offset] = mod_sym_pilot[int(N_sc_use/2):];
tx_ofdm_sym[-int(N_sc_use/2):] = mod_sym_pilot[0:int(N_sc_use/2)];

#time_ofdm_sym_pilot = ifft(tx_ofdm_sym) * sqrt(N_fft) * signal_scale;
#time_ofdm_sym_cp_pilot = [time_ofdm_sym_pilot(end - CP_len + 1:end); time_ofdm_sym_pilot_rep];

time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_sym, N_fft, 0, norm="ortho")
time_ofdm_sym_cp_pilot = np.concatenate(( time_ofdm_sym_pilot[-CP_len:] , time_ofdm_sym_pilot))


# data frame
#data_stream = randi([0, 1], BlockSize_d, 1);
data_stream = rng.randint(0, 2, size=(BlockSize_d, 1))
mod_sym = 1 - 2 * data_stream
mod_sym = np.complex64(mod_sym)


#tx_ofdm_sym = zeros(N_fft, 1);
tx_ofdm_sym = np.zeros((N_fft, 1), dtype=np.complex64)
tx_ofdm_sym[0+do_dc_offset:int(N_sc_use/2) + do_dc_offset] = mod_sym[int(N_sc_use/2):]
tx_ofdm_sym[-int(N_sc_use/2):] = mod_sym[0:int(N_sc_use/2)]

time_ofdm_sym = np.fft.ifft(tx_ofdm_sym, N_fft, 0, norm="ortho")

time_ofdm_sym_cp = np.concatenate((time_ofdm_sym[-CP_len:], time_ofdm_sym))

guard_frame = np.zeros((guard_length, 1), dtype=np.complex64)

#full_frame = [preamble_full_cp; time_ofdm_sym_cp_pilot; time_ofdm_sym_cp; guard_frame];
full_frame = np.concatenate((preamble_full_cp, time_ofdm_sym_cp_pilot, time_ofdm_sym_cp, guard_frame))

frame_len = full_frame.shape[0]
print(f'FrameLen={frame_len}')

num_rep = 3
full_frame_rep = np.matlib.repmat(full_frame, num_rep, 1)

FrameSize = full_frame_rep.shape[0]
sig_len = full_frame_rep.shape[0]

plt.figure(1)
x_arr = np.arange(0, FrameSize)
plt.figure(1)
plt.plot(x_arr, np.real(full_frame_rep))
plt.grid()
#plt.show()


if use_sdr:
    sdr.tx_buffer_size = FrameSize
    sdr.tx(full_frame_rep[:,0] * 10024.0)
    
    #data = np.arange(1, 10, 3)
    # Send
    #sdr.tx(data)
    print('Success tx : enter if exit')
    try:
        ue_in = input()
        if ue_in == 'E' and ue_in == 'e':
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        exit(1)
        ff = 1


# dummy receiver
if 0:
    
    tx_sig = full_frame_rep[:,0];
    
    L = int(CP_len/8) # channel impulse response
    #L = 1
    h_time = rng.randn(L, 1) + 1j * rng.randn(L, 1);
    h_time = h_time[:,0]
    
    rx_sig = np.convolve(tx_sig, h_time, mode='full')
    rx_sig = np.complex64(rx_sig)
    
    # generate noise    
    Es = np.real(np.dot( np.conj(rx_sig) , rx_sig )) / rx_sig.shape[0]
    sigma0 = Es * 10**( -SNR / 10.0 )
    print(f'sigma0={sigma0} Es={Es}')
    L_tot = len(rx_sig)
    
    print(f'TxLen={FrameSize} afterConvolve={L_tot} ChanLen={L}')
    noise_vect = np.sqrt(sigma0) * ( rng.randn(L_tot, 1) + 1j * rng.randn(L_tot, 1) )
    noise_vect = np.complex64(noise_vect[:, 0])
    
    # add awgn
    rx_sig = rx_sig + noise_vect;
    
    # add carrier frequency offset
    cfo_idl_max = 1000e-6;
    
    cfo_idl = cfo_idl_max * (1.00 + rng.rand(1, 1)[0])
          
    x_arr1 = np.arange(0, L_tot)
    cfo_sig = np.exp(1j * cfo_idl * x_arr1)
    
    # add cfo signal
    rx_sig = rx_sig * cfo_sig
        
    x_arr = np.arange(0, L_tot)
    plt.figure(2)
    plt.plot(x_arr[:], np.real(rx_sig[:]))
    plt.grid()
    
    # find start position of frame
    start_idx = int(frame_len/4) + rng.randint(0, frame_len/2, 1)[0]
    
    #start_idx = 0
    
    corr_list = []
    for idx_search in range(start_idx, start_idx + frame_len):
        
        first_part = rx_sig[idx_search : idx_search + pream_len]
        second_part = rx_sig[idx_search + pream_len : idx_search + 2*pream_len]
        
        corr_now = np.dot( np.conj(first_part), second_part)        
        corr_list.append(corr_now)
        
    x_arr_corr = np.arange(0, len(corr_list))
    plt.figure(3)
    plt.plot(x_arr_corr, np.array(corr_list))
    plt.grid()
    
    rel_idx = np.argmax(np.abs(corr_list), axis=0)
    idx_max = rel_idx + start_idx
    
    angle_cfo = np.angle(corr_list[rel_idx]) / pream_len
    print(f'MaxCorrIdx={idx_max} CFOidl={cfo_idl[0]} CFOest={angle_cfo}')
    
    frame_receive = rx_sig[idx_max : idx_max + frame_len]
    
    # apply deCFO
    if do_cfo_corr:
        cfo_comp_sig = np.exp(1j * (-angle_cfo * np.arange(0, frame_len)) )
        frame_receive = frame_receive * cfo_comp_sig
    
    plt.figure(4)
    x_arr = np.arange(0, frame_len)
    plt.plot(x_arr, np.real(frame_receive))
    plt.grid()
        
        
    pilot_receive = frame_receive[N_fft + CP_len : 2*N_fft + CP_len]
    
    pilot_freq = np.fft.fft(pilot_receive, N_fft, 0, norm="ortho")
    
    plt.figure(5)
    x_arr = np.arange(0, N_fft)
    plt.plot(x_arr, np.real(pilot_freq))
    plt.grid()
    
    rec_sym_pilot = np.zeros_like(mod_sym_pilot)
    rec_sym_pilot[int(N_sc_use/2):, 0] = pilot_freq[0+do_dc_offset:int(N_sc_use/2) + do_dc_offset]
    rec_sym_pilot[0:int(N_sc_use/2), 0] = pilot_freq[-int(N_sc_use/2):]
    
    h_ls = rec_sym_pilot[:, 0] / mod_sym_pilot[:, 0]
    
    ce_len = len(h_ls)
    
    # apply CE
    h_time = np.fft.ifft(h_ls, N_fft, 0, norm='ortho')
    
    plt.figure(100)
    plt.plot(np.arange(0, N_fft), np.abs(h_time))
    plt.grid()
    
    W_spead = int(CP_len/8)
    W_sync_err = int(CP_len)
    W_max = W_spead + W_sync_err
    W_min = W_sync_err
    
    eta_denoise = np.zeros_like(h_time)
    eta_denoise[-W_min:] = 1.0 
    eta_denoise[0:W_max] = 1.0
    
    print(f'Wmax={W_max} Wmin={W_min} CP={CP_len}')
    
    h_time_denoise = h_time * eta_denoise
    #h_time_denoise = h_time
    
    h_hw = np.fft.fft(h_time_denoise, N_fft, 0, norm='ortho')
    if do_ce:
        h_ls = h_hw[0:ce_len]
    
    
    # SNR estimation
    noise_arr = pilot_freq[int(N_sc_use/2) : -int(N_sc_use/2)]
    sigma0_freq = np.real( np.dot( np.conj(noise_arr) , noise_arr ) ) / len(noise_arr)
    
    Es_freq = np.real( np.dot( np.conj(rec_sym_pilot[:,0]) , rec_sym_pilot[:,0] ) ) / len(rec_sym_pilot[:, 0])
    
    SNR_est = 10.0 * np.log10(Es_freq / sigma0_freq)
    
    print(f'SNRest={SNR_est}')
    
    plt.figure(6)
    x_arr = np.arange(0, N_sc_use)
    plt.plot(x_arr, np.real(h_ls))
    plt.grid()
    
    rec_data_sym = frame_receive[2 * (N_fft + CP_len) : 2 * (N_fft + CP_len) + N_fft]
    
    rec_data_sym_freq = np.fft.fft(rec_data_sym, N_fft, 0, norm="ortho")
    
    plt.figure(7)
    x_arr = np.arange(0, N_fft)
    plt.plot(x_arr, np.real(rec_data_sym_freq))
    plt.grid()
    
    rec_sym_data = np.zeros_like(mod_sym)
    rec_sym_data[int(N_sc_use/2):, 0] = rec_data_sym_freq[0+do_dc_offset:int(N_sc_use/2) + do_dc_offset]
    rec_sym_data[0:int(N_sc_use/2), 0] = rec_data_sym_freq[-int(N_sc_use/2):]
    
    # simple ZF equalizer
    eq_data = rec_sym_data[:, 0] / h_ls
    
    
    # BPSK demodulation
    bit_arr = []    
    for idx in range(0, N_sc_use):
        
        if (np.real(eq_data[idx]) > 0):
            bit_curr = 0
        else:
            bit_curr = 1
            
        bit_arr.append(bit_curr)
    
    print(data_stream[:10,0])
    print(bit_arr[:10])
    
    err_num = 0
    
    for idx in range(0, N_sc_use):
        
        if (data_stream[idx, 0] != bit_arr[idx]):
            err_num = err_num + 1
    
    ber = 1.0 * err_num / N_sc_use
    print(f'BER={ber}')
    
    
    

# legacy code to save mat file receive PLUTO
if 0:
    rx_sig = sdr.rx() # receive samples off Pluto
    print(rx_sig.shape)
    print(rx_sig[0:10])
    path_save = f'C:\dev\git_tutor\OFDM_PLUTO\misc\pluto_{int(center_freq)}.mat'
    print(path_save)
    
    scipy.io.savemat(path_save, {'rx_sig': rx_sig})
    
    print(rx_sig[0:10])

