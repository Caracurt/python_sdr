# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 21:52:51 2023

@author: trefi
"""

import adi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy.matlib

# test mat load

# pluto_2000000000.mat
if 0:
    path_load = f'C:\dev\git_tutor\python_sdr\pluto_2000000000.mat'
    print(path_load)
    mat = scipy.io.loadmat(path_load)
    data = mat['data']
    print(len(data[0]), len(data[1]))

# load_idx = 1, 2
load_idx = 2
path_load = f'C:\dev\git_tutor\python_sdr\pluto_{load_idx}.mat'

# receiver params
do_cfo_corr = 1
do_ce = 1
SNR = 5.0

# fullIRC
do_full_irc = 0
do_rand_proj = 0

use_sdr = 1

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
delta_t = 1 / sample_rate;

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


data_stream = rng.randint(0, 2, size=(int(BlockSize), 1));
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

FrameSize = full_frame_rep.shape[0];
sig_len = full_frame_rep.shape[0];

plt.figure(1)
x_arr = np.arange(0, FrameSize)
plt.figure(1)
plt.plot(x_arr, np.real(full_frame_rep))
plt.grid()
#plt.show()


if 0:
    sdr.tx_buffer_size = FrameSize
    sdr.tx(full_frame_rep[:,0] * 10024.0)
    
    #data = np.arange(1, 10, 3)
    # Send
    #sdr.tx(data)
    print('Success tx')
    ff = 1


# dummy receiver
if 1:
    
    # configure RxSDR
    # Create radio
    if 0:
        sdr = adi.ad9361(uri='ip:192.168.2.1')
        samp_rate = sample_rate    # must be <=30.72 MHz if both channels are enabled
        num_samps = FrameSize      # number of samples per buffer.  Can be different for Rx and Tx
        rx_lo = int(center_freq)
        rx_mode = "slow_attack"  # can be "manual" or "slow_attack"
        rx_gain0 = 40
        rx_gain1 = 40
        tx_lo = rx_lo
        tx_gain0 = -10
        tx_gain1 = -10
        
        '''Configure Rx properties'''
        sdr.rx_enabled_channels = [0, 1]
        sdr.sample_rate = int(samp_rate)
        sdr.rx_lo = int(rx_lo)
        sdr.gain_control_mode = rx_mode
        sdr.rx_hardwaregain_chan0 = int(rx_gain0)
        #sdr.rx_hardwaregain_chan1 = int(rx_gain1)
        sdr.rx_buffer_size = int(num_samps)
    
    
    ts = 1 / float(sample_rate)
    
    data = sdr.rx()
    
    
    # save log in matfile
    #path_save = f'C:\dev\git_tutor\python_sdr\pluto_{int(center_freq)}.mat'
    #print(path_save)    
    #scipy.io.savemat(path_save, {'data': data})    
    #mat = scipy.io.loadmat('file.mat')
    
    
    print(path_load)
    mat = scipy.io.loadmat(path_load)
    data = mat['data']
    

    Rx_0 = data[0]
    Rx_1 = data[1]
    
    # measure power for each Rx    
    
    # process covariance matrix
    R = data @ np.conj(data).T
    R = R / data.shape[1]
    
    print(f'powRx1={np.abs(R[0,0])}, powRx2={np.abs(R[1,1])}')
    
    [u1, s1, v1] = np.linalg.svd(R)
    
    Rx_total = Rx_0 + Rx_1
    Rx_total = Rx_0
        
    
    if do_full_irc:
        u_main = u1[:, 0]
        
        if do_rand_proj:
            
            u_main = rng.randn(2, 1) + 1.0 * 1j*rng.randn(2, 1)
            u_main = u_main[:,0]
        
        data_eig = np.conj(u_main).T @ data
    
        Rx_total = data_eig
    
    NumSamples = len(Rx_total)
    win = np.hamming(NumSamples)
    y = Rx_total * win
    sp = np.absolute(np.fft.fft(y))
    sp = sp[1:-1]
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / (np.sum(win)/2)    # Scale FFT by window and /2 since we are using half the FFT spectrum
    s_dbfs = 20*np.log10(s_mag/(2**12))     # Pluto is a 12 bit ADC, so use that to convert to dBFS
    xf = np.fft.fftfreq(NumSamples, ts)
    xf = np.fft.fftshift(xf[1:-1])/1e6
    plt.figure(10)
    plt.plot(xf, s_dbfs)
    plt.xlabel("frequency [MHz]")
    plt.ylabel("dBfs")
    #plt.draw()
    plt.show()
    
    
    if 1:
                                
        L = int(CP_len/8) # channel impulse response        
        
        rx_sig = data
        L_tot = rx_sig.shape[1]
        N_rx = rx_sig.shape[0]
        
            
        x_arr = np.arange(0, L_tot)
        if 0:
            plt.figure(2)
            plt.plot(x_arr[:], np.real(rx_sig[:, 0]))
            plt.grid()
        
        # find start position of frame
        #start_idx = int(frame_len/4) + rng.randint(0, frame_len/2, 1)[0]
        
        start_idx = 0
        
        corr_list = []
        for idx_search in range(start_idx, start_idx + frame_len):
            
            first_part = rx_sig[:, idx_search : idx_search + pream_len]
            second_part = rx_sig[:, idx_search + pream_len : idx_search + 2*pream_len]
            
            corr_now_mx = np.dot( np.conj(first_part), np.transpose(second_part) )
            corr_now = corr_now_mx[0][0] + corr_now_mx[1][1]
            
            
            corr_list.append(corr_now)
            
        x_arr_corr = np.arange(0, len(corr_list))
        plt.figure(3)
        plt.plot(x_arr_corr, np.array(np.abs(corr_list)))
        plt.grid()
        
        rel_idx = np.argmax(np.abs(corr_list), axis=0)
        idx_max = rel_idx + start_idx
        
        angle_cfo = np.angle(corr_list[rel_idx]) / pream_len
        #print(f'MaxCorrIdx={idx_max} CFOidl={cfo_idl[0]} CFOest={angle_cfo}')
        print(f'MaxCorrIdx={idx_max} CFOest={angle_cfo}')
        
        frame_receive = rx_sig[:, idx_max : idx_max + frame_len]
        
        # apply deCFO
        if do_cfo_corr:
            cfo_comp_sig = np.exp(1j * (-angle_cfo * np.arange(0, frame_len)) )
            cfo_mat = np.matlib.repmat(cfo_comp_sig, 2, 1)
            frame_receive = frame_receive * cfo_mat
        
        #frame_receive = frame_receive[0, :]
        
        plt.figure(4)
        x_arr = np.arange(0, frame_len)
        plt.plot(x_arr, np.real(frame_receive[0, :]))
        plt.grid()
            
        
        h_ls_fin = np.zeros((N_rx, N_sc_use), dtype=np.complex64)
        
        for rx_idx in np.arange(0, N_rx):
            
            pilot_receive = frame_receive[rx_idx, N_fft + CP_len : 2*N_fft + CP_len]
            
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
            
            print(f'RxIdx={rx_idx} SNRest={SNR_est}')
            
            plt.figure(6)
            x_arr = np.arange(0, N_sc_use)
            plt.plot(x_arr, np.real(h_ls))
            plt.grid()
            
            h_ls_fin[rx_idx, :] = h_ls
        
        rec_data_sym = frame_receive[:, 2 * (N_fft + CP_len) : 2 * (N_fft + CP_len) + N_fft]
        
        rec_data_freq = np.zeros((N_rx, N_sc_use), dtype=np.complex64)
        
        for rx_idx in np.arange(0, N_rx):
                
            rec_data_sym_freq = np.fft.fft(rec_data_sym[rx_idx, :], N_fft, 0, norm="ortho")
            
            plt.figure(7)
            x_arr = np.arange(0, N_fft)
            plt.plot(x_arr, np.real(rec_data_sym_freq))
            plt.grid()
            
            rec_sym_data = np.zeros_like(mod_sym)
            rec_sym_data[int(N_sc_use/2):, 0] = rec_data_sym_freq[0+do_dc_offset:int(N_sc_use/2) + do_dc_offset]
            rec_sym_data[0:int(N_sc_use/2), 0] = rec_data_sym_freq[-int(N_sc_use/2):]
            
            #rec_sym_data
            rec_data_freq[rx_idx, :] = rec_sym_data.flatten()
            
            #for i in range(0, N_sc_use):
                #rec_data_freq[rx_idx, i] = rec_sym_data[i, 0]
            

        # BPSK demodulation
        bit_arr = []    
        eq_data = np.zeros(N_sc_use)
        
        norm_ = np.linalg.norm(h_ls_fin, ord=2, axis=-2, keepdims=True)
        w = h_ls_fin / norm_
        w = w.conj().T
        w = w[:, None, :]

        rec_data_freq = rec_data_freq.T
        rec_data_freq = rec_data_freq[:, :, None]
        
        eq_data = (w @ rec_data_freq)[:, :,0]
        bit_arr = np.ones_like(eq_data)
        bit_arr[eq_data > 0] = 0
        
        
        
        ber = np.mean(bit_arr != data_stream )
        aa1 = bit_arr != data_stream
        print(aa1[:10])
        
        #ber = 1.0 * err_num / N_sc_use
        print(f'BER={ber}')