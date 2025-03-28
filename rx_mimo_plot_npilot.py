import re

import adi
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import scipy
import time
import pylab as pl
from IPython import display

import sionna as sn
import tensorflow as tf

#from tx_mimo import create_data

from matplotlib.animation import FuncAnimation

# tested receiver cfgs
#cfg_test =[(4, 'IRC'), (3, 'WMMSE'), (2, 'MMSE'), (1, 'EigRx'), (0, 'SumRx')]
#cfg_test =[(4, 'IRC'), (0, 'MMSE')]
#cfg_test =[(4, 'IRC'), (2, 'MMSE')]
#cfg_test =[(14, 'IRC_SMMSE'), (4, 'IRC'), (12, 'MMSE_SMMSE'), (2, 'MMSE')]
#cfg_test =[(102, 'MMSE_SW_rep1'), (2, 'MMSE_rep1')]
cfg_test =[(2, 'MMSE_rep2'), (2, 'MMSE_rep1')]

Ntx = 1

num_cfg = len(cfg_test)
line_arr = list()
line_snr_arr = list()

num_subplots = 4
fig, ax = plt.subplots(num_subplots)
ax[0].grid() # ber plot
ax[1].grid() # SNR plot
ax[2].grid() # correlation plot
ax[3].grid() # throughput plot

colors = ['r-', 'g-', 'b-', 'c-', 'm-', 'y-']
for idx, (mimo_mode, name_mimo) in enumerate(cfg_test):
    #line_c, = ax.plot([], [], colors[idx], label=name_mimo)
    line_c, = ax[0].semilogy([], [], colors[idx], label=name_mimo)
    line_arr.append(line_c)

    line_c, = ax[1].plot([], [], colors[idx], label=name_mimo)
    line_arr.append(line_c)

    line_c, = ax[2].plot([], [], colors[idx], label=name_mimo)
    line_arr.append(line_c)

    line_c, = ax[3].plot([], [], colors[idx], label=name_mimo)
    line_arr.append(line_c)

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()

################## INIT START
sample_rate = 5e6
center_freq = 2.0e9

delta_t = 1 / sample_rate
delta_f = 30e3
frac_guard = 0.5

N_sc_av = int(sample_rate / delta_f)
N_fft = 2 ** (int(np.log2(N_sc_av)))
N_sc_use = int(N_fft * frac_guard)
guard_length = int(0.5 * N_fft)
CP_len = int(N_fft * 0.2)
num_bits_sym = 2

BlockSize = N_sc_use * num_bits_sym

Thr_max = (N_sc_use * num_bits_sym) * (N_sc_use / N_fft) * delta_f / 1e3 # kbit/sec

do_cfo_corr = 1
do_ce = 1
dc_offset = 0

# repeatition of pilot feature
pilot_repeat = 2

do_load_file = False
do_save = False

# init Thr
alpha_avg = 0.1
Thr_dict = dict()
for idx, (mimo_mode, name_mimo) in enumerate(cfg_test):
    Thr_dict[name_mimo] = 0.0

def create_preamble(N_fft, CP_len, N_repeat=2):
    preamble = 1 - 2 * np.random.randint(0, 2, size=(int(N_fft / 2), 1))
    preamble = np.complex64(preamble)
    preamble_full = np.tile(preamble, (N_repeat, 1))
    preamble_full_cp = np.concatenate((preamble_full[-CP_len:], preamble_full))
    return preamble_full_cp, preamble


##############
def create_data(N_sc, N_fft, CP_len, mod_dict_data, dc_offset = False, return_freq_data = False, aditional_return = False, comb_start=0, comb_step=1):

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

    if not aditional_return:
        return time_ofdm_sym_cp_pilot, mod_sym_pilot
    else:
        return time_ofdm_sym_cp_pilot, mod_sym_pilot, data_stream


#############################################################################
def find_edges(rx_sig, frame_len, preamble_len, CP_len, preamble_core, start_idx):
    corr_list = []

    if False:
        ax1 = plt.gca()
        ax1.plot(range(rx_sig.shape[1]), np.abs(rx_sig[0, :]))
        ax1.grid()
        plt.show()

    for idx_search in range(start_idx, start_idx + frame_len):
        corr_fin = 0.0
        for rx_idx in range(rx_sig.shape[0]):
            first_part = rx_sig[rx_idx, idx_search: idx_search + preamble_len]
            second_part = rx_sig[rx_idx, idx_search + preamble_len: idx_search + 2 * preamble_len]

            if True:
                first_part = first_part * np.conj(preamble_core[:, 0])
                second_part = second_part * np.conj(preamble_core[:, 0])

            corr_now = np.dot(np.conj(first_part), second_part)

            corr_fin = corr_fin + corr_now

        corr_list.append(corr_fin)

    corr_list_new = np.abs(np.array(corr_list))
    rel_idx = np.argmax(np.abs(corr_list_new), axis=0)
    idx_max = rel_idx + start_idx



    filter_cp = np.ones(CP_len) / np.sqrt(CP_len)
    corr_list_filt = np.convolve(filter_cp, corr_list_new, mode='same')

    rel_idx_filt = np.argmax(np.abs(corr_list_filt), axis=0)
    idx_max_filt = rel_idx_filt + start_idx
    prot_shift = 10 # heuristic value
    idx_max_filt = idx_max_filt + int(CP_len/2) - prot_shift

    if False:
        plt.figure(10)
        ax_corr = plt.gca()
        ax_corr.plot(range(len(corr_list)), np.array(corr_list_new), label='Orig')
        ax_corr.plot(range(len(corr_list)), corr_list_filt, label='filt')
        ax_corr.grid()
        plt.legend()

    # estimation of SNR
    frame_tmp = rx_sig[:, idx_max_filt : idx_max_filt + frame_len]

    frame_tmp_noise = frame_tmp[:, -2*CP_len:-CP_len] # save region of pure noise

    sigma_arr = np.diag(frame_tmp_noise @ frame_tmp_noise.conj().T) / frame_tmp_noise.shape[1]
    frame_preamb =frame_tmp[:, :2*preamble_len]
    Es_arr = np.diag(frame_preamb @ frame_preamb.conj().T) / frame_preamb.shape[1]

    SNR_guard = 10.0 * np.log10( np.abs(Es_arr) / np.abs(sigma_arr)  )

    #print(f'Guard SNR={SNR_guard} sigma_arr={sigma_arr}')

    if False:
        ax1 = plt.gca()
        for rx_idx in range(frame_tmp.shape[0]):
            ax1.plot(range(frame_tmp.shape[1]), np.abs(frame_tmp[rx_idx, :]))
            ax1.grid()

            ax2 = plt.gca()
            ax2.plot(range(frame_tmp_noise.shape[1]), np.abs(frame_tmp_noise[rx_idx, :]))
            ax2.grid()

    #plt.show()

    #plt.show()

    if True:
        idx_max = idx_max_filt
        rel_idx = rel_idx_filt

    SNR_mean = np.mean(SNR_guard)

    return idx_max, corr_list[rel_idx], sigma_arr, SNR_mean


def cfo(frame_receive, corr_value, preamble_len):
    angle_cfo = np.angle(corr_value) / preamble_len
    cfo_comp_sig = np.exp(1j * (-1.0*angle_cfo * np.arange(0, frame_len)))
    frame_receive = frame_receive * cfo_comp_sig
    return frame_receive


def baseband_freq_domian(pilot_freq, N_sc_use):
    rec_sym_pilot = np.zeros((N_sc_use, 1), dtype=np.complex64)
    rec_sym_pilot[int(N_sc_use / 2):, 0] = pilot_freq[0 + dc_offset:int(N_sc_use / 2) + dc_offset]
    rec_sym_pilot[0:int(N_sc_use / 2), 0] = pilot_freq[-int(N_sc_use / 2):]

    return rec_sym_pilot


def channel_estimation(h_ls, CP_len, N_fft, comb_step=1, sigma_0=0.0, ce_mode=0):
    h_time = np.fft.ifft(h_ls, N_fft, 0, norm='ortho')

    if ce_mode == 1:
        h_time_denoise = np.zeros_like(h_time)

        pos_high = np.where(np.abs(h_time)**2 > sigma_0)

        if len(pos_high) > 0:
            eta_sw = ( np.abs(h_time[pos_high])**2 - sigma_0 ) / np.abs(h_time[pos_high])**2

            h_time_denoise[pos_high] = eta_sw * h_time[pos_high]

            h_time = h_time_denoise



    ce_len = len(h_ls) * comb_step

    W_spead = int(CP_len / 2 / comb_step)
    W_sync_err = int(CP_len)
    W_max = W_spead + W_sync_err
    W_min = W_sync_err

    eta_denoise = np.zeros_like(h_time)
    eta_denoise[-W_min:] = 1.0
    eta_denoise[0:W_max] = 1.0

    h_time_denoise = h_time * eta_denoise

    # zero padding
    h_time_denoise_us = np.hstack((h_time_denoise[:int(N_fft/2)], np.zeros((comb_step-1)*N_fft), h_time_denoise[int(N_fft/2):]))

    h_hw = np.sqrt(comb_step) * np.fft.fft(h_time_denoise_us, N_fft * comb_step, 0, norm='ortho')
    h_ls = h_hw[0:ce_len]

    if False:
        display.clear_output(wait=True)
        plt.figure(100)
        plt.plot(np.arange(0, N_fft), np.abs(h_time))
        plt.plot(np.arange(0, N_fft), np.abs(h_time_denoise))
        plt.title('Channel response time domain')
        plt.grid();



    return h_ls


def estimate_SNR(pilot_freq, rec_sym_pilot, N_sc_use):
    Es_freq = 0.0
    sigma0_freq = 0.0
    for rx_idx in range(pilot_freq.shape[0]):
        noise_arr = pilot_freq[rx_idx, int(N_sc_use / 2): -int(N_sc_use / 2)]
        sigma0_freq = sigma0_freq + np.real(np.dot(np.conj(noise_arr), noise_arr)) / len(noise_arr)
        Es_freq = Es_freq + np.real(np.dot(np.conj(rec_sym_pilot[:, rx_idx]), rec_sym_pilot[:, rx_idx])) / len(rec_sym_pilot[:, rx_idx])

    SNR_est = 10.0 * np.log10(Es_freq / sigma0_freq)
    return SNR_est


def demodulate(eq_data, N_sc_use, mod_dict):
    bit_arr = []

    if mod_dict_data['num_bit'] == 1:
        for idx in range(0, N_sc_use):

            if (np.real(eq_data[idx]) > 0):
                bit_curr = 0
            else:
                bit_curr = 1

            bit_arr.append(bit_curr)

        bit_arr = np.array(bit_arr)
        bit_arr = bit_arr[..., np.newaxis]
    else:
        eq_data_n = eq_data[..., np.newaxis]
        eq_data_t = tf.cast(eq_data_n.T, np.complex64)

        no = 1.0
        llr_tf = demapper([eq_data_t, no])
        llr = llr_tf.numpy()
        bit_arr_t = np.zeros_like(llr)
        bit_arr_t[llr > 0] = 1

        bit_arr = bit_arr_t.T

    return bit_arr


def get_ber(data_stream, bit_arr, N_sc_use):
    err_num = 0

    if False:
        for idx in range(0, N_sc_use):
            if (data_stream[idx, 0] != bit_arr[idx]):
                err_num = err_num + 1
        ber = 1.0 * err_num / N_sc_use

    ber = np.sum(data_stream != bit_arr) / len(data_stream)

    return ber


#############################################################################
# Try Sionna
NUM_BITS_PER_SYMBOL = num_bits_sym

binary_source = sn.utils.BinarySource()

if num_bits_sym > 1:
    constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True) # The constellation is set to be trainable

    # Mapper and demapper
    mapper = sn.mapping.Mapper(constellation=constellation)
    demapper = sn.mapping.Demapper("maxlog", constellation=constellation)
else:
    mapper = list()
    demapper = list()

np.random.seed(123)
tf.random.set_seed(123)
preamble, preamble_core = create_preamble(N_fft, CP_len, 2)

pilot_tx = list()
data_tx = list()
bite_stream_tx = list()
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
    data, mod_data, bite_stream = create_data(N_sc_use, N_fft, CP_len, mod_dict_data, False, True, True)
    data_tx.append(data)
    bite_stream_tx.append(bite_stream)

    guard = np.zeros((guard_length, 1), dtype=np.complex64)

    pilot_rep = np.tile(pilot, reps=(pilot_repeat, 1))

    frame = np.concatenate((preamble, pilot_rep, data, guard))

    repeated_frame = np.tile(frame, reps = (3,1))

    if len(repeated_frame_tx) == 0:
        repeated_frame_tx = repeated_frame
    else:
        repeated_frame_tx = np.hstack((repeated_frame_tx, repeated_frame))


FrameSize = repeated_frame_tx.shape[0]



#guard = np.zeros((guard_length, 1), dtype=np.complex64)
#frame = np.concatenate((preamble, pilot, data, guard))

#repeated_frame = np.tile(frame, reps=(3, 1))

frame_len = len(frame)
preamble_len = N_fft // 2

#sdr = adi.Pluto(uri='ip:192.168.1.1')
#sdr = adi.ad9361(uri='ip:192.168.1.1')
if not do_load_file:
    sdr = adi.ad9361(uri='ip:192.168.2.2')
    samp_rate = sample_rate  # must be <=30.72 MHz if both channels are enabled
    num_samps = len(repeated_frame)  # number of samples per buffer.  Can be different for Rx and Tx
    rx_lo = int(center_freq)
    rx_mode = "slow_attack"  # can be "manual" or "slow_attack"
    rx_gain0 = 70
    rx_gain1 = 70
    tx_lo = rx_lo
    tx_gain0 = -10
    tx_gain1 = -10

    sdr.rx_enabled_channels = [0, 1]
    #sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(samp_rate)
    sdr.rx_lo = int(rx_lo)
    sdr.gain_control_mode = rx_mode
    sdr.rx_hardwaregain_chan0 = int(rx_gain0)
    sdr.rx_hardwaregain_chan1 = int(rx_gain1)
    sdr.rx_buffer_size = int(num_samps)

#### INIT part is finished
def receiver_MIMO(data, mimo_mode_in, iNtx, pilot_rep_use=1):
    data = np.array(data)

    # parse mimo_mode
    mimo_mode = mimo_mode_in % 10 # extract MIMO detection
    smmse_mode = int(mimo_mode_in / 10) % 10 # extract SMME mode
    ce_mode = int(mimo_mode_in / 100) % 10 # CE mode

    if mimo_mode == 0:
        rx_sig = np.zeros((1, data.shape[1]), dtype=np.complex64)
        rx_sig[0, :] = data[0, :] + data[1, :]

    elif mimo_mode == 1:

        R = data @ data.conj().T
        U1, s1, V1 = np.linalg.svd(R)
        U_main = U1[:, 0:1]
        rx_sig = U_main.conj().T @ data

        # rx_sig = rx_sig[0]

    elif (mimo_mode == 2) or (mimo_mode == 3) or (mimo_mode == 4):

        rx_sig = data

    # rx_sig = data
    num_rx, rx_len = rx_sig.shape

    idx_max, corr_value, sigma_arr, SNR_guard = find_edges(rx_sig, frame_len, preamble_len, CP_len, preamble_core, start_idx=0)
    frame_receive = rx_sig[:, idx_max: idx_max + frame_len]

    frame_receive = cfo(frame_receive, corr_value, preamble_len) if do_cfo_corr else frame_receive

    pilot_freq_rep = np.zeros( (pilot_repeat, num_rx, N_fft), dtype=np.complex64 )

    frame_cp_len = N_fft + CP_len
    start_time = N_fft + CP_len
    for rep_idx in range(pilot_repeat):
        pilot_receive = frame_receive[:, start_time + (rep_idx)* frame_cp_len : start_time + (rep_idx)* frame_cp_len + N_fft]
        pilot_freq = np.fft.fft(pilot_receive, N_fft, axis=1, norm="ortho")

        pilot_freq_rep[rep_idx, :, :] = pilot_freq


    h_ls_all = np.zeros((iNtx, num_rx, N_sc_use), dtype=np.complex64)
    rec_sym_pilot_all = np.zeros((pilot_repeat, num_rx, N_sc_use), dtype=np.complex64)

    Ruu = np.zeros((num_rx, num_rx), dtype=np.complex64)
    for tx_idx in range(iNtx):

        comb_start = tx_idx
        comb_step = iNtx

        ls_len = int(N_sc_use/comb_step)

        for rx_idx in range(num_rx):

            h_ls_rep = np.zeros( (pilot_repeat, num_rx, ls_len), dtype=np.complex64 )
            for rep_idx in range(pilot_repeat):

                rec_sym_pilot = baseband_freq_domian(pilot_freq_rep[rep_idx, rx_idx, :], N_sc_use)
                rec_sym_pilot_all[rep_idx, rx_idx, :] = rec_sym_pilot[:, 0]

                # apply CE
                h_ls = rec_sym_pilot[comb_start::comb_step, 0] / pilot_tx[tx_idx][comb_start::comb_step, 0]

                h_ls_rep[rep_idx, rx_idx, :] = h_ls

            # LS average
            h_ls = h_ls_rep[0, rx_idx, :]
            if True:
                for rep_idx in range(1, pilot_rep_use):
                    h_ls_c = h_ls_rep[rep_idx, rx_idx, :]
                    h_ls = h_ls + h_ls_c
                h_ls = h_ls / pilot_rep_use



            h_ls_all[tx_idx, rx_idx, :] = channel_estimation(h_ls, CP_len, N_fft, comb_step, sigma_arr[rx_idx], ce_mode) if do_ce else h_ls

        # usamples estimation
        for rep_idx in range(pilot_repeat):
            u_mx = rec_sym_pilot_all[rep_idx, :, comb_start::comb_step] - h_ls_all[tx_idx, :, comb_start::comb_step] * pilot_tx[tx_idx][comb_start::comb_step, 0]
            Ruu_c = u_mx @ u_mx.conj().T / u_mx.shape[1]


        Ruu = Ruu + (Ruu_c / pilot_repeat)



        #print(f'S_t={S_t}')

    Ruu = Ruu / iNtx
    U_t, S_t, v_t = np.linalg.svd(Ruu)
    Ruu_inv = np.linalg.inv(Ruu)

    # receive baseband frame
    rec_data_sym = frame_receive[:, (1 + pilot_repeat) * (N_fft + CP_len): (1 + pilot_repeat) * (N_fft + CP_len) + N_fft]


    #rec_data_sym = frame_receive[:, 2 * (N_fft + CP_len): 2 * (N_fft + CP_len) + N_fft]
    rec_data_sym_freq = np.fft.fft(rec_data_sym, N_fft, axis=1, norm="ortho")

    rec_sym_data_all = np.zeros((num_rx, N_sc_use), dtype=np.complex64)

    rho_su = 1.0 # init correlation between SU weights

    for rx_idx in range(num_rx):
        rx_tmp = baseband_freq_domian(rec_data_sym_freq[rx_idx, :], N_sc_use)
        rec_sym_data_all[rx_idx, :] = rx_tmp[:, 0]

    if mimo_mode == 0 or mimo_mode == 1:
        eq_data = rec_sym_data_all[0, :] / h_ls_all[0, :]

    else:
        eq_data = np.zeros( (iNtx, N_sc_use), dtype=np.complex64)
        rho_avg = list()

        # SU weights correlation
        V_su = np.zeros((num_rx, iNtx), dtype=np.complex64)
        for tx_idx in range(iNtx):
            H_c = h_ls_all[tx_idx, :, :]
            R_hh = H_c @ H_c.conj().T / H_c.shape[1]
            U_c, S_c, V_c = np.linalg.svd(R_hh)

            V_su[:, tx_idx] = U_c[:, 0]

            # SMME start
            if (smmse_mode > 0):
                alpha_ruu = 0.3
                W_smmse = R_hh @ np.linalg.inv(R_hh + alpha_ruu * Ruu)

                h_ls_all[tx_idx, :, :] = W_smmse @ h_ls_all[tx_idx, :, :]


        if Ntx > 1:
            R_corr = V_su.conj().T @ V_su
            rho_su = R_corr[0, 1]
        else:
            rho_su = 1.0

        for sc_idx in range(rec_sym_data_all.shape[1]):

            h_c = h_ls_all[:, :, sc_idx]

            h_c = h_c.T

            r_c = rec_sym_data_all[:, sc_idx:sc_idx + 1]
            # MRC
            if mimo_mode == 2:
                x_c = np.linalg.inv(h_c.conj().T @ h_c) @ h_c.conj().T @ r_c
            elif mimo_mode == 3:
                # W-MRC // IRC

                Ruu_d = np.diag(sigma_arr)
                Ruu_d = Ruu_d.astype(np.complex64)
                Ruu_d_inv = np.linalg.inv(Ruu_d)
                x_c = np.linalg.inv(h_c.conj().T @ Ruu_d_inv @ h_c) @ h_c.conj().T @ Ruu_d_inv @ r_c

            elif mimo_mode == 4:
                A = h_c.conj().T @ Ruu_inv @ h_c
                D = np.diag(1.0 / np.sqrt(np.diag(A)))
                A1 = D @ A @ D
                A2 = (np.abs(A1))

                if iNtx > 1:
                    rho_c = A2[0, 1]
                else:
                    rho_c = 1.0

                rho_avg.append(rho_c)

                x_c = np.linalg.inv(h_c.conj().T @ Ruu_inv @ h_c) @ h_c.conj().T @ Ruu_inv @ r_c

            eq_data[:, sc_idx:sc_idx+1] = x_c

    #rho_avg_plot = np.mean(rho_avg)
    rho_avg_plot = rho_su
    #rho_avg_plot = np.max(rho_avg)

    ber_arr = list()
    for tx_idx in range(iNtx):
        bit_arr = demodulate(eq_data[tx_idx, :], N_sc_use, mod_dict_data)
        ber = get_ber(bite_stream_tx[tx_idx], bit_arr, N_sc_use)

        ber_arr.append(ber)

        #SNR_est = estimate_SNR(pilot_freq, rec_sym_pilot_all, N_sc_use)

    return ber_arr, SNR_guard, rho_avg_plot

#### RECEIVER PART start - test reception chain
if not do_load_file:
    data = sdr.rx()

    print(f'SDR reception is OK!!!')

    if do_save:
        #path_save = f'C:\\dev\\git_tutor\\python_sdr\\tmp.mat'
        path_save = f'tmp.mat'
        print(path_save)
        data = np.array(data)
        scipy.io.savemat(path_save, {'data': data})

    print(data[0].shape)
else:
    print(f'Reception from file is OK!!!')

    path_save = f'tmp.mat'
    mat = scipy.io.loadmat(path_save)
    data = mat['data']
    data = np.array(data)

    # test receiver

# FOR DEBUG first frame
if True:
    #while True:
        for mimo, rec_name in cfg_test:

            try:
                pilot_rep_use = int(re.findall('_rep(\d+)', rec_name)[0])
            except:
                pilot_rep_use = 1

            ber_c, snr_c, rho_avg_plot = receiver_MIMO(data, mimo, Ntx, pilot_rep_use)

            print(f'mimo={mimo} rec_name={rec_name}, snr_c={snr_c}')

            for tx_idx in range(Ntx):
                print(f'Layer={tx_idx} ber={ber_c[tx_idx]}')

            time.sleep(2)


def update(frame1):

    ### SDR reception
    if not do_load_file:
        data = sdr.rx()

        if do_save:
            path_save = f'tmp.mat'
            print(path_save)
            data = np.array(data)
            scipy.io.savemat(path_save, {'data': data})
            print(data[0].shape)

    else:
        path_save = f'tmp.mat'
        mat = scipy.io.loadmat(path_save)
        data = mat['data']
        data = np.array(data)


    for idx, (mimo, rec_name) in enumerate(cfg_test):

        ber_c, snr_c, rho_avg_plot = receiver_MIMO(data, mimo, Ntx)

        thr_c =  Thr_max * ( Ntx - np.sum( np.array(ber_c) ))



        x1, y1 = line_arr[num_subplots * idx].get_data()
        #x1, y1 = line_c.get_data()

        x1 = np.append(x1, frame1)
        y1 = np.append(y1, np.mean(ber_c))
        line_arr[num_subplots  * idx].set_data(x1, y1)

        x1, y1 = line_arr[num_subplots * idx + 1].get_data()
        # x1, y1 = line_c.get_data()

        x1 = np.append(x1, frame1)
        y1 = np.append(y1, snr_c)
        line_arr[num_subplots * idx + 1].set_data(x1, y1)

        x1, y1 = line_arr[num_subplots * idx + 2].get_data()
        x1 = np.append(x1, frame1)
        y1 = np.append(y1, rho_avg_plot)
        line_arr[num_subplots * idx + 2].set_data(x1, y1)

        # plot throughput
        x1, y1 = line_arr[num_subplots * idx + 3].get_data()
        x1 = np.append(x1, frame1)

        if len(x1) == 0:
            Thr_dict[rec_name] = thr_c
        else:
            Thr_dict[rec_name] = thr_c * alpha_avg + (1.0 - alpha_avg) * Thr_dict[rec_name]

        y1 = np.append(y1, Thr_dict[rec_name])
        line_arr[num_subplots * idx + 3].set_data(x1, y1)


    return (*line_arr,)


def init():
    ax[0].set_xlabel('Time')
    ax[0].set_xlim(0, 2*np.pi)
    ax[0].set_ylabel(f'BER@QAM{2**NUM_BITS_PER_SYMBOL}')
    ax[0].set_ylim(10 ** (-4), 10 ** (-0))

    ax[1].set_xlabel('Time')
    ax[1].set_ylabel(f'SNRguard')
    ax[1].set_xlim(0, 2 * np.pi)
    ax[1].set_ylim(0, 45)

    ax[2].set_xlabel('Time')
    ax[2].set_ylabel(f'SU correlation')
    ax[2].set_xlim(0, 2 * np.pi)
    ax[2].set_ylim(0.0, 1.0)

    ax[3].set_xlabel('Time')
    ax[3].set_ylabel(f'Throughput')
    ax[3].set_xlim(0, 2 * np.pi)
    ax[3].set_ylim(0.0, 1.2 * Thr_max * Ntx)



    return (*line_arr,)

def main():
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 1024), init_func=init, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
