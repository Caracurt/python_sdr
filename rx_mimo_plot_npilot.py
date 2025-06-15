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

from tx_mimo_npilot import create_data, create_preamble, create_data_frame, init_tx_dict
from system_tx import SysParUL
import json

# tested receiver cfgs
#cfg_test =[(4, 'IRC'), (3, 'WMMSE'), (2, 'MMSE'), (1, 'EigRx'), (0, 'SumRx')]
#cfg_test =[(4, 'IRC'), (0, 'MMSE')]
#cfg_test =[(4, 'IRC'), (2, 'MMSE')]
#cfg_test =[(14, 'IRC_SMMSE'), (4, 'IRC'), (12, 'MMSE_SMMSE'), (2, 'MMSE')]
#cfg_test =[(102, 'MMSE_SW_rep1'), (2, 'MMSE_rep1')]
cfg_test =[(2, 'MMSE_rep4'), (2, 'MMSE_rep2'), (2, 'MMSE_rep1')]



num_cfg = len(cfg_test)
line_arr = list()
line_snr_arr = list()

# set list of metric to plot
NUM_FRAMES = 2* np.pi # length of plot
#metrics_to_plot = ['SNR_guard', 'EVM', 'BER']
metrics_to_plot = ['BER', 'EVM']

num_subplots = len(metrics_to_plot)

fig, ax = plt.subplots(num_subplots)

for p_idx in range(num_subplots):
    ax[p_idx].grid() # ber plot


# define lines on plots per tested CFG
colors = ['r-', 'g-', 'b-', 'c-', 'm-', 'y-']
for idx, (mimo_mode, name_mimo) in enumerate(cfg_test):
    #line_c, = ax.plot([], [], colors[idx], label=name_mimo)
    for p_idx in range(num_subplots):

        if metrics_to_plot[p_idx] == 'BER':
            line_c, = ax[p_idx].semilogy([], [], colors[idx], label=name_mimo)
        else:
            line_c, = ax[p_idx].plot([], [], colors[idx], label=name_mimo)

        line_arr.append(line_c)

# create legend
for p_idx in range(num_subplots):
    ax[p_idx].legend()



# util functions
def calc_evm(dat_idl, dat_est):
    evm_c = 10.0 * np.log10( np.linalg.norm(dat_idl.flatten()))**2 / np.linalg.norm((dat_idl - dat_est).flatten()**2 )
    return evm_c


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

    # debug only
    #idx_max_filt = CP_len # perfect location in case of dummyRx

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

    if pilot_freq.ndim == 1:
        rec_sym_pilot = np.zeros((N_sc_use,), dtype=np.complex64)
        rec_sym_pilot[int(N_sc_use / 2):] = pilot_freq[0 + inPar.dc_offset:int(N_sc_use / 2) + inPar.dc_offset]
        rec_sym_pilot[0:int(N_sc_use / 2)] = pilot_freq[-int(N_sc_use / 2):]
    else:
        rec_sym_pilot = np.zeros((N_sc_use, pilot_freq.shape[1]), dtype=np.complex64)
        rec_sym_pilot[int(N_sc_use / 2):, :] = pilot_freq[0 + inPar.dc_offset:int(N_sc_use / 2) + inPar.dc_offset, :]
        rec_sym_pilot[0:int(N_sc_use / 2), :] = pilot_freq[-int(N_sc_use / 2):, :]

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

    if mod_dict['num_bit'] == 1:
        for idx in range(0, N_sc_use):

            if (np.real(eq_data[idx]) > 0):
                bit_curr = 0
            else:
                bit_curr = 1

            bit_arr.append(bit_curr)

        bit_arr = np.array(bit_arr)
        bit_arr = bit_arr[..., np.newaxis]
    else:
        #eq_data_n = eq_data[..., np.newaxis]
        eq_data_n = eq_data
        eq_data_t = tf.cast(eq_data_n.T, np.complex64)

        no = 1.0
        #llr_tf = demapper([eq_data_t, no])
        llr_tf = mod_dict["demapper"]([eq_data_t, no])
        llr = llr_tf.numpy()
        bit_arr_t = np.zeros_like(llr)
        bit_arr_t[llr > 0] = 1
        bit_arr = bit_arr_t.T

        # perform FEC decoding
        if inPar.do_fec:
            llr_tf_use = np.reshape(llr_tf.numpy(), (1, llr_tf.shape[0] * llr_tf.shape[1]), order='C')
            decoder = mod_dict["decoder"]
            bit_uncode = decoder(tf.convert_to_tensor(llr_tf_use))
            bit_uncode = bit_uncode.numpy().T
        else:
            bit_uncode = bit_arr

    return bit_arr, bit_uncode


def get_ber(data_stream, bit_arr, N_sc_use):
    ber = np.sum(data_stream != bit_arr) / len(data_stream)
    return ber

#do_load_file = False
inPar = init_tx_dict()
# imitate tranmitted
(repeated_frame_tx, repeated_frame, frame_len, preamble_len, preamble_core, mod_dict_data, pilot_tx,
 bite_stream_tx, mod_data_tx, bite_stream_tx_uncode)  \
        = create_data_frame(inPar)
# end imitate

#Thr_max = (inPar.N_sc_use * inPar.num_bits_sym) * (inPar.N_sc_use / inPar.N_fft) * inPar.delta_f / 1e3  # kbit/sec
Thr_max = inPar.N_sc_use * inPar.num_bits_sym * inPar.Ndata / 1e3 # kbit per package


if not inPar.dummyRx:
    # additional params

    do_load_file = False # should be False
    do_save = False

    # init Thr
    alpha_avg = 0.1
    Thr_dict = dict()
    for idx, (mimo_mode, name_mimo) in enumerate(cfg_test):
        Thr_dict[name_mimo] = 0.0

    sdr = adi.ad9361(uri='ip:192.168.2.2')
    samp_rate = inPar.sample_rate  # must be <=30.72 MHz if both channels are enabled
    num_samps = len(repeated_frame) * 1  # number of samples per buffer.  Can be different for Rx and Tx
    rx_lo = int(inPar.center_freq)
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
else:
    # emulate Ntx
    repeated_frame_tx = repeated_frame_tx.T
    Nrx_test = 2
    repeated_frame_tx = np.tile(repeated_frame_tx, (Nrx_test, 1))

    # add AWGN channel for Tx signal
    Es_tmp = np.linalg.norm(repeated_frame_tx.flatten())**2 / len(repeated_frame_tx.flatten())

    SNR = inPar.SNR_dummy
    sigma_noise_tmp = Es_tmp * 10**(-SNR/10)

    noise_arr = np.sqrt(sigma_noise_tmp / 2) * ( (np.random.normal(0, 1, repeated_frame_tx.shape)) + 1j * (np.random.normal(0, 1, repeated_frame_tx.shape)) )

    data_rx_dummy = repeated_frame_tx + noise_arr
    SNR_est = 10.0 * np.log10(np.linalg.norm(repeated_frame_tx.flatten())**2 / np.linalg.norm(noise_arr.flatten())**2)
    print(f'dummy Tx SNRset={SNR:.2f} SNR={SNR_est:.2f}')

    #data = data_rx_dummy

    sdr = []

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

        rx_sig = data # Ntx x Nsamples

    # rx_sig = data
    num_rx, rx_len = rx_sig.shape

    frame_len_use = frame_len
    idx_max, corr_value, sigma_arr, SNR_guard = find_edges(rx_sig, frame_len_use, preamble_len, inPar.CP_len, preamble_core, start_idx=0)
    frame_receive = rx_sig[:, idx_max: idx_max + frame_len_use]

    frame_receive = cfo(frame_receive, corr_value, preamble_len) if inPar.do_cfo_corr else frame_receive

    pilot_freq_rep = np.zeros( (inPar.pilot_repeat, num_rx, inPar.N_fft), dtype=np.complex64 )

    frame_cp_len = inPar.N_fft + inPar.CP_len
    start_time = inPar.N_fft + inPar.CP_len
    for rep_idx in range(inPar.pilot_repeat):
        pilot_receive = frame_receive[:, start_time + (rep_idx)* frame_cp_len : start_time + (rep_idx)* frame_cp_len + inPar.N_fft]
        pilot_freq = np.fft.fft(pilot_receive, inPar.N_fft, axis=1, norm="ortho")
        pilot_freq_rep[rep_idx, :, :] = pilot_freq


    h_ls_all = np.zeros((iNtx, num_rx, inPar.N_sc_use), dtype=np.complex64)
    rec_sym_pilot_all = np.zeros((inPar.pilot_repeat, num_rx, inPar.N_sc_use), dtype=np.complex64)

    Ruu = np.zeros((num_rx, num_rx), dtype=np.complex64)

    # CE part
    for tx_idx in range(iNtx):

        comb_start = tx_idx
        comb_step = iNtx

        ls_len = int(inPar.N_sc_use/comb_step)

        for rx_idx in range(num_rx):

            h_ls_rep = np.zeros( (inPar.pilot_repeat, num_rx, ls_len), dtype=np.complex64 )
            for rep_idx in range(inPar.pilot_repeat):

                rec_sym_pilot = baseband_freq_domian(pilot_freq_rep[rep_idx, rx_idx, :], inPar.N_sc_use)
                rec_sym_pilot_all[rep_idx, rx_idx, :] = rec_sym_pilot

                # apply CE
                h_ls = rec_sym_pilot[comb_start::comb_step] / pilot_tx[tx_idx][comb_start::comb_step, 0]

                h_ls_rep[rep_idx, rx_idx, :] = h_ls

            # LS average
            h_ls = h_ls_rep[0, rx_idx, :]
            if True:
                for rep_idx in range(1, pilot_rep_use):
                    h_ls_c = h_ls_rep[rep_idx, rx_idx, :]
                    h_ls = h_ls + h_ls_c
                h_ls = h_ls / pilot_rep_use

            h_ls_all[tx_idx, rx_idx, :] = channel_estimation(h_ls, inPar.CP_len, inPar.N_fft, comb_step, sigma_arr[rx_idx], ce_mode) if inPar.do_ce else h_ls

        # usamples estimation
        for rep_idx in range(inPar.pilot_repeat):
            u_mx = rec_sym_pilot_all[rep_idx, :, comb_start::comb_step] - h_ls_all[tx_idx, :, comb_start::comb_step] * pilot_tx[tx_idx][comb_start::comb_step, 0]
            Ruu_c = u_mx @ u_mx.conj().T / u_mx.shape[1]


        Ruu = Ruu + (Ruu_c / inPar.pilot_repeat)



        #print(f'S_t={S_t}')

    Ruu = Ruu / iNtx
    U_t, S_t, v_t = np.linalg.svd(Ruu)
    Ruu_inv = np.linalg.inv(Ruu)


    # rewrite CP removal
    data_freq_rep = np.zeros((inPar.Ndata, num_rx, inPar.N_fft), dtype=np.complex64)
    frame_cp_len = inPar.N_fft + inPar.CP_len
    start_time = inPar.N_fft + (frame_cp_len) * inPar.pilot_repeat + inPar.CP_len

    # remove CP per TTi
    for rep_idx in range(inPar.Ndata):
        data_receive = frame_receive[:,
                        start_time + (rep_idx) * frame_cp_len: start_time + (rep_idx) * frame_cp_len + inPar.N_fft]
        data_freq = np.fft.fft(data_receive, inPar.N_fft, axis=1, norm="ortho")
        data_freq_rep[rep_idx, :, :] = data_freq

    rec_data_sym_freq = np.transpose(data_freq_rep, axes=(1, 2, 0))


    rec_sym_data_all = np.zeros((num_rx, inPar.N_sc_use, inPar.Ndata), dtype=np.complex64)

    rho_su = 1.0 # init correlation between SU weights

    for rx_idx in range(num_rx):
        rx_tmp = baseband_freq_domian(rec_data_sym_freq[rx_idx, :, :], inPar.N_sc_use)
        rec_sym_data_all[rx_idx, :, :] = rx_tmp[:, :]

    if mimo_mode == 0 or mimo_mode == 1:
        # outdated modes
        eq_data = rec_sym_data_all[0, :] / h_ls_all[0, :]
    else:
        eq_data = np.zeros( (iNtx, inPar.N_sc_use, inPar.Ndata), dtype=np.complex64)
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


        if inPar.Ntx > 1:
            R_corr = V_su.conj().T @ V_su
            rho_su = R_corr[0, 1]
        else:
            rho_su = 1.0

        for sc_idx in range(rec_sym_data_all.shape[1]):

            h_c = h_ls_all[:, :, sc_idx]

            h_c = h_c.T

            r_c = rec_sym_data_all[:, sc_idx, :] # Nrx x Ndata
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

            eq_data[:, sc_idx, :] = x_c

    # equalization points are ready
    # calculate EVM
    evm_arr = np.zeros((iNtx,), dtype=np.float32)
    for tx_idx in range(iNtx):
        evm_arr[tx_idx] = calc_evm(mod_data_tx[tx_idx], eq_data)

    #rho_avg_plot = np.mean(rho_avg)
    rho_avg_plot = rho_su
    #rho_avg_plot = np.max(rho_avg)

    # calculate BER/FER
    ber_arr = list()
    for tx_idx in range(iNtx):

        if inPar.do_sc_fdm:
            bit_arr, bit_arr_uncode = demodulate(inPar.T_prec.conj().T @ eq_data[tx_idx, :], inPar.N_sc_use, mod_dict_data)
        else:
            bit_arr, bit_arr_uncode = demodulate(eq_data[tx_idx, :], inPar.N_sc_use, mod_dict_data)

        # unwrap data to bitstream
        bit_arr = np.reshape(bit_arr, newshape=(bit_arr.shape[0] * bit_arr.shape[1], 1), order='F') # Nbit_total x 1

        ber = get_ber(bite_stream_tx[tx_idx], bit_arr, inPar.N_sc_use)

        if inPar.do_fec:
            ber_uncode = get_ber(bite_stream_tx_uncode[tx_idx], bit_arr_uncode, inPar.N_sc_use)
            ber_arr.append(ber_uncode)
        else:
            ber_arr.append(ber)


        #SNR_est = estimate_SNR(pilot_freq, rec_sym_pilot_all, N_sc_use)

    return ber_arr, SNR_guard, rho_avg_plot, evm_arr

#### RECEIVER PART start - test reception chain
if not inPar.dummyRx:
    if not do_load_file:

        data = sdr.rx()
        data = np.array(data)


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
else:
    # code for dummy receeption
    # input array data set globally
    data = data_rx_dummy
    print('Dummpy Rx mode')
    #exit(0)

# FOR DEBUG first frame
for mimo, rec_name in cfg_test:

    # parse configs
    try:
        pilot_rep_use = int(re.findall('_rep(\d+)', rec_name)[0])
    except:
        pilot_rep_use = 1

    ber_c, snr_c, rho_avg_plot, evm_rx = receiver_MIMO(data, mimo, inPar.Ntx, pilot_rep_use)

    print(f'mimo={mimo} rec_name={rec_name}, snr_c={snr_c}, evm={np.mean(evm_rx)}')

    for tx_idx in range(inPar.Ntx):
        print(f'Layer={tx_idx} ber={ber_c[tx_idx]}')

    time.sleep(0.1)


def update(frame1):

    ### SDR reception
    if not inPar.dummyRx:
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
    else:
        #print('Dummpy Rx mode')
        #exit(1)
        data = data_rx_dummy
        pass

    for idx, (mimo, rec_name) in enumerate(cfg_test):

        # parse configs
        try:
            pilot_rep_use = int(re.findall('_rep(\d+)', rec_name)[0])
        except:
            pilot_rep_use = 1


        ber_c, snr_c, rho_avg_plot, evm_arr = receiver_MIMO(data, mimo, inPar.Ntx, pilot_rep_use)

        thr_c =  Thr_max * ( inPar.Ntx - np.sum( np.array(ber_c) ))

        for p_idx in range(num_subplots):

            x1, y1 = line_arr[num_subplots * idx + p_idx].get_data()
            x1 = np.append(x1, frame1)

            if metrics_to_plot[p_idx] == 'SNR_guard':
                y1 = np.append(y1, snr_c)
            elif metrics_to_plot[p_idx] == 'EVM':
                evm_plot = np.mean(evm_arr)
                if evm_plot > 3.5:
                    evm_plot = 3.5
                y1 = np.append(y1, evm_plot)
            elif metrics_to_plot[p_idx] == 'BER':
                y1 = np.append(y1, np.mean(ber_c))

            line_arr[num_subplots  * idx + p_idx].set_data(x1, y1)

        # x1, y1 = line_arr[num_subplots * idx + 1].get_data()
        # # x1, y1 = line_c.get_data()
        #
        # x1 = np.append(x1, frame1)
        # y1 = np.append(y1, snr_c)
        # line_arr[num_subplots * idx + 1].set_data(x1, y1)
        #
        # x1, y1 = line_arr[num_subplots * idx + 2].get_data()
        # x1 = np.append(x1, frame1)
        # y1 = np.append(y1, rho_avg_plot)
        # line_arr[num_subplots * idx + 2].set_data(x1, y1)
        #
        # # plot throughput
        # x1, y1 = line_arr[num_subplots * idx + 3].get_data()
        # x1 = np.append(x1, frame1)
        #
        # if len(x1) == 0:
        #     Thr_dict[rec_name] = thr_c
        # else:
        #     Thr_dict[rec_name] = thr_c * alpha_avg + (1.0 - alpha_avg) * Thr_dict[rec_name]
        #
        # y1 = np.append(y1, Thr_dict[rec_name])
        # line_arr[num_subplots * idx + 3].set_data(x1, y1)


    return (*line_arr,)


# init plots
def init():
    #metrics_to_plot = ['SNR_guard', 'EVM', 'BER']
    for p_idx in range(num_subplots):

        if metrics_to_plot[p_idx] == "SNR_guard":
            ax[p_idx].set_xlabel('Time')
            ax[p_idx].set_ylabel(f'SNRguard')
            ax[p_idx].set_xlim(0, NUM_FRAMES)
            ax[p_idx].set_ylim(0, 40)
        elif metrics_to_plot[p_idx] == "EVM":
            ax[p_idx].set_xlabel('Time')
            ax[p_idx].set_ylabel(f'EVM')
            ax[p_idx].set_xlim(0, NUM_FRAMES)
            ax[p_idx].set_ylim(0, 4.0)
        elif metrics_to_plot[p_idx] == "BER":
            ax[p_idx].set_xlabel('Time')
            ax[p_idx].set_ylabel(f'BER@QAM{2**inPar.num_bits_sym}')
            ax[p_idx].set_xlim(0, NUM_FRAMES)
            ax[p_idx].set_ylim(10 ** (-2), 10 ** (-0))

    # ax[0].set_xlabel('Time')
    # ax[0].set_xlim(0, 2*np.pi)
    # ax[0].set_ylabel(f'BER@QAM{2**inPar.num_bits_sym}')
    # ax[0].set_ylim(10 ** (-4), 10 ** (-0))
    #
    # ax[1].set_xlabel('Time')
    # ax[1].set_ylabel(f'SNRguard')
    # ax[1].set_xlim(0, 2 * np.pi)
    # ax[1].set_ylim(0, 45)
    #
    # ax[2].set_xlabel('Time')
    # ax[2].set_ylabel(f'SU correlation')
    # ax[2].set_xlim(0, 2 * np.pi)
    # ax[2].set_ylim(0.0, 1.0)
    #
    # ax[3].set_xlabel('Time')
    # ax[3].set_ylabel(f'Throughput')
    # ax[3].set_xlim(0, 2 * np.pi)
    # ax[3].set_ylim(0.0, 1.2 * Thr_max * inPar.Ntx)



    return (*line_arr,)

def main():
    ani = FuncAnimation(fig, update, frames=np.linspace(0, NUM_FRAMES, 1024), init_func=init, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
