import re
import pickle
from pathlib import Path

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
from tensorflow import newaxis

from tx_mimo_npilot import create_data, create_preamble, create_data_frame, init_tx_dict
from system_tx import SysParUL
import json

def calc_evm(dat_idl, dat_est):
    evm_c = 10.0 * np.log10( np.mean( np.abs(dat_idl.flatten())**2 )  / np.mean( np.abs(dat_idl.flatten() - dat_est.flatten())**2 )  )
    return evm_c


def _hard_quantize_to_constellation(sym, constellation_points):
    """Map complex symbols to nearest constellation point."""
    sym_flat = sym.reshape(-1)
    const = np.asarray(constellation_points, dtype=np.complex64).reshape(1, -1)
    dist = np.abs(sym_flat[:, None] - const) ** 2
    idx = np.argmin(dist, axis=1)
    sym_hat = const[0, idx]
    return sym_hat.reshape(sym.shape)


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


def cfo(frame_receive, corr_value, preamble_len, frame_len):
    angle_cfo = np.angle(corr_value) / preamble_len

    #print(f'CFOest={angle_cfo} CFO_near_idl_2antsdr=-0.0038')

    cfo_comp_sig = np.exp(1j * (-1.0*angle_cfo * np.arange(0, frame_len)))
    frame_receive = frame_receive * cfo_comp_sig
    return frame_receive


def baseband_freq_domian(pilot_freq, N_sc_use, inPar : SysParUL):

    if pilot_freq.ndim == 1:
        rec_sym_pilot = np.zeros((N_sc_use,), dtype=np.complex64)
        rec_sym_pilot[int(N_sc_use / 2):] = pilot_freq[0 + inPar.dc_offset:int(N_sc_use / 2) + inPar.dc_offset]
        rec_sym_pilot[0:int(N_sc_use / 2)] = pilot_freq[-int(N_sc_use / 2):]
    else:
        rec_sym_pilot = np.zeros((N_sc_use, pilot_freq.shape[1]), dtype=np.complex64)
        rec_sym_pilot[int(N_sc_use / 2):, :] = pilot_freq[0 + inPar.dc_offset:int(N_sc_use / 2) + inPar.dc_offset, :]
        rec_sym_pilot[0:int(N_sc_use / 2), :] = pilot_freq[-int(N_sc_use / 2):, :]

        #noise_arr = pilot_freq[int(N_sc_use / 2) + inPar.dc_offset : -int(N_sc_use / 2)]

        #sigma_noise = np.mean(np.abs(noise_arr.flatten()) ** 2)
        #signal_power = np.mean(np.abs(rec_sym_pilot.flatten()) ** 2)

        #SNR_bb = 10.0 * np.log10(signal_power / sigma_noise)



    return rec_sym_pilot


def _interpolate_dc_region(h_freq, dc_mask_half_width):
    """Replace DC and neighboring bins by linear interpolation to reduce DC spike impact."""
    if dc_mask_half_width <= 0 or len(h_freq) < 3:
        return h_freq
    n = len(h_freq)
    dc_idx = n // 2
    lo = max(0, dc_idx - dc_mask_half_width)
    hi = min(n, dc_idx + dc_mask_half_width + 1)
    if lo >= hi:
        return h_freq
    left_val = h_freq[lo - 1] if lo > 0 else h_freq[hi] if hi < n else 0.0
    right_val = h_freq[hi] if hi < n else h_freq[lo - 1] if lo > 0 else 0.0
    denom = hi - lo
    for i in range(lo, hi):
        t = (i - lo) / denom if denom > 0 else 0.0
        h_freq[i] = (1.0 - t) * left_val + t * right_val
    return h_freq


def _esprit_delays(h_ls, n_taps, order=None):
    """
    ESPRIT super-resolution: estimate n_taps complex poles z_l from frequency snapshot h_ls.
    Model: h_ls(k) = sum_l a_l * z_l^k + noise. Returns z_poles (length n_taps) for MMSE projection.
    """
    n = len(h_ls)
    if n_taps < 1 or n < n_taps + 2:
        return np.array([1.0], dtype=np.complex64)  # fallback single pole
    # ESPRIT order M: need M >= n_taps+1 and 2*M-2 <= n => M <= (n+2)//2
    if order is None:
        order = min(n_taps + 6, (n + 2) // 2)
    order = max(n_taps + 1, min(order, (n + 2) // 2))
    # Hankel: row i = [h_ls(i), h_ls(i+1), ..., h_ls(i+order-1)], i=0..n-order
    n_rows = n - order + 1
    X1 = np.zeros((n_rows, order - 1), dtype=np.complex64)
    X2 = np.zeros((n_rows, order - 1), dtype=np.complex64)
    for i in range(n_rows):
        for j in range(order - 1):
            X1[i, j] = h_ls[i + j]
            X2[i, j] = h_ls[i + j + 1]
    # Z = [X1; X2], 2*n_rows x (order-1)
    Z = np.vstack([X1, X2])
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    # Signal subspace: first n_taps left singular vectors
    L = min(n_taps, order - 1, U.shape[1])
    Us = U[:, :L]
    Us1 = Us[:n_rows, :]
    Us2 = Us[n_rows:, :]
    # Phi such that Us2 ≈ Us1 @ Phi  =>  Phi = pinv(Us1) @ Us2
    Phi = np.linalg.lstsq(Us1, Us2, rcond=None)[0]
    z_poles = np.linalg.eigvals(Phi)
    z_poles = np.array(z_poles, dtype=np.complex64)
    # Prefer poles inside or near unit circle (physical delays)
    inside = np.abs(z_poles) <= 1.0 + 0.1
    z_poles = z_poles[inside] if np.any(inside) else z_poles
    # Take up to n_taps, ordered by magnitude (strongest first)
    if len(z_poles) > n_taps:
        idx = np.argsort(np.abs(z_poles))[::-1][:n_taps]
        z_poles = z_poles[idx]
    if len(z_poles) < n_taps:
        pad = np.exp(-2j * np.pi * np.arange(1, n_taps - len(z_poles) + 1) / n)
        z_poles = np.concatenate([z_poles, pad]).astype(np.complex64)
    return z_poles[:n_taps]


def _detect_dc_spike(h_ls, dc_mask_half_width, threshold):
    """True if DC region power exceeds threshold * median power of other subcarriers."""
    if threshold <= 0 or len(h_ls) < 3:
        return False
    n = len(h_ls)
    dc_idx = n // 2
    lo = max(0, dc_idx - dc_mask_half_width)
    hi = min(n, dc_idx + dc_mask_half_width + 1)
    p_dc = np.mean(np.abs(h_ls[lo:hi]) ** 2)
    other_idx = np.concatenate([np.arange(lo), np.arange(hi, n)])
    if len(other_idx) == 0:
        return False
    p_other = np.median(np.abs(h_ls[other_idx]) ** 2)
    if p_other <= 0:
        return True
    return (p_dc / p_other) > threshold


def channel_estimation(h_ls, CP_len, N_fft, comb_step=1, sigma_0=0.0, ce_mode=0, dc_mask_half_width=0, dc_adaptive_threshold=0.0, n_taps_esprit=5):
    # Adaptive DC handling: only interpolate when DC spike is detected (avoids hurting BER when no spike)
    h_ls = np.array(h_ls, dtype=np.complex64, copy=True)
    if dc_mask_half_width > 0 and dc_adaptive_threshold > 0 and _detect_dc_spike(h_ls, dc_mask_half_width, dc_adaptive_threshold):
        h_ls = _interpolate_dc_region(h_ls, dc_mask_half_width)
    h_time = np.fft.ifft(h_ls, N_fft, 0, norm='ortho')

    if ce_mode == 4:
        # CE_mode=4: ESPRIT super-resolution delay profile + MMSE projection (LS + diagonal loading)
        N_sc_in = h_ls.shape[0]
        L = max(1, min(n_taps_esprit, N_sc_in // 2 - 1))
        z_poles = _esprit_delays(h_ls, L)
        L = len(z_poles)
        # Vandermonde: A[k,l] = z_l^k  =>  frequency response of tap l at subcarrier k
        k_arr = np.arange(N_sc_in, dtype=np.int32)
        A_esprit = np.zeros((N_sc_in, L), dtype=np.complex64)
        for l in range(L):
            A_esprit[:, l] = np.power(z_poles[l], k_arr)
        # LS tap gains: g = (A^H A)^{-1} A^H h_ls
        AHA = A_esprit.conj().T @ A_esprit
        g_ls = np.linalg.solve(AHA, A_esprit.conj().T @ h_ls)
        # Diagonal loading: R = A^H A + diag(sigma_0 / (|g_l|^2 + eps))
        pdp = np.abs(g_ls) ** 2
        eps = np.finfo(np.float32).eps * (1.0 + np.max(pdp))
        D_load = np.diag((sigma_0 / (pdp + eps)).astype(np.float32)).astype(np.complex64)
        R_mmse = AHA + D_load
        W_ce = A_esprit @ np.linalg.solve(R_mmse, A_esprit.conj().T)
        h_ce_out = (W_ce @ h_ls).ravel()
        return h_ce_out

    if ce_mode >= 1:
        # new code
        # first find time shift
        N_sc_in = h_ls.shape[0]

        h_first = h_ls[:-2]
        h_second = h_ls[1:-1]

        avg_angle = np.angle( h_first.conj().T @ h_second )

        x_arr = np.arange(0, N_sc_in)
        ta_comp = np.exp(1j * -avg_angle * x_arr )
        ta_comp_inv = np.exp(1j * avg_angle * x_arr)

        h_ls_ta = h_ls * ta_comp

        h_ls_ta = h_ls_ta[..., np.newaxis] # Nsc x 1
        ff = 1

        A_dft = np.fft.fft(np.eye(N_fft, dtype=np.complex64), norm='ortho')
        A_idft = A_dft[:N_sc_in, :].conj().T

        h_time_check = A_idft @ h_ls_ta

        if False:
            plt.plot(range(len(h_time_check)), np.abs(h_time_check))
            plt.show()

        W_max = 5
        W_min = 1

        A_pdft_max = A_dft[:N_sc_in, :W_max]

        if W_min > 0:
            A_pdft_min = A_dft[:N_sc_in, -W_min:]
            A_pdft_join = np.hstack((A_pdft_max, A_pdft_min))
        else:
            A_pdft_join = A_pdft_max

        # SVD transformation
        if ce_mode == 3:

            # SVD
            U_c, S_c, V_c = np.linalg.svd(A_pdft_join, full_matrices=True)

            n_take = W_max + W_min
            U_c = U_c[:, :n_take]

            A_pdft_join = U_c


        R_cov_tt = A_pdft_join.conj().T @ A_pdft_join

        if (ce_mode == 2) or (ce_mode == 3):
            pdp_est = A_pdft_join.conj().T @ h_ls_ta
            eta_fact = 1.0
            pdp_power = np.abs(pdp_est[:, 0]) ** 2
            eps = np.finfo(np.complex64).eps * (1.0 + np.max(pdp_power))
            D_snr = np.diag((eta_fact * sigma_0 / (pdp_power + eps)).astype(np.float32)).astype(np.complex64)
            R_cov_tt = R_cov_tt + D_snr

        W_ce = A_pdft_join @ np.linalg.inv(R_cov_tt) @ A_pdft_join.conj().T

        h_ce_rez = W_ce @ h_ls_ta

        h_ce_out = h_ce_rez[:, 0] * ta_comp_inv

        if False:
            fig1, ax1 = plt.subplots()
            ax1.plot(range(N_sc_in), h_ce_out, label='CEout')
            ax1.plot(range(N_sc_in), h_ls, label='CEin')
            ax1.grid()
            ax1.legend()
            plt.show()

        ff = 1

        return h_ce_out

        # old code soft window
        if False:
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


def demodulate(eq_data, N_sc_use, mod_dict, inPar : SysParUL):
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

def receiver_MIMO_v2(data, mimo_mode_in, iNtx, pilot_rep_use=1, ce_mode=None, smmse_mode=None, dc_mask_half_width=0, dc_adaptive_threshold=0.0, robust_pilot_avg=False, exclude_dc_from_ruu=True, n_taps_esprit=5, return_channel_for_plot=False, turbo_enable=False, turbo_iters=1, turbo_pilot_weight=0.5):

    inPar = init_tx_dict()
    # imitate tranmitted
    (repeated_frame_tx, repeated_frame, frame_len, preamble_len, preamble_core, mod_dict_data, pilot_tx,
     bite_stream_tx, mod_data_tx, bite_stream_tx_uncode) \
        = create_data_frame(inPar)

    data = np.array(data)

    # parse mimo_mode
    mimo_mode = mimo_mode_in % 10 # extract MIMO detection
    # SMMSE mode: use provided parameter, or extract from mimo_mode_in for backward compatibility
    if smmse_mode is None:
        smmse_mode = int(mimo_mode_in / 10) % 10 # extract SMME mode
    # CE mode: use provided parameter, or extract from mimo_mode_in for backward compatibility
    if ce_mode is None:
        ce_mode = int(mimo_mode_in / 100) % 10 # CE mode

    if mimo_mode == 0:
        rx_sig = np.zeros((1, data.shape[1]), dtype=np.complex64)
        rx_sig[0, :] = data[0, :] + data[1, :]

    elif mimo_mode == 5:

        rx_sig = np.zeros((1, data.shape[1]), dtype=np.complex64)
        rx_sig[0, :] = data[0, :]

    elif mimo_mode == 1:

        R = data @ data.conj().T
        U1, s1, V1 = np.linalg.svd(R)
        U_main = U1[:, 0:1]
        rx_sig = U_main.conj().T @ data

        # rx_sig = rx_sig[0]

    elif (mimo_mode == 2) or (mimo_mode == 3) or (mimo_mode == 4):

        rx_sig = data # Ntx x Nsamples
    else:
        rx_sig = data

    # rx_sig = data
    num_rx, rx_len = rx_sig.shape

    frame_len_use = frame_len
    idx_max, corr_value, sigma_arr, SNR_guard = find_edges(rx_sig, frame_len_use, preamble_len, inPar.CP_len, preamble_core, start_idx=0)
    frame_receive = rx_sig[:, idx_max: idx_max + frame_len_use]

    frame_receive = cfo(frame_receive, corr_value, preamble_len, frame_len_use) if inPar.do_cfo_corr else frame_receive

    pilot_freq_rep = np.zeros( (inPar.pilot_repeat, num_rx, inPar.N_fft), dtype=np.complex64 )

    frame_cp_len = inPar.N_fft + inPar.CP_len
    start_time = 2 * preamble_len + inPar.CP_len

    cfo_cp = 0 # CFO correction using CP # should be zero
    for rep_idx in range(inPar.pilot_repeat):
        pilot_receive = frame_receive[:, start_time + (rep_idx)* frame_cp_len : start_time + (rep_idx)* frame_cp_len + inPar.N_fft]


        # try CFO correction based on CP
        if cfo_cp:
            sig_cp_start = frame_receive[:, start_time + (rep_idx)* frame_cp_len - inPar.CP_len : start_time + (rep_idx)* frame_cp_len]
            sig_cp_end = pilot_receive[:, -inPar.CP_len:]

            conf_len = int(inPar.CP_len / 2)
            sig_cp_start_conf = sig_cp_start[:, -conf_len:]
            sig_cp_end_conf = sig_cp_end[:, -conf_len:]

            corr_tmp = sig_cp_start_conf @ sig_cp_end_conf.conj().T
            corr_avg = np.mean(np.diag(corr_tmp))
            angle_cfo_delta = np.angle(corr_avg)

            angle_cfo = angle_cfo_delta / inPar.N_fft
            cfo_comp_sig = np.exp(1j * ( angle_cfo * np.arange(0, inPar.N_fft)))

            pilot_receive = pilot_receive * cfo_comp_sig
            # end CFO processing

        pilot_freq = np.fft.fft(pilot_receive, inPar.N_fft, axis=1, norm="ortho")
        pilot_freq_rep[rep_idx, :, :] = pilot_freq


    h_ls_all = np.zeros((iNtx, num_rx, inPar.N_sc_use), dtype=np.complex64)
    rec_sym_pilot_all = np.zeros((inPar.pilot_repeat, num_rx, inPar.N_sc_use), dtype=np.complex64)
    h_ls_pilot_full = None
    if return_channel_for_plot or turbo_enable:
        h_ls_pilot_full = np.full((iNtx, num_rx, inPar.N_sc_use), np.nan, dtype=np.complex64)

    Ruu = np.zeros((num_rx, num_rx), dtype=np.complex64)

    # CE part
    for tx_idx in range(iNtx):

        comb_start = tx_idx
        comb_step = iNtx

        ls_len = int(inPar.N_sc_use/comb_step)

        for rx_idx in range(num_rx):

            h_ls_rep = np.zeros( (inPar.pilot_repeat, num_rx, ls_len), dtype=np.complex64 )
            for rep_idx in range(inPar.pilot_repeat):

                rec_sym_pilot = baseband_freq_domian(pilot_freq_rep[rep_idx, rx_idx, :], inPar.N_sc_use, inPar)
                rec_sym_pilot_all[rep_idx, rx_idx, :] = rec_sym_pilot

                # apply CE
                h_ls = rec_sym_pilot[comb_start::comb_step] / pilot_tx[tx_idx][comb_start::comb_step, 0]

                h_ls_rep[rep_idx, rx_idx, :] = h_ls

            # LS average over pilot repeats (mean or robust median)
            if robust_pilot_avg and pilot_rep_use > 1:
                h_ls = np.median(h_ls_rep[:pilot_rep_use, rx_idx, :], axis=0).astype(np.complex64)
            else:
                h_ls = h_ls_rep[0, rx_idx, :]
                for rep_idx in range(1, pilot_rep_use):
                    h_ls = h_ls + h_ls_rep[rep_idx, rx_idx, :]
                h_ls = h_ls / pilot_rep_use

            if h_ls_pilot_full is not None:
                h_ls_pilot_full[tx_idx, rx_idx, comb_start::comb_step] = h_ls

            h_ls_all[tx_idx, rx_idx, :] = channel_estimation(h_ls, inPar.CP_len, inPar.N_fft, comb_step, sigma_arr[rx_idx], ce_mode, dc_mask_half_width, dc_adaptive_threshold, n_taps_esprit) if inPar.do_ce else h_ls

        # Interference covariance from pilot residuals; exclude DC columns so spike does not inflate Ruu
        for rep_idx in range(inPar.pilot_repeat):
            u_mx = rec_sym_pilot_all[rep_idx, :, comb_start::comb_step] - h_ls_all[tx_idx, :, comb_start::comb_step] * pilot_tx[tx_idx][comb_start::comb_step, 0]
            if exclude_dc_from_ruu:
                dc_baseband = inPar.N_sc_use // 2
                valid_cols = [k for k in range(u_mx.shape[1]) if (comb_start + k * comb_step) != dc_baseband]
                if len(valid_cols) > 0:
                    u_mx_ruu = u_mx[:, valid_cols]
                    Ruu_c = u_mx_ruu @ u_mx_ruu.conj().T / u_mx_ruu.shape[1]
                else:
                    Ruu_c = u_mx @ u_mx.conj().T / u_mx.shape[1]
            else:
                Ruu_c = u_mx @ u_mx.conj().T / u_mx.shape[1]

            Ruu = Ruu + (Ruu_c / inPar.pilot_repeat)



        #print(f'S_t={S_t}')

    Ruu = Ruu / iNtx
    U_t, S_t, v_t = np.linalg.svd(Ruu)
    Ruu_inv = np.linalg.inv(Ruu)


    # rewrite CP removal
    data_freq_rep = np.zeros((inPar.Ndata, num_rx, inPar.N_fft), dtype=np.complex64)
    frame_cp_len = inPar.N_fft + inPar.CP_len
    start_time = 2 * preamble_len + (frame_cp_len) * inPar.pilot_repeat + inPar.CP_len

    # remove CP per TTi
    for rep_idx in range(inPar.Ndata):
        data_receive = frame_receive[:,
                        start_time + (rep_idx) * frame_cp_len: start_time + (rep_idx) * frame_cp_len + inPar.N_fft]

        # try CFO correction based on CP
        if cfo_cp:
            sig_cp_start = frame_receive[:,
                           start_time + (rep_idx) * frame_cp_len - inPar.CP_len: start_time + (rep_idx) * frame_cp_len]
            sig_cp_end = data_receive[:, -inPar.CP_len:]

            conf_len = int(inPar.CP_len / 2)
            sig_cp_start_conf = sig_cp_start[:, -conf_len:]
            sig_cp_end_conf = sig_cp_end[:, -conf_len:]

            corr_tmp = sig_cp_start_conf @ sig_cp_end_conf.conj().T
            corr_avg = np.mean(np.diag(corr_tmp))
            angle_cfo_delta = np.angle(corr_avg)

            angle_cfo = angle_cfo_delta / inPar.N_fft
            cfo_comp_sig = np.exp(1j * (angle_cfo * np.arange(0, inPar.N_fft)))

            data_receive = data_receive * cfo_comp_sig
            # end CFO processing




        data_freq = np.fft.fft(data_receive, inPar.N_fft, axis=1, norm="ortho")
        data_freq_rep[rep_idx, :, :] = data_freq

    rec_data_sym_freq = np.transpose(data_freq_rep, axes=(1, 2, 0))


    rec_sym_data_all = np.zeros((num_rx, inPar.N_sc_use, inPar.Ndata), dtype=np.complex64)

    rho_su = 1.0 # init correlation between SU weights

    for rx_idx in range(num_rx):
        rx_tmp = baseband_freq_domian(rec_data_sym_freq[rx_idx, :, :], inPar.N_sc_use, inPar)
        rec_sym_data_all[rx_idx, :, :] = rx_tmp[:, :]

    def _equalize_with_channel(h_ls_in):
        h_ls_work = np.array(h_ls_in, dtype=np.complex64, copy=True)
        if mimo_mode == 0 or mimo_mode == 1 or mimo_mode == 5:
            # outdated modes
            rec_data_perm = np.transpose(rec_sym_data_all, axes=(2, 0, 1))
            eq_data_local = rec_data_perm / h_ls_work
            eq_data_local = np.transpose(eq_data_local, axes=(1, 2, 0))
            rho_su_local = 1.0
            H_c_local = h_ls_work[0, :, :]
            R_hh_local = H_c_local @ H_c_local.conj().T / H_c_local.shape[1]
            return eq_data_local, rho_su_local, R_hh_local

        eq_data_local = np.zeros((iNtx, inPar.N_sc_use, inPar.Ndata), dtype=np.complex64)
        rho_avg = list()

        # SU weights correlation
        V_su = np.zeros((num_rx, iNtx), dtype=np.complex64)
        for tx_idx in range(iNtx):
            H_c = h_ls_work[tx_idx, :, :]
            R_hh = H_c @ H_c.conj().T / H_c.shape[1]
            U_c, S_c, V_c = np.linalg.svd(R_hh)

            V_su[:, tx_idx] = U_c[:, 0]

            # SMMSE start
            if (smmse_mode > 0):
                alpha_ruu = 0.3
                W_smmse = R_hh @ np.linalg.inv(R_hh + alpha_ruu * Ruu)

                h_ls_work[tx_idx, :, :] = W_smmse @ h_ls_work[tx_idx, :, :]

        if inPar.Ntx > 1:
            R_corr = V_su.conj().T @ V_su
            rho_su_local = R_corr[0, 1]
        else:
            rho_su_local = 1.0

        # Rhh calc common
        H_c = h_ls_work[0, :, :]
        R_hh_local = H_c @ H_c.conj().T / H_c.shape[1]

        for sc_idx in range(rec_sym_data_all.shape[1]):

            h_c = h_ls_work[:, :, sc_idx]
            h_c = h_c.T

            r_c = rec_sym_data_all[:, sc_idx, :] # Nrx x Ndata
            # MRC
            if mimo_mode == 2:
                try:
                    x_c = np.linalg.inv(h_c.conj().T @ h_c) @ h_c.conj().T @ r_c
                except:
                    x_c = h_c.conj().T @ r_c
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

            eq_data_local[:, sc_idx, :] = x_c

        return eq_data_local, rho_su_local, R_hh_local

    eq_data, rho_su, R_hh = _equalize_with_channel(h_ls_all)

    # Turbo receiver: decision-directed channel re-estimation and second-pass equalization
    if turbo_enable and mod_dict_data['num_bit'] > 1:
        const_obj = mod_dict_data.get('constellation', None)
        if const_obj is None:
            const_obj = sn.mapping.Constellation("qam", inPar.num_bits_sym, trainable=False)
        const_points = const_obj.points
        if hasattr(const_points, "numpy"):
            const_points = const_points.numpy()

        turbo_iters_use = max(1, int(turbo_iters))
        for _ in range(turbo_iters_use):
            x_hat = np.zeros_like(eq_data)
            for tx_idx in range(iNtx):
                x_hat[tx_idx, :, :] = _hard_quantize_to_constellation(eq_data[tx_idx, :, :], const_points)

            h_ls_data = np.zeros_like(h_ls_all)
            for sc_idx in range(inPar.N_sc_use):
                X = x_hat[:, sc_idx, :]  # iNtx x Ndata
                R = rec_sym_data_all[:, sc_idx, :]  # num_rx x Ndata
                XXH = X @ X.conj().T
                eps = np.finfo(np.float32).eps * (1.0 + np.trace(XXH).real)
                H = R @ X.conj().T @ np.linalg.inv(XXH + eps * np.eye(iNtx, dtype=np.complex64))
                h_ls_data[:, :, sc_idx] = H.T

            # Denoise/smooth via channel_estimation on full grid
            for tx_idx in range(iNtx):
                for rx_idx in range(num_rx):
                    if h_ls_pilot_full is not None:
                        w = float(np.clip(turbo_pilot_weight, 0.0, 1.0))
                        h_ls_pilot_use = np.nan_to_num(h_ls_pilot_full[tx_idx, rx_idx, :], nan=0.0)
                        h_ls_mix = w * h_ls_pilot_use + (1.0 - w) * h_ls_data[tx_idx, rx_idx, :]
                    else:
                        h_ls_mix = h_ls_data[tx_idx, rx_idx, :]
                    h_ls_all[tx_idx, rx_idx, :] = channel_estimation(
                        h_ls_mix,
                        inPar.CP_len,
                        inPar.N_fft,
                        1,
                        sigma_arr[rx_idx],
                        ce_mode,
                        dc_mask_half_width,
                        dc_adaptive_threshold,
                        n_taps_esprit,
                    )

            eq_data, rho_su, R_hh = _equalize_with_channel(h_ls_all)

    # equalization points are ready
    # calculate EVM

    #a = mod_data_tx[0]
    #b = eq_data[0, :, :]
    #c = 10.0 * np.log10(1.0 / np.abs(a - b) ** 2)
    #print(f'EVMsym={np.mean(c, axis=0)}')

    if False:
        plt.imshow(np.abs(c))
        plt.show()

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
            bit_arr, bit_arr_uncode = demodulate(inPar.T_prec.conj().T @ eq_data[tx_idx, :], inPar.N_sc_use, mod_dict_data, inPar)
        else:
            bit_arr, bit_arr_uncode = demodulate(eq_data[tx_idx, :], inPar.N_sc_use, mod_dict_data, inPar)

        # unwrap data to bitstream
        bit_arr = np.reshape(bit_arr, newshape=(bit_arr.shape[0] * bit_arr.shape[1], 1), order='F') # Nbit_total x 1

        ber = get_ber(bite_stream_tx[tx_idx], bit_arr, inPar.N_sc_use)

        if inPar.do_fec:
            ber_uncode = get_ber(bite_stream_tx_uncode[tx_idx], bit_arr_uncode, inPar.N_sc_use)
            ber_arr.append(ber_uncode)
        else:
            ber_arr.append(ber)


        #SNR_est = estimate_SNR(pilot_freq, rec_sym_pilot_all, N_sc_use)

    if return_channel_for_plot:
        return ber_arr, SNR_guard, rho_avg_plot, evm_arr, R_hh, h_ls_pilot_full, h_ls_all, ce_mode
    return ber_arr, SNR_guard, rho_avg_plot, evm_arr, R_hh