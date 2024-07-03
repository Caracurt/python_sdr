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

sample_rate = 5e6
center_freq = 1.5e9

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

do_cfo_corr = 1
do_ce = 1
dc_offset = 0


def create_preamble(N_fft, CP_len, N_repeat=2):
    preamble = 1 - 2 * np.random.randint(0, 2, size=(int(N_fft / 2), 1))
    preamble = np.complex64(preamble)
    preamble_full = np.tile(preamble, (N_repeat, 1))
    preamble_full_cp = np.concatenate((preamble_full[-CP_len:], preamble_full))
    return preamble_full_cp


def create_data(N_sc, N_fft, CP_len, mod_dict_data : dict, dc_offset=False, aditional_return=None, ):

    if mod_dict_data['num_bit'] == 1:
        data_stream = np.random.randint(0, 2, size=(N_sc, 1))
        data_stream_mod = 1 - 2 * data_stream
        mod_sym_pilot = np.complex64(data_stream_mod)
    else:
        binary_source = mod_dict_data['binary_source']
        mapper = mod_dict_data['mapper']

        bits = binary_source([1, N_sc * mod_dict_data['num_bit']])

        data_stream = bits.numpy().T

        mod_sym_pilot = mapper(bits)

        mod_sym_pilot = mod_sym_pilot.numpy().T




    tx_ofdm_sym = np.zeros((N_fft, 1), dtype=np.complex64)
    dc = int(dc_offset)
    tx_ofdm_sym[dc: N_sc // 2 + dc] = mod_sym_pilot[N_sc // 2:]
    tx_ofdm_sym[-N_sc // 2:] = mod_sym_pilot[0: N_sc // 2]

    time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_sym, axis=0, norm='ortho')
    time_ofdm_sym_cp_pilot = np.concatenate((time_ofdm_sym_pilot[-CP_len:], time_ofdm_sym_pilot))

    if aditional_return == 'mod_sym_pilot':
        return time_ofdm_sym_cp_pilot, mod_sym_pilot
    if aditional_return == 'data_stream':
        return time_ofdm_sym_cp_pilot, data_stream, mod_sym_pilot


#############################################################################
def find_edges(rx_sig, frame_len, preamble_len, CP_len, start_idx):
    corr_list = []
    for idx_search in range(start_idx, start_idx + frame_len):
        first_part = rx_sig[idx_search: idx_search + preamble_len]
        second_part = rx_sig[idx_search + preamble_len: idx_search + 2 * preamble_len]

        corr_now = np.dot(np.conj(first_part), second_part)
        corr_list.append(corr_now)

    corr_list_new = np.abs(np.array(corr_list))
    rel_idx = np.argmax(np.abs(corr_list_new), axis=0)
    idx_max = rel_idx + start_idx

    plt.figure(10)
    ax_corr = plt.gca()
    ax_corr.plot(range(len(corr_list)), np.array(corr_list_new), label='Orig')

    filter_cp = np.ones(CP_len) / np.sqrt(CP_len)
    corr_list_filt = np.convolve(filter_cp, corr_list_new, mode='same')

    rel_idx_filt = np.argmax(np.abs(corr_list_filt), axis=0)
    idx_max_filt = rel_idx_filt + start_idx
    idx_max_filt = idx_max_filt + int(CP_len/2)

    ax_corr.plot(range(len(corr_list)), corr_list_filt, label='filt')
    ax_corr.grid()
    plt.legend()

    #plt.show()

    if True:
        idx_max = idx_max_filt
        rel_idx = rel_idx_filt

    return idx_max, corr_list[rel_idx]


def cfo(frame_receive, corr_value, preamble_len):
    angle_cfo = np.angle(corr_value) / preamble_len
    #cfo_comp_sig = np.exp(1j * (-angle_cfo * np.arange(0, frame_len)))
    cfo_comp_sig = np.exp(1j * (-1.0*angle_cfo * np.arange(0, frame_len)))
    frame_receive = frame_receive * cfo_comp_sig
    return frame_receive


def baseband_freq_domian(pilot_freq, N_sc_use):
    rec_sym_pilot = np.zeros((N_sc_use, 1), dtype=np.complex64)
    rec_sym_pilot[int(N_sc_use / 2):, 0] = pilot_freq[0 + dc_offset:int(N_sc_use / 2) + dc_offset]
    rec_sym_pilot[0:int(N_sc_use / 2), 0] = pilot_freq[-int(N_sc_use / 2):]

    return rec_sym_pilot


def channel_estimation(h_ls, CP_len, N_fft, mimo):
    h_time = np.fft.ifft(h_ls, N_fft, 0, norm='ortho')
    ce_len = len(h_ls)

    W_spead = int(CP_len / 2)
    W_sync_err = int(CP_len)
    W_max = W_spead + W_sync_err
    W_min = W_sync_err

    eta_denoise = np.zeros_like(h_time)
    eta_denoise[-W_min:] = 1.0
    eta_denoise[0:W_max] = 1.0

    h_time_denoise = h_time * eta_denoise

    h_hw = np.fft.fft(h_time_denoise, N_fft, 0, norm='ortho')
    h_ls = h_hw[0:ce_len]

    display.clear_output(wait=True)

    if True:
        plt.figure(100)
        plt.plot(np.arange(0, N_fft), np.abs(h_time), label=f'mimo={mimo}')
        plt.plot(np.arange(0, N_fft), np.abs(h_time_denoise), label=f'mimo={mimo}')
        plt.title('Channel response time domain')
        plt.legend()
        plt.grid()

        #plt.show()



    return h_ls


def estimate_SNR(pilot_freq, rec_sym_pilot, N_sc_use):
    noise_arr = pilot_freq[int(N_sc_use / 2): -int(N_sc_use / 2)]
    sigma0_freq = np.real(np.dot(np.conj(noise_arr), noise_arr)) / len(noise_arr)
    Es_freq = np.real(np.dot(np.conj(rec_sym_pilot[:, 0]), rec_sym_pilot[:, 0])) / len(rec_sym_pilot[:, 0])
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


############################################################################
# Try Sionna
NUM_BITS_PER_SYMBOL = num_bits_sym

binary_source = sn.utils.BinarySource()
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=True) # The constellation is set to be trainable

# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("maxlog", constellation=constellation)

np.random.seed(123)
tf.random.set_seed(123)
preamble = create_preamble(N_fft, CP_len, 2)

mod_dict_pilot = dict()
mod_dict_pilot['num_bit'] = 1
pilot, pilot_freq_ = create_data(N_sc_use, N_fft, CP_len, mod_dict_pilot, aditional_return='mod_sym_pilot')

mod_dict_data = dict()
mod_dict_data['binary_source'] = binary_source
mod_dict_data['mapper'] = mapper
mod_dict_data['demapper'] = demapper
mod_dict_data['num_bit'] = num_bits_sym
data, data_stream, mod_sym_data = create_data(N_sc_use, N_fft, CP_len, mod_dict_data, aditional_return='data_stream')


guard = np.zeros((guard_length, 1), dtype=np.complex64)
frame = np.concatenate((preamble, pilot, data, guard))

repeated_frame = np.tile(frame, reps=(3, 1))

frame_len = len(frame)
preamble_len = N_fft // 2


for i in range(1):

    file_name = f'C:\\dev\\git_tutor\\python_sdr\\t_sdr_greencourt.mat'
    file_name = f'C:\\dev\\git_tutor\\python_sdr\\cfo_prob.mat'
    print(file_name)
    mat = scipy.io.loadmat(file_name)
    data = mat['data']
    data = np.array(data)


    print(data[0].shape)

    for mimo in [0, 1]:

        if mimo == 0:
            rx_sig = data[0] + data[1]
        else:
            data = np.array(data)
            R = data @ data.conj().T
            U1, s1, V1 = np.linalg.svd(R)
            U_main = U1[:,0:1]
            rx_sig = U_main.conj().T @ data
            rx_sig = rx_sig[0]


        #rx_sig = data
        rx_len = len(rx_sig)

        idx_max, corr_value = find_edges(rx_sig, frame_len, preamble_len, CP_len,  start_idx=0)
        frame_receive = rx_sig[idx_max: idx_max + frame_len]

        frame_receive = cfo(frame_receive, corr_value, preamble_len) if do_cfo_corr else frame_receive
        pilot_receive = frame_receive[N_fft + CP_len: 2 * N_fft + CP_len]

        pilot_freq = np.fft.fft(pilot_receive, N_fft, 0, norm="ortho")
        rec_sym_pilot = baseband_freq_domian(pilot_freq, N_sc_use)

        # apply CE
        h_ls = rec_sym_pilot[:, 0] / pilot_freq_[:, 0]
        h_ls = channel_estimation(h_ls, CP_len, N_fft, mimo) if do_ce else h_ls

        rec_data_sym = frame_receive[2 * (N_fft + CP_len): 2 * (N_fft + CP_len) + N_fft]
        rec_data_sym_freq = np.fft.fft(rec_data_sym, N_fft, 0, norm="ortho")
        rec_sym_data = baseband_freq_domian(rec_data_sym_freq, N_sc_use)
        eq_data = rec_sym_data[:, 0] / h_ls

        bit_arr = demodulate(eq_data, N_sc_use, mod_dict_data)
        ber = get_ber(data_stream, bit_arr, N_sc_use)
        SNR_est = estimate_SNR(pilot_freq, rec_sym_pilot, N_sc_use)

        plt.figure(figsize=(8, 8))
        plt.axes().set_aspect(1.0)
        plt.grid(True)
        plt.scatter(np.real(eq_data), np.imag(eq_data), label='Output')
        plt.scatter(np.real(mod_sym_data), np.imag(mod_sym_data), label='Input')
        plt.legend(fontsize=20);

        #print(data_stream[:10, 0])
        #print(bit_arr[:10])
        print(f'SNRest={SNR_est}')
        print(f'mimo={mimo} BER={ber}')

    #time.sleep(1)
    plt.show()