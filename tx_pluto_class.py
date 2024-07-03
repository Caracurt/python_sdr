import adi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy.matlib

class SDR_cfg:
    def __init__(self, fc, fs, delta_f, frac_guard, num_bits_sym_pilot, num_bits_sym_data):

        self.fc = fc
        self.fs = fs
        self.delta_f = delta_f
        self.frac_guard = frac_guard
        self.num_bits_pilot = num_bits_sym_pilot
        self.num_bits_data = num_bits_sym_data

    def generate_tx_packet(self):
        rng = np.random.RandomState(123)

        tti_len = 1024
        num_samps = tti_len * 20

        # OFDM sig parameters
        delta_t = 1 / self.fs

        delta_f = self.delta_f
        frac_guard = self.frac_guard

        N_sc_av = int(np.fix(self.fs / delta_f))
        N_fft = int(2 ** (np.fix(np.log2(N_sc_av))))
        N_sc_use = int(np.fix(N_fft * frac_guard))
        guard_length = int(np.fix(0.5 * N_fft))

        self.N_fft = N_fft
        self.N_sc_use = N_sc_use
        self.guard_length = guard_length

        CP_len = int(np.fix(N_fft * 0.2))

        do_dc_offset = 0

        BlockSize = N_sc_use *  self.num_bits_pilot
        BlockSize_d = N_sc_use * self.num_bits_data

        N_repeat = 10
        num_rep_pilot = 1
        signal_scale = 1.0

        # Generate sync preambule
        preamble = 1 - 2 * rng.randint(0, 2, size=(int(N_fft / 2), 1))
        preamble = np.complex64(preamble)
        preamble_full = np.matlib.repmat(preamble, 2, 1)
        preamble_full_cp = np.concatenate((preamble_full[-CP_len:], preamble_full))

        pream_len = preamble.shape[0]

        # Generate pilot
        data_stream = rng.randint(0, 2, size=(int(BlockSize), 1))

        mod_sym_pilot = 1.0 - 2.0 * np.complex64(data_stream) # BPSK mod
        self.mod_sym_pilot = mod_sym_pilot

        tx_ofdm_sym = np.zeros((N_fft, 1), dtype=np.complex64);
        tx_ofdm_sym[0 + do_dc_offset:int(N_sc_use / 2) + do_dc_offset] = mod_sym_pilot[int(N_sc_use / 2):];
        tx_ofdm_sym[-int(N_sc_use / 2):] = mod_sym_pilot[0:int(N_sc_use / 2)];

        # time_ofdm_sym_pilot = ifft(tx_ofdm_sym) * sqrt(N_fft) * signal_scale;
        # time_ofdm_sym_cp_pilot = [time_ofdm_sym_pilot(end - CP_len + 1:end); time_ofdm_sym_pilot_rep];

        time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_sym, N_fft, 0, norm="ortho")
        time_ofdm_sym_cp_pilot = np.concatenate((time_ofdm_sym_pilot[-CP_len:], time_ofdm_sym_pilot))

        # data frame
        # data_stream = randi([0, 1], BlockSize_d, 1);
        data_stream = rng.randint(0, 2, size=(BlockSize_d, 1))
        self.data_stream = data_stream

        mod_sym = 1 - 2 * data_stream
        mod_sym = np.complex64(mod_sym)
        self.mod_sym = mod_sym

        # tx_ofdm_sym = zeros(N_fft, 1);
        tx_ofdm_sym = np.zeros((N_fft, 1), dtype=np.complex64)
        tx_ofdm_sym[0 + do_dc_offset:int(N_sc_use / 2) + do_dc_offset] = mod_sym[int(N_sc_use / 2):]
        tx_ofdm_sym[-int(N_sc_use / 2):] = mod_sym[0:int(N_sc_use / 2)]

        time_ofdm_sym = np.fft.ifft(tx_ofdm_sym, N_fft, 0, norm="ortho")

        time_ofdm_sym_cp = np.concatenate((time_ofdm_sym[-CP_len:], time_ofdm_sym))

        guard_frame = np.zeros((guard_length, 1), dtype=np.complex64)

        # full_frame = [preamble_full_cp; time_ofdm_sym_cp_pilot; time_ofdm_sym_cp; guard_frame];
        full_frame = np.concatenate((preamble_full_cp, time_ofdm_sym_cp_pilot, time_ofdm_sym_cp, guard_frame))

        frame_len = full_frame.shape[0]
        self.frame_len = frame_len
        self.full_frame = full_frame

        print(f'FrameLen={frame_len}')

        num_rep = 3
        full_frame_rep = np.matlib.repmat(full_frame, num_rep, 1)

        FrameSize = full_frame_rep.shape[0]
        sig_len = full_frame_rep.shape[0]

        self.frame_len_rep = sig_len
        self.full_frame_rep = full_frame_rep