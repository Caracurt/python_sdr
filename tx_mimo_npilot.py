import adi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import numpy.matlib

import tensorflow as tf
import sionna as sn
import json
from system_tx import SysParUL


def init_tx_dict():

    with open('inPar_default.json', 'r') as f:
        inPar_dict = json.load(f)

    inPar = SysParUL(inPar_dict)

    return inPar


def create_preamble(N_fft, CP_len, N_repeat = 2, inPar : SysParUL = None):

    if inPar != None:
        n_factor = inPar.pream_n_fact # increase length of preambule
    else:
        n_factor = 1

    preamble = 1 - 2 * np.random.randint(0, 2, size=(int(N_fft/2) * n_factor, 1))
    preamble = np.complex64(preamble)
    preamble_full = np.tile(preamble, (N_repeat, 1))

    #preamble_full_cp = np.concatenate((preamble_full[-CP_len:], preamble_full))
    preamble_full_cp = np.concatenate((preamble_full[-CP_len:] * 1.0, preamble_full)) # preambule without CP

    return preamble_full_cp, preamble

def create_data(N_sc, N_fft, CP_len, mod_dict_data, do_sc_fdm, T_prec, inPar : SysParUL, dc_offset = False, return_freq_data = False, return_data = False, comb_start=0, comb_step=1):

    if mod_dict_data['num_bit'] == 1:
        # pilot tx
        data_stream = (1.0 / np.sqrt(2)) * ( ( 1 - 2 * np.random.randint(0, 2, size=(N_sc, 1)) ) + 1j * (1 - 2 * np.random.randint(0, 2, size=(N_sc, 1))))
        data_stream_uncode = data_stream
        mod_sym_pilot_whole = np.complex64(data_stream)

        # SC-FDM for pilot
        if do_sc_fdm:
            mod_sym_pilot_whole = T_prec @ mod_sym_pilot_whole

        mod_sym_pilot = np.zeros_like(mod_sym_pilot_whole)
        mod_sym_pilot[comb_start::comb_step, :] = mod_sym_pilot_whole[comb_start::comb_step, :]

        Nsym_use = 1 # tmp

    else:
        # data tx
        binary_source = mod_dict_data['binary_source']
        mapper = mod_dict_data['mapper']
        encoder = mod_dict_data['encoder']
        decoder = mod_dict_data['decoder']
        info_total = mod_dict_data['info_total']
        bits_total = mod_dict_data['bits_total']



        Nsym_use = inPar.Ndata

        if not inPar.do_fec:
            bits = binary_source([1, N_sc * mod_dict_data['num_bit'] * inPar.Ndata])
            bits_uncode = bits
        else:
            bits_uncode = binary_source([1, info_total])
            bits = encoder(bits_uncode)

            # check decoder
            #llr = 2.0 * bits.numpy() - 1.0
            #llr = tf.convert_to_tensor(llr)
            #dec_bits = decoder(llr)



        data_stream = bits.numpy().T
        data_stream_uncode = bits_uncode.numpy().T

        mod_sym_pilot = mapper(bits)

        mod_sym_pilot = mod_sym_pilot.numpy().T

        # debug
        #mod_sym_pilot = np.ones_like(mod_sym_pilot)


        mod_sym_pilot = np.reshape(mod_sym_pilot, (N_sc, inPar.Ndata), order='F')

        # SC-FDM
        if do_sc_fdm:
            mod_sym_pilot = T_prec @ mod_sym_pilot

    tx_ofdm_sym = np.zeros((N_fft, Nsym_use), dtype = np.complex64)
    dc = int(dc_offset)
    tx_ofdm_sym[dc : N_sc//2 + dc, :] = mod_sym_pilot[ N_sc//2: , :]
    tx_ofdm_sym[-N_sc//2: , :] = mod_sym_pilot[ 0 : N_sc//2, :]

    time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_sym, axis = 0, norm = 'ortho')

    #time_ofdm_sym_cp_pilot = np.concatenate( (time_ofdm_sym_pilot[-CP_len:], time_ofdm_sym_pilot) )
    time_ofdm_sym_cp_pilot = np.vstack((time_ofdm_sym_pilot[-CP_len:, :], time_ofdm_sym_pilot[:, :]))


    if return_freq_data:
        return time_ofdm_sym_cp_pilot, mod_sym_pilot, data_stream, data_stream_uncode
    else:
        return time_ofdm_sym_cp_pilot, mod_sym_pilot, data_stream, data_stream_uncode


def create_data_frame(inPar : SysParUL):
    np.random.seed(123)
    tf.random.set_seed(123)

    # preambule is the same for both channels
    preamble, preamble_core = create_preamble(inPar.N_fft, inPar.CP_len, 2, inPar)

    binary_source = sn.utils.BinarySource()

    if inPar.num_bits_sym > 1:
        constellation = sn.mapping.Constellation("qam", inPar.num_bits_sym, trainable=True)
        mapper = sn.mapping.Mapper(constellation=constellation)
        demapper = sn.mapping.Demapper("maxlog", constellation=constellation)

        # fec params
        if inPar.do_fec:
            bits_total = inPar.Ndata * inPar.num_bits_sym * inPar.N_sc_use
            info_total = int(bits_total * inPar.CR)
            encoder = sn.fec.ldpc.LDPC5GEncoder(info_total, bits_total)
            decoder = sn.fec.ldpc.LDPC5GDecoder(encoder, cn_type="minsum", hard_out=True, num_iter=8)
        else:
            info_total = inPar.Ndata * inPar.num_bits_sym * inPar.N_sc_use
            bits_total = inPar.Ndata * inPar.num_bits_sym * inPar.N_sc_use
            encoder = []
            decoder = []

    else:
        info_total = inPar.Ndata * inPar.num_bits_sym * inPar.N_sc_use
        bits_total = inPar.Ndata * inPar.num_bits_sym * inPar.N_sc_use
        mapper = list()
        demapper = list()
        encoder = []
        decoder = []

    pilot_tx = list()
    data_tx = list()
    repeated_frame_tx = []
    bite_stream_tx = list()
    bite_stream_tx_uncode = list()

    # init params
    mod_dict_data = dict()
    mod_dict_data['binary_source'] = binary_source
    mod_dict_data['mapper'] = mapper
    mod_dict_data['demapper'] = demapper
    mod_dict_data['num_bit'] = inPar.num_bits_sym
    mod_dict_data['encoder'] = encoder
    mod_dict_data['decoder'] = decoder
    mod_dict_data['info_total'] = info_total
    mod_dict_data['bits_total'] = bits_total

    # mod data tx output
    mod_data_tx = list() # per Tx , complex array Nsc_avg x Ndatas

    for tx_idx in range(inPar.Ntx):


        # parameters for pilto generation
        mod_dict_pilot = dict()
        mod_dict_pilot['num_bit'] = 1

        # simple scheme for 2layer multiplex
        comb_start = tx_idx
        comb_step = inPar.Ntx


        pilot, pilot_freq, _, _ = create_data(inPar.N_sc_use, inPar.N_fft, inPar.CP_len, mod_dict_pilot, inPar.do_sc_fdm,
                                           inPar.T_prec, inPar,False, True,
                                           False, comb_start, comb_step)

        pilot_tx.append(pilot_freq)


        data, mod_data, bite_stream, bite_stream_uncode = create_data(inPar.N_sc_use, inPar.N_fft, inPar.CP_len, mod_dict_data, inPar.do_sc_fdm,
                                 inPar.T_prec, inPar)
        data_tx.append(data)
        bite_stream_tx.append(bite_stream)
        bite_stream_tx_uncode.append(bite_stream_uncode)
        mod_data_tx.append(mod_data)

        guard = np.zeros((inPar.guard_length, 1), dtype=np.complex64)

        pilot_rep = np.tile(pilot, reps=(inPar.pilot_repeat, 1))

        # unwrap data
        data_unwrap = np.reshape(data, newshape=(data.shape[0] * data.shape[1], 1), order='F')

        frame = np.concatenate((preamble, pilot_rep, data_unwrap, guard))

        repeated_frame = np.tile(frame, reps=(2, 1)) # could be 2 to avoid edge effects during transmission

        if len(repeated_frame_tx) == 0:
            repeated_frame_tx = repeated_frame
        else:
            repeated_frame_tx = np.hstack((repeated_frame_tx, repeated_frame))

    frame_len = len(frame)
    #preamble_len = inPar.N_fft // 2
    preamble_len = len(preamble_core)

    return repeated_frame_tx, repeated_frame, frame_len, preamble_len, preamble_core, mod_dict_data, pilot_tx, bite_stream_tx, mod_data_tx, bite_stream_tx_uncode

def main():

    # fetch config
    inPar = init_tx_dict()

    # create tx frame
    (repeated_frame_tx, repeated_frame, frame_len, preamble_len, preamble_core,
     mod_dict_data, pilot_tx, bite_stream_tx, mod_data, bite_stream_tx_uncode)\
        = create_data_frame(inPar)

    # actual transmission
    # Ntx= 1 (FrameLen,)
    # Ntx= 2 (Ntx, FrameLen)
    FrameSize = repeated_frame_tx.shape[0] # repeated_frame_tx is FrameLen x Ntx

    if inPar.Ntx > 1:
        repeated_frame_tx = repeated_frame_tx.T
    else:
        repeated_frame_tx = repeated_frame[:, 0]

    tx_gain0 = 0

    if not inPar.dummyTx:
        #sdr = adi.Pluto('ip:192.168.3.3') # interfere cfg
        if inPar.device_type !='ANTsdr':
            sdr = adi.Pluto('ip:192.168.1.1')
            #sdr = adi.ad9363(uri='ip:192.168.1.1')

            sdr.gain_control_mode_chan0 = 'manual'
            sdr.rx_hardwaregain_chan0 = 70.0  # dB
            sdr.rx_lo = int(inPar.center_freq)
            sdr.sample_rate = int(inPar.sample_rate)
            sdr.rx_rf_bandwidth = int(inPar.sample_rate)  # filter width, just set it to the same as sample rate for now
            sdr.rx_buffer_size = FrameSize

            '''Configure Tx properties'''
            sdr.tx_rf_bandwidth = int(inPar.sample_rate)
            sdr.tx_lo = int(inPar.center_freq)
            sdr.tx_cyclic_buffer = True
            sdr.tx_hardwaregain_chan0 = int(tx_gain0)

            sdr.tx_buffer_size = FrameSize
            sdr.tx(repeated_frame[:, 0] * 10024.0)
        else:
            # adi setup
            sdr = adi.ad9363(uri='ip:192.168.1.1')
            samp_rate = inPar.sample_rate  # must be <=30.72 MHz if both channels are enabled
            num_samps = len(repeated_frame)  # number of samples per buffer.  Can be different for Rx and Tx
            rx_lo = int(inPar.center_freq)
            rx_mode = "slow_attack"  # can be "manual" or "slow_attack"
            rx_gain0 = 70
            rx_gain1 = 70
            tx_lo = rx_lo
            tx_gain0 = 0
            tx_gain1 = 0


            sdr.tx_enabled_channels = list(range(inPar.Ntx))
            #sdr.rx_enabled_channels = [0]
            sdr.sample_rate = int(inPar.sample_rate)
            sdr.rx_lo = int(rx_lo)
            sdr.gain_control_mode = rx_mode
            sdr.rx_hardwaregain_chan0 = int(rx_gain0)
            sdr.rx_hardwaregain_chan1 = int(rx_gain1)
            sdr.rx_buffer_size = int(num_samps)
            # end adi setup

            sdr.tx_rf_bandwidth = int(inPar.sample_rate)
            sdr.tx_lo = int(inPar.center_freq)
            sdr.tx_cyclic_buffer = True
            sdr.tx_hardwaregain_chan0 = int(tx_gain0)
            sdr.tx_hardwaregain_chan1 = int(tx_gain1)

            sdr.tx_buffer_size = FrameSize

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

if __name__ == '__main__':
    main()